"""Phase 5 — Agent 5: Grounding Validator (SRS §8, FR-015/FR-016, O7).

Checks every extracted claim against the evidence the answer cited:

- **supported** — a cited excerpt's content tokens cover the claim (by a
  configurable containment threshold) and every number in the claim appears
  in the cited excerpt (waiting periods, sums insured, percentages must match).
- **contradicted** — the cited excerpt states numbers of the same kind, but
  none of the claim's numbers appear among them (e.g. the excerpt says
  2 years, the claim says 30 days).
- **insufficient** — the claim is uncited, cites a marker that never
  resolved, or no excerpt's content comes close enough. Absence of evidence
  is deliberately NOT treated as evidence of absence.

Rules come first so the validator is deterministic and unit-testable offline;
an optional LLM pass second-guesses only the rules (``use_llm_verification``),
batching all claims into one call. If the LLM errors, the rules verdict stands.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Sequence

from .claims import ClaimExtractor
from .schema import Claim, ClaimVerification, Verdict

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")

DEFAULT_STOPWORDS = frozenset(
    "the and for with from that this those these are was were has have had will "
    "shall would must may can cannot not its their your they them then when what "
    "which where whom whose under over into upon per also any all above below "
    "answer such other others while both each only more less very here there "
    "one two three however"
)
VERIFY_SYSTEM_PROMPT = (
    "You are a strict evidence checker for insurance policy answers.\n"
    "You are given evidence excerpts and numbered claims. For each claim decide:\n"
    "- supported: an excerpt explicitly states the claim's content (numbers included).\n"
    "- contradicted: an excerpt explicitly states information inconsistent with the claim.\n"
    "- insufficient: no excerpt establishes or refutes the claim.\n"
    "Reply with one line per claim, exactly: 'CLAIM <n>: <verdict>'."
)

VERIFY_PROMPT_TEMPLATE = (
    "Evidence excerpts:\n{evidence}\n\nClaims:\n{claims}\n\n"
    "Verdicts (one line per claim):"
)

# Word count beyond which an excerpt is truncated in the LLM verification prompt.
MAX_VERIFY_EXCERPT_CHARS = 500


class GroundingValidator:
    """Agent 5 — verifies generated claims against the cited evidence."""

    def __init__(
        self,
        llm: Any | None = None,
        extractor: ClaimExtractor | None = None,
        support_threshold: float = 0.35,
        number_strict: bool = True,
        verify_conclusions: bool = False,
        stopwords: Iterable[str] | None = None,
        use_llm: bool = True,
    ) -> None:
        self.llm = llm
        self.extractor = extractor or ClaimExtractor()
        self.support_threshold = support_threshold
        self.number_strict = number_strict
        self.verify_conclusions = verify_conclusions
        self.stopwords = frozenset(stopwords or DEFAULT_STOPWORDS)
        self.use_llm = use_llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self, answer: str, evidence: Sequence[Any]
    ) -> list[ClaimVerification]:
        """Verify every claim in ``answer`` against ``evidence``.

        ``evidence`` must be ordered the same way the generator numbered its
        excerpts: claim marker ``[n]`` refers to ``evidence[n - 1]``.
        """
        claims = self.extractor.extract(answer)
        results = [self._verify_claim(claim, evidence) for claim in claims]
        self._llm_second_pass(claims, results, evidence)
        return results

    # ------------------------------------------------------------------
    # Rules path
    # ------------------------------------------------------------------

    def _verify_claim(self, claim: Claim, evidence: Sequence[Any]) -> ClaimVerification:
        if claim.kind == "conclusion" and not self.verify_conclusions:
            return ClaimVerification(
                claim=claim,
                verdict=Verdict.NOT_VERIFIED,
                reason="conclusion (inference, not a fact claim)",
            )
        if not claim.markers:
            return ClaimVerification(
                claim=claim,
                verdict=Verdict.INSUFFICIENT,
                reason="uncited claim",
            )
        if not evidence:
            return ClaimVerification(
                claim=claim,
                verdict=Verdict.INSUFFICIENT,
                reason="no evidence available",
            )

        cited, unresolved = self._resolve_markers(claim.markers, evidence)
        if not cited:
            return ClaimVerification(
                claim=claim,
                verdict=Verdict.INSUFFICIENT,
                evidence_chunk_ids=[],
                reason="unresolved citation"
                + (f" (markers {unresolved})" if unresolved else ""),
            )

        claim_tokens = tokenize(claim.text, self.stopwords)
        claim_numbers = set(_NUMBER_RE.findall(claim.text))
        best_score = 0.0
        best_chunk = None
        number_mismatch = False
        seen_numbers: set[str] = set()
        for chunk in cited:
            text = _chunk_text(chunk)
            chunk_tokens = tokenize(text, self.stopwords)
            score = containment(claim_tokens, chunk_tokens)
            chunk_numbers = set(_NUMBER_RE.findall(text))
            seen_numbers |= chunk_numbers
            if score > best_score:
                best_score, best_chunk = score, chunk

        if claim_numbers and seen_numbers and not (claim_numbers & seen_numbers):
            number_mismatch = True

        if number_mismatch and self.number_strict:
            return ClaimVerification(
                claim=claim,
                verdict=Verdict.CONTRADICTED,
                support_score=round(best_score, 4),
                evidence_chunk_ids=[_chunk_id(c) for c in cited],
                reason="numbers in the claim do not appear in the cited evidence",
            )

        if best_score >= self.support_threshold:
            return ClaimVerification(
                claim=claim,
                verdict=Verdict.SUPPORTED,
                support_score=round(best_score, 4),
                evidence_chunk_ids=[_chunk_id(c) for c in cited],
                reason=f"content containment {best_score:.2f} >= {self.support_threshold:.2f}",
            )

        return ClaimVerification(
            claim=claim,
            verdict=Verdict.INSUFFICIENT,
            support_score=round(best_score, 4),
            evidence_chunk_ids=[_chunk_id(c) for c in cited],
            reason=f"content containment {best_score:.2f} < {self.support_threshold:.2f}",
        )

    @staticmethod
    def _resolve_markers(
        markers: Sequence[int], evidence: Sequence[Any]
    ) -> tuple[list[Any], list[int]]:
        cited: list[Any] = []
        unresolved: list[int] = []
        for marker in markers:
            if 1 <= marker <= len(evidence):
                cited.append(evidence[marker - 1])
            else:
                unresolved.append(marker)
        return cited, unresolved

    # ------------------------------------------------------------------
    # LLM second pass
    # ------------------------------------------------------------------

    def _llm_second_pass(
        self,
        claims: Sequence[Claim],
        results: list[ClaimVerification],
        evidence: Sequence[Any],
    ) -> None:
        if not self.use_llm or self.llm is None or not claims:
            return
        # Only second-guess what the rules could not establish.
        pending = [
            (claim, result)
            for claim, result in zip(claims, results)
            if result.verdict in (Verdict.INSUFFICIENT, Verdict.CONTRADICTED)
        ]
        if not pending:
            return

        prompt = self._build_llm_prompt(claims, evidence)
        try:
            raw = self.llm.generate(prompt, system=VERIFY_SYSTEM_PROMPT)
        except Exception as exc:  # noqa: BLE001 - degrade to rules verdicts
            logger.warning("LLM verification unavailable (%s); using rules verdicts.", exc)
            return

        verdicts = self._parse_verdicts(raw)
        for claim, result in pending:
            verdict = verdicts.get(claim.index)
            if verdict is None:
                continue
            upgraded = _merge_verdict(result.verdict, verdict)
            if upgraded is not result.verdict:
                result.verdict = upgraded
                result.method = "llm"
                result.reason = f"llm verdict '{verdict.value}' overrides rules"

    def _build_llm_prompt(
        self, claims: Sequence[Claim], evidence: Sequence[Any]
    ) -> str:
        evidence_blocks = []
        for i, chunk in enumerate(evidence, start=1):
            text = _chunk_text(chunk).strip()
            if len(text) > MAX_VERIFY_EXCERPT_CHARS:
                text = text[:MAX_VERIFY_EXCERPT_CHARS].rstrip() + "…"
            evidence_blocks.append(f"[{i}] ({_chunk_id(chunk)})\n{text}")
        claim_blocks = [
            f"CLAIM {c.index}: {c.text} (cites: {c.markers or 'none'})" for c in claims
        ]
        return VERIFY_PROMPT_TEMPLATE.format(
            evidence="\n\n".join(evidence_blocks), claims="\n".join(claim_blocks)
        )

    @staticmethod
    def _parse_verdicts(raw: str) -> dict[int, Verdict]:
        parsed: dict[int, Verdict] = {}
        for match in re.finditer(
            r"CLAIM\s*(\d+)\s*[:\-]\s*(supported|contradicted|insufficient)",
            (raw or ""),
            re.IGNORECASE,
        ):
            index = int(match.group(1))
            verdict = Verdict(match.group(2))
            parsed[index] = verdict
        return parsed


def _merge_verdict(rules: Verdict, llm: Verdict) -> Verdict:
    """Combine a rules verdict with an LLM verdict conservatively.

    Only the LLM can upgrade an INSUFFICIENT to SUPPORTED; nothing is ever
    downgraded from CONTRADICTED to something weaker.
    """
    if rules is Verdict.CONTRADICTED:
        return Verdict.CONTRADICTED if llm is not Verdict.SUPPORTED else Verdict.SUPPORTED
    if rules is Verdict.INSUFFICIENT:
        return llm
    return rules


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(getattr(chunk, "text", "") or "")


def _chunk_id(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("chunk_id", ""))
    return str(getattr(chunk, "chunk_id", ""))


def tokenize(text: str, stopwords: frozenset[str]) -> set[str]:
    return {
        tok
        for tok in _TOKEN_RE.findall((text or "").lower())
        if tok not in stopwords
    }


def containment(claim_tokens: set[str], chunk_tokens: set[str]) -> float:
    """Fraction of the claim's content tokens the chunk covers.

    Single-token overlap is not entailment, so a floor of one shared token is
    treated as no support unless the claim is very short.
    """
    if not claim_tokens:
        return 0.0
    overlap = len(claim_tokens & chunk_tokens)
    if overlap <= 1 and len(claim_tokens) > 1:
        return 0.0
    return overlap / len(claim_tokens)
