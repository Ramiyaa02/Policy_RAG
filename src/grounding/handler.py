"""Phase 5 — Response shaping: unsupported claims, abstention, clarification
(SRS FR-016/FR-017/FR-018, O7/O8, §9).

Takes the raw generated answer plus Agent 5's per-claim verdicts and produces
the final user-facing response:

- **unsupported-claim handling (FR-016)** — fact claims that are not supported
  are cut out of the answer rather than presented as policy fact; what was
  removed is disclosed in a grounding note instead of silently hidden.
- **contradictory evidence (FR-017)** — a contradicted claim is not just
  dropped; the answer states that the cited evidence conflicts with it.
- **abstention (O8)** — when no fact claim survives verification, the handler
  abstains rather than shipping an answer made of unverified sentences.
- **clarification (FR-018)** — an abstention asks one targeted question toward
  the most likely missing piece (product, procedure or criterion), never a
  vague "please provide more information".
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .schema import Claim, ClaimVerification, VerifiedResponse, Verdict

logger = logging.getLogger(__name__)

ABSTENTION_PREFIX = (
    "I could not verify enough of the answer against the policy documents, "
    "so I will not risk presenting policy facts I cannot support."
)

MAX_NOTE_CLAIMS = 5
MAX_NOTE_CHARS = 160


class GroundingHandler:
    """Shapes the verified final response from verification results."""

    def __init__(
        self,
        strip_unsupported: bool = True,
        min_supported_rate: float = 0.5,
        max_note_claims: int = MAX_NOTE_CLAIMS,
    ) -> None:
        self.strip_unsupported = strip_unsupported
        self.min_supported_rate = min_supported_rate
        self.max_note_claims = max_note_claims

    # ------------------------------------------------------------------

    def build(
        self,
        raw_response: dict[str, Any],
        claims: Sequence[Claim],
        verifications: Sequence[ClaimVerification],
        query_type: str = "",
    ) -> VerifiedResponse:
        """Combine a raw generation result with its verification report."""
        answer = raw_response.get("answer", "") or ""
        supported = [
            v for v in verifications if v.verdict is Verdict.SUPPORTED
        ]
        unsupported = [
            v.claim
            for v in verifications
            if v.verdict is Verdict.INSUFFICIENT and v.claim.kind == "fact"
        ]
        contradicted = [
            v.claim for v in verifications if v.verdict is Verdict.CONTRADICTED
        ]
        fact_verdicts = [
            v
            for v in verifications
            if v.claim.kind == "fact" and v.verdict is not Verdict.NOT_VERIFIED
        ]
        supported_rate = (
            len(supported) / len(fact_verdicts) if fact_verdicts else 0.0
        )
        response = VerifiedResponse(
            query=raw_response.get("query", ""),
            query_type=query_type,
            method="verified",
            answer=answer,
            abstained=bool(raw_response.get("abstained", False)),
            citations=raw_response.get("citations", []),
            products=raw_response.get("products", []),
            criteria=raw_response.get("criteria", []),
            steps=raw_response.get("steps", []),
            evidence_count=raw_response.get("evidence_count", 0),
            model=raw_response.get("model", ""),
            claims=list(claims),
            verifications=list(verifications),
            supported_rate=round(supported_rate, 4),
            unsupported_claims=[c.text for c in unsupported],
            contradicted_claims=[c.text for c in contradicted],
        )

        if not claims and _is_refusal_answer(answer):
            # The generator already declined; keep its message as the abstention.
            response.abstained = True
            response.notes.append("Generator abstained: no evidence in the corpus.")
            response.final_answer = answer
            return response

        if self.strip_unsupported and (unsupported or contradicted):
            final = _strip_claims(answer, unsupported)
            final = _strip_claims(final, contradicted)
            response.final_answer = _collapse(final)

        if contradicted:
            response.notes.append(
                _conflict_note(contradicted)
            )
        if unsupported:
            response.notes.append(_unsupported_note(unsupported, self.max_note_claims))

        # O8 — abstain when nothing factual survived verification.
        abstain = bool(fact_verdicts) and (
            not supported
            or supported_rate < self.min_supported_rate
        )
        if abstain:
            response.abstained = True
            response.clarification = build_clarification(
                response.query, query_type, response.evidence_count
            )
            response.final_answer = f"{ABSTENTION_PREFIX} {response.clarification}"
        elif not response.final_answer:
            response.final_answer = answer
        return response


# ----------------------------------------------------------------------
# Answer rewriting
# ----------------------------------------------------------------------


def _strip_claims(answer: str, claims: Sequence[Claim]) -> str:
    """Remove the answer lines the given claims came from.

    Claims keep the line number they were extracted from, so removal is
    line-based: one bad sentence drops its whole line (a stray fragment left
    behind reads worse than the slightly broader removal).
    """
    if not claims or not answer:
        return answer
    drop = {c.line for c in claims if c.line}
    kept: list[str] = []
    for line_no, line in enumerate(answer.splitlines(), start=1):
        if line_no not in drop:
            kept.append(line)
    return "\n".join(kept)


def _collapse(text: str) -> str:
    """Drop empty lines created by removals and trim the result."""
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _conflict_note(contradicted: Sequence[Claim]) -> str:
    claim = contradicted[0].text
    if len(contradicted) == 1:
        return (
            "Note: the statement \""
            + _shorten(claim)
            + "\" conflicts with the cited policy evidence (FR-017) and was removed."
        )
    return (
        f"Note: {len(contradicted)} statements conflict with the cited policy "
        f"evidence, including \"{_shorten(claim)}\", and were removed."
    )


def _unsupported_note(unsupported: Sequence[Claim], max_notes: int) -> str:
    shown = [_shorten(c.text) for c in unsupported[:max_notes]]
    extra = len(unsupported) - len(shown)
    body = "; ".join(f"\"{s}\"" for s in shown)
    if extra > 0:
        body += f"; and {extra} more"
    return (
        "Note: the following statements could not be verified against the "
        f"policy documents and were removed: {body}."
    )


def _shorten(text: str, limit: int = MAX_NOTE_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _is_refusal_answer(answer: str) -> bool:
    lowered = (answer or "").lower()
    return (
        "do not contain this information" in lowered
        or "does not contain this information" in lowered
    )


# ----------------------------------------------------------------------
# Clarification (FR-018)
# ----------------------------------------------------------------------

_QUERY_TYPE_QUESTIONS = {
    "coverage": "Could you name the treatment or expense you want coverage checked for (e.g. cataract, maternity, room rent)?",
    "waiting_period": "Could you name the specific procedure or illness whose waiting period you need?",
    "exclusion": "Could you name the treatment, condition or activity you want checked for exclusion?",
    "eligibility": "Could you share the applicant's age (and the product, if you have one in mind)?",
    "policy_information": "Could you restate the question with the specific section or term you need (e.g. free look period, grace period)?",
    "multi_step": "Could you split the question so I can verify each part against the policy documents?",
}

COMPARISON_CLARIFICATION = (
    "Could you name which two or more products the comparison should cover?"
)


def build_clarification(query: str, query_type: str, evidence_count: int) -> str:
    """One targeted clarification question toward the missing information."""
    question = _QUERY_TYPE_QUESTIONS.get(query_type or "")
    if not question:
        question = (
            "Could you restate the question with the specific term, procedure or "
            "product it should be checked against?"
        )
    if evidence_count == 0:
        lead = "No matching policy evidence was retrieved for this question."
    else:
        lead = "No retrieved excerpt supports the key part of this question."
    return f"{lead} {question}"
