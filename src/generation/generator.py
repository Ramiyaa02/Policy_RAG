"""Phase 2 — Basic grounded generation with citations (SRS Phase 2).

The baseline answers strictly from retrieved policy excerpts:
- The prompt contains numbered evidence blocks built from RetrievedChunks.
- The model is instructed to cite evidence as [1], [2] and to say the corpus
  does not contain the answer when it does not (light abstention; the full
  grounding/abstention machinery is Phase 5).
- ``[n]`` markers in the answer are resolved into Citation objects carrying
  document, product, pages, section and clause ids, so every statement stays
  traceable to its source (FR-004 traceability).

Phase 2 STOP boundary — no query classification, planning, multi-step
retrieval, claim verification or comparison logic (Phases 3-5).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .llm import OllamaLLM

logger = logging.getLogger(__name__)

_MARKER = re.compile(r"\[(\d{1,2})\]")

SYSTEM_PROMPT = (
    "You are an insurance policy assistant. Answer ONLY from the numbered "
    "policy excerpts provided in the user message.\n"
    "Rules:\n"
    "1. Cite the excerpt number in square brackets after every factual "
    "statement, e.g. [1] or [2][3].\n"
    "2. If the excerpts do not contain the answer, reply exactly: "
    "\"The policy documents provided do not contain this information.\" "
    "Do not guess and do not use outside knowledge.\n"
    "3. Quote waiting periods, percentages and currency amounts exactly.\n"
    "4. Be concise; use short sentences or bullet points.\n"
    "5. Cite ONLY with the bracketed excerpt number, e.g. [1]. Never cite "
    "section numbers, roman numerals or any other style."
)

USER_PROMPT_TEMPLATE = (
    "Policy excerpts:\n{evidence}\n"
    "---\n"
    "Question: {query}\n"
    "Answer with [n] citations to the excerpts above."
)

EVIDENCE_TEMPLATE = (
    "[{n}] Source: {label}\n{quote}"
)

# Characters of excerpt text shown to the model per chunk.
MAX_EXCERPT_CHARS = 1200


@dataclass
class Citation:
    """A resolved [n] marker pointing at one retrieved chunk."""

    marker: int
    chunk_id: str
    document_id: str
    filename: str
    product: str
    insurer: str
    uin: str
    pages: str
    section: str
    clause_ids: list[str] = field(default_factory=list)
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedAnswer:
    """Answer plus resolved citations and retrieval context."""

    query: str
    answer: str
    citations: list[Citation]
    retrieved: list[dict[str, Any]]
    model: str
    used_markers: list[int]
    abstained: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved": self.retrieved,
            "model": self.model,
            "used_markers": self.used_markers,
            "abstained": self.abstained,
        }


class AnswerGenerator:
    """Grounded baseline generation with [n] citation resolution."""

    def __init__(
        self,
        llm: OllamaLLM,
        max_evidence_chars: int = 6000,
    ) -> None:
        self.llm = llm
        self.max_evidence_chars = max_evidence_chars

    # ------------------------------------------------------------------

    def build_prompt(
        self, query: str, retrieved: list[Any]
    ) -> tuple[str, list[str]]:
        """Return (user_prompt, evidence_labels). Label i <-> marker i+1."""
        evidence_parts: list[str] = []
        labels: list[str] = []
        budget = self.max_evidence_chars
        for i, chunk in enumerate(retrieved, start=1):
            label = chunk.citation_label()
            labels.append(label)
            quote = (chunk.text or "").strip()
            if len(quote) > MAX_EXCERPT_CHARS:
                quote = quote[:MAX_EXCERPT_CHARS].rstrip() + "…"
            block = EVIDENCE_TEMPLATE.format(n=i, label=label, quote=quote)
            if budget - len(block) < 0 and evidence_parts:
                logger.debug("Evidence budget exhausted after %d chunks.", i - 1)
                break
            budget -= len(block)
            evidence_parts.append(block)
        prompt = USER_PROMPT_TEMPLATE.format(
            evidence="\n\n".join(evidence_parts), query=query
        )
        return prompt, labels

    def answer(self, query: str, retrieved: list[Any]) -> GeneratedAnswer:
        """Generate an answer for ``query`` grounded in ``retrieved`` chunks."""
        if not retrieved:
            return GeneratedAnswer(
                query=query,
                answer="The policy documents provided do not contain this information.",
                citations=[],
                retrieved=[],
                model=self.llm.model,
                used_markers=[],
                abstained=True,
            )

        prompt, _labels = self.build_prompt(query, retrieved)
        raw = self.llm.generate(prompt, system=SYSTEM_PROMPT)

        citations, used_markers = self._resolve_citations(raw, retrieved)
        abstained = _is_refusal(raw)
        return GeneratedAnswer(
            query=query,
            answer=raw,
            citations=citations,
            retrieved=[
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "product": c.product,
                    "similarity": round(float(getattr(c, "similarity", 0.0)), 4),
                    "section": c.section,
                    "pages": f"{c.page_start}-{c.page_end}",
                }
                for c in retrieved
            ],
            model=self.llm.model,
            used_markers=used_markers,
            abstained=abstained,
        )

    # ------------------------------------------------------------------

    def resolve_citations(
        self, answer_text: str, retrieved: list[Any]
    ) -> tuple[list[Citation], list[int]]:
        """Public entry point for citation resolution.

        Exposed so the Phase 4 comparison generator can resolve ``[n]`` markers
        against a flat evidence list without duplicating the logic.
        """
        return self._resolve_citations(answer_text, retrieved)

    def _resolve_citations(
        self, answer_text: str, retrieved: list[Any]
    ) -> tuple[list[Citation], list[int]]:
        """Map [n] markers in the answer to Citation objects."""
        used = sorted({int(m.group(1)) for m in _MARKER.finditer(answer_text)})
        citations: list[Citation] = []
        for marker in used:
            if marker < 1 or marker > len(retrieved):
                continue  # hallucinated marker — logged, dropped
                # (kept out of citations; Phase 5 handles unsupported claims)
            chunk = retrieved[marker - 1]
            pages = (
                f"p. {chunk.page_start}"
                if chunk.page_start == chunk.page_end
                else f"pp. {chunk.page_start}-{chunk.page_end}"
            )
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    product=chunk.product,
                    insurer=chunk.insurer,
                    uin=chunk.uin,
                    pages=pages,
                    section=chunk.section or "",
                    clause_ids=list(chunk.clause_ids),
                    excerpt=(chunk.text or "")[:300],
                )
            )
        if used and not citations:
            logger.warning("Answer cited markers %s but none resolved.", used)
        return citations, used


_REFUSAL_PATTERNS = (
    "do not contain this information",
    "does not contain this information",
    "not mentioned in the",
    "no information about",
)


def is_refusal(text: str) -> bool:
    """True when an answer declines to answer for lack of evidence."""
    lowered = (text or "").lower()
    return any(p in lowered for p in _REFUSAL_PATTERNS)


# Backwards-compatible private alias (used by the Phase 2 tests).
_is_refusal = is_refusal
