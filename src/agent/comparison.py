"""Phase 4 — Agent 4/6: Policy comparison and response (SRS FR-012, §9, §12).

Given evidence gathered per product and criterion, produce one answer that
compares the policies. The prompt enforces the SRS §9 recommendation structure
and the §26 principle: facts (stated in an excerpt) must be separated from
conclusions (the model's inference), missing criteria must be reported rather
than guessed, and a recommendation must not be presented as an absolute decision.

Evidence is **round-robined across products** before the context budget is
applied, so one policy cannot starve another of excerpts. The exact same ordered
list is used for both prompt numbering and citation resolution, so a ``[n]``
marker always points at the excerpt it was shown as.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from src.generation.generator import AnswerGenerator, GeneratedAnswer, is_refusal

logger = logging.getLogger(__name__)

MAX_EXCERPT_CHARS = 1200

COMPARISON_SYSTEM = (
    "You compare insurance policies using ONLY the numbered policy excerpts "
    "provided in the user message.\n"
    "Rules:\n"
    "1. Organise the answer by policy and cover the requested comparison criteria.\n"
    "2. Cite the excerpt number in square brackets after every factual statement, "
    "e.g. [1] or [2][3].\n"
    "3. Clearly separate FACTS (stated in an excerpt) from CONCLUSIONS (your own "
    "inference from those facts); label conclusions as such.\n"
    "4. If no excerpt covers a policy/criterion, say the provided documents do not "
    "state it. Never guess.\n"
    "5. Do not present a recommendation as an absolute decision; note limitations "
    "and missing information.\n"
    "6. If no excerpt supports the comparison, reply exactly: \"The policy documents "
    "provided do not contain this information.\""
)

COMPARISON_TEMPLATE = (
    "Policies to compare: {products}\n"
    "Comparison criteria: {criteria}\n\n"
    "Policy excerpts:\n{evidence}\n"
    "---\n"
    "Question: {query}\n"
    "Answer with [n] citations to the excerpts above."
)

EVIDENCE_TEMPLATE = "[{n}] Source: {label}\n{quote}"


class ComparisonGenerator:
    """Builds and cites a grounded multi-policy comparison."""

    def __init__(
        self,
        llm: Any,
        max_evidence_chars: int = 8000,
        max_excerpt_chars: int = MAX_EXCERPT_CHARS,
    ) -> None:
        self.llm = llm
        self.max_evidence_chars = max_evidence_chars
        self.max_excerpt_chars = max_excerpt_chars
        self._resolver = AnswerGenerator(llm, max_evidence_chars=max_evidence_chars)

    # ------------------------------------------------------------------

    @staticmethod
    def _ordered_evidence(chunks: Sequence[Any], products: Sequence[str]) -> list[Any]:
        """Interleave evidence round-robin across products."""
        by_product: dict[str, list[Any]] = {}
        for chunk in chunks:
            by_product.setdefault(getattr(chunk, "product", "") or "(unspecified)", []).append(chunk)
        ordered_products = [p for p in products if p in by_product]
        ordered_products += [p for p in by_product if p not in ordered_products]

        queues = {p: list(by_product[p]) for p in ordered_products}
        interleaved: list[Any] = []
        while any(queues[p] for p in ordered_products):
            for product in ordered_products:
                if queues[product]:
                    interleaved.append(queues[product].pop(0))
        return interleaved

    def _prepare(
        self,
        query: str,
        products: Sequence[str],
        criteria: Sequence[str],
        chunks: Sequence[Any],
    ) -> tuple[str, list[Any]]:
        """Return ``(prompt, used_chunks)`` — numbering matches ``used_chunks``."""
        ordered = self._ordered_evidence(chunks, products)
        blocks: list[str] = []
        used: list[Any] = []
        budget = self.max_evidence_chars
        marker = 0
        for chunk in ordered:
            marker += 1
            label = chunk.citation_label()
            quote = (getattr(chunk, "text", "") or "").strip()
            if len(quote) > self.max_excerpt_chars:
                quote = quote[: self.max_excerpt_chars].rstrip() + "…"
            block = EVIDENCE_TEMPLATE.format(n=marker, label=label, quote=quote)
            if budget - len(block) < 0 and blocks:
                logger.debug("Comparison evidence budget exhausted after %d chunks.", marker - 1)
                break
            budget -= len(block)
            blocks.append(block)
            used.append(chunk)

        prompt = COMPARISON_TEMPLATE.format(
            products=", ".join(products) or "(all provided)",
            criteria=", ".join(criteria) or "(as stated)",
            evidence="\n\n".join(blocks),
            query=query,
        )
        return prompt, used

    def build_prompt(
        self,
        query: str,
        products: Sequence[str],
        criteria: Sequence[str],
        chunks: Sequence[Any],
    ) -> str:
        """Numbered evidence interleaved across products, plus a criteria header."""
        return self._prepare(query, products, criteria, chunks)[0]

    def answer(
        self,
        query: str,
        products: Sequence[str],
        criteria: Sequence[str],
        chunks: Sequence[Any],
    ) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                query=query,
                answer="The policy documents provided do not contain this information.",
                citations=[],
                retrieved=[],
                model=self.llm.model,
                used_markers=[],
                abstained=True,
            )

        prompt, used = self._prepare(query, products, criteria, chunks)
        raw = self.llm.generate(prompt, system=COMPARISON_SYSTEM)
        # Resolve against the *same* ordering used to number the prompt.
        citations, markers = self._resolver.resolve_citations(raw, used)
        return GeneratedAnswer(
            query=query,
            answer=raw,
            citations=citations,
            retrieved=[
                {
                    "chunk_id": getattr(c, "chunk_id", ""),
                    "document_id": getattr(c, "document_id", ""),
                    "product": getattr(c, "product", ""),
                    "section": getattr(c, "section", ""),
                    "pages": f"{getattr(c, 'page_start', 0)}-{getattr(c, 'page_end', 0)}",
                }
                for c in used
            ],
            model=self.llm.model,
            used_markers=markers,
            abstained=is_refusal(raw),
        )
