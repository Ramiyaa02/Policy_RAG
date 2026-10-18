"""Phase 6 — Failure taxonomy (SRS §15, AC-012).

Every per-query evaluation failure is classified with one or more codes from
the SRS §15 taxonomy instead of being reported only as an overall failure
percentage.

Code definitions as implemented here (evaluator context):

- F1  Retrieval Failure            — answerable query with no relevant chunk
                                     among the evidence the pipeline retrieved.
- F2  Missing Relevant Evidence    — partial: only some of the relevant chunks
                                     made it into the retrieved evidence.
- F3  Incorrect Chunk Selection    — relevant chunk was retrieved but not
                                     selected as final evidence by the agent.
- F4  Query Classification Failure — answerable query classified
                                     ``out_of_domain`` (or type mismatch with
                                     the frozen-set category).
- F5  Agent Planning Failure       — plan produced no retrieval steps (and did
                                     not request clarification).
- F6  Reasoning Failure            — generation produced an answer with no
                                     extractable claims (degenerate output).
- F7  Unsupported Generation       — ≥1 fact claim in the raw generation is
                                     unsupported by the cited evidence.
- F8  Citation Failure             — a claim cites a marker that did not
                                     resolve to a retrieved chunk.
- F9  Grounding Verification Failure — a contradicted claim is present in the
                                     FINAL answer (verification missed it).
- F10 Incorrect Abstention         — answerable query with relevant evidence
                                     retrieved, yet the system abstained.
- F11 Unnecessary Abstention       — answerable query abstained with evidence
                                     present and no grounding failure forcing
                                     it (verifier stripped everything).
- F12 Product Comparison Failure   — comparison/recommendation query answered
                                     from evidence of fewer than 2 products.
- F13 Contradictory Evidence Handling Failure — contradicted evidence conflated
                                     with support (kept without a conflict note).
- F14 Out-of-Domain Detection Failure — OOD probe was answered instead of
                                     abstaining/refusing.

Success codes returned for corrected runs (kept out of failure counts):

- F15 GROUNDING_CORRECTED          — unsupported/contradicted claims were
                                     detected and removed before the response.
- F16 ABSTENTION_CORRECT           — OOD probe correctly abstained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

CATEGORY_TO_TYPE = {
    "factual": "policy_information",
    "coverage": "coverage",
    "eligibility": "eligibility",
    "exclusion": "exclusion",
    "waiting_period": "waiting_period",
    "comparison": "product_comparison",
}

# Type values the taxonomy considers "comparison-flavoured" for F12.
_COMPARISON_TYPES = {"product_comparison", "recommendation", "multi_step"}

# SRS §15 code list, in order, with short human labels.
CODES: dict[str, str] = {
    "F1": "retrieval_failure",
    "F2": "missing_relevant_evidence",
    "F3": "incorrect_chunk_selection",
    "F4": "query_classification_failure",
    "F5": "agent_planning_failure",
    "F6": "reasoning_failure",
    "F7": "unsupported_generation",
    "F8": "citation_failure",
    "F9": "grounding_verification_failure",
    "F10": "incorrect_abstention",
    "F11": "unnecessary_abstention",
    "F12": "product_comparison_failure",
    "F13": "contradictory_evidence_handling_failure",
    "F14": "out_of_domain_detection_failure",
    "F15": "grounding_corrected",
    "F16": "abstention_correct",
}


@dataclass
class FailureCode:
    """One taxonomy assignment with its evidence note."""

    code: str
    note: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "label": CODES.get(self.code, ""), "note": self.note}


@dataclass
class QueryFailureReport:
    """Per-query taxonomy result."""

    qid: str
    codes: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return any(c in CODES and not c.startswith(("F15", "F16")) for c in self.codes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "failed": self.failed,
            "codes": self.codes,
            "details": [{"code": c, "label": CODES.get(c, ""), "note": self.notes.get(c, "")} for c in self.codes],
        }


def classify_query(
    qid: str,
    category: str,
    answerable: bool,
    result: dict[str, Any],
) -> QueryFailureReport:
    """Classify one evaluated query run.

    ``result`` is the evaluator's per-query record; expected keys (all
    optional — missing keys simply skip that check):

    - ``query_type``: type the pipeline classified the query as.
    - ``abstained`` / ``clarification``: abstention outcome.
    - ``evidence_chunk_ids``: chunk ids of the evidence actually used.
    - ``retrieved_chunk_ids``: ordered chunk ids retrieved (pre-selection).
    - ``relevant_chunk_ids``: frozen-set ground truth (chunk level).
    - ``claims`` (int)/``supported``/``contradicted``/``insufficient``: counts.
    - ``unresolved_citations`` (int): citations whose [n] never resolved.
    - ``unsupported_stripped`` (int): unsupported claims removed from the answer.
    - ``contradicted_kept`` (int): contradicted claims left in the final answer.
    - ``final_answer``: what the user would see.
    """
    report = QueryFailureReport(qid=qid)
    relevant = set(result.get("relevant_chunk_ids") or [])
    evidence = set(result.get("evidence_chunk_ids") or [])
    retrieved = set(result.get("retrieval_top_chunk_ids") or result.get("evidence_chunk_ids") or [])
    abstained = bool(result.get("abstained"))
    query_type = str(result.get("query_type") or "")
    claims = int(result.get("claims") or 0)
    supported = int(result.get("supported") or 0)
    contradicted = int(result.get("contradicted") or 0)
    insufficient = int(result.get("insufficient") or 0)
    unresolved = int(result.get("unresolved_citations") or 0)
    stripped = int(result.get("unsupported_stripped") or 0)
    contradicted_kept = int(result.get("contradicted_kept") or 0)
    products_seen = set(result.get("evidence_products") or [])

    if answerable:
        hits = relevant & evidence
        if not hits:
            if relevant:
                report.codes.append("F1")
                report.notes["F1"] = "no relevant chunk in the evidence used"
            if relevant & retrieved:
                report.codes.append("F3")
                report.notes["F3"] = (
                    f"{len(relevant & retrieved)} relevant chunk(s) retrieved but not selected as evidence"
                )
        elif len(relevant) > len(hits):
            report.codes.append("F2")
            report.notes["F2"] = (
                f"{len(relevant) - len(hits)} of {len(relevant)} relevant chunk(s) missing from evidence"
            )

        if answerable and str(result.get("expected_category", category)) and query_type == "out_of_domain":
            report.codes.append("F4")
            report.notes["F4"] = f"answerable query (category={category}) classified out_of_domain"
        if answerable and not result.get("steps") and not abstained and result.get("plan_empty"):
            report.codes.append("F5")
            report.notes["F5"] = "plan produced no retrieval steps"

        if abstained:
            report.codes.append("F10")
            note = "abstained although relevant evidence was retrieved" if hits else "abstained although query is answerable"
            report.notes["F10"] = note
            if hits and contradicted == 0 and insufficient == 0 and claims == 0:
                report.codes.append("F11")
                report.notes["F11"] = "no verification pressure: evidence was there and nothing was stripped"
        if claims == 0 and (not abstained or result.get("degenerate")):
            # Reasoning failure: either the answer went out with nothing
            # verify-able, or the generated text was so degenerate that the
            # handler had to abstain. Recorded regardless of the final outcome.
            report.codes.append("F6")
            report.notes["F6"] = "generation produced no extractable (verify-able) claims"
        elif claims > 0 and (insufficient + contradicted) > 0 and not abstained:
            report.codes.append("F7")
            report.notes["F7"] = (
                f"{insufficient + contradicted} of {claims} fact claim(s) unsupported by the cited evidence"
            )
        if unresolved:
            report.codes.append("F8")
            report.notes["F8"] = f"{unresolved} citation marker(s) never resolved to a chunk"
        if contradicted_kept:
            report.codes.append("F9")
            report.notes["F9"] = f"{contradicted_kept} contradicted claim(s) kept in the final answer"
        if contradicted:
            report.codes.append("F13")
            report.notes["F13"] = f"{contradicted} claim(s) conflicted with the cited evidence"
        if category == "comparison" and query_type in _COMPARISON_TYPES and len(products_seen) < 2 and not abstained:
            report.codes.append("F12")
            report.notes["F12"] = f"comparison answered from {len(products_seen)} product(s)"
    else:
        if not abstained:
            report.codes.append("F14")
            report.notes["F14"] = "out-of-domain probe was answered instead of abstaining"
        else:
            report.codes.append("F16")
            report.notes["F16"] = "out-of-domain probe correctly abstained"

    if stripped:
        report.codes.append("F15")
        report.notes["F15"] = f"{stripped} unsupported claim(s) detected and removed before the response"

    # Dedupe, keep taxonomy order.
    order = list(CODES)
    report.codes = sorted(set(report.codes), key=lambda c: order.index(c))
    return report


def aggregate(reports: Sequence[QueryFailureReport]) -> dict[str, Any]:
    """Taxonomy-wide counts: how many queries carried each code."""
    counts = {code: 0 for code in CODES}
    for r in reports:
        for c in r.codes:
            counts[c] = counts.get(c, 0) + 1
    failed = sum(1 for r in reports if r.failed)
    return {
        "num_queries": len(reports),
        "num_failed": failed,
        "failure_rate": round(failed / len(reports), 4) if reports else 0.0,
        "codes": {c: n for c, n in counts.items() if n},
    }


def render_markdown(aggregate_: dict[str, Any]) -> str:
    lines = ["# Phase 6 — Failure taxonomy (SRS §15)", ""]
    a = aggregate_
    lines.append(f"- Queries: {a.get('num_queries', 0)} · classified failed: {a.get('num_failed', 0)} (rate {a.get('failure_rate', 0):.2%})")
    lines.append("")
    lines.append("| code | label | queries |")
    lines.append("|---|---|---|")
    for code, n in sorted(a.get("codes", {}).items(), key=lambda kv: int(kv[0][1:])):  # F1..F16 order
        lines.append(f"| {code} | {CODES.get(code, '')} | {n} |")
    lines.append("")
    lines.append(
        "_F15/F16 are positive outcomes (grounding correction detected unsupported claims, "
        "correct OOD abstention); they are recorded per query but excluded from the failure rate._"
    )
    return "\n".join(lines)
