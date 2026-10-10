"""Phase 3 — Retrieval evaluation (SRS Phase 3 "retrieval evaluation", §14.1).

Measures retrieval quality on a frozen query set with ground-truth relevant
chunk ids (a small qrels), independently of any LLM. Metrics follow §14.1:

- Precision@K — fraction of the top-K that is relevant
- Recall@K    — fraction of all relevant chunks found in the top-K
- MRR         — reciprocal rank of the first relevant chunk
- nDCG@K      — rank-weighted gain with binary relevance
- Hit@K       — at least one relevant chunk in the top-K

Metrics are chunk-level (the unit the retriever returns) and macro-averaged
across answerable queries. Unanswerable queries (empty relevant set, e.g.
out-of-domain probes) are reported separately — standard IR metrics are
undefined for them, and abstention quality is a Phase 5 concern.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

# A retrieve function maps (query, k) -> ordered list of chunk ids.
RetrieveFn = Callable[[str, int], Sequence[str]]


@dataclass
class EvalQuery:
    """One frozen evaluation query with its relevant chunk ids."""

    qid: str
    query: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    category: str = "factual"
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.relevant_chunk_ids)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalQuery":
        return cls(
            qid=data.get("id") or data.get("qid") or "",
            query=data["query"],
            relevant_chunk_ids=list(data.get("relevant_chunk_ids") or []),
            category=data.get("category", "factual"),
            notes=data.get("notes", ""),
        )


def load_eval_set(path: str) -> list[EvalQuery]:
    """Load a frozen evaluation set from JSON (``{"queries": [...]}`` or a list)."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("queries", payload) if isinstance(payload, dict) else payload
    queries = [EvalQuery.from_dict(row) for row in rows]
    logger.info("Loaded %d evaluation queries from %s", len(queries), path)
    return queries


# ----------------------------------------------------------------------
# Metrics (binary relevance)
# ----------------------------------------------------------------------

def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for cid in top if cid in relevant) / float(len(top))


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    return sum(1 for cid in top if cid in relevant) / float(len(relevant))


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@K: DCG = Σ rel_i / log2(i + 1)."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(ranked[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / float(len(relevant))


def score_query(ranked: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    """All metrics for one query at cutoff ``k``."""
    return {
        f"precision@{k}": precision_at_k(ranked, relevant, k),
        f"recall@{k}": recall_at_k(ranked, relevant, k),
        f"hit@{k}": hit_rate_at_k(ranked, relevant, k),
        f"ndcg@{k}": ndcg_at_k(ranked, relevant, k),
        "mrr": reciprocal_rank(ranked, relevant),
        "map": average_precision(ranked, relevant),
    }


# ----------------------------------------------------------------------
# Evaluation runner
# ----------------------------------------------------------------------

@dataclass
class QueryResult:
    qid: str
    query: str
    category: str
    answerable: bool
    ranked_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    missed_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate(
    retrieve_fn: RetrieveFn,
    queries: Sequence[EvalQuery],
    k: int = 5,
    label: str = "",
) -> dict[str, Any]:
    """Run one retriever over the frozen set and return per-query + aggregate metrics.

    ``retrieve_fn(query, k)`` must return an ordered list of chunk ids.
    """
    per_query: list[QueryResult] = []
    for q in queries:
        ranked = list(retrieve_fn(q.query, k))
        relevant = set(q.relevant_chunk_ids)
        missed = [cid for cid in q.relevant_chunk_ids if cid not in ranked[:k]]
        result = QueryResult(
            qid=q.qid,
            query=q.query,
            category=q.category,
            answerable=q.answerable,
            ranked_chunk_ids=ranked,
            relevant_chunk_ids=q.relevant_chunk_ids,
            metrics=score_query(ranked, relevant, k) if q.answerable else {},
            missed_chunk_ids=missed,
        )
        per_query.append(result)

    return {
        "label": label,
        "k": k,
        "num_queries": len(queries),
        "num_answerable": sum(1 for q in queries if q.answerable),
        "aggregate": aggregate(per_query),
        "by_category": aggregate_by_category(per_query),
        "per_query": [r.to_dict() for r in per_query],
    }


def aggregate(per_query: Sequence[QueryResult]) -> dict[str, float]:
    """Macro-average metrics over answerable queries."""
    scored = [r for r in per_query if r.answerable and r.metrics]
    if not scored:
        return {}
    keys = list(scored[0].metrics.keys())
    return {
        key: round(sum(r.metrics.get(key, 0.0) for r in scored) / len(scored), 4)
        for key in keys
    }


def aggregate_by_category(per_query: Sequence[QueryResult]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[QueryResult]] = {}
    for r in per_query:
        if r.answerable:
            buckets.setdefault(r.category, []).append(r)
    return {
        category: {"count": len(items), **aggregate(items)}
        for category, items in sorted(buckets.items())
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _metric_columns(k: int) -> list[str]:
    return [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", f"hit@{k}", "mrr", "map"]


def _metric_header(columns: list[str]) -> str:
    return "| " + " | ".join(
        c.split("@")[0].upper() + ("@" + c.split("@")[1] if "@" in c else "") for c in columns
    ) + " |"


def render_markdown(report: dict[str, Any], k: int) -> str:
    """Markdown for a single-mode evaluation report."""
    lines: list[str] = [f"### {report.get('label') or 'retrieval'}", ""]
    agg = report.get("aggregate", {})
    if agg:
        columns = _metric_columns(k)
        lines.append(_metric_header(columns))
        lines.append("|" + "---|" * len(columns))
        lines.append("| " + " | ".join(f"{agg.get(c, 0.0):.3f}" for c in columns) + " |")
    else:
        lines.append("_no answerable queries_")
    misses = [r for r in report.get("per_query", []) if r.get("missed_chunk_ids")]
    if report.get("num_answerable"):
        lines.append("")
        lines.append(f"Missed-relevant queries: {len(misses)}/{report['num_answerable']}")
    return "\n".join(lines)


def render_comparison(combined: dict[str, Any]) -> str:
    """Markdown for a multi-mode report (list under ``modes``)."""
    k = int(combined.get("k", 5))
    modes = combined.get("modes", [])
    columns = _metric_columns(k)
    lines = ["# Phase 3 Retrieval Evaluation", ""]
    lines.append(f"- Frozen set: `{combined.get('eval_set', '')}`")
    lines.append(f"- Cutoff K: {k} · candidate pool: {combined.get('candidate_k', '')} · fusion: {combined.get('fusion', '')}")
    if modes:
        lines.append(f"- Answerable queries: {modes[0].get('num_answerable', 0)} of {modes[0].get('num_queries', 0)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| mode | " + " | ".join(_metric_header(columns).strip("| ").split(" | ")) + " |")
    lines.append("|" + "---|" * (len(columns) + 1))
    for report in modes:
        agg = report.get("aggregate", {})
        lines.append(
            f"| {report.get('label', '')} | "
            + " | ".join(f"{agg.get(c, 0.0):.3f}" for c in columns)
            + " |"
        )
    baseline = next((r for r in modes if r.get("label") == "semantic"), None)
    hybrid = next((r for r in modes if str(r.get("label", "")).startswith("hybrid")), None)
    if baseline and hybrid:
        lines.append("")
        lines.append("## Baseline vs hybrid (delta)")
        lines.append("")
        lines.append("| metric | semantic | hybrid | delta |")
        lines.append("|---|---|---|---|")
        for c in (f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "mrr"):
            b = baseline["aggregate"].get(c, 0.0)
            h = hybrid["aggregate"].get(c, 0.0)
            lines.append(f"| {c} | {b:.3f} | {h:.3f} | {h - b:+.3f} |")
    lines.append("")
    lines.append("## By category")
    for report in modes:
        lines.append("")
        lines.append(f"### {report.get('label', '')}")
        lines.append("")
        lines.append("| category | count | " + " | ".join(_metric_header(columns).strip("| ").split(" | ")) + " |")
        lines.append("|" + "---|" * (len(columns) + 2))
        for category, stats in (report.get("by_category") or {}).items():
            lines.append(
                f"| {category} | {stats.get('count', 0)} | "
                + " | ".join(f"{stats.get(c, 0.0):.3f}" for c in columns)
                + " |"
            )
    return "\n".join(lines)


def write_report(report: dict[str, Any], json_path: str, md_path: str | None = None) -> None:
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Retrieval evaluation written to %s", json_path)
    if md_path:
        os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
        markdown = (
            render_comparison(report)
            if "modes" in report
            else render_markdown(report, int(report.get("k", 5)))
        )
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown + "\n")
