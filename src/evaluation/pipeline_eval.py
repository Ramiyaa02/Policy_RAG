"""Phase 6 — Pipeline evaluation (SRS Phase 6, §14, §15, AC-010..AC-012).

Runs whole pipelines (not retrievers) over the frozen query set:

- ``full``     — the Phase 5 verified pipeline (analyze → plan → multi-step
                 hybrid retrieval → generate → verify → shape).
- ``baseline`` — the Phase 2 baseline (semantic-only retrieval → direct
                 grounded generation; no agent, no verification).

Because the study is single-annotator, answer correctness (§14.2) uses the
frozen-set chunk qrels as a proxy: a non-abstaining answer backed by ≥1
relevant chunk counts as ``answered_correct``. Grounding (§14.5) and
unsupported-claim rate (§14.4) use Agent 5's rules validator as an offline
scoring instrument for BOTH systems — for the baseline it is deliberately not
part of the system, just the measurement.

Every query also gets latency (§14.7) and token usage (NFR-005, local models
cost ≈ $0; recorded as tokens) and a §15 failure-taxonomy classification.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable, Sequence

from .failure_taxonomy import QueryFailureReport, aggregate as taxonomy_aggregate, classify_query

logger = logging.getLogger(__name__)

# A run_fn turns a query into a normalized result dict. Both the baseline and
# the full pipeline runners in run_phase6.py produce this shape.
RunFn = Callable[[str], dict[str, Any]]


@dataclass
class PipelineQueryResult:
    """One evaluated query through one pipeline."""

    label: str
    qid: str
    query: str
    category: str
    answerable: bool
    # Behaviour
    query_type: str = ""
    abstained: bool = False
    clarification: str | None = None
    final_answer: str = ""
    # Grounding
    claims: int = 0
    supported: int = 0
    contradicted: int = 0
    insufficient: int = 0
    supported_rate: float = 0.0
    unsupported_stripped: int = 0
    contradicted_kept: int = 0
    unresolved_citations: int = 0
    # Evidence / retrieval
    relevant_chunk_ids: list[str] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)
    retrieval_top_chunk_ids: list[str] = field(default_factory=list)
    evidence_products: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    # Operate
    latency_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    # Derived
    answered_with_relevant_evidence: bool = False
    failure: QueryFailureReport | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure"] = self.failure.to_dict() if self.failure else None
        return d


def _diff_usage(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        "calls": after.get("calls", 0) - before.get("calls", 0),
        "prompt_tokens": after.get("prompt_tokens", 0) - before.get("prompt_tokens", 0),
        "completion_tokens": after.get("completion_tokens", 0) - before.get("completion_tokens", 0),
    }


def mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


class PipelineEvaluator:
    """Evaluate one pipeline (``run_fn``) over the frozen query set.

    Pass ``checkpoint_path`` to make long LLM-in-loop runs resumable per
    query: every finished query is appended (flushed + fsynced) to the
    checkpoint as one JSON line, and a restarted evaluation replays all
    previously recorded results instead of re-running them.
    """

    def __init__(self, label: str, run_fn: RunFn, checkpoint_path: str | None = None) -> None:
        self.label = label
        self.run_fn = run_fn
        self.checkpoint_path = checkpoint_path

    # ------------------------------------------------------------------

    def run_query(self, q) -> PipelineQueryResult:  # q: EvalQuery
        run = dict(self.run_fn(q.query))
        verifications = run.get("verifications") or []
        fact_ver = [v for v in verifications if v.get("kind", v["claim"].get("kind")) == "fact"]
        supported = sum(1 for v in fact_ver if v.get("verdict") == "supported")
        contradicted = sum(1 for v in fact_ver if v.get("verdict") == "contradicted")
        insufficient = sum(1 for v in fact_ver if v.get("verdict") == "insufficient")
        fact_claims = len(fact_ver)

        evidence_ids = list(dict.fromkeys(run.get("evidence_chunk_ids") or []))
        relevant = list(q.relevant_chunk_ids)
        abstained = bool(run.get("abstained"))
        hit = bool(q.answerable) and any(cid in set(relevant) for cid in evidence_ids)
        answered_correct = q.answerable and not abstained and hit

        result = PipelineQueryResult(
            label=self.label,
            qid=q.qid,
            query=q.query,
            category=q.category,
            answerable=q.answerable,
            query_type=str(run.get("query_type") or ""),
            abstained=abstained,
            clarification=run.get("clarification"),
            final_answer=str(run.get("final_answer") or run.get("answer") or ""),
            claims=fact_claims,
            supported=supported,
            contradicted=contradicted,
            insufficient=insufficient,
            supported_rate=float(run.get("supported_rate") or 0.0),
            unsupported_stripped=len(run.get("unsupported_claims") or []),
            contradicted_kept=0,
            unresolved_citations=int(run.get("unresolved_citations") or 0),
            relevant_chunk_ids=relevant,
            evidence_chunk_ids=evidence_ids,
            retrieval_top_chunk_ids=list(run.get("retrieval_top_chunk_ids") or evidence_ids),
            evidence_products=list(run.get("evidence_products") or []),
            steps=list(run.get("steps") or []),
            latency_ms=dict(run.get("latency_ms") or {}),
            usage=dict(run.get("usage") or {}),
            answered_with_relevant_evidence=answered_correct,
        )
        result.failure = classify_query(
            q.qid,
            q.category,
            q.answerable,
            {
                "query_type": result.query_type,
                "abstained": abstained,
                "clarification": result.clarification,
                "evidence_chunk_ids": evidence_ids,
                "retrieval_top_chunk_ids": result.retrieval_top_chunk_ids,
                "relevant_chunk_ids": relevant,
                "claims": fact_claims,
                "supported": supported,
                "contradicted": contradicted,
                "insufficient": insufficient,
                "unresolved_citations": result.unresolved_citations,
                "unsupported_stripped": result.unsupported_stripped,
                "contradicted_kept": result.contradicted_kept,
                "evidence_products": result.evidence_products,
                "steps": result.steps,
                "degenerate": bool(run.get("degenerate")),
            },
        )
        return result

    # ------------------------------------------------------------------

    def evaluate(self, queries: Sequence[Any]) -> dict[str, Any]:
        done = self._load_checkpoint()
        per_query: list[PipelineQueryResult] = []
        for q in queries:
            cached = done.get(q.qid)
            if cached is not None:
                print(f"  resume qid={q.qid}", flush=True)
                per_query.append(cached)
                continue
            result = self.run_query(q)
            per_query.append(result)
            if self.checkpoint_path:
                self._append_checkpoint(result)
        return {
            "label": self.label,
            "num_queries": len(queries),
            "aggregate": self.aggregate(per_query),
            "by_category": self.aggregate_by_category(per_query),
            "taxonomy": taxonomy_aggregate([r.failure for r in per_query if r.failure]),
            "per_query": [r.to_dict() for r in per_query],
        }

    # ------------------------------------------------------------------
    # Per-query checkpointing (crash-safe resume for long runs)
    # ------------------------------------------------------------------

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> PipelineQueryResult:
        failure = d.get("failure")
        result = PipelineQueryResult(**{k: v for k, v in d.items() if k != "failure"})
        if failure:
            allowed = {f.name for f in fields(QueryFailureReport)}
            result.failure = QueryFailureReport(**{k: v for k, v in failure.items() if k in allowed})
        return result

    def _load_checkpoint(self) -> dict[str, PipelineQueryResult]:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return {}
        done: dict[str, PipelineQueryResult] = {}
        with open(self.checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("label") != self.label:
                        continue
                    done[d["qid"]] = self._from_dict(d)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue  # torn/partial last line is tolerated
        return done

    def _append_checkpoint(self, result: PipelineQueryResult) -> None:
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        with open(self.checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------

    @staticmethod
    def aggregate(results: Sequence[PipelineQueryResult]) -> dict[str, Any]:
        """SRS §14 metrics for one pipeline."""
        answerable = [r for r in results if r.answerable]
        ood = [r for r in results if not r.answerable]
        answered = [r for r in answerable if not r.abstained]
        abstained_ans = [r for r in answerable if r.abstained]

        # §14.4 unsupported claim rate over all fact claims *evaluated*.
        total_claims = sum(r.claims for r in results)
        unsupported_unverified = sum(r.contradicted + r.insufficient for r in results)
        unsupported_rate = (
            round(unsupported_unverified / total_claims, 4) if total_claims else 0.0
        )

        lat = [r.latency_ms.get("total", 0.0) for r in results if r.latency_ms.get("total")]
        components: dict[str, list[float]] = {}
        for r in results:
            for comp, ms in (r.latency_ms or {}).items():
                if isinstance(ms, (int, float)) and ms > 0:
                    components.setdefault(comp, []).append(ms)

        tokens = {
            "prompt_tokens": sum(r.usage.get("prompt_tokens", 0) for r in results),
            "completion_tokens": sum(r.usage.get("completion_tokens", 0) for r in results),
            "calls": sum(r.usage.get("calls", 0) for r in results),
        }
        return {
            # §14.6 — reported separately for answerable vs OOD probes.
            "answerable_queries": len(answerable),
            "answered": len(answered),
            "answer_rate": round(len(answered) / len(answerable), 4) if answerable else 0.0,
            "ood_probes": len(ood),
            "ood_abstained": sum(1 for r in ood if r.abstained),
            "abstentions_on_answerable": len(abstained_ans),
            # §14.2 proxy (single-annotator): answered AND relevant evidence used.
            "answered_correct": sum(1 for r in answered if r.answered_with_relevant_evidence),
            "answered_correct_rate": (
                round(sum(1 for r in answered if r.answered_with_relevant_evidence) / len(answered), 4)
                if answered
                else 0.0
            ),
            # §14.4 / §14.5
            "total_fact_claims": total_claims,
            "unsupported_claim_rate": unsupported_rate,
            "mean_supported_rate": mean([r.supported_rate for r in answered]),
            "queries_with_unsupported_claims": sum(1 for r in results if r.unsupported_stripped or r.contradicted),
            # §14.7
            "latency": {
                "total_ms": {"mean": mean(lat), "median": round(statistics.median(lat), 1) if lat else 0.0},
                "components": {
                    comp: {"mean": mean(vals), "max": round(max(vals), 1)}
                    for comp, vals in sorted(components.items())
                },
            },
            # §14.8 / NFR-005 — tokens consumed; local Ollama cost ≈ $0.
            "tokens": tokens,
            "cost_estimate_usd": 0.0,
            "cost_note": "local Ollama inference: no API cost; tokens recorded instead",
        }

    @staticmethod
    def aggregate_by_category(results: Sequence[PipelineQueryResult]) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[PipelineQueryResult]] = {}
        for r in results:
            buckets.setdefault(r.category, []).append(r)
        out = {}
        for cat, items in sorted(buckets.items()):
            answered = [r for r in items if r.answerable and not r.abstained]
            total_claims = sum(r.claims for r in items)
            unverified = sum(r.contradicted + r.insufficient for r in items)
            out[cat] = {
                "count": len(items),
                "answered": len(answered),
                "abstained": sum(1 for r in items if r.abstained),
                "answered_correct": sum(1 for r in answered if r.answered_with_relevant_evidence),
                "answered_correct_rate": round(
                    sum(1 for r in answered if r.answered_with_relevant_evidence) / len(answered), 4
                )
                if answered
                else 0.0,
                "unsupported_claim_rate": round(unverified / total_claims, 4) if total_claims else 0.0,
                "mean_total_ms": mean([r.latency_ms.get("total", 0.0) for r in items if r.latency_ms.get("total")]),
            }
        return out


# ----------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------

PIPELINE_METRIC_ROWS = (
    ("answered_correct_rate", "answer correctness (proxy) ↑", "{:.3f}"),
    ("answer_rate", "answered (answerable) ↑", "{:.3f}"),
    ("abstentions_on_answerable", "abstentions on answerable", "{:d}"),
    ("ood_abstained", "OOD probes abstained ↑", "{:d}"),
    ("unsupported_claim_rate", "unsupported claim rate ↓", "{:.3f}"),
    ("mean_supported_rate", "mean grounding supported rate ↑", "{:.3f}"),
)


def render_markdown(reports: list[dict[str, Any]], taxonomy: dict[str, Any] | None = None) -> str:
    """Human report comparing evaluated pipelines side by side."""
    lines = ["# Phase 6 — Pipeline evaluation (SRS §14)", ""]
    if reports:
        first = reports[0]
        lines.append(f"- Frozen set: {first.get('eval_set', '')}")
        lines.append(f"- Queries: {first.get('num_queries', 0)} · generated: {first.get('generated', '')}")
    lines.append("")
    lines.append("| metric | " + " | ".join(r.get("label", "") for r in reports) + " |")
    lines.append("|---|" + "---|" * len(reports))
    for key, title, fmt in PIPELINE_METRIC_ROWS:
        cells = []
        for r in reports:
            agg = r.get("aggregate", {})
            cells.append(fmt.format(agg.get(key, 0)))
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    for r in reports:
        lat = (r.get("aggregate", {}) or {}).get("latency", {})
        tok = (r.get("aggregate", {}) or {}).get("tokens", {})
        comp = lat.get("components", {})
        lines.append("")
        lines.append(f"## {r.get('label', '')} — latency & cost")
        lines.append("")
        lines.append(
            f"- total: mean {lat.get('total_ms', {}).get('mean', 0):.0f} ms, "
            f"median {lat.get('total_ms', {}).get('median', 0):.0f} ms"
        )
        for comp_name, stats in comp.items():
            lines.append(f"- {comp_name}: mean {stats['mean']:.0f} ms (max {stats['max']:.0f})")
        lines.append(
            f"- tokens: prompt {tok.get('prompt_tokens', 0)}, completion {tok.get('completion_tokens', 0)}, "
            f"LLM calls {tok.get('calls', 0)} — cost ≈ $0 (local Ollama)"
        )
    if taxonomy:
        lines.append("")
        lines.append("## Failure taxonomy")
        lines.append("")
        lines.append("| code | label | queries |")
        lines.append("|---|---|---|")
        for code, n in sorted(taxonomy.get("codes", {}).items(), key=lambda kv: int(kv[0][1:])):
            from .failure_taxonomy import CODES

            lines.append(f"| {code} | {CODES.get(code, '')} | {n} |")
    return "\n".join(lines)


def write_json(payload: dict[str, Any], path: str) -> None:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Phase 6 report written to %s", path)
