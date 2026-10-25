#!/usr/bin/env python
"""Phase 6 — Insurance Policy RAG: Evaluation.

Implements the SRS Phase 6 bundle on top of the Phase 5 verified pipeline:

    baseline comparison (AC-010) · ablation experiments
    · failure taxonomy (§15, AC-012) · accuracy / grounding / latency / cost
    (§14 + NFR-004/005, AC-011)

The comparison target is the Phase 2 *baseline RAG* — semantic-only retrieval
and one direct grounded generation pass, with no query classification, no
multi-step agent, no hybrid retrieval, no reranking and no claim verification.
The Phase 5 rules validator is used as an offline scoring instrument for BOTH
systems (for the baseline it is not part of the system), so the unsupported
claim rate (§14.4) is measured like-for-like.

Usage::

    python run_phase6.py evaluate [--modes full,baseline] [--no-llm] [--limit N]
    python run_phase6.py ablate [--no-llm] [--limit N]
    python run_phase6.py report

Settings (frozen set path, report dir) live in ``configs/phase6.json``,
overlaying the phase 5 → 4 → 3 → 2 chain.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

logger = logging.getLogger("phase6")

DEFAULT_CONFIG = "configs/phase6.json"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    from run_phase3 import load_config as _load  # shared merge logic

    return _load(path)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ----------------------------------------------------------------------
# Token recorder — NFR-005 (usage agnostic to how many OllamaLLM instances
# the pipeline creates internally)
# ----------------------------------------------------------------------

class UsageRecorder:
    """Global token accounting for every OllamaLLM.generate call."""

    def __init__(self) -> None:
        self.totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def install(self) -> None:
        from src.generation.llm import OllamaLLM

        recorder = self
        original = OllamaLLM.generate

        def recorded(self, prompt, system=None):  # noqa: ANN001
            out = original(self, prompt, system)
            recorder.totals["calls"] += 1
            recorder.totals["prompt_tokens"] += int(getattr(self, "last_usage", {}).get("prompt_eval_count") or 0)
            recorder.totals["completion_tokens"] += int(getattr(self, "last_usage", {}).get("eval_count") or 0)
            return out

        OllamaLLM.generate = recorded
        self.original = original

    def snapshot(self) -> dict[str, int]:
        return dict(self.totals)

    def diff(self, before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        return {k: after.get(k, 0) - before.get(k, 0) for k in self.totals}


# ----------------------------------------------------------------------
# Runners: one per compared pipeline, each returning the normalized result
# dict the evaluator consumes.
# ----------------------------------------------------------------------

def build_full_run_fn(config: dict, no_llm: bool, usage: UsageRecorder):
    """Phase 5 verified pipeline (agent + hybrid retrieval + verification)."""
    from run_phase5 import build_agent

    agent, llms = build_agent(
        config,
        use_extractive=no_llm,
        use_verify_llm=None if not no_llm else False,
        return_llm=True,
    )

    def run(query: str) -> dict:
        before = usage.snapshot()
        response = agent.run(query)
        usage_after = usage.diff(before, usage.snapshot())

        d = response.to_dict()
        verifications = d.get("verifications", [])
        citations = d.get("citations", []) or []
        products = sorted({c.get("product", "") for c in citations if c.get("product")})
        evidence_ids = sorted(
            {c.get("chunk_id", "") for c in citations if c.get("chunk_id")}
        )
        # Chunk ids Agent 5 actually checked (its own alignment with markers).
        verif_ids = sorted(
            {cid for v in verifications for cid in (v.get("evidence_chunk_ids") or [])}
        )
        unresolved = sum(1 for v in verifications if str(v.get("reason", "")).startswith("unresolved"))
        claims = d.get("counts", {}).get("claims", 0)
        steps = d.get("steps", []) or []
        return {
            "query_type": d.get("query_type", ""),
            "abstained": bool(d.get("abstained")),
            "clarification": d.get("clarification"),
            "final_answer": d.get("final_answer") or d.get("answer", ""),
            "answer": d.get("answer", ""),
            "verifications": verifications,
            "supported_rate": d.get("supported_rate", 0.0),
            "unsupported_claims": d.get("unsupported_claims", []),
            "contradicted_claims": d.get("contradicted_claims", []),
            "unresolved_citations": unresolved,
            "evidence_chunk_ids": sorted(set(evidence_ids) | set(verif_ids)),  # full
            "retrieval_top_chunk_ids": [c.get("chunk_id", "") for c in citations],
            "evidence_products": products,
            "steps": steps,
            "plan_empty": (not steps) and not d.get("abstained"),
            "degenerate": claims == 0 and not d.get("abstained"),
            "latency_ms": d.get("latency_ms", {}) or {},
            "usage": usage_after,
            "model": d.get("model", ""),
            "evidence_count": d.get("evidence_count", 0),
        }

    return run


def build_baseline_run_fn(config: dict, no_llm: bool, usage: UsageRecorder):
    """Phase 2 baseline: semantic-only retrieval → one grounded generation.

    No agent, no hybrid fusion, no reranking, no claim verification. The
    rules validator still scores its claims OFFLINE (§14.4 like-for-like).
    """
    import run_phase3
    from src.agent.schema import QueryType
    from run_phase5 import ExtractiveGenerator
    from src.generation.generator import AnswerGenerator, is_refusal
    from src.grounding.verifier import GroundingValidator

    retriever, _bm25 = run_phase3.build_hybrid_retriever(config, want_reranker=False)

    gen_cfg = config.get("generation", {})
    if no_llm:
        generator = ExtractiveGenerator()
        llm = None
    else:
        from src.generation.llm import OllamaLLM

        llm = OllamaLLM(
            model=gen_cfg.get("model", "llama3"),
            host=gen_cfg.get("ollama_host", "http://localhost:11434"),
            temperature=float(gen_cfg.get("temperature", 0.2)),
            num_predict=int(gen_cfg.get("num_predict", 512)),
            num_gpu=gen_cfg.get("num_gpu"),
            timeout_seconds=int(gen_cfg.get("timeout_seconds", 180)),
        )
        generator = AnswerGenerator(llm, max_evidence_chars=int(gen_cfg.get("max_evidence_chars", 6000)))

    scorer = GroundingValidator(use_llm=False)  # offline measuring instrument only
    k_default = int(config.get("retrieval", {}).get("default_k", 5))

    def run(query: str) -> dict:
        before = usage.snapshot()
        t0 = time.time()
        t = time.time()
        hits = retriever.retrieve(query, k=k_default, mode="semantic", rerank=False)
        retrieval_ms = round((time.time() - t) * 1000, 1)
        chunks = list(hits)  # RetrievedChunk objects, in prompt-marker order

        t = time.time()
        result = generator.answer(query, chunks)
        generation_ms = round((time.time() - t) * 1000, 1)

        usage_after = usage.diff(before, usage.snapshot())
        answer = result.answer
        abstained = bool(getattr(result, "abstained", False)) or is_refusal(answer)

        evidence_ids = [getattr(c, "chunk_id", "") for c in chunks]
        products = sorted({getattr(c, "product", "") for c in chunks})
        cited_ids = sorted({c.chunk_id for c in (getattr(result, "citations", []) or [])})
        verifications = []
        if abstained:
            verifications = []
        else:
            verifications = [v.to_dict() for v in scorer.verify(answer, chunks)]
        fact_ver = [v for v in verifications if v["claim"].get("kind") == "fact"]
        supported = sum(1 for v in fact_ver if v["verdict"] == "supported")
        total_fact = len(fact_ver)
        unresolved = sum(1 for v in verifications if str(v.get("reason", "")).startswith("unresolved"))
        unsupported = [
            v["claim"]["text"] for v in fact_ver if v["verdict"] in ("insufficient", "contradicted")
        ]
        latency = {
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "verification": 0.0,
            "total": round((time.time() - t0) * 1000, 1),
        }
        return {
            "query_type": QueryType.POLICY_INFORMATION.value,  # baseline has no classifier
            "abstained": abstained,
            "clarification": None,
            "final_answer": answer,
            "answer": answer,
            "verifications": verifications,
            "supported_rate": round(supported / total_fact, 4) if total_fact else 0.0,
            "unsupported_claims": unsupported,
            "contradicted_claims": [],
            "unresolved_citations": unresolved,
            "evidence_chunk_ids": cited_ids or evidence_ids,
            "retrieval_top_chunk_ids": evidence_ids,
            "evidence_products": products,
            "steps": [],
            "plan_empty": False,
            "degenerate": (not abstained) and total_fact == 0,
            "latency_ms": latency,
            "usage": usage_after,
            "model": getattr(result, "model", "baseline"),
            "evidence_count": len(chunks),
        }

    return run


# ----------------------------------------------------------------------
# Command: evaluate — baseline vs full on the frozen set
# ----------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    from src.evaluation.pipeline_eval import PipelineEvaluator, render_markdown, write_json
    from src.evaluation.retrieval_eval import load_eval_set

    config = load_config(args.config)
    ecfg = config.get("evaluation", {})
    eval_set_path = args.eval_set or ecfg.get("eval_set", "data/eval/retrieval_eval.json")
    queries = load_eval_set(eval_set_path)
    if args.limit:
        queries = queries[: int(args.limit)]

    usage = UsageRecorder()
    usage.install()

    modes = [m.strip() for m in (args.modes or "").split(",") if m.strip()]
    if not modes:
        modes = ecfg.get("modes", ["full", "baseline"])

    out_dir = ecfg.get("report_dir", "reports")
    out_path = args.out or os.path.join(out_dir, "phase6_pipeline_eval.json")
    md_path = os.path.splitext(out_path)[0] + ".md"

    # Resume support — a long LLM run may die mid-way; completed pipeline
    # reports are checkpointed after each mode and reused in order.
    reports: list[dict] = []
    checkpoint_path = os.path.join(out_dir, "phase6_eval_checkpoint.json")
    if not args.fresh:
        try:
            with open(checkpoint_path, encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("eval_set") == eval_set_path and prev.get("no_llm") == bool(args.no_llm):
                prev_reports = prev.get("reports", [])
                if prev_reports and [r.get("label") for r in prev_reports] == modes[: len(prev_reports)]:
                    reports = list(prev_reports)
                    print(f"resumed {len(reports)} completed pipeline(s) from {checkpoint_path}", flush=True)
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass

    builders = {
        "full": lambda: build_full_run_fn(config, args.no_llm, usage),
        "baseline": lambda: build_baseline_run_fn(config, args.no_llm, usage),
    }

    for label in modes[len(reports):]:
        if label not in builders:
            print(f"Unknown mode '{label}' (choose from full, baseline)", file=sys.stderr)
            return 2
        print(f"--- evaluating pipeline '{label}' over {len(queries)} queries ---", flush=True)
        try:
            run_fn = builders[label]()
        except Exception as exc:  # noqa: BLE001
            print(f"Could not build pipeline '{label}': {exc}", file=sys.stderr)
            return 1
        t0 = time.time()
        qckpt = os.path.join(
            out_dir,
            f"phase6_eval_queries_{label}{'_offline' if args.no_llm else ''}.jsonl",
        )
        evaluator = PipelineEvaluator(label=label, run_fn=run_fn, checkpoint_path=qckpt)
        report = evaluator.evaluate(queries)
        done_qids = set(evaluator._load_checkpoint())
        if os.path.exists(qckpt) and done_qids >= {q.qid for q in queries}:
            # every query recorded -> the crash-resume file is no longer needed
            try:
                os.remove(qckpt)
            except OSError:
                pass
        report["eval_set"] = eval_set_path
        report["generated"] = datetime.now(timezone.utc).isoformat()
        report["wall_s"] = round(time.time() - t0, 2)
        report["no_llm"] = bool(args.no_llm)
        reports.append(report)
        write_json({"reports": reports, "eval_set": eval_set_path, "no_llm": bool(args.no_llm)}, checkpoint_path)
        print(f"checkpointed '{label}' ({len(reports)}/{len(modes)}) -> {checkpoint_path}", flush=True)

    payload = {"reports": reports, "eval_set": eval_set_path, "no_llm": bool(args.no_llm)}
    write_json(payload, out_path)
    try:
        os.remove(checkpoint_path) if os.path.exists(checkpoint_path) else None
    except OSError:  # noqa: PERF203 — best-effort cleanup
        pass

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            render_markdown(
                reports,
                taxonomy=reports[0].get("taxonomy") if reports else None,
            )
            + "\n"
        )

    _print_reports(reports)
    print(f"\nReports: {out_path}\n         {md_path}")
    return 0


def _print_reports(reports: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("PHASE 6 PIPELINE EVALUATION (SRS §14)")
    print("=" * 78)
    header = (
        f"{'pipeline':<22} {'answered↑':>10} {'correct↑':>9} {'ood-abst↑':>10} "
        f"{'unsup↓':>8} {'grounding↑':>11} {'ms/query':>10} {'tok':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        agg = r["aggregate"]
        tok = agg.get("tokens", {})
        print(
            f"{r['label']:<22} "
            f"{agg.get('answer_rate', 0):>10.3f} "
            f"{agg.get('answered_correct_rate', 0):>9.3f} "
            f"{agg.get('ood_abstained', 0):>10d}/{agg.get('ood_probes', 0)} "
            f"{agg.get('unsupported_claim_rate', 0):>8.3f} "
            f"{agg.get('mean_supported_rate', 0):>11.3f} "
            f"{agg.get('latency', {}).get('total_ms', {}).get('mean', 0):>10.0f} "
            f"{tok.get('prompt_tokens', 0) + tok.get('completion_tokens', 0):>7d}"
        )
    full = next((r for r in reports if r["label"] == "full"), None)
    baseline = next((r for r in reports if r["label"] == "baseline"), None)
    if full and baseline:
        print("\nBaseline (Phase 2) vs full (Phase 5 pipeline):")
        for metric in ("answered_correct_rate", "unsupported_claim_rate", "mean_supported_rate"):
            b = baseline["aggregate"].get(metric, 0.0)
            h = full["aggregate"].get(metric, 0.0)
            print(f"  {metric:<26} baseline={b:.3f}  full={h:.3f}  delta={h - b:+.3f}")
        bms = baseline["aggregate"].get("latency", {}).get("total_ms", {}).get("mean", 0.0)
        hms = full["aggregate"].get("latency", {}).get("total_ms", {}).get("mean", 0.0)
        print(f"  latency ms/query           baseline={bms:.0f}  full={hms:.0f}  overhead={hms / bms if bms else 0:+.1f}x")
    for r in reports:
        tax = r.get("taxonomy", {})
        print(f"\nTaxonomy [{r['label']}]: failed {tax.get('num_failed', 0)}/{tax.get('num_queries', 0)}")
        for code, n in sorted(tax.get("codes", {}).items(), key=lambda kv: int(kv[0][1:])):
            print(f"  {code}: {n}")


# ----------------------------------------------------------------------
# Command: ablate — pipeline-level ablation experiments (SRS §14.5 style)
# ----------------------------------------------------------------------

ABLATIONS: dict[str, tuple[str, dict]] = {
    # name: (description, config overlay applied on top of phase5 config)
    "no_rerank": ("hybrid retrieval without the cross-encoder reranker", {"reranker": {"enabled": False}}),
    "keyword_only": ("drop the dense retriever; BM25 keyword only", {"retrieval": {"mode": "keyword"}}),
    "semantic_only": ("drop BM25; semantic retrieval only (≈ baseline retrieval)", {"retrieval": {"mode": "semantic"}}),
    "no_strip": ("keep unsupported claims in the answer (grounding handler off)", {"handler": {"strip_unsupported": False}}),
    "rules_only_verify": ("drop the LLM second pass; rules-only verification", {"grounding": {"use_llm_verification": False}}),
}

_ABLATION_AGG_KEYS = (
    "answered_correct_rate",
    "unsupported_claim_rate",
    "mean_supported_rate",
    "abstentions_on_answerable",
)


def _overlay(base: dict, patch: dict) -> dict:
    import copy

    out = copy.deepcopy(base)
    for section, values in patch.items():
        out.setdefault(section, {})
        if isinstance(values, dict):
            out[section].update(values)
        else:
            out[section] = values
    return out


def cmd_ablate(args: argparse.Namespace) -> int:
    from src.evaluation.pipeline_eval import PipelineEvaluator, write_json
    from src.evaluation.retrieval_eval import load_eval_set

    config = load_config(args.config)
    ecfg = config.get("evaluation", {})
    eval_set_path = args.eval_set or ecfg.get("eval_set", "data/eval/retrieval_eval.json")
    queries = load_eval_set(eval_set_path)
    if args.limit:
        queries = queries[: int(args.limit)]

    usage = UsageRecorder()
    usage.install()

    # The configured default (full pipeline) is the reference point.
    names = [n.strip() for n in (args.ablations or "").split(",") if n.strip()] or list(ABLATIONS)

    rows = []
    for name in ["default", *names]:
        if name != "default" and name not in ABLATIONS:
            print(f"Unknown ablation '{name}'", file=sys.stderr)
            return 2
        desc = "full pipeline as configured (hybrid + rerank + grounding)"
        run_config = config
        if name != "default":
            desc = ABLATIONS[name][0]
            run_config = _overlay(config, ABLATIONS[name][1])
        print(f"--- ablation '{name}': {desc} ---", flush=True)
        try:
            run_fn = build_full_run_fn(run_config, args.no_llm, usage)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not build ablation '{name}': {exc}", file=sys.stderr)
            continue
        report = PipelineEvaluator(label=name, run_fn=run_fn).evaluate(queries)
        report["description"] = desc
        report["generated"] = datetime.now(timezone.utc).isoformat()
        rows.append(report)

    payload = {"ablations": rows, "eval_set": eval_set_path, "no_llm": bool(args.no_llm)}
    out_dir = ecfg.get("report_dir", "reports")
    out_path = args.out or os.path.join(out_dir, "phase6_ablations.json")
    write_json(payload, out_path)

    print("\n" + "=" * 78)
    print("PHASE 6 ABLATIONS (SRS Phase 6 'ablation experiments')")
    print("=" * 78)
    header = f"{'variant':<20} {'correct↑':>9} {'unsup↓':>8} {'grounding↑':>11} {'abst':>5} {'ms/query':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        agg = r["aggregate"]
        print(
            f"{r['label']:<20} "
            f"{agg.get('answered_correct_rate', 0):>9.3f} "
            f"{agg.get('unsupported_claim_rate', 0):>8.3f} "
            f"{agg.get('mean_supported_rate', 0):>11.3f} "
            f"{agg.get('abstentions_on_answerable', 0):>5d} "
            f"{agg.get('latency', {}).get('total_ms', {}).get('mean', 0):>10.0f}"
        )
    print(f"\nReport: {out_path}")
    return 0


# ----------------------------------------------------------------------
# Command: report — roll everything up for AC-010..AC-012
# ----------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    from src.evaluation.failure_taxonomy import CODES, aggregate as tax_agg

    config = load_config(args.config)
    out_dir = config.get("evaluation", {}).get("report_dir", "reports")

    def load_json(path: str) -> dict | None:
        if not os.path.exists(path):
            logger.warning("Missing report: %s", path)
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    pipeline = load_json(os.path.join(out_dir, "phase6_pipeline_eval.json")) or {}
    ablations = load_json(os.path.join(out_dir, "phase6_ablations.json")) or {}
    retrieval = load_json(os.path.join(out_dir, "phase3_retrieval_eval.json")) or {}

    lines: list[str] = ["# Phase 6 — Evaluation report (SRS Phase 6, §14, §15, AC-010..AC-012)", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    # AC-010 — baseline comparison
    reports = pipeline.get("reports", [])
    full = next((r for r in reports if r["label"] == "full"), None)
    baseline = next((r for r in reports if r["label"] == "baseline"), None)
    lines.append("## AC-010 — Enhanced system vs baseline RAG")
    lines.append("")
    if full and baseline:
        fb = baseline["aggregate"]
        ff = full["aggregate"]
        lines.append("| metric | baseline (Phase 2) | full (Phase 5 pipeline) | delta |")
        lines.append("|---|---|---|---|")
        for metric in ("answered_correct_rate", "unsupported_claim_rate", "mean_supported_rate"):
            b, h = fb.get(metric, 0.0), ff.get(metric, 0.0)
            lines.append(f"| {metric} | {b:.3f} | {h:.3f} | {h - b:+.3f} |")
        bms = fb.get("latency", {}).get("total_ms", {}).get("mean", 0.0)
        hms = ff.get("latency", {}).get("total_ms", {}).get("mean", 0.0)
        lines.append(f"| latency ms/query | {bms:.0f} | {hms:.0f} | {hms - bms:+.0f} |")
    else:
        lines.append("_Run `python run_phase6.py evaluate` first._")
    lines.append("")

    # AC-011 — latency & cost tables
    lines.append("## AC-011 — Latency and token/cost record")
    lines.append("")
    for r in reports:
        agg = r["aggregate"]
        lat = agg.get("latency", {})
        tok = agg.get("tokens", {})
        lines.append(f"### {r['label']}")
        lines.append("")
        lines.append("| component | mean ms | max ms |")
        lines.append("|---|---|---|")
        for comp, stats in lat.get("components", {}).items():
            lines.append(f"| {comp} | {stats['mean']:.0f} | {stats['max']:.0f} |")
        lines.append(f"| **total** | {lat.get('total_ms', {}).get('mean', 0):.0f} | |")
        lines.append("")
        lines.append(
            f"Tokens: prompt {tok.get('prompt_tokens', 0)}, completion {tok.get('completion_tokens', 0)}, "
            f"LLM calls {tok.get('calls', 0)} — estimated cost **$0.00/query** (local Ollama; "
            "NFR-005 records tokens where an API cost does not apply)."
        )
        lines.append("")

    # Ablations
    rows = ablations.get("ablations", [])
    if rows:
        lines.append("## Ablation experiments")
        lines.append("")
        lines.append("| variant | correct ↑ | unsupported ↓ | grounding ↑ | abstentions | mean ms |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            agg = r["aggregate"]
            lines.append(
                f"| {r['label']} | {agg.get('answered_correct_rate', 0):.3f} | "
                f"{agg.get('unsupported_claim_rate', 0):.3f} | "
                f"{agg.get('mean_supported_rate', 0):.3f} | "
                f"{agg.get('abstentions_on_answerable', 0)} | "
                f"{agg.get('latency', {}).get('total_ms', {}).get('mean', 0):.0f} |"
            )
        for r in rows:
            if r.get("description"):
                lines.append(f"- `{r['label']}`: {r['description']}")
        lines.append("")
    else:
        lines.append("## Ablation experiments")
        lines.append("")
        lines.append("_Run `python run_phase6.py ablate` first._")

    # Retrieval-quality reference (Phase 3 frozen-set eval)
    if retrieval.get("modes"):
        lines.append("## Retrieval quality reference (Phase 3 frozen-set eval)")
        lines.append("")
        k = retrieval.get("k", 5)
        lines.append("| mode | P@k | R@k | nDCG@k | MRR |")
        lines.append("|---|---|---|---|---|")
        for r in retrieval["modes"]:
            agg = r.get("aggregate", {})
            lines.append(
                f"| {r.get('label', '')} | {agg.get(f'precision@{k}', 0):.3f} | "
                f"{agg.get(f'recall@{k}', 0):.3f} | {agg.get(f'ndcg@{k}', 0):.3f} | "
                f"{agg.get('mrr', 0):.3f} |"
            )
        lines.append("")

    # AC-012 — taxonomy roll-up
    taxo_src = full or (reports[0] if reports else None)
    if taxo_src and taxo_src.get("taxonomy"):
        codes = taxo_src["taxonomy"].get("codes", {})
        tax = tax_agg([])  # just to reuse the code table
        lines.append("## AC-012 — Failure taxonomy (full pipeline)")
        lines.append("")
        lines.append("| code | label | queries |")
        lines.append("|---|---|---|")
        for code, n in sorted(codes.items(), key=lambda kv: int(kv[0][1:])):
            lines.append(f"| {code} | {CODES.get(code, '')} | {n} |")
        lines.append("")
        lines.append(
            "F15/F16 are positive detections (unsupported-claim correction, correct OOD abstention) "
            "recorded per query but excluded from the failure rate."
        )
        lines.append("")

    payload = {"config": DEFAULT_CONFIG, "eval_set": pipeline.get("eval_set"), "reports": reports, "ablations": rows}
    out_path = args.out or os.path.join(out_dir, "phase6_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    md_path = os.path.splitext(out_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Phase 6 report: {out_path}\n                {md_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6: Evaluation")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Phase 6 config JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Baseline vs full pipeline on the frozen set")
    p_eval.add_argument("query", nargs="?", help=argparse.SUPPRESS)
    p_eval.add_argument("--modes", help="Comma-separated: full,baseline")
    p_eval.add_argument("--eval-set", help="Frozen evaluation set JSON")
    p_eval.add_argument("--limit", help="Evaluate only the first N queries (smoke runs)")
    p_eval.add_argument("--no-llm", action="store_true", help="Extractive/fast offline mode (no Ollama)")
    p_eval.add_argument("--out", help="Write JSON here")
    p_eval.add_argument(
        "--fresh", action="store_true", help="Ignore any previous evaluate checkpoint and rebuild reports"
    )

    p_ab = sub.add_parser("ablate", help="Ablation experiments over the frozen set")
    p_ab.add_argument("--ablations", help="Comma-separated names: " + ",".join(ABLATIONS))
    p_ab.add_argument("--eval-set")
    p_ab.add_argument("--limit")
    p_ab.add_argument("--no-llm", action="store_true")
    p_ab.add_argument("--out")

    p_rep = sub.add_parser("report", help="Roll up AC-010..AC-012 into a single report")
    p_rep.add_argument("--out")

    args = parser.parse_args()
    setup_logging(args.verbose)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.command == "evaluate":
        return cmd_evaluate(args)
    if args.command == "ablate":
        return cmd_ablate(args)
    if args.command == "report":
        return cmd_report(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
