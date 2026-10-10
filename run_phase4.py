#!/usr/bin/env python
"""Phase 4 — Insurance Policy RAG: Agentic Layer.

Adds the SRS Phase 4 scope on top of the Phase 3 hybrid retriever:

    query classification · requirement extraction · query planning ·
    multi-step retrieval · product comparison

Usage:
    python run_phase4.py analyze "QUESTION" [--llm] [--config configs/phase4.json]
    python run_phase4.py query "QUESTION" [--k 5] [--no-llm] [--out PATH]
                                          [--mode hybrid] [--reranker lexical]
    python run_phase4.py products

Commands
--------
analyze   Show the query classification, extracted requirements and the retrieval
          plan without retrieving anything (no index/embedder needed). Add
          ``--llm`` to exercise the LLM classification fallback.
query     Run the full agent: analyze -> plan -> multi-step retrieval -> grounded
          answer or product comparison. ``--no-llm`` stops after retrieval.
products  List the products and aliases the analyzer recognises.

Settings come from ``configs/phase4.json`` (agent caps, criteria, aliases),
which merges ``configs/phase3.json`` and ``configs/phase2.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import run_phase3 as r3

logger = logging.getLogger("phase4")

DEFAULT_CONFIG = "configs/phase4.json"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

def _build_llm(config: dict, num_gpu: int | None = None):
    from src.generation.llm import OllamaLLM

    gen = config.get("generation", {})
    llm = OllamaLLM(
        model=gen.get("model", "llama3"),
        host=gen.get("ollama_host", "http://localhost:11434"),
        temperature=float(gen.get("temperature", 0.2)),
        num_predict=int(gen.get("num_predict", 512)),
        num_gpu=(num_gpu if num_gpu is not None else gen.get("num_gpu")),
        timeout_seconds=int(gen.get("timeout_seconds", 180)),
    )
    return llm


def build_products_matcher(config: dict):
    from src.agent import ProductMatcher
    from src.retrieval.bm25 import load_chunk_corpus

    corpus_path = config.get("retrieval", {}).get("chunk_corpus", "data/chunks/chunks.jsonl")
    chunks = load_chunk_corpus(corpus_path)
    aliases = config.get("agent", {}).get("product_aliases") or {}
    return ProductMatcher.from_chunks(chunks) if not aliases else ProductMatcher(
        [c["product"] for c in chunks if c.get("product")], extra_aliases=aliases
    )


def build_analyzer(config: dict, matcher, llm=None, use_llm_analysis: bool = True):
    from src.agent import QueryAnalyzer

    agent_cfg = config.get("agent", {})
    return QueryAnalyzer(
        llm=llm,
        matcher=matcher,
        use_llm=use_llm_analysis,
        llm_confidence_threshold=float(agent_cfg.get("llm_confidence_threshold", 0.55)),
    )


def build_planner(config: dict, matcher):
    from src.agent import RetrievalPlanner

    agent_cfg = config.get("agent", {})
    retrieval_cfg = config.get("retrieval", {})
    return RetrievalPlanner(
        products=matcher.products,
        default_k=int(retrieval_cfg.get("default_k", 5)),
        max_steps=int(agent_cfg.get("max_steps", 24)),
        max_products=int(agent_cfg.get("max_products", 4)),
    )


def build_executor(config: dict, reranker_kind: str | None = None, rerank: bool = True):
    from src.agent import PlanExecutor

    retrieval_cfg = config.get("retrieval", {})
    agent_cfg = config.get("agent", {})
    retriever, _ = r3.build_hybrid_retriever(
        config, want_reranker=rerank, reranker_kind=reranker_kind
    )
    return PlanExecutor(
        retriever,
        mode=retrieval_cfg.get("mode", "hybrid"),
        fusion=retrieval_cfg.get("fusion", "rrf"),
        candidate_k=int(retrieval_cfg.get("candidate_k", 20)),
        rerank=rerank and retriever.reranker is not None,
        max_chunks=int(agent_cfg.get("max_evidence_chunks", 80)),
    )


def build_agent(config: dict, matcher, llm, executor, use_llm_analysis: bool = True):
    from src.agent import ComparisonGenerator, PolicyRAGAgent
    from src.generation.generator import AnswerGenerator

    gen_cfg = config.get("generation", {})
    cmp_cfg = config.get("comparison", {})
    generator = AnswerGenerator(
        llm, max_evidence_chars=int(gen_cfg.get("max_evidence_chars", 6000))
    )
    comparison = ComparisonGenerator(
        llm, max_evidence_chars=int(cmp_cfg.get("max_evidence_chars", 8000))
    )
    return PolicyRAGAgent(
        analyzer=build_analyzer(config, matcher, llm, use_llm_analysis),
        planner=build_planner(config, matcher),
        executor=executor,
        generator=generator,
        comparison=comparison,
    )


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> int:
    config = r3.load_config(args.config)
    matcher = build_products_matcher(config)
    llm = None
    if args.llm:
        llm = _build_llm(config, args.num_gpu)
        if not llm.is_available():
            print(f"Ollama not reachable at {llm.host}; using rules only.", file=sys.stderr)
            llm = None
    analyzer = build_analyzer(config, matcher, llm, use_llm_analysis=bool(llm))
    planner = build_planner(config, matcher)

    analysis = analyzer.analyze(args.query)
    plan = planner.plan(analysis, k=args.k)
    result = {"analysis": analysis.to_dict(), "plan": plan.to_dict()}

    print("=" * 64)
    print(f"QUERY: {args.query}")
    print("=" * 64)
    print(f"type        : {analysis.query_type.value}")
    print(f"method      : {analysis.method}  (confidence {analysis.confidence:.2f})")
    print(f"complex     : {analysis.is_complex}")
    print(f"products    : {analysis.products or '-'}")
    print(f"criteria    : {analysis.criteria or '-'}")
    print(f"cues        : {', '.join(analysis.matched_cues) or '-'}")
    req = analysis.requirements
    print(
        "requirements: "
        + (", ".join(f"{k}={v}" for k, v in req.to_dict().items() if v and k != "raw") or "-")
    )
    print("-" * 64)
    if plan.needs_clarification:
        print(f"CLARIFICATION: {plan.clarification_question}")
    elif analysis.query_type.value == "out_of_domain":
        print("OUT OF DOMAIN: no insurance evidence retrieval.")
    else:
        print(f"PLAN: {len(plan.steps)} step(s)"
              + (" (comparison)" if plan.needs_comparison else ""))
        for step in plan.steps:
            filt = f" product={step.product!r}" if step.product else ""
            print(f"  [{step.step_id}] {step.description}{filt}")
            print(f"       query: {step.query!r}")
    print("=" * 64)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Written to {args.out}")
    return 0


def cmd_products(args: argparse.Namespace) -> int:
    config = r3.load_config(args.config)
    matcher = build_products_matcher(config)
    print("=" * 64)
    print("PHASE 4 KNOWN PRODUCTS")
    print("=" * 64)
    for product in matcher.products:
        aliases = sorted(a for a, p in matcher.alias_to_product.items() if p == product)
        print(f"- {product}")
        print(f"    aliases: {', '.join(aliases)}")
    print("=" * 64)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    config = r3.load_config(args.config)
    matcher = build_products_matcher(config)

    use_llm = not args.no_llm
    llm = None
    if use_llm:
        llm = _build_llm(config, args.num_gpu)
        if not llm.is_available():
            print(
                f"Ollama not reachable at {llm.host}; retrieval-only run. "
                "Start it with: ollama serve",
                file=sys.stderr,
            )
            llm = None

    executor = build_executor(config, reranker_kind=args.reranker, rerank=not args.no_rerank)

    if llm is None:
        # Retrieval-only: analyze + plan + execute, no generation.
        analyzer = build_analyzer(config, matcher, None, use_llm_analysis=False)
        planner = build_planner(config, matcher)
        analysis = analyzer.analyze(args.query)
        plan = planner.plan(analysis, k=args.k)
        print("=" * 64)
        print(f"QUERY: {args.query}")
        print("=" * 64)
        print(f"type={analysis.query_type.value} method={analysis.method} "
              f"products={analysis.products or '-'} criteria={plan.criteria or '-'}")
        if plan.needs_clarification:
            print(f"CLARIFICATION: {plan.clarification_question}")
            return 0
        if analysis.query_type.value == "out_of_domain":
            print("OUT OF DOMAIN: no retrieval performed.")
            return 0
        bundles = executor.execute(plan)
        chunks = executor.flatten(bundles)
        print(f"\nPLAN: {len(plan.steps)} step(s); retrieved {len(chunks)} unique chunks")
        for bundle in bundles:
            print(f"  [{bundle.step.step_id}] {bundle.step.description} -> {len(bundle.chunks)} chunks")
            for chunk in bundle.chunks:
                print(f"        {chunk.chunk_id} {chunk.document_id} p{chunk.page_start} "
                      f"[{(chunk.product or '')[:22]}] {chunk.text[:48]!r}...")
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "query": args.query,
                        "analysis": analysis.to_dict(),
                        "plan": plan.to_dict(),
                        "bundles": [b.to_dict() for b in bundles],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"Written to {args.out}")
        return 0

    agent = build_agent(config, matcher, llm, executor, use_llm_analysis=True)
    response = agent.run(args.query, k=args.k)

    print("=" * 64)
    print(f"QUERY: {args.query}")
    print("=" * 64)
    print(f"type={response.query_type} method={response.method} "
          f"products={response.products or '-'} criteria={response.criteria or '-'}")
    if response.clarification:
        print(f"\nCLARIFICATION NEEDED:\n{response.answer}")
    else:
        print(f"\nANSWER ({response.model}):\n")
        print(response.answer)
        if response.citations:
            print("\nCITATIONS:")
            for c in response.citations:
                print(
                    f"  [{c['marker']}] {c['product'] or c['filename']} ({c['pages']})"
                    f" — {c['section'] or 'n/a'}"
                    + (f" — clauses: {', '.join(c['clause_ids'])}" if c["clause_ids"] else "")
                )
        elif response.used_markers:
            print("\n(markers cited but unresolved)")
    lat = response.latency_ms
    print(
        f"\nevidence={response.evidence_count} chunks  steps={len(response.steps)}  "
        f"latency[analysis={lat.get('analysis')}ms plan={lat.get('planning')}ms "
        f"retrieval={lat.get('retrieval')}ms gen={lat.get('generation')}ms "
        f"total={lat.get('total')}ms]"
    )
    print("=" * 64)

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "phase4_query_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(response.to_dict(), ensure_ascii=False) + "\n")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(response.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Full response written to {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4: Agentic Layer")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Phase 4 config JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Classify + plan a query (no retrieval)")
    p_analyze.add_argument("query")
    p_analyze.add_argument("--llm", action="store_true", help="Enable LLM classification fallback")
    p_analyze.add_argument("--k", type=int, help="Top-k per step")
    p_analyze.add_argument("--num-gpu", type=int, default=None)
    p_analyze.add_argument("--out", help="Write analysis + plan JSON here")

    p_query = sub.add_parser("query", help="Run the full agent pipeline")
    p_query.add_argument("query")
    p_query.add_argument("--k", type=int, help="Top-k chunks per step")
    p_query.add_argument("--mode", help="semantic | keyword | hybrid")
    p_query.add_argument("--reranker", choices=["cross-encoder", "lexical", "none"])
    p_query.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    p_query.add_argument("--no-llm", action="store_true", help="Retrieval + plan only")
    p_query.add_argument("--num-gpu", type=int, default=None)
    p_query.add_argument("--out", help="Write the full AgentResponse JSON here")

    sub.add_parser("products", help="List recognised products and aliases")

    args = parser.parse_args()
    setup_logging(args.verbose)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "products":
        return cmd_products(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
