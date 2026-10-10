#!/usr/bin/env python
"""Phase 3 — Insurance Policy RAG: Retrieval Improvement.

Builds on the Phase 2 baseline (semantic retrieval + grounded generation) with
the four Phase 3 items from the SRS:

    metadata filtering · hybrid retrieval · reranking · retrieval evaluation

Usage:
    python run_phase3.py build  [--reset] [--config configs/phase3.json]
    python run_phase3.py query "QUESTION" [--mode hybrid] [--k 5]
                                          [--candidates 20] [--no-rerank]
                                          [--document-id DOC-006] [--product "..."]
                                          [--chunk-type prose] [--no-llm] [--out PATH]
    python run_phase3.py eval   [--k 5] [--modes semantic,keyword,hybrid,hybrid+rerank]
                                [--out reports/phase3_retrieval_eval.json]
    python run_phase3.py stats  [--config configs/phase3.json]

Commands
--------
build   Build and cache the BM25 keyword index over data/chunks/chunks.jsonl.
query   Hybrid (semantic + BM25, rank-fused, optionally reranked) retrieval and
        a grounded answer with [n] citations.
eval    Retrieval evaluation against the frozen query set: Precision@K,
        Recall@K, MRR, nDCG@K, and a baseline (semantic) vs hybrid comparison.
stats   BM25 corpus stats plus the dense vector-index stats.

The dense index and generation model come from Phase 2 (configs/phase2.json),
so run `python run_phase2.py build` first; this phase only adds the lexical
half, fusion, reranking and evaluation on top.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

logger = logging.getLogger("phase3")

DEFAULT_CONFIG = "configs/phase3.json"


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (nested dicts merged)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> dict:
    """Load a phase config, overlaying it on its ``base_config`` chain.

    Follows ``base_config`` recursively, so ``configs/phase4.json`` -> phase3 ->
    phase2 all merge into one settings tree.
    """
    config = load_json(path)
    base_path = config.get("base_config")
    if base_path and os.path.exists(base_path) and os.path.abspath(base_path) != os.path.abspath(path):
        config = deep_merge(load_config(base_path), config)
    return config


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


# ----------------------------------------------------------------------
# Retriever assembly
# ----------------------------------------------------------------------

def load_or_build_bm25(config: dict, rebuild: bool = False):
    from src.retrieval.bm25 import BM25Index, build_bm25_from_chunks, load_chunk_corpus

    rcfg = config.get("retrieval", {})
    corpus_path = rcfg.get("chunk_corpus", "data/chunks/chunks.jsonl")
    index_path = rcfg.get("bm25_index", "data/bm25/bm25_index.json")
    bm25_cfg = rcfg.get("bm25", {})

    if not rebuild and os.path.exists(index_path):
        try:
            return BM25Index.load(index_path)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning("Could not load cached BM25 index (%s); rebuilding.", exc)

    chunks = load_chunk_corpus(corpus_path)
    bm25 = build_bm25_from_chunks(
        chunks, k1=float(bm25_cfg.get("k1", 1.5)), b=float(bm25_cfg.get("b", 0.75))
    )
    # Cache only when the corpus exists on disk (query/eval convenience).
    try:
        bm25.save(index_path)
    except OSError as exc:  # pragma: no cover - filesystem issue
        logger.warning("Could not cache BM25 index: %s", exc)
    return bm25


def build_hybrid_retriever(
    config: dict,
    rebuild_bm25: bool = False,
    want_reranker: bool = True,
    reranker_kind: str | None = None,
):
    """Assemble a :class:`HybridRetriever` from the merged config.

    Returns ``(retriever, bm25)``. The dense retriever and reranker are lazy:
    if the Phase 2 index or an embedding model is unavailable the lexical and
    reranking paths still work, which keeps `stats`/`keyword` usable offline.
    """
    from src.retrieval.bm25 import load_chunk_corpus
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.reranker import build_reranker

    rcfg = config.get("retrieval", {})
    rrcfg = config.get("reranker", {})
    corpus_path = rcfg.get("chunk_corpus", "data/chunks/chunks.jsonl")

    bm25 = load_or_build_bm25(config, rebuild=rebuild_bm25)
    chunks = load_chunk_corpus(corpus_path)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    semantic = None
    try:
        from src.embeddings import Embedder
        from src.retrieval.retriever import Retriever
        from src.retrieval.vector_store import PolicyVectorStore

        embed_cfg = config.get("embeddings", {})
        store_cfg = config.get("vector_store", {})
        store = PolicyVectorStore(
            persist_dir=store_cfg.get("persist_dir", "data/chroma"),
            collection=store_cfg.get("collection", "policy_chunks"),
        )
        if store.count() == 0:
            logger.warning("Vector index is empty; semantic retrieval disabled.")
        else:
            embedder = Embedder(
                model_name=embed_cfg.get("model", "all-MiniLM-L6-v2"),
                device=embed_cfg.get("device"),
                batch_size=int(embed_cfg.get("batch_size", 64)),
            )
            semantic = Retriever(embedder, store)
    except Exception as exc:  # noqa: BLE001 - degrade to keyword-only
        logger.warning("Semantic retriever unavailable (%s); keyword-only mode.", exc)

    reranker = None
    if want_reranker and rrcfg.get("enabled", True):
        reranker = build_reranker(
            reranker_kind or rrcfg.get("kind", "cross-encoder"),
            index=bm25,
            model_name=rrcfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=rrcfg.get("device"),
            batch_size=int(rrcfg.get("batch_size", 32)),
            lexical_weight=float(rrcfg.get("lexical_weight", 0.20)),
            semantic_weight=float(rrcfg.get("semantic_weight", 0.80)),
            coverage_weight=float(rrcfg.get("coverage_weight", 0.0)),
        )
    return HybridRetriever(semantic, bm25, chunks_by_id, reranker), bm25


def parse_mode(spec: str) -> tuple[str, bool]:
    """'hybrid+rerank' -> ('hybrid', True); 'semantic' -> ('semantic', False)."""
    spec = spec.strip().lower()
    if "+" in spec:
        mode, _, suffix = spec.partition("+")
        return mode, suffix in {"rerank", "reranked"}
    return spec, False


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rcfg = config.get("retrieval", {})
    t0 = time.time()
    bm25 = load_or_build_bm25(config, rebuild=True)
    stats = bm25.stats()
    print("\n" + "=" * 60)
    print("PHASE 3 KEYWORD INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Corpus:               {rcfg.get('chunk_corpus', 'data/chunks/chunks.jsonl')}")
    print(f"Documents indexed:    {stats['documents']}")
    print(f"Unique terms:         {stats['unique_terms']}")
    print(f"Avg doc length:       {stats['avg_doc_length']} tokens")
    print(f"BM25 k1/b:            {stats['k1']} / {stats['b']}")
    print(f"Cached at:            {rcfg.get('bm25_index', 'data/bm25/bm25_index.json')}")
    print(f"Built in:             {time.time() - t0:.1f}s")
    print("=" * 60)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rcfg = config.get("retrieval", {})
    gen_cfg = config.get("generation", {})

    want_reranker = not args.no_rerank
    retriever, bm25 = build_hybrid_retriever(
        config, want_reranker=want_reranker, reranker_kind=args.reranker
    )

    mode, rerank_from_mode = parse_mode(args.mode or rcfg.get("mode", "hybrid"))
    rerank = rerank_from_mode or (not args.no_rerank and retriever.reranker is not None)
    k = args.k or int(rcfg.get("default_k", 5))
    candidate_k = args.candidates or int(rcfg.get("candidate_k", 20))
    fusion = args.fusion or rcfg.get("fusion", "rrf")

    if retriever.bm25.n_docs == 0:
        print("Keyword index is empty. Run: python run_phase3.py build", file=sys.stderr)
        return 1

    t0 = time.time()
    retrieved = retriever.retrieve(
        args.query,
        k=k,
        mode=mode,
        candidate_k=candidate_k,
        fusion=fusion,
        alpha=float(rcfg.get("alpha", 0.5)),
        rrf_k=int(rcfg.get("rrf_k", 60)),
        rerank=rerank,
        document_id=args.document_id,
        product=args.product,
        chunk_type=args.chunk_type,
    )
    retrieval_ms = (time.time() - t0) * 1000
    if not retrieved:
        print("No chunks matched the query/filters.", file=sys.stderr)
        return 1

    result: dict = {
        "query": args.query,
        "mode": mode,
        "reranked": bool(rerank and retriever.reranker is not None),
        "fusion": fusion,
        "candidate_k": candidate_k,
        "retrieval_ms": round(retrieval_ms, 1),
        "k": k,
        "retrieved": [r.__dict__ for r in retrieved],
    }

    if not args.no_llm:
        from src.generation.generator import AnswerGenerator
        from src.generation.llm import OllamaLLM

        llm = OllamaLLM(
            model=gen_cfg.get("model", "llama3"),
            host=gen_cfg.get("ollama_host", "http://localhost:11434"),
            temperature=float(gen_cfg.get("temperature", 0.2)),
            num_predict=int(gen_cfg.get("num_predict", 512)),
            num_gpu=(args.num_gpu if args.num_gpu is not None else gen_cfg.get("num_gpu")),
            timeout_seconds=int(gen_cfg.get("timeout_seconds", 180)),
        )
        if not llm.is_available():
            print(
                f"Ollama server not reachable at {llm.host}. Start it with: ollama serve",
                file=sys.stderr,
            )
            return 1
        generator = AnswerGenerator(
            llm, max_evidence_chars=int(gen_cfg.get("max_evidence_chars", 6000))
        )
        t0 = time.time()
        answer = generator.answer(args.query, retrieved)
        result.update(
            {
                "answer": answer.answer,
                "citations": [c.to_dict() for c in answer.citations],
                "used_markers": answer.used_markers,
                "abstained": answer.abstained,
                "model": answer.model,
                "generation_s": round(time.time() - t0, 2),
            }
        )

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "phase3_query_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "query": args.query,
                    "mode": mode,
                    "reranked": result["reranked"],
                    "fusion": fusion,
                    "k": k,
                    "retrieval_ms": result["retrieval_ms"],
                    "chunk_ids": [r.chunk_id for r in retrieved],
                    "abstained": result.get("abstained", False),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print("\n" + "=" * 60)
    print(f"QUERY: {args.query}")
    print("=" * 60)
    print(
        f"mode={mode} fusion={fusion} reranked={result['reranked']} "
        f"candidate_k={candidate_k} k={k}"
    )
    if "answer" in result:
        print(f"\nANSWER ({result['model']}, {result['generation_s']}s):\n")
        print(result["answer"])
        print("\nCITATIONS:")
        for c in result["citations"]:
            print(
                f"  [{c['marker']}] {c['product'] or c['filename']} ({c['pages']})"
                f" — {c['section'] or 'n/a'}"
                + (f" — clauses: {', '.join(c['clause_ids'])}" if c["clause_ids"] else "")
                + f"  [{c['document_id']} {c['chunk_id']}]"
            )
    print(f"\nRETRIEVED {len(retrieved)} chunks in {result['retrieval_ms']} ms:")
    for i, r in enumerate(retrieved, start=1):
        score = (
            f"rerank={r.rerank_score:.3f}"
            if r.rerank_score is not None
            else f"fused={r.fused_score:.4f}"
            if r.fused_score is not None
            else f"sim={r.similarity:.3f}"
        )
        print(
            f"  [{i}] {score} kw={r.keyword_score or 0.0:.2f} "
            f"{r.document_id} {r.chunk_id} pp.{r.page_start}-{r.page_end} "
            f"src={','.join(r.sources) or 'semantic'} | {r.text[:55]!r}..."
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nFull result written to {args.out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from src.evaluation.retrieval_eval import evaluate, load_eval_set, write_report

    config = load_config(args.config)
    rcfg = config.get("retrieval", {})
    eval_cfg = config.get("evaluation", {})

    eval_set_path = eval_cfg.get("eval_set", "data/eval/retrieval_eval.json")
    if not os.path.exists(eval_set_path):
        print(f"Evaluation set not found: {eval_set_path}", file=sys.stderr)
        return 1
    queries = load_eval_set(eval_set_path)

    mode_specs = (
        args.modes.split(",")
        if args.modes
        else eval_cfg.get("modes", ["semantic", "keyword", "hybrid", "hybrid+rerank"])
    )
    k = args.k or int(eval_cfg.get("k", 5))
    candidate_k = args.candidates or int(rcfg.get("candidate_k", 20))
    fusion = args.fusion or rcfg.get("fusion", "rrf")

    retriever, _ = build_hybrid_retriever(
        config, want_reranker=True, reranker_kind=args.reranker
    )

    reports = []
    for spec in mode_specs:
        spec = spec.strip()
        if not spec:
            continue
        mode, rerank = parse_mode(spec)
        want_rerank = rerank or spec.endswith("+rerank")
        if mode == "semantic" and retriever.semantic is None:
            logger.warning("Skipping mode '%s': dense retriever unavailable.", spec)
            continue

        def retrieve_fn(query: str, kk: int, _mode=mode, _rr=want_rerank) -> list[str]:
            hits = retriever.retrieve(
                query,
                k=kk,
                mode=_mode,
                candidate_k=candidate_k,
                fusion=fusion,
                alpha=float(rcfg.get("alpha", 0.5)),
                rrf_k=int(rcfg.get("rrf_k", 60)),
                rerank=_rr,
            )
            return [c.chunk_id for c in hits]

        t0 = time.time()
        report = evaluate(retrieve_fn, queries, k=k, label=spec)
        report["elapsed_s"] = round(time.time() - t0, 2)
        reports.append(report)

    if not reports:
        print("No retrieval modes could be evaluated.", file=sys.stderr)
        return 1

    out_path = args.out or os.path.join(
        eval_cfg.get("report_dir", "reports"), "phase3_retrieval_eval.json"
    )
    md_path = os.path.splitext(out_path)[0] + ".md"
    combined = {
        "eval_set": eval_set_path,
        "k": k,
        "candidate_k": candidate_k,
        "fusion": fusion,
        "generated": datetime.now(timezone.utc).isoformat(),
        "modes": reports,
    }
    write_report(combined, out_path, md_path)

    print("\n" + "=" * 78)
    print(f"PHASE 3 RETRIEVAL EVALUATION (k={k}, candidate_k={candidate_k}, fusion={fusion})")
    print("=" * 78)
    print(f"Frozen set: {eval_set_path}  ({len(queries)} queries)")
    header = f"{'mode':<16} {'P@k':>7} {'R@k':>7} {'nDCG@k':>7} {'Hit@k':>7} {'MRR':>7} {'MAP':>7}"
    print(header)
    print("-" * len(header))
    for report in reports:
        agg = report["aggregate"]
        print(
            f"{report['label']:<16} "
            f"{agg.get(f'precision@{k}', 0):>7.3f} "
            f"{agg.get(f'recall@{k}', 0):>7.3f} "
            f"{agg.get(f'ndcg@{k}', 0):>7.3f} "
            f"{agg.get(f'hit@{k}', 0):>7.3f} "
            f"{agg.get('mrr', 0):>7.3f} "
            f"{agg.get('map', 0):>7.3f}"
        )
    print("-" * len(header))
    baseline = next((r for r in reports if r["label"] == "semantic"), None)
    hybrid = next((r for r in reports if r["label"].startswith("hybrid")), None)
    if baseline and hybrid:
        print("\nBaseline (semantic) vs hybrid:")
        for metric in (f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "mrr"):
            b = baseline["aggregate"].get(metric, 0.0)
            h = hybrid["aggregate"].get(metric, 0.0)
            print(f"  {metric:<14} semantic={b:.3f}  hybrid={h:.3f}  delta={h - b:+.3f}")
    print(f"\nReports: {out_path}\n         {md_path}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _, bm25 = build_hybrid_retriever(config, want_reranker=False)
    stats = bm25.stats()
    print("=" * 60)
    print("PHASE 3 RETRIEVAL STATS")
    print("=" * 60)
    print("Keyword index (BM25):")
    print(f"  Documents:     {stats['documents']}")
    print(f"  Unique terms:  {stats['unique_terms']}")
    print(f"  Avg doc len:   {stats['avg_doc_length']} tokens")
    print(f"  k1 / b:        {stats['k1']} / {stats['b']}")
    try:
        from src.retrieval.vector_store import PolicyVectorStore

        store_cfg = config.get("vector_store", {})
        store = PolicyVectorStore(
            persist_dir=store_cfg.get("persist_dir", "data/chroma"),
            collection=store_cfg.get("collection", "policy_chunks"),
        )
        dense = store.stats()
        print("Dense index (Chroma):")
        print(f"  Collection:    {dense.collection}")
        print(f"  Chunks:        {dense.count}")
        print(f"  Documents:     {len(dense.documents)}")
    except Exception as exc:  # noqa: BLE001
        print(f"Dense index unavailable: {exc}")
    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: Retrieval Improvement")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Phase 3 config JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build and cache the BM25 keyword index")
    p_build.add_argument("--reset", action="store_true", help="Rebuild even if cached")

    p_query = sub.add_parser("query", help="Hybrid retrieval (+ optional answer)")
    p_query.add_argument("query")
    p_query.add_argument("--mode", help="semantic | keyword | hybrid (append '+rerank')")
    p_query.add_argument("--k", type=int, help="Final top-k chunks")
    p_query.add_argument("--candidates", type=int, help="Per-retriever candidate pool size")
    p_query.add_argument("--fusion", choices=["rrf", "weighted"], help="Fusion strategy")
    p_query.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    p_query.add_argument(
        "--reranker",
        choices=["cross-encoder", "lexical", "none"],
        help="Override the configured reranker (lexical works offline)",
    )
    p_query.add_argument("--document-id", help="Filter to one document (e.g. DOC-006)")
    p_query.add_argument("--product", help="Filter by product name")
    p_query.add_argument("--chunk-type", help="Filter by chunk type (prose/table)")
    p_query.add_argument("--no-llm", action="store_true", help="Retrieval only")
    p_query.add_argument("--num-gpu", type=int, default=None, help="Ollama GPU layers (0=CPU)")
    p_query.add_argument("--out", help="Write full JSON result to this path")

    p_eval = sub.add_parser("eval", help="Retrieval evaluation on the frozen query set")
    p_eval.add_argument("--k", type=int, help="Cutoff K")
    p_eval.add_argument("--candidates", type=int, help="Per-retriever candidate pool size")
    p_eval.add_argument("--fusion", choices=["rrf", "weighted"])
    p_eval.add_argument("--modes", help="Comma-separated modes, e.g. semantic,keyword,hybrid")
    p_eval.add_argument(
        "--reranker",
        choices=["cross-encoder", "lexical", "none"],
        help="Override the configured reranker (lexical works offline)",
    )
    p_eval.add_argument("--out", help="Write the evaluation report JSON here")

    sub.add_parser("stats", help="Show keyword + dense index stats")

    args = parser.parse_args()
    setup_logging(args.verbose)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.command == "build":
        return cmd_build(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "eval":
        return cmd_eval(args)
    if args.command == "stats":
        return cmd_stats(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
