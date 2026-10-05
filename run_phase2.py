#!/usr/bin/env python
"""Phase 2 — Insurance Policy RAG: Baseline RAG.

Usage:
    python run_phase2.py build  [--reset] [--model NAME] [--config configs/phase2.json]
    python run_phase2.py query "QUESTION" [--k 5] [--document-id DOC-001]
                                       [--product "..."] [--chunk-type prose]
                                       [--no-llm] [--out reports/answer.json]
    python run_phase2.py stats  [--config configs/phase2.json]

Commands
--------
build   Chunk the normalized corpus, embed chunks and index them into the
        ChromaDB vector store. Idempotent (chunk_id upserts); --reset rebuilds.
query   Retrieve top-k chunks for a question and generate a grounded answer
        with [n] citations (requires a running Ollama server; --no-llm skips
        generation and returns retrieval only).
stats   Show index statistics.

Phase 2 STOP boundary — no hybrid retrieval, reranking, agentic planning,
claim verification, or comparison workflows (Phases 3-5).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

logger = logging.getLogger("phase2")

DEFAULT_CONFIG = "configs/phase2.json"


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def cmd_build(args: argparse.Namespace) -> int:
    from src.chunking import chunk_corpus
    from src.embeddings import Embedder
    from src.retrieval.vector_store import PolicyVectorStore

    config = load_config(args.config)
    chunk_cfg = {**config.get("chunking", {})}
    embed_cfg = {**config.get("embeddings", {})}
    store_cfg = {**config.get("vector_store", {})}

    if args.model:
        embed_cfg["model"] = args.model
    if args.device:
        embed_cfg["device"] = args.device

    # 1. Chunk
    t0 = time.time()
    chunks = chunk_corpus(args.normalized_dir, **chunk_cfg)
    if not chunks:
        logger.error("No chunks produced — is data/normalized populated? Run phase 1 first.")
        return 1
    logger.info("Chunked %d chunks in %.1fs", len(chunks), time.time() - t0)

    # 2. Embed
    embedder = Embedder(
        model_name=embed_cfg.get("model", "all-MiniLM-L6-v2"),
        device=embed_cfg.get("device"),
        batch_size=int(embed_cfg.get("batch_size", 64)),
    )
    t0 = time.time()
    vectors = embedder.embed_texts([c.context_text for c in chunks])
    logger.info(
        "Embedded %d chunks (dim=%d) in %.1fs",
        len(chunks),
        vectors.shape[1],
        time.time() - t0,
    )

    # 3. Index
    store = PolicyVectorStore(
        persist_dir=store_cfg.get("persist_dir", "data/chroma"),
        collection=store_cfg.get("collection", "policy_chunks"),
        distance=store_cfg.get("distance", "cosine"),
    )
    if args.reset:
        store.reset()
    store.upsert_chunks(chunks, vectors)

    stats = store.save_stats(
        os.path.join(store_cfg.get("persist_dir", "data/chroma"), "index_stats.json"),
        embedding_model=embedder.model_name,
        dimension=embedder.dimension,
    )
    for chunk in chunks:
        chunk.embedding_model = embedder.model_name

    # 4. Persist chunks + embedding descriptor for reproducibility/debugging
    from src.chunking.chunker import chunks_to_jsonl

    os.makedirs("data/chunks", exist_ok=True)
    n_written = chunks_to_jsonl(chunks, os.path.join("data", "chunks", "chunks.jsonl"))

    print("\n" + "=" * 60)
    print("PHASE 2 INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Documents:            {len(stats.documents)}")
    print(f"Chunks indexed:       {stats.count} (jsonl: {n_written})")
    print(f"Embedding model:      {stats.embedding_model} (dim {stats.dimension})")
    print(f"Vector store:         {store_cfg.get('persist_dir', 'data/chroma')}")
    print("=" * 60)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from src.embeddings import Embedder
    from src.retrieval.retriever import Retriever
    from src.retrieval.vector_store import PolicyVectorStore

    config = load_config(args.config)
    embed_cfg = {**config.get("embeddings", {})}
    store_cfg = {**config.get("vector_store", {})}
    gen_cfg = {**config.get("generation", {})}
    retrieval_cfg = {**config.get("retrieval", {})}

    store = PolicyVectorStore(
        persist_dir=store_cfg.get("persist_dir", "data/chroma"),
        collection=store_cfg.get("collection", "policy_chunks"),
    )
    if store.count() == 0:
        print("Index is empty. Run: python run_phase2.py build", file=sys.stderr)
        return 1

    embedder = Embedder(
        model_name=embed_cfg.get("model", "all-MiniLM-L6-v2"),
        device=embed_cfg.get("device"),
    )
    retriever = Retriever(embedder, store)
    k = args.k or int(retrieval_cfg.get("default_k", 5))

    t0 = time.time()
    retrieved = retriever.retrieve(
        args.query,
        k=k,
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
            num_gpu=(
                args.num_gpu
                if args.num_gpu is not None
                else gen_cfg.get("num_gpu", None)
            ),
            timeout_seconds=int(gen_cfg.get("timeout_seconds", 180)),
        )
        if not llm.is_available():
            print(
                f"Ollama server not reachable at {llm.host}. "
                "Start it with: ollama serve",
                file=sys.stderr,
            )
            return 1
        generator = AnswerGenerator(
            llm, max_evidence_chars=int(gen_cfg.get("max_evidence_chars", 6000))
        )
        t0 = time.time()
        answer = generator.answer(args.query, retrieved)
        generation_s = time.time() - t0
        result.update(
            {
                "answer": answer.answer,
                "citations": [c.to_dict() for c in answer.citations],
                "used_markers": answer.used_markers,
                "abstained": answer.abstained,
                "model": answer.model,
                "generation_s": round(generation_s, 2),
            }
        )

    # Observability (NFR-009): append the query log
    os.makedirs("reports", exist_ok=True)
    log_path = os.path.join("reports", "phase2_query_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "query": args.query,
                    "k": k,
                    "retrieval_ms": result.get("retrieval_ms"),
                    "used_markers": result.get("used_markers", []),
                    "abstained": result.get("abstained", False),
                    "model": result.get("model"),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    # Pretty print
    print("\n" + "=" * 60)
    print(f"QUERY: {args.query}")
    print("=" * 60)
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
        if not result["citations"] and result["used_markers"]:
            print("  (markers cited but unresolved)")
    print(f"\nRETRIEVED {len(retrieved)} chunks in {result['retrieval_ms']} ms:")
    for i, r in enumerate(retrieved, start=1):
        print(
            f"  [{i}] sim={r.similarity:.3f} {r.document_id} {r.chunk_id}"
            f" pp.{r.page_start}-{r.page_end} {r.product}"
            f" | {(r.section or 'n/a')[:50]} | {r.text[:60]!r}..."
        )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nFull result written to {args.out}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from src.retrieval.vector_store import PolicyVectorStore

    config = load_config(args.config)
    store_cfg = {**config.get("vector_store", {})}
    store = PolicyVectorStore(
        persist_dir=store_cfg.get("persist_dir", "data/chroma"),
        collection=store_cfg.get("collection", "policy_chunks"),
    )
    stats = store.stats()
    print("=" * 60)
    print("PHASE 2 INDEX STATS")
    print("=" * 60)
    print(f"Collection:      {stats.collection}")
    print(f"Chunks:          {stats.count}")
    print(f"Documents:       {len(stats.documents)}")
    for doc in stats.documents:
        print(f"  - {doc}")
    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: Baseline RAG")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Phase 2 config JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Chunk, embed and index the corpus")
    p_build.add_argument("--normalized-dir", default="data/normalized")
    p_build.add_argument("--reset", action="store_true", help="Rebuild the collection")
    p_build.add_argument("--model", help="Override embedding model name")
    p_build.add_argument("--device", help="Override torch device (cpu/cuda)")

    p_query = sub.add_parser("query", help="Ask a question against the index")
    p_query.add_argument("query", help="The question to ask")
    p_query.add_argument("--k", type=int, help="Top-k chunks to retrieve")
    p_query.add_argument("--document-id", help="Filter to one document (e.g. DOC-001)")
    p_query.add_argument("--product", help="Filter by product name")
    p_query.add_argument("--chunk-type", help="Filter by chunk type (prose/table)")
    p_query.add_argument("--no-llm", action="store_true", help="Retrieval only")
    p_query.add_argument(
        "--num-gpu",
        type=int,
        default=None,
        help=(
            "Ollama GPU layers (0 = CPU only). Use 0 when the local GPU "
            "cannot run the model."
        ),
    )
    p_query.add_argument("--out", help="Write full JSON result to this path")

    sub.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Windows consoles default to cp1252; policy text is Unicode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.command == "build":
        return cmd_build(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "stats":
        return cmd_stats(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
