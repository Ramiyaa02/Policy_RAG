#!/usr/bin/env python
"""Phase 5 — Insurance Policy RAG: Grounding & Verification.

Adds the SRS Phase 5 layer on top of the Phase 4 agentic pipeline:

    claim extraction · evidence verification (Agent 5)
    · unsupported-claim detection · abstention · clarification

Usage:
    python run_phase5.py verify "QUESTION" [--no-llm] [--no-verify-llm]
                                          [--k 5] [--out PATH] [--raw]

Commands
--------
verify  Run the full pipeline (analyze -> plan -> multi-step retrieval ->
        generate -> claim verification -> response shaping) and print the
        verified answer, per-claim verdicts and the grounding report.
        --no-llm replaces generation with a deterministic extractive answer
        built from the retrieved excerpts, so the whole grounding loop runs
        offline (no Ollama server needed).

Settings live in ``configs/phase5.json``; retrieval, agent and generation
settings are inherited from ``configs/phase4.json`` -> ``phase3`` -> ``phase2``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

logger = logging.getLogger("phase5")

DEFAULT_CONFIG = "configs/phase5.json"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    """Load a phase config, overlaying it on its ``base_config`` chain."""
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
# Offline extractive answer (--no-llm)
# ----------------------------------------------------------------------

EXTRACTIVE_SENTENCES = 2


def extractive_answer(query: str, chunks: list) -> tuple[str, list[str]]:
    """Deterministic stand-in for the LLM: quote the top excerpts with [n].

    Only useful for exercising the grounding machinery without an Ollama
    server; it copies up to two sentences straight from each excerpt, so its
    claims are, by construction, supported unless retrieval failed.
    """
    from src.generation.generator import is_refusal

    fallback = "The policy documents provided do not contain this information."
    if not chunks:
        return fallback, []
    lines = []
    for i, chunk in enumerate(chunks[:6], start=1):
        text = (getattr(chunk, "text", "") or "").strip().replace("\n", " ")
        parts = [s.strip() for s in text.split(". ") if len(s.strip()) >= 15]
        for sentence in parts[:EXTRACTIVE_SENTENCES]:
            sentence = sentence.rstrip(".")
            lines.append(f"{sentence}. [{i}]")
    answer = "\n".join(lines)
    if not answer or is_refusal(answer):
        return fallback, []
    return answer, []


class ExtractiveGenerator:
    """Fake AnswerGenerator built on :func:`extractive_answer`."""

    model = "extractive"

    def answer(self, query: str, chunks: list):  # noqa: ANN001
        from src.generation.generator import GeneratedAnswer

        answer, _ = extractive_answer(query, chunks)
        abstained = "do not contain" in answer
        return GeneratedAnswer(
            query=query,
            answer=answer,
            citations=list(result_citations(query, chunks)),
            retrieved=[],
            model=self.model,
            used_markers=[],
            abstained=abstained,
        )


def result_citations(query: str, chunks: list):  # noqa: ANN001
    from src.generation.generator import AnswerGenerator

    # Citations for the extractive answer: marker i -> chunk i for i in 1..k.
    from src.generation.generator import Citation

    out = []
    for i, chunk in enumerate(chunks[:6], start=1):
        pages = (
            f"p. {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pp. {chunk.page_start}-{chunk.page_end}"
        )
        out.append(
            Citation(
                marker=i,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                product=chunk.product,
                insurer=chunk.insurer,
                uin=chunk.uin,
                pages=pages,
                section=chunk.section or "",
                clause_ids=list(chunk.clause_ids or []),
                excerpt=(chunk.text or "")[:300],
            )
        )
    return out


# ----------------------------------------------------------------------
# Pipeline assembly
# ----------------------------------------------------------------------


def build_agent(config: dict, use_extractive: bool, use_verify_llm: bool | None = None):
    from run_phase3 import build_hybrid_retriever
    from src.agent.classifier import ProductMatcher, QueryAnalyzer
    from src.agent.comparison import ComparisonGenerator
    from src.agent.executor import PlanExecutor
    from src.agent.planner import RetrievalPlanner
    from src.generation.generator import AnswerGenerator
    from src.generation.llm import OllamaLLM
    from src.grounding.handler import GroundingHandler
    from src.grounding.pipeline import VerifiedAgent
    from src.grounding.verifier import GroundingValidator
    from src.retrieval.bm25 import load_chunk_corpus

    gcfg = config.get("grounding", {})
    hcfg = config.get("handler", {})
    acfg = config.get("agent", {})
    gen_cfg = config.get("generation", {})

    retriever, _bm25 = build_hybrid_retriever(config, want_reranker=True)
    chunks = load_chunk_corpus(config.get("retrieval", {}).get("chunk_corpus", "data/chunks/chunks.jsonl"))
    products = sorted({c.get("product", "") for c in chunks if c.get("product")})
    alias_map = acfg.get("product_aliases", {}) or {}

    llm = OllamaLLM(
        model=gen_cfg.get("model", "llama3"),
        host=gen_cfg.get("ollama_host", "http://localhost:11434"),
        temperature=float(gen_cfg.get("temperature", 0.2)),
        num_predict=int(gen_cfg.get("num_predict", 512)),
        num_gpu=gen_cfg.get("num_gpu"),
        timeout_seconds=int(gen_cfg.get("timeout_seconds", 180)),
    )
    llm_ready = (not use_extractive) and llm.is_available()
    if not llm_ready and not use_extractive:
        logger.warning("Ollama not reachable at %s; falling back to extractive answers.", llm.host)

    generator = (
        AnswerGenerator(llm, max_evidence_chars=int(gen_cfg.get("max_evidence_chars", 6000)))
        if llm_ready
        else ExtractiveGenerator()
    )
    comparison = (
        ComparisonGenerator(llm, max_evidence_chars=int(config.get("comparison", {}).get("max_evidence_chars", 8000)))
        if llm_ready
        else None
    )

    verify_llm = llm if (llm_ready and use_verify_llm is not False) else None
    restore_num_gpu = None
    if verify_llm is not None:
        # Verification expects short outputs; do not inherit generation's.
        restore_num_gpu = verify_llm.num_gpu
        verify_llm = OllamaLLM(
            model=gen_cfg.get("model", "llama3"),
            host=gen_cfg.get("ollama_host", "http://localhost:11434"),
            temperature=0.0,
            num_predict=64,
            num_gpu=restore_num_gpu,
            timeout_seconds=int(gen_cfg.get("timeout_seconds", 180)),
        )

    validator = GroundingValidator(
        llm=verify_llm,
        support_threshold=float(gcfg.get("support_threshold", 0.35)),
        number_strict=bool(gcfg.get("number_strict", True)),
        verify_conclusions=bool(gcfg.get("verify_conclusions", False)),
        stopwords=gcfg.get("stopwords") or None,
        use_llm=bool(gcfg.get("use_llm_verification", True)) and use_verify_llm is not False,
    )
    handler = GroundingHandler(
        strip_unsupported=bool(hcfg.get("strip_unsupported", True)),
        min_supported_rate=float(hcfg.get("min_supported_rate", 0.5)),
        max_note_claims=int(hcfg.get("max_note_claims", 5)),
    )

    analyzer = QueryAnalyzer(
        llm=llm if (llm_ready and acfg.get("use_llm_analysis", True)) else None,
        matcher=ProductMatcher(products + list(alias_map)),
        use_llm=llm_ready and bool(acfg.get("use_llm_analysis", True)),
    )
    planner = RetrievalPlanner(
        products=products,
        default_k=int(config.get("retrieval", {}).get("default_k", 5)),
        max_steps=int(acfg.get("max_steps", 24)),
        max_products=int(acfg.get("max_products", 4)),
    )
    executor = PlanExecutor(
        retriever,
        mode=config.get("retrieval", {}).get("mode", "hybrid"),
        candidate_k=int(config.get("retrieval", {}).get("candidate_k", 20)),
        rerank=True,
        max_chunks=int(acfg.get("max_evidence_chunks", 80)),
    )
    return VerifiedAgent(
        analyzer=analyzer,
        planner=planner,
        executor=executor,
        generator=generator,
        comparison=comparison,
        validator=validator,
        handler=handler,
    )


# ----------------------------------------------------------------------
# Command: verify
# ----------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    from src.grounding.schema import Verdict

    config = load_config(args.config)
    agent = build_agent(config, use_extractive=args.no_llm, use_verify_llm=None if not args.no_verify_llm else False)

    t0 = time.time()
    response = agent.run(args.query, k=args.k)
    elapsed = time.time() - t0

    result = response.to_dict()
    result["wall_s"] = round(elapsed, 2)

    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("reports", "phase5_query_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "query": args.query,
                    "query_type": response.query_type,
                    "abstained": response.abstained,
                    "supported_rate": response.supported_rate,
                    "counts": result.get("counts", {}),
                    "latency_ms": result.get("latency_ms", {}),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print("\n" + "=" * 60)
    print(f"QUERY: {args.query}")
    print("=" * 60)
    print(
        f"type={response.query_type} evidence={response.evidence_count} "
        f"latency[retrieval={response.latency_ms.get('retrieval', 0)}ms "
        f"gen={response.latency_ms.get('generation', 0)}ms "
        f"verify={response.latency_ms.get('verification', 0)}ms] "
        f"total={response.latency_ms.get('total', 0)}ms"
    )

    if response.abstained:
        print("\nABSTAINED" + (" (clarification asked)" if response.clarification else ""))
    print("\nVERIFIED ANSWER:\n")
    print(response.final_answer or response.answer)

    counts = result.get("counts", {})
    print(
        f"\nGROUNDING: {counts.get('supported', 0)} supported / "
        f"{counts.get('contradicted', 0)} contradicted / "
        f"{counts.get('insufficient', 0)} insufficient "
        f"of {counts.get('claims', 0)} claims (supported_rate={response.supported_rate:.2f})"
    )
    if response.verifications:
        print("\nCLAIMS:")
        for v in response.verifications:
            marker = {
                Verdict.SUPPORTED: "OK  ",
                Verdict.CONTRADICTED: "FAIL",
                Verdict.INSUFFICIENT: "MISS",
                Verdict.NOT_VERIFIED: "----",
            }[v.verdict]
            print(
                f"  [{marker}] ({v.verdict.value:<12} score={v.support_score:.2f}, {v.method}) "
                f"{v.claim.text[:100]}"
            )
    for note in response.notes:
        print(f"\n{note}")
    if response.citations:
        print("\nCITATIONS:")
        for c in response.citations:
            print(
                f"  [{c['marker']}] {c.get('product') or c.get('filename')} ({c.get('pages')})"
                f" — {c.get('section') or 'n/a'}"
                + (f" — clauses: {', '.join(c.get('clause_ids') or [])}" if c.get("clause_ids") else "")
                + f"  [{c.get('document_id')} {c.get('chunk_id')}]"
            )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nFull result written to {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5: Grounding & Verification")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Phase 5 config JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="Verified query end-to-end")
    p_verify.add_argument("query")
    p_verify.add_argument("--k", type=int, help="Top-k chunks per retrieval step")
    p_verify.add_argument("--no-llm", action="store_true", help="Extractive answer, no Ollama")
    p_verify.add_argument(
        "--no-verify-llm", action="store_true", help="Rules-only verification (no LLM pass)"
    )
    p_verify.add_argument("--out", help="Write full JSON result to this path")

    args = parser.parse_args()
    setup_logging(args.verbose)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.command == "verify":
        return cmd_verify(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
