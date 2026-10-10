"""Tests for Phase 3 — Retrieval Improvement.

Covers the four Phase 3 items: metadata filtering, hybrid (BM25 + semantic)
retrieval, reranking and retrieval evaluation. No model downloads: the dense
retriever, reranker and cross-encoder are replaced with fakes/synthetic data.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.retrieval_eval import (
    EvalQuery,
    QueryResult,
    aggregate,
    evaluate,
    load_eval_set,
    ndcg_at_k,
    render_comparison,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    render_markdown,
    write_report,
)
from src.retrieval.bm25 import BM25Index, build_bm25_from_chunks, tokenize
from src.retrieval.hybrid import HybridRetriever, _build_where
from src.retrieval.reranker import LexicalReranker, build_reranker
from src.retrieval.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_bm25():
    docs = [
        ("A::c0", "waiting period for cataract surgery is two years", {"document_id": "DOC-1", "product": "Alpha", "chunk_type": "prose"}),
        ("A::c1", "cataract surgery coverage details and limits", {"document_id": "DOC-1", "product": "Alpha", "chunk_type": "prose"}),
        ("B::c0", "grace period for premium payment is thirty days", {"document_id": "DOC-2", "product": "Beta", "chunk_type": "prose"}),
    ]
    return build_bm25_from_chunks(
        [
            {"chunk_id": cid, "text": text, "context_text": text, **meta}
            for cid, text, meta in docs
        ]
    )


def make_chunks_by_id():
    return {
        "A::c0": {"chunk_id": "A::c0", "text": "waiting period for cataract surgery is two years", "document_id": "DOC-1", "product": "Alpha", "page_start": 3, "page_end": 3, "section": "Waiting Periods", "clause_ids": "4", "chunk_type": "prose"},
        "A::c1": {"chunk_id": "A::c1", "text": "cataract surgery coverage details and limits", "document_id": "DOC-1", "product": "Alpha", "page_start": 5, "page_end": 5, "section": "Coverage", "clause_ids": "", "chunk_type": "prose"},
        "B::c0": {"chunk_id": "B::c0", "text": "grace period for premium payment is thirty days", "document_id": "DOC-2", "product": "Beta", "page_start": 9, "page_end": 9, "section": "General Terms", "clause_ids": "1", "chunk_type": "prose"},
    }


class FakeSemantic:
    """Returns a scripted ranked list and records the filter kwargs it received."""

    def __init__(self, ranked_ids, similarities=None):
        self.ranked_ids = ranked_ids
        self.similarities = similarities or [0.9 - 0.1 * i for i in range(len(ranked_ids))]
        self.calls = []

    def retrieve(self, query, k=5, document_id=None, product=None, chunk_type=None):
        self.calls.append(
            {"query": query, "k": k, "document_id": document_id, "product": product, "chunk_type": chunk_type}
        )
        records = make_chunks_by_id()
        hits = []
        for i, cid in enumerate(self.ranked_ids[:k]):
            chunk = RetrievedChunk.from_chunk(records[cid], similarity=self.similarities[i], source="semantic")
            hits.append(chunk)
        return hits


class FakeReranker:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def rerank(self, query, candidates, top_k):
        self.calls += 1
        out = list(reversed(list(candidates)))
        for i, chunk in enumerate(out):
            chunk.rerank_score = float(100 - i)
        return out[:top_k]


# ---------------------------------------------------------------------------
# BM25 keyword retrieval
# ---------------------------------------------------------------------------

class TestBM25:
    def test_tokenizer_drops_stopwords_and_single_chars(self):
        assert tokenize("The AYUSH a and PED") == ["ayush", "ped"]

    def test_search_returns_relevant_docs_ranked(self):
        bm25 = make_bm25()
        hits = bm25.search("cataract surgery", k=5)
        ids = [cid for cid, _ in hits]
        assert "A::c0" in ids and "A::c1" in ids
        assert "B::c0" not in ids  # no shared terms

    def test_metadata_filter_narrows_search(self):
        bm25 = make_bm25()
        hits = bm25.search("period", k=5, where={"document_id": {"$eq": "DOC-2"}})
        assert [cid for cid, _ in hits] == ["B::c0"]
        assert bm25.search("period", k=5, where={"$and": [{"product": {"$eq": "Alpha"}}]}) != []

    def test_empty_query_returns_nothing(self):
        assert make_bm25().search("the and of", k=5) == []

    def test_roundtrip_persistence(self, tmp_path):
        bm25 = make_bm25()
        path = str(tmp_path / "bm25.json")
        bm25.save(path)
        loaded = BM25Index.load(path)
        assert loaded.stats()["documents"] == bm25.stats()["documents"]
        assert loaded.search("cataract", k=3) == bm25.search("cataract", k=3)

    def test_stats(self):
        stats = make_bm25().stats()
        assert stats["documents"] == 3
        assert stats["unique_terms"] > 0
        assert stats["avg_doc_length"] > 0

    def test_build_validates_lengths(self):
        with pytest.raises(ValueError):
            BM25Index().build(["a"], ["one", "two"])


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

class TestReranker:
    def _candidates(self):
        return [
            RetrievedChunk.from_chunk(make_chunks_by_id()["B::c0"], source="keyword"),
            RetrievedChunk.from_chunk(make_chunks_by_id()["A::c0"], source="keyword"),
        ]

    def test_lexical_reranker_promotes_query_overlap(self):
        rr = LexicalReranker(index=make_bm25())
        out = rr.rerank("cataract surgery", self._candidates(), top_k=2)
        assert out[0].chunk_id == "A::c0"
        assert all(c.rerank_score is not None for c in out)

    def test_lexical_reranker_blends_dense_similarity(self):
        """Identical text -> lexical/coverage tie, so dense similarity must decide."""
        record = make_chunks_by_id()["A::c0"]
        low = RetrievedChunk.from_chunk(record, similarity=0.1, source="semantic")
        low.chunk_id = "low"
        high = RetrievedChunk.from_chunk(record, similarity=0.9, source="semantic")
        high.chunk_id = "high"
        out = LexicalReranker(index=make_bm25()).rerank("cataract surgery", [low, high], top_k=2)
        assert out[0].chunk_id == "high"

    def test_lexical_reranker_differs_from_bm25_on_semantic_evidence(self):
        """Not a BM25 echo: dense similarity + coverage can outrank a high lexical score."""
        lexical_only = RetrievedChunk.from_chunk(  # top BM25 hit, no shared terms, no dense
            make_chunks_by_id()["B::c0"], source="keyword"
        )
        lexical_only.keyword_score = 10.0
        lexical_only.similarity = 0.0
        dense = RetrievedChunk.from_chunk(  # contains the query terms and a strong dense score
            make_chunks_by_id()["A::c0"], source="semantic", similarity=0.95
        )
        dense.keyword_score = 0.0
        out = LexicalReranker(index=make_bm25()).rerank(
            "cataract surgery", [lexical_only, dense], top_k=1
        )
        assert out[0].chunk_id == "A::c0"

    def test_lexical_reranker_rejects_bad_weights(self):
        with pytest.raises(ValueError):
            LexicalReranker(lexical_weight=-1)
        with pytest.raises(ValueError):
            LexicalReranker(lexical_weight=0, semantic_weight=0, coverage_weight=0)

    def test_factory_none_and_lexical(self):
        assert build_reranker("none") is None
        assert isinstance(build_reranker("lexical", index=make_bm25()), LexicalReranker)

    def test_factory_falls_back_when_cross_encoder_unavailable(self, monkeypatch):
        # Simulate a cross-encoder that cannot load (no cache / no network).
        import src.retrieval.reranker as rr_mod

        def boom(*args, **kwargs):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(rr_mod, "CrossEncoderReranker", boom)
        rr = build_reranker("cross-encoder", index=make_bm25(), model_name="cross-encoder/fake")
        assert isinstance(rr, LexicalReranker)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            build_reranker("bogus")


# ---------------------------------------------------------------------------
# Hybrid retrieval + fusion + metadata filtering
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _retriever(self, semantic, reranker=None):
        return HybridRetriever(semantic, make_bm25(), make_chunks_by_id(), reranker)

    def test_semantic_mode_uses_only_semantic(self):
        sem = FakeSemantic(["B::c0", "A::c1"])
        out = self._retriever(sem).retrieve("cataract", k=5, mode="semantic")
        assert [c.chunk_id for c in out] == ["B::c0", "A::c1"]
        assert all(c.sources == ["semantic"] for c in out)

    def test_keyword_mode_uses_only_bm25(self):
        sem = FakeSemantic(["B::c0"])
        out = self._retriever(sem).retrieve("cataract surgery", k=5, mode="keyword")
        assert {c.chunk_id for c in out} == {"A::c0", "A::c1"}
        assert all("keyword" in c.sources for c in out)

    def test_rrf_fusion_agreement_wins_and_sources_merge(self):
        # Both retrievers rank A::c0 first -> it must top the fused list and be
        # deduplicated into a single entry carrying both sources.
        sem = FakeSemantic(["A::c0", "B::c0"])
        out = self._retriever(sem).retrieve("cataract surgery", k=5, mode="hybrid")
        assert out[0].chunk_id == "A::c0"
        ids = [c.chunk_id for c in out]
        assert len(ids) == len(set(ids))
        assert set(out[0].sources) == {"semantic", "keyword"}

    def test_weighted_fusion_respects_alpha(self):
        sem = FakeSemantic(["B::c0", "A::c1"], similarities=[0.9, 0.1])
        out = self._retriever(sem).retrieve(
            "cataract surgery", k=5, mode="hybrid", fusion="weighted", alpha=0.99
        )
        assert out[0].chunk_id == "B::c0"  # semantic dominates

    def test_metadata_filter_reaches_both_paths(self):
        sem = FakeSemantic(["B::c0"])
        retriever = self._retriever(sem)
        out = retriever.retrieve("period", k=5, mode="hybrid", document_id="DOC-2")
        assert [c.chunk_id for c in out] == ["B::c0"]
        assert sem.calls[-1]["document_id"] == "DOC-2"

    def test_rerank_flag_invokes_reranker(self):
        sem = FakeSemantic(["A::c0", "B::c0"])
        rr = FakeReranker()
        out = self._retriever(sem, reranker=rr).retrieve(
            "cataract", k=2, mode="hybrid", rerank=True
        )
        assert rr.calls == 1
        assert all(c.rerank_score is not None for c in out)

    def test_invalid_mode_or_fusion_raises(self):
        retriever = self._retriever(FakeSemantic([]))
        with pytest.raises(ValueError):
            retriever.retrieve("q", mode="nope")
        with pytest.raises(ValueError):
            retriever.retrieve("q", mode="hybrid", fusion="nope")

    def test_where_builder(self):
        assert _build_where() is None
        assert _build_where(document_id="DOC-1") == {"document_id": {"$eq": "DOC-1"}}
        assert _build_where(document_id="DOC-1", product="A") == {
            "$and": [{"document_id": {"$eq": "DOC-1"}}, {"product": {"$eq": "A"}}]
        }


# ---------------------------------------------------------------------------
# Retrieval evaluation metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_precision_at_k(self):
        assert precision_at_k(["a", "b", "c", "d"], {"b", "c"}, 4) == pytest.approx(0.5)
        assert precision_at_k(["a", "b"], {"b"}, 2) == pytest.approx(0.5)
        assert precision_at_k([], {"a"}, 3) == 0.0

    def test_recall_at_k(self):
        assert recall_at_k(["a", "b", "c"], {"b", "c", "d"}, 3) == pytest.approx(2 / 3)
        assert recall_at_k(["a"], {"a", "b"}, 1) == pytest.approx(0.5)
        assert recall_at_k(["a"], set(), 1) == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)
        assert reciprocal_rank(["a"], {"a"}) == 1.0
        assert reciprocal_rank(["x"], {"a"}) == 0.0

    def test_ndcg(self):
        assert ndcg_at_k(["a", "b"], {"a"}, 2) == pytest.approx(1.0)
        assert ndcg_at_k(["b", "a"], {"a"}, 2) == pytest.approx(1 / 1.5849625, rel=1e-4)
        assert ndcg_at_k(["x"], {"a"}, 1) == 0.0


class TestEvaluationRunner:
    def _eval_set(self):
        return [
            EvalQuery("q1", "cataract surgery", ["A::c0", "A::c1"], category="coverage"),
            EvalQuery("q2", "grace period", ["B::c0"], category="factual"),
            EvalQuery("q3", "stock market", [], category="out_of_domain"),
        ]

    def test_evaluate_aggregates_answerable_queries_only(self):
        ranked = {"cataract surgery": ["A::c0", "B::c0", "A::c1"], "grace period": ["B::c0"], "stock market": ["A::c0"]}
        report = evaluate(lambda q, k: ranked[q], self._eval_set(), k=3, label="test")
        assert report["num_answerable"] == 2
        agg = report["aggregate"]
        # q1: 2 relevant of 3 returned -> P@3 = 2/3 ; q2: 1 of 1 -> P@3 = 1.0
        assert agg["precision@3"] == pytest.approx((2 / 3 + 1.0) / 2, rel=1e-3)
        assert agg["recall@3"] == pytest.approx(1.0)
        assert agg["mrr"] == pytest.approx(1.0)
        # unanswerable query is reported but has no metrics
        ood = next(r for r in report["per_query"] if r["qid"] == "q3")
        assert ood["answerable"] is False and ood["metrics"] == {}

    def test_missed_relevant_ids_reported(self):
        report = evaluate(lambda q, k: ["B::c0"], [EvalQuery("q1", "cataract", ["A::c0"])], k=5)
        assert report["per_query"][0]["missed_chunk_ids"] == ["A::c0"]

    def test_aggregate_by_category(self):
        ranked = {"cataract surgery": ["A::c0"], "grace period": ["B::c0"], "stock market": []}
        report = evaluate(lambda q, k: ranked[q], self._eval_set(), k=3)
        assert set(report["by_category"]) == {"coverage", "factual"}

    def test_aggregate_empty(self):
        assert aggregate([]) == {}
        unanswerable = QueryResult("q", "x", "out_of_domain", False, [], [])
        assert aggregate([unanswerable]) == {}

    def test_markdown_and_write_report(self, tmp_path):
        ranked = {"cataract surgery": ["A::c0"], "grace period": ["B::c0"], "stock market": []}
        report = evaluate(lambda q, k: ranked[q], self._eval_set(), k=3, label="hybrid")
        md = render_markdown(report, 3)
        assert "hybrid" in md and "PRECISION@3" in md
        json_path = tmp_path / "r.json"
        md_path = tmp_path / "r.md"
        write_report(report, str(json_path), str(md_path))
        assert json.loads(json_path.read_text(encoding="utf-8"))["label"] == "hybrid"
        assert md_path.read_text(encoding="utf-8").strip()

    def test_combined_report_markdown(self, tmp_path):
        ranked = {"cataract surgery": ["A::c0"], "grace period": ["B::c0"], "stock market": []}
        report = evaluate(lambda q, k: ranked[q], self._eval_set(), k=3, label="hybrid")
        combined = {
            "eval_set": "data/eval/retrieval_eval.json",
            "k": 3,
            "candidate_k": 20,
            "fusion": "rrf",
            "modes": [report],
        }
        md = render_comparison(combined)
        assert "Phase 3 Retrieval Evaluation" in md
        assert "hybrid" in md
        json_path = tmp_path / "c.json"
        md_path = tmp_path / "c.md"
        write_report(combined, str(json_path), str(md_path))
        assert "Summary" in md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Frozen evaluation set (integration)
# ---------------------------------------------------------------------------

class TestFrozenEvalSet:
    def test_eval_set_loads_and_is_well_formed(self):
        path = os.path.join(PROJECT_ROOT, "data", "eval", "retrieval_eval.json")
        if not os.path.exists(path):
            pytest.skip("frozen eval set not present")
        queries = load_eval_set(path)
        assert len(queries) >= 10
        assert all(q.qid and q.query for q in queries)
        assert any(q.answerable for q in queries)

    def test_relevant_chunk_ids_exist_in_corpus(self):
        corpus = os.path.join(PROJECT_ROOT, "data", "chunks", "chunks.jsonl")
        eval_path = os.path.join(PROJECT_ROOT, "data", "eval", "retrieval_eval.json")
        if not (os.path.exists(corpus) and os.path.exists(eval_path)):
            pytest.skip("corpus or eval set not present")
        ids = {json.loads(line)["chunk_id"] for line in open(corpus, encoding="utf-8")}
        queries = load_eval_set(eval_path)
        for q in queries:
            for cid in q.relevant_chunk_ids:
                assert cid in ids, f"{q.qid} references unknown chunk {cid}"
