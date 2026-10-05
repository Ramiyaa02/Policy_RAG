"""Tests for Phase 2 — Baseline RAG (chunking, vector store, retrieval, generation).

These tests avoid model downloads: the vector-store and generator tests use
synthetic embeddings and a fake LLM. End-to-end index building is exercised by
run_phase2.py against the real corpus.
"""

import json
import os
import sys

import numpy as np
import pytest

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.chunking import Chunk, Chunker, chunk_corpus
from src.generation.generator import AnswerGenerator, _is_refusal
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import PolicyVectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_block(
    text,
    page,
    order,
    btype="paragraph",
    subtype=None,
    clause_id=None,
    excluded=False,
    heading_level=None,
):
    return {
        "type": btype,
        "subtype": subtype,
        "text": text,
        "page_number": page,
        "reading_order": order,
        "clause_id": clause_id,
        "include_in_chunk_text": False if excluded else True,
        "heading_level": heading_level,
    }


def make_doc(blocks, doc_id="DOC-T", product="Test Product", pages=1):
    page_blocks = {}
    for b in blocks:
        page_blocks.setdefault(b["page_number"], []).append(b)
    return {
        "document_id": doc_id,
        "filename": f"{doc_id}.pdf",
        "source_path": f"/tmp/{doc_id}.pdf",
        "insurer": "Test Insurer",
        "product": product,
        "uin": "TEST123",
        "document_type": "Policy Wording",
        "metadata": {"insurer": "Test Insurer", "product": product, "uin": "TEST123"},
        "pages": [
            {"page_number": p, "blocks": sorted(bs, key=lambda b: b["reading_order"])}
            for p, bs in sorted(page_blocks.items())
        ],
        "detected_tables": [],
    }


@pytest.fixture(scope="module")
def real_chunks():
    """Chunks from the real corpus (requires data/normalized)."""
    normalized_dir = os.path.join(PROJECT_ROOT, "data", "normalized")
    if not os.path.isdir(normalized_dir) or not any(
        f.endswith(".json") for f in os.listdir(normalized_dir)
    ):
        pytest.skip("data/normalized not populated")
    return chunk_corpus(normalized_dir)


# ---------------------------------------------------------------------------
# Chunking (FR-004)
# ---------------------------------------------------------------------------

class TestChunking:
    def test_headers_footers_excluded(self):
        blocks = [
            make_block("Go Digit General Insurance Ltd.", 1, 0, btype="header", excluded=True),
            make_block("Body text about coverage.", 1, 1),
            make_block("1", 1, 2, btype="footer", excluded=True),
        ]
        chunks = Chunker().chunk_document(make_doc(blocks))
        assert len(chunks) == 1
        assert "Go Digit" not in chunks[0].text
        assert "Body text about coverage." in chunks[0].text

    def test_clause_not_split_when_it_fits(self):
        """A single clause shorter than max_chars must stay in one chunk."""
        clause = ("13. Day Care Treatment: means medical treatment carried out. " * 8).strip()
        assert len(clause) < 1300
        blocks = [make_block(clause, 3, 0, btype="clause", clause_id="13")]
        chunks = Chunker().chunk_document(make_doc(blocks))
        assert len(chunks) == 1
        assert chunks[0].clause_ids == ["13"]
        assert chunks[0].text == clause

    def test_clause_boundary_preferred_over_mid_clause_cut(self):
        """Once target size is reached, the next clause starts a new chunk."""
        c1 = "10. Coverage details. " + "word " * 190  # ~970 chars >= target
        c2 = "11. Exclusions apply under certain conditions. " + "word " * 30
        blocks = [
            make_block(c1.strip(), 1, 0, btype="clause", clause_id="10"),
            make_block(c2.strip(), 1, 1, btype="clause", clause_id="11"),
        ]
        chunks = Chunker().chunk_document(make_doc(blocks))
        assert len(chunks) == 2
        assert chunks[0].clause_ids == ["10"]
        assert chunks[1].clause_ids == ["11"]

    def test_oversized_block_split(self):
        """A block longer than max_chars is split at sentence boundaries."""
        text = "Sentence one is here. " * 300  # ~6600 chars, sentence-splittable
        blocks = [make_block(text, 2, 0)]
        chunks = Chunker().chunk_document(make_doc(blocks))
        assert len(chunks) > 1
        tolerance = Chunker().max_chars + Chunker().min_chars
        assert all(c.char_count <= tolerance for c in chunks)

    def test_oversized_block_without_punctuation_hard_split(self):
        """A punctuation-free monster block still gets split (no 10k chunks)."""
        text = "insuranceterms" * 1000  # 14000 chars, no spaces or punctuation
        blocks = [make_block(text, 2, 0)]
        chunks = Chunker().chunk_document(make_doc(blocks))
        tolerance = Chunker().max_chars + Chunker().min_chars
        assert all(c.char_count <= tolerance for c in chunks)

    def test_section_tracking(self):
        blocks = [
            make_block("B. DEFINITIONS Some marketing blurb follows here", 1, 0, btype="clause", clause_id="B"),
            make_block("Hospital means an institution.", 1, 1),
            make_block("ANNEXURE II", 2, 0, btype="heading", subtype="section_heading"),
            make_block("Annex content.", 2, 1),
        ]
        chunks = Chunker().chunk_document(make_doc(blocks))
        assert chunks[0].section == "B. DEFINITIONS"
        assert chunks[0].text.startswith("B. DEFINITIONS")
        assert chunks[1].section == "ANNEXURE II"
        assert chunks[1].subsection is None

    def test_section_prefix_not_duplicated_in_context(self):
        blocks = [
            make_block("B. DEFINITIONS title", 1, 0, btype="clause", clause_id="B"),
            make_block("Definition body text.", 1, 1),
        ]
        chunks = Chunker().chunk_document(make_doc(blocks))
        ctx = chunks[0].context_text
        assert ctx.count("B. DEFINITIONS title") == 1

    def test_chunk_metadata_complete(self):
        blocks = [make_block("Some policy text.", 4, 0)]
        chunks = Chunker().chunk_document(make_doc(blocks))
        c = chunks[0]
        assert c.document_id == "DOC-T"
        assert c.product == "Test Product"
        assert c.insurer == "Test Insurer"
        assert c.uin == "TEST123"
        assert c.page_start == c.page_end == 4
        meta = c.vector_store_metadata()
        for key in (
            "document_id", "filename", "insurer", "product", "uin",
            "document_type", "page_start", "page_end", "section",
            "subsection", "clause_ids", "chunk_type", "char_count",
        ):
            assert key in meta
        # Chroma metadata must be scalar
        assert all(np.isscalar(v) for v in meta.values())

    def test_deterministic_chunk_ids(self):
        blocks = [make_block("Text one.", 1, 0), make_block("Text two.", 1, 1)]
        doc = make_doc(blocks)
        a = Chunker().chunk_document(json.loads(json.dumps(doc)))
        b = Chunker().chunk_document(json.loads(json.dumps(doc)))
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
        assert a[0].chunk_id == "DOC-T::c0000"

    def test_cross_page_continuity_recorded(self):
        blocks = [
            make_block("Clause text starts on page one and", 1, 5),
            make_block("continues on page two.", 2, 0),
        ]
        chunks = Chunker().chunk_document(make_doc(blocks))
        assert len(chunks) == 1
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 2

    def test_real_corpus_chunk_sizes(self, real_chunks):
        """The unbounded-chunk regression: nothing may exceed max+min."""
        chunker = Chunker()
        assert real_chunks, "expected chunks from the real corpus"
        assert all(c.char_count <= chunker.max_chars + chunker.min_chars for c in real_chunks)
        # Tiny chunks may survive merging only at section boundaries; they are
        # bounded and rare.
        tiny = [c for c in real_chunks if c.char_count < chunker.min_chars]
        assert len(tiny) < len(real_chunks) * 0.05
        ids = [c.chunk_id for c in real_chunks]
        assert len(ids) == len(set(ids))

    def test_real_corpus_traceability(self, real_chunks):
        """FR-004: every chunk carries document/product/section/page info."""
        assert all(c.document_id.startswith("DOC-") for c in real_chunks)
        assert all(c.product for c in real_chunks)
        assert all(c.page_start >= 1 for c in real_chunks)
        assert sum(1 for c in real_chunks if c.section) > len(real_chunks) * 0.5


# ---------------------------------------------------------------------------
# Vector store (FR-006)
# ---------------------------------------------------------------------------

class FakeChunk:
    """Minimal duck-typed chunk for store tests."""

    def __init__(self, chunk_id, text, metadata):
        self.chunk_id = chunk_id
        self.context_text = text
        self._meta = metadata

    def vector_store_metadata(self):
        return self._meta


@pytest.fixture()
def tmp_store(tmp_path):
    return PolicyVectorStore(persist_dir=str(tmp_path / "chroma"), collection="test_chunks")


class TestVectorStore:
    def test_upsert_and_count(self, tmp_store):
        chunks = [
            FakeChunk("D::c0000", "first text", {"document_id": "D", "page_start": 1, "page_end": 1, "chunk_type": "prose", "clause_ids": "", "section": ""}),
            FakeChunk("D::c0001", "second text", {"document_id": "D", "page_start": 2, "page_end": 2, "chunk_type": "prose", "clause_ids": "", "section": ""}),
        ]
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert tmp_store.upsert_chunks(chunks, vecs) == 2
        assert tmp_store.count() == 2

    def test_topk_similarity_ordering(self, tmp_store):
        chunks = [
            FakeChunk("D::c0000", "about waiting periods", {"document_id": "D", "chunk_type": "prose", "clause_ids": "", "section": ""}),
            FakeChunk("D::c0001", "about cataract surgery", {"document_id": "D", "chunk_type": "prose", "clause_ids": "", "section": ""}),
        ]
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        tmp_store.upsert_chunks(chunks, vecs)
        hits = tmp_store.query(np.array([0.9, 0.1], dtype=np.float32), k=2)
        assert len(hits) == 2
        assert hits[0]["chunk_id"] == "D::c0000"  # closer to [1, 0]
        assert hits[0]["similarity"] > hits[1]["similarity"]

    def test_metadata_filtering(self, tmp_store):
        """FR-006: metadata filtering narrows the search space."""
        chunks = [
            FakeChunk("D1::c0000", "waiting period text", {"document_id": "DOC-001", "product": "Alpha", "chunk_type": "prose", "clause_ids": "", "section": ""}),
            FakeChunk("D2::c0000", "cataract waiting period", {"document_id": "DOC-002", "product": "Beta", "chunk_type": "prose", "clause_ids": "", "section": ""}),
        ]
        vecs = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        tmp_store.upsert_chunks(chunks, vecs)
        hits = tmp_store.query(np.array([1.0, 1.0], dtype=np.float32), k=5, where={"document_id": {"$eq": "DOC-002"}})
        assert len(hits) == 1
        assert hits[0]["metadata"]["document_id"] == "DOC-002"

    def test_document_level_identification(self, tmp_store):
        """FR-006: stats() reports which documents are in the index."""
        chunks = [
            FakeChunk("D1::c0000", "a", {"document_id": "DOC-001", "chunk_type": "prose", "clause_ids": "", "section": ""}),
            FakeChunk("D2::c0000", "b", {"document_id": "DOC-002", "chunk_type": "prose", "clause_ids": "", "section": ""}),
        ]
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        tmp_store.upsert_chunks(chunks, vecs)
        stats = tmp_store.stats()
        assert stats.count == 2
        assert stats.documents == ["DOC-001", "DOC-002"]

    def test_reset(self, tmp_store):
        chunks = [FakeChunk("D::c0000", "x", {"document_id": "D", "chunk_type": "prose", "clause_ids": "", "section": ""})]
        tmp_store.upsert_chunks(chunks, np.array([[1.0, 0.0]], dtype=np.float32))
        tmp_store.reset()
        assert tmp_store.count() == 0

    def test_length_mismatch_rejected(self, tmp_store):
        chunks = [FakeChunk("D::c0000", "x", {"document_id": "D", "chunk_type": "prose", "clause_ids": "", "section": ""})]
        with pytest.raises(ValueError):
            tmp_store.upsert_chunks(chunks, np.zeros((2, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# Retriever (FR-008 semantic portion)
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Maps keywords to fixed vectors so retrieval order is predictable."""

    def embed_query(self, text):
        if "waiting" in text.lower():
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


class FakeRetrieverStore:
    def __init__(self):
        self.last_where = None

    def query(self, vector, k=5, where=None):
        self.last_where = where
        base = [
            {"chunk_id": "D1::c0000", "text": "waiting period is 36 months", "distance": 0.1, "metadata": {"document_id": "DOC-001", "product": "Alpha", "insurer": "Ins A", "uin": "U1", "document_type": "Policy Wording", "filename": "a.pdf", "page_start": 5, "page_end": 5, "section": "Waiting Periods", "subsection": "", "clause_ids": "4", "chunk_type": "prose"}},
            {"chunk_id": "D2::c0000", "text": "cataract coverage details", "distance": 0.4, "metadata": {"document_id": "DOC-002", "product": "Beta", "insurer": "Ins B", "uin": "U2", "document_type": "Policy Wording", "filename": "b.pdf", "page_start": 9, "page_end": 10, "section": "Coverage", "subsection": "Eye", "clause_ids": "", "chunk_type": "prose"}},
        ]
        return base[:k]


class TestRetriever:
    def test_returns_retrieved_chunks_with_metadata(self):
        r = Retriever(FakeEmbedder(), FakeRetrieverStore())
        hits = r.retrieve("what is the waiting period?", k=2)
        assert len(hits) == 2
        first = hits[0]
        assert first.chunk_id == "D1::c0000"
        assert first.document_id == "DOC-001"
        assert first.clause_ids == ["4"]
        assert first.similarity == pytest.approx(0.9)
        assert "Alpha" in first.citation_label()
        assert "p. 5" in first.citation_label()

    def test_filter_builds_chroma_where(self):
        store = FakeRetrieverStore()
        r = Retriever(FakeEmbedder(), store)
        r.retrieve("q", document_id="DOC-001", product="Alpha", chunk_type="prose")
        assert store.last_where == {
            "$and": [
                {"document_id": {"$eq": "DOC-001"}},
                {"product": {"$eq": "Alpha"}},
                {"chunk_type": {"$eq": "prose"}},
            ]
        }

    def test_single_filter_is_equality(self):
        store = FakeRetrieverStore()
        r = Retriever(FakeEmbedder(), store)
        r.retrieve("q", document_id="DOC-003")
        assert store.last_where == {"document_id": {"$eq": "DOC-003"}}


# ---------------------------------------------------------------------------
# Generation + citations
# ---------------------------------------------------------------------------

class FakeLLM:
    def __init__(self, response="The waiting period is 36 months [1]."):
        self.response = response
        self.model = "fake-llm"
        self.last_prompt = ""
        self.last_system = ""

    def generate(self, prompt, system=None):
        self.last_prompt = prompt
        self.last_system = system or ""
        return self.response


class TestGeneration:
    def _retrieved(self):
        store = FakeRetrieverStore()
        return Retriever(FakeEmbedder(), store).retrieve("waiting period?", k=2)

    def test_prompt_contains_numbered_evidence_and_query(self):
        llm = FakeLLM()
        gen = AnswerGenerator(llm)
        gen.answer("what is the waiting period?", self._retrieved())
        assert "[1]" in llm.last_prompt
        assert "[2]" in llm.last_prompt
        assert "what is the waiting period?" in llm.last_prompt
        assert "ONLY" in llm.last_system  # grounding instruction

    def test_citations_resolved_from_markers(self):
        gen = AnswerGenerator(FakeLLM("It is 36 months [1]. See also [2]."))
        result = gen.answer("waiting period?", self._retrieved())
        assert [c.marker for c in result.citations] == [1, 2]
        c1 = result.citations[0]
        assert c1.document_id == "DOC-001"
        assert c1.pages == "p. 5"
        assert c1.clause_ids == ["4"]
        assert result.used_markers == [1, 2]
        assert not result.abstained

    def test_hallucinated_marker_dropped(self):
        gen = AnswerGenerator(FakeLLM("Claim [7] and [1]."))
        result = gen.answer("q", self._retrieved())
        assert [c.marker for c in result.citations] == [1]
        assert result.used_markers == [1, 7]

    def test_abstention_answer(self):
        gen = AnswerGenerator(FakeLLM("The policy documents provided do not contain this information."))
        result = gen.answer("quantum gravity?", self._retrieved())
        assert result.abstained

    def test_no_evidence_abstains_without_llm_call(self):
        llm = FakeLLM()
        gen = AnswerGenerator(llm)
        result = gen.answer("anything?", [])
        assert result.abstained
        assert result.citations == []
        assert llm.last_prompt == ""  # LLM never called

    def test_refusal_detector(self):
        assert _is_refusal("The policy documents provided do not contain this information.")
        assert not _is_refusal("The waiting period is 24 months [1].")

    def test_evidence_budget_limits_context(self):
        llm = FakeLLM()
        gen = AnswerGenerator(llm, max_evidence_chars=120)
        gen.answer("q", self._retrieved())
        # Only the first chunk fits in the 120-char budget
        assert "[1]" in llm.last_prompt
        assert "[2]" not in llm.last_prompt
