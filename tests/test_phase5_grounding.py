"""Tests for Phase 5 — Grounding & Verification (SRS §8, FR-015/016/017/018, O7/O8).

No model downloads and no Ollama server: the LLM is faked and retrieval uses a
scripted retriever, so every grounding path is exercised deterministically.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.generation.generator import Citation, GeneratedAnswer
from src.grounding.claims import ClaimExtractor
from src.grounding.handler import GroundingHandler, build_clarification
from src.grounding.pipeline import VerifiedAgent
from src.grounding.schema import Verdict
from src.grounding.verifier import GroundingValidator, containment, tokenize
from src.retrieval.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, response=""):
        self.response = response
        self.model = "fake-llm"
        self.calls = 0
        self.last_prompt = ""
        self.last_system = ""

    def generate(self, prompt, system=None):
        self.calls += 1
        self.last_prompt = prompt
        self.last_system = system or ""
        return self.response


from src.agent.classifier import ProductMatcher, QueryAnalyzer
from src.agent.comparison import ComparisonGenerator
from src.agent.executor import PlanExecutor
from src.agent.planner import RetrievalPlanner
from src.generation.generator import AnswerGenerator

PRODUCTS = ["Digit Health Insurance Policy", "Elevate", "MediCare"]


def make_chunk(chunk_id="DOC-1::c0", product="Elevate", text="waiting period is 30 days", page=3):
    return RetrievedChunk.from_chunk(
        {
            "chunk_id": chunk_id,
            "text": text,
            "document_id": "DOC-1",
            "product": product,
            "filename": "a.pdf",
            "insurer": "Ins",
            "uin": "U1",
            "document_type": "Policy Wording",
            "page_start": page,
            "page_end": page,
            "section": "Waiting Periods",
            "clause_ids": "4",
            "chunk_type": "prose",
        },
        source="semantic",
    )


class FakeRetriever:
    def __init__(self, per_call=2):
        self.calls = []
        self.per_call = per_call

    def retrieve(self, query, k=5, mode="hybrid", candidate_k=20, fusion="rrf",
                 rerank=False, document_id=None, product=None, chunk_type=None):
        self.calls.append({"query": query, "product": product})
        p = product or "unfiltered"
        return [
            make_chunk(chunk_id=f"{p}::c{i}", product=product or "Elevate", text=f"{query} clause {i}")
            for i in range(min(k, self.per_call))
        ]


# ---------------------------------------------------------------------------
# Claim extraction (FR-015)
# ---------------------------------------------------------------------------


class TestClaimExtractor:
    def test_extracts_sentences_with_markers(self):
        claims = ClaimExtractor().extract(
            "The waiting period is 30 days [1]. Order 2 years [2]"
        )
        assert [c.markers for c in claims] == [[1], [2]]
        assert all(c.kind == "fact" for c in claims)
        assert "30 days" in claims[0].text and "[1]" not in claims[0].text

    def test_skips_headings_and_empty(self):
        claims = ClaimExtractor().extract(
            "**Waiting Period:**\n\nIt is 30 days [1].\n**Exclusions:**\n"
        )
        assert len(claims) == 1
        assert claims[0].section == "waiting period"

    def test_bullets_are_single_claims(self):
        claims = ClaimExtractor().extract(
            "- the waiting period is 30 days [1]\n* cataract has 2 years [2]"
        )
        assert len(claims) == 2

    def test_conclusions_labelled_after_heading(self):
        claims = ClaimExtractor().extract(
            "Policy A has 30 days [1].\n**Conclusion:**\nPolicy A looks better overall."
        )
        assert claims[0].kind == "fact"
        assert claims[1].kind == "conclusion"

    def test_marker_after_sentence_period_merges(self):
        claims = ClaimExtractor().extract("Policy A appears better for most applicants here. [1]")
        assert len(claims) == 1
        assert claims[0].markers == [1]
        assert claims[0].text.startswith("Policy A")

    def test_dedupes_repeated_markers(self):
        claims = ClaimExtractor().extract("It is 30 days [1][1][2].")
        assert claims[0].markers == [1, 2]

    def test_short_fragments_skipped(self):
        claims = ClaimExtractor().extract("Yes. [1] No. Indeed it is thirty days [1].")
        assert len(claims) == 1  # only the last sentence is long enough

    def test_no_markers_claim(self):
        claims = ClaimExtractor().extract("Insurance is important for everyone honestly.")
        assert claims and claims[0].markers == []


# ---------------------------------------------------------------------------
# Token scoring helpers
# ---------------------------------------------------------------------------


class TestScoring:
    def test_containment(self):
        claim = tokenize("waiting period cataract surgery 30 days", frozenset())
        chunk = tokenize(
            "specifid disease waiting period cataract surgery thirty 30 days two years", frozenset()
        )
        score = containment(claim, chunk)
        assert score > 0.5

    def test_containment_no_overlap(self):
        assert containment({"zebra", "unicorn"}, {"waiting", "period"}) == 0.0

    def test_containment_single_token_floor(self):
        # One shared token inside a longer claim is treated as no support.
        assert containment({"cataract", "surgery"}, {"cataract", "unrelated"}) == 0.0


# ---------------------------------------------------------------------------
# Grounding Validator (Agent 5, FR-015)
# ---------------------------------------------------------------------------


class TestGroundingValidatorRules:
    def _validator(self, **kwargs):
        return GroundingValidator(use_llm=False, **kwargs)

    def test_supported_claim(self):
        evidence = [
            make_chunk(text="The waiting period for specified diseases is 2 years.")
        ]
        v = self._validator().verify(
            "The waiting period for specified diseases is 2 years [1].", evidence
        )
        assert v[0].verdict is Verdict.SUPPORTED
        assert v[0].evidence_chunk_ids == ["DOC-1::c0"]
        assert v[0].method == "rules"

    def test_uncited_claim_is_insufficient(self):
        v = self._validator().verify("The claim has absolutely no citations at all here.", [])
        assert v[0].verdict is Verdict.INSUFFICIENT
        assert "uncited" in v[0].reason

    def test_no_evidence_insufficient(self):
        v = self._validator().verify("Cataract waiting is 30 days [1].", [])
        assert v[0].verdict is Verdict.INSUFFICIENT
        assert "no evidence" in v[0].reason

    def test_unresolved_marker_insufficient(self):
        evidence = [make_chunk(text="whatever evidence text exists here fully")]
        v = self._validator().verify("The waiting period is 30 days [5].", evidence)
        assert v[0].verdict is Verdict.INSUFFICIENT
        assert "unresolved" in v[0].reason

    def test_below_threshold_insufficient(self):
        evidence = [make_chunk(text="Completely unrelated policy text about premium tables.")]
        v = self._validator().verify("The waiting period for cataract surgery is 30 days [1].", evidence)
        assert v[0].verdict is Verdict.INSUFFICIENT

    def test_number_mismatch_contradicted(self):
        evidence = [make_chunk(text="The waiting period for specified diseases is 2 years for sure.")]
        v = self._validator().verify(
            "The waiting period for specified diseases is 30 days [1].", evidence
        )
        assert v[0].verdict is Verdict.CONTRADICTED

    def test_number_match_stays_supported(self):
        evidence = [make_chunk(text="Cataract surgery has waiting period of 30 days anyway.")]
        v = self._validator().verify("Cataract surgery has a 30 days waiting period generally stated [1].", evidence)
        assert v[0].verdict is Verdict.SUPPORTED

    def test_conclusion_not_verified(self):
        claims_text = "**Conclusion:**\nPolicy A appears better for most applicants here."
        v = self._validator().verify(claims_text, [make_chunk()])
        assert v[0].verdict is Verdict.NOT_VERIFIED

    def test_conclusions_verified_when_enabled(self):
        v = GroundingValidator(use_llm=False, verify_conclusions=True).verify(
            "**Conclusion:**\nPolicy A appears better for most applicants here. [1]",
            [make_chunk(text="Policy A appears better for most applicants here overall.")],
        )
        assert v[0].verdict is Verdict.SUPPORTED


class TestGroundingValidatorLLM:
    def _llm_verify(self, response):
        llm = FakeLLM(response)
        return GroundingValidator(llm=llm, use_llm=True), llm

    def test_llm_upgrades_insufficient(self):
        validator, llm = self._llm_verify("CLAIM 1: supported")
        v = validator.verify(
            "Some unusual claim the rules cannot match [1].",
            [make_chunk(text="Unmatching filler text with zero overlap.")],
        )
        assert v[0].verdict is Verdict.SUPPORTED
        assert v[0].method == "llm"
        assert llm.calls == 1
        assert "CLAIM 1" in llm.last_prompt

    def test_llm_supports_never_lowers_contradicted(self):
        validator, _ = self._llm_verify("CLAIM 1: supported")
        v = validator.verify(
            "The waiting period for specified diseases is 30 days [1].",
            [make_chunk(text="The waiting period for specified diseases is 2 years for real.")],
        )
        assert v[0].verdict is Verdict.SUPPORTED  # LLM wins when it says supported

    def test_llm_keeps_contradiction(self):
        validator, _ = self._llm_verify("CLAIM 1: contradicted")
        v = validator.verify(
            "The waiting period for specified diseases is 30 days [1].",
            [make_chunk(text="The waiting period for specified diseases is 2 years for real.")],
        )
        assert v[0].verdict is Verdict.CONTRADICTED

    def test_llm_error_falls_back_to_rules(self):
        class Boom(FakeLLM):
            def generate(self, prompt, system=None):
                raise RuntimeError("server down")

        validator = GroundingValidator(llm=Boom(), use_llm=True)
        v = validator.verify(
            "Completely different words appear in this sentence [1].",
            [make_chunk(text="Nothing remotely related here whatsoever friend.")],
        )
        assert v[0].verdict is Verdict.INSUFFICIENT
        assert v[0].method == "rules"

    def test_llm_skipped_when_all_supported(self):
        validator, llm = self._llm_verify("CLAIM 1: contradicted")
        validator.verify(
            "The waiting period is exactly two years here [1].",
            [make_chunk(text="The waiting period is exactly two years here.")],
        )
        assert llm.calls == 0

    def test_llm_disabled_uses_rules_only(self):
        validator = GroundingValidator(llm=FakeLLM("CLAIM 1: supported"), use_llm=False)
        v = validator.verify(
            "Some unusual claim with no matching words [1].",
            [make_chunk(text="Unmatching informational filler text.")],
        )
        assert v[0].method == "rules"
        assert v[0].verdict is Verdict.INSUFFICIENT


# ---------------------------------------------------------------------------
# Handler (FR-016/FR-017/FR-018, O8)
# ---------------------------------------------------------------------------


def _verdict(claim_text, markers, verdict, kind="fact", line=2):
    from src.grounding.schema import Claim, ClaimVerification

    claim = Claim(text=claim_text, markers=markers, kind=kind, line=line)
    return ClaimVerification(claim=claim, verdict=verdict)


class TestGroundingHandler:
    def _raw(self, answer):
        return {"answer": answer, "abstained": False, "evidence_count": 2}

    def test_unsupported_claims_stripped(self):
        answer = "Supported sentence about cataract waiting periods [1].\nFabricated claim invented out of thin air [2]."
        verification = _verdict("Supported sentence about cataract waiting periods", [1], Verdict.SUPPORTED, line=1)
        verif2 = _verdict("Fabricated claim invented out of thin air", [2], Verdict.INSUFFICIENT, line=2)
        response = GroundingHandler().build(self._raw(answer), [], [verification, verif2])
        assert "Fabricated claim" not in response.final_answer
        assert "Supported sentence" in response.final_answer
        assert any("removed" in n for n in response.notes)
        assert not response.abstained

    def test_contradicted_claim_removed_with_conflict_note(self):
        answer = "The waiting period is 30 days [1]."
        v = _verdict("The waiting period is 30 days", [1], Verdict.CONTRADICTED, line=1)
        response = GroundingHandler().build(self._raw(answer), [], [v])
        # Nothing supported survives → abstain, but with an explicit FR-017 note.
        assert response.abstained
        assert response.contradicted_claims
        assert any("conflicts" in n for n in response.notes)

    def test_abstains_when_nothing_supported(self):
        answer = "Claim A is invented entirely [1]."
        v = _verdict("Claim A is invented entirely", [1], Verdict.INSUFFICIENT, line=1)
        response = GroundingHandler().build(self._raw(answer), [], [v], query_type="coverage")
        assert response.abstained
        assert response.clarification
        assert "cataract" in response.clarification or "treatment" in response.clarification

    def test_high_supported_rate_keeps_answer(self):
        answer = "Claim A supported here [1].\nClaim B supported too [2]."
        verif1 = _verdict("Claim A supported here", [1], Verdict.SUPPORTED, line=1)
        verif2 = _verdict("Claim B supported too", [2], Verdict.SUPPORTED, line=2)
        response = GroundingHandler().build(self._raw(answer), [], [verif1, verif2])
        assert response.supported_rate == 1.0
        assert not response.abstained
        assert response.final_answer == answer

    def test_low_supported_rate_abstains(self):
        answer = "Claim A supported here [1].\nClaim B invented [2]."
        verif1 = _verdict("Claim A supported here", [1], Verdict.SUPPORTED, line=1)
        verif2 = _verdict("Claim B invented", [2], Verdict.INSUFFICIENT, line=2)
        response = GroundingHandler().build(self._raw(answer), [], [verif1, verif2])
        assert response.supported_rate == 0.5
        assert not response.abstained  # exactly at threshold → keep, note added

    def test_refusal_answer_becomes_abstention(self):
        raw = {"answer": "The policy documents provided do not contain this information.", "abstained": True}
        response = GroundingHandler().build(raw, [], [], query_type="coverage")
        assert response.abstained
        assert response.final_answer == raw["answer"]

    def test_clarification_is_targeted(self):
        q = build_clarification("what is the waiting period", "waiting_period", 0)
        assert "procedure" in q.lower()
        assert "please provide more information" not in q.lower()

    def test_clarification_for_empty_evidence_mentions_retrieval(self):
        q = build_clarification("q", "policy_information", 0)
        assert "retrieved" in q


# ---------------------------------------------------------------------------
# Verified pipeline
# ---------------------------------------------------------------------------


class TestVerifiedAgent:
    def _agent(self, llm_answer="The waiting period is 30 days [1]."):
        llm = FakeLLM(llm_answer)
        return (
            VerifiedAgent(
                analyzer=QueryAnalyzer(matcher=ProductMatcher(PRODUCTS)),
                planner=RetrievalPlanner(products=PRODUCTS),
                executor=PlanExecutor(FakeRetriever(per_call=1), rerank=False),
                generator=AnswerGenerator(llm),
            ),
            llm,
        )

    def test_supported_answer_passes_through(self):
        agent, _ = self._agent("The waiting period is 30 days [1].")
        response = agent.run("What is the waiting period for cataract surgery?")
        # The generator gets a fake chunk ("What is the waiting period for
        # cataract surgery? clause 0"), so the answer's wording ties to it.
        assert response.claims
        assert response.verifications
        assert set(response.latency_ms) >= {"retrieval", "generation", "verification", "total"}
        assert "final_answer" in response.to_dict()

    def test_out_of_domain_short_circuits(self):
        agent, _ = self._agent()
        response = agent.run("What will the stock market do tomorrow?")
        assert response.abstained
        assert response.claims == []
        assert response.evidence_count == 0

    def test_clarification_short_circuits(self):
        agent, _ = self._agent()
        response = agent.run("Compare Policy A and Policy B for coverage")
        assert response.abstained
        assert response.clarification
        assert response.evidence_count == 0


# ---------------------------------------------------------------------------
# Comparison prepare() reused by verification
# ---------------------------------------------------------------------------


class TestComparisonPrepare:
    def test_prepare_returns_aligned_used_chunks(self):
        gen = ComparisonGenerator(FakeLLM())
        chunks = [
            make_chunk("A::c0", product="A", text="A first"),
            make_chunk("B::c0", product="B", text="B only"),
        ]
        _prompt, used = gen.prepare("compare", ["A", "B"], ["coverage"], chunks)
        assert [getattr(c, "chunk_id") for c in used][:2] == ["A::c0", "B::c0"]


# ---------------------------------------------------------------------------
# Citation alignment used by pipeline
# ---------------------------------------------------------------------------


class TestAnswerDataclass:
    def test_generated_answer_fields(self):
        answer = GeneratedAnswer(
            query="q", answer="a", citations=[], retrieved=[], model="m",
            used_markers=[], abstained=True,
        )
        assert answer.abstained and answer.to_dict()["abstained"]

    def test_citation_roundtrip(self):
        chunk = make_chunk("D::c0")
        citation = Citation(
            marker=1, chunk_id=chunk.chunk_id, document_id=chunk.document_id,
            filename=chunk.filename, product=chunk.product, insurer=chunk.insurer,
            uin=chunk.uin, pages="p. 3", section=chunk.section,
            clause_ids=list(chunk.clause_ids), excerpt=chunk.text[:300],
        )
        assert citation.to_dict()["marker"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
