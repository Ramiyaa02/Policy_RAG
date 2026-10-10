"""Tests for Phase 4 — Agentic Layer (SRS FR-007/012/013/014/018/019).

No model downloads and no Ollama server: the LLM is faked and retrieval uses a
scripted retriever, so every agent path is exercised deterministically.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.agent.agent import OUT_OF_DOMAIN_ANSWER, PolicyRAGAgent
from src.agent.classifier import ProductMatcher, QueryAnalyzer, extract_requirements
from src.agent.comparison import ComparisonGenerator
from src.agent.executor import PlanExecutor
from src.agent.planner import RetrievalPlanner
from src.agent.schema import QueryType, RetrievalPlan, UserRequirements
from src.generation.generator import AnswerGenerator
from src.retrieval.retriever import RetrievedChunk


PRODUCTS = ["Digit Health Insurance Policy", "Elevate", "MediCare", "my:Optima Secure", "Activ One"]


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------

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
    """Returns a per-product chunk and records every call's filters."""

    def __init__(self, per_call=2):
        self.calls = []
        self.per_call = per_call

    def retrieve(self, query, k=5, mode="hybrid", candidate_k=20, fusion="rrf",
                 rerank=False, document_id=None, product=None, chunk_type=None):
        self.calls.append({"query": query, "k": k, "product": product, "document_id": document_id})
        p = product or "unfiltered"
        return [
            make_chunk(chunk_id=f"{p}::c{i}", product=product or "Elevate", text=f"{query} [{i}]")
            for i in range(min(k, self.per_call))
        ]


class FakeLLM:
    def __init__(self, response="The waiting period is 30 days [1]."):
        self.response = response
        self.model = "fake-llm"
        self.last_prompt = ""
        self.last_system = ""
        self.calls = 0

    def generate(self, prompt, system=None):
        self.calls += 1
        self.last_prompt = prompt
        self.last_system = system or ""
        return self.response


def analyzer_with(products=PRODUCTS, llm=None, use_llm=False):
    return QueryAnalyzer(llm=llm, matcher=ProductMatcher(products), use_llm=use_llm)


# ---------------------------------------------------------------------------
# Product matching
# ---------------------------------------------------------------------------

class TestProductMatcher:
    def test_alias_matching(self):
        matcher = ProductMatcher(PRODUCTS)
        assert matcher.find("Compare Digit and Elevate for waiting periods") == [
            "Digit Health Insurance Policy",
            "Elevate",
        ]
        assert matcher.find("my optima secure room rent") == ["my:Optima Secure"]

    def test_word_boundary_avoids_substring_false_positive(self):
        # "care" must not match inside "MediCare" or "start".
        matcher = ProductMatcher(["MediCare", "Ultimate Care"])
        assert matcher.find("does Medicare cover this") == ["MediCare"]
        assert matcher.find("start a claim") == []

    def test_from_chunks(self):
        matcher = ProductMatcher.from_chunks(
            [{"product": "A"}, {"product": "B"}, {"product": "A"}, {}]
        )
        assert matcher.products == ["A", "B"]

    def test_no_product_mention(self):
        assert ProductMatcher(PRODUCTS).find("what is the waiting period") == []


# ---------------------------------------------------------------------------
# Requirement extraction (FR-013)
# ---------------------------------------------------------------------------

class TestRequirements:
    def test_age_and_sum_insured(self):
        req = extract_requirements(
            "recommend a policy for a 45-year-old with 10 lakh coverage"
        )
        assert req.age == 45
        assert req.sum_insured == "10 lakh"

    def test_crore_and_coverage_objective(self):
        req = extract_requirements("a 1 crore maternity cover for age 30")
        assert req.sum_insured == "1 crore"
        assert req.coverage_objective == "maternity"
        assert req.age == 30

    def test_priorities_and_tenure(self):
        req = extract_requirements(
            "low waiting period and cheaper premium, 2 year tenure"
        )
        assert "low waiting period" in req.priorities
        assert "cheaper premium" in req.priorities
        assert req.tenure == "2 years"

    def test_empty(self):
        assert extract_requirements("what is the waiting period?").is_empty


# ---------------------------------------------------------------------------
# Query Analyzer (FR-007)
# ---------------------------------------------------------------------------

class TestQueryAnalyzerRules:
    def test_comparison(self):
        a = analyzer_with().analyze("Compare the Digit Health Insurance Policy and Elevate")
        assert a.query_type is QueryType.PRODUCT_COMPARISON
        assert a.is_complex
        assert set(a.products) == {"Digit Health Insurance Policy", "Elevate"}

    def test_recommendation(self):
        a = analyzer_with().analyze("Which policy should I buy? I am 45 years old")
        assert a.query_type is QueryType.RECOMMENDATION
        assert a.requirements.age == 45

    def test_waiting_period(self):
        a = analyzer_with().analyze("What is the waiting period for cataract surgery?")
        assert a.query_type is QueryType.WAITING_PERIOD

    def test_exclusion(self):
        a = analyzer_with().analyze("Are dental treatments excluded?")
        assert a.query_type is QueryType.EXCLUSION

    def test_eligibility(self):
        a = analyzer_with().analyze("What is the entry age limit for this policy?")
        assert a.query_type is QueryType.ELIGIBILITY

    def test_coverage(self):
        a = analyzer_with().analyze("Is AYUSH treatment covered?")
        assert a.query_type is QueryType.COVERAGE

    def test_multi_step(self):
        a = analyzer_with().analyze("Is AYUSH covered and what is the waiting period?")
        assert a.query_type is QueryType.MULTI_STEP
        assert a.is_complex
        assert len(a.detected_types) >= 2

    def test_policy_information_default(self):
        a = analyzer_with().analyze("What is the free look period?")
        assert a.query_type is QueryType.POLICY_INFORMATION

    def test_out_of_domain(self):
        assert analyzer_with().analyze(
            "What will the stock market do tomorrow?"
        ).query_type is QueryType.OUT_OF_DOMAIN
        assert analyzer_with().analyze(
            "Write a Python function to sort a list"
        ).query_type is QueryType.OUT_OF_DOMAIN

    def test_criteria_detected(self):
        a = analyzer_with().analyze("compare waiting period and exclusions")
        assert "waiting period" in a.criteria
        assert "exclusions" in a.criteria


class TestQueryAnalyzerLLMFallback:
    def test_llm_used_when_low_confidence(self):
        # In-domain (mentions "policy") but carries no category cue, so the rules
        # path is low-confidence and the LLM is consulted.
        llm = FakeLLM('{"query_type": "coverage", "criteria": ["coverage"]}')
        a = analyzer_with(llm=llm, use_llm=True).analyze("Tell me about my policy")
        assert a.method == "llm"
        assert a.query_type is QueryType.COVERAGE

    def test_llm_skipped_when_confident(self):
        llm = FakeLLM('{"query_type": "exclusion"}')
        a = analyzer_with(llm=llm, use_llm=True).analyze("Compare Digit and Elevate")
        assert a.method == "rules"
        assert llm.calls == 0

    def test_unparseable_llm_falls_back(self):
        llm = FakeLLM("I think it is about coverage")
        a = analyzer_with(llm=llm, use_llm=True).analyze("Tell me about my policy")
        assert a.method == "rules"

    def test_unknown_type_falls_back(self):
        llm = FakeLLM('{"query_type": "banana"}')
        a = analyzer_with(llm=llm, use_llm=True).analyze("Tell me about my policy")
        assert a.method == "rules"

    def test_llm_error_falls_back(self):
        class Boom(FakeLLM):
            def generate(self, prompt, system=None):
                raise RuntimeError("server down")

        a = analyzer_with(llm=Boom(), use_llm=True).analyze("Tell me about my policy")
        assert a.method == "rules"

    def test_llm_requirements_merged(self):
        llm = FakeLLM('{"query_type": "recommendation", "requirements": {"age": 60}}')
        a = analyzer_with(llm=llm, use_llm=True).analyze("Tell me about my policy options")
        assert a.method == "llm"
        assert a.requirements.age == 60


# ---------------------------------------------------------------------------
# Retrieval Planner (FR-014, FR-018)
# ---------------------------------------------------------------------------

class TestRetrievalPlanner:
    def _planner(self):
        return RetrievalPlanner(products=PRODUCTS, max_products=4, max_steps=24)

    def test_simple_plan_single_step(self):
        a = analyzer_with().analyze("What is the waiting period for cataract surgery?")
        plan = self._planner().plan(a)
        assert len(plan.steps) == 1
        assert plan.steps[0].query == a.query
        assert not plan.needs_clarification

    def test_comparison_plan_is_product_by_criterion(self):
        a = analyzer_with().analyze("Compare Digit and Elevate for waiting period and exclusions")
        plan = self._planner().plan(a, k=4)
        assert plan.needs_comparison
        assert {s.product for s in plan.steps} == {"Digit Health Insurance Policy", "Elevate"}
        assert len(plan.steps) == 4  # 2 products x 2 criteria
        assert all(s.top_k == 4 for s in plan.steps)
        assert all(s.criteria for s in plan.steps)

    def test_comparison_with_unspecified_products_uses_all_capped(self):
        a = analyzer_with().analyze("Compare the policies for waiting period")
        plan = self._planner().plan(a)
        assert len(plan.products) == 4  # capped at max_products
        assert not plan.needs_clarification

    def test_comparison_unknown_named_products_asks_clarification(self):
        a = analyzer_with().analyze("Compare Policy A and Policy B for hospitalization")
        plan = self._planner().plan(a)
        assert plan.needs_clarification
        assert plan.clarification_question

    def test_recommendation_plan_uses_all_candidates(self):
        a = analyzer_with().analyze(
            "Which policy should I buy for a 45-year-old with 10 lakh coverage?"
        )
        plan = self._planner().plan(a)
        assert plan.needs_comparison
        assert plan.products  # candidate set
        assert all(s.product for s in plan.steps)

    def test_recommendation_without_requirements_asks_clarification(self):
        a = analyzer_with().analyze("Please recommend a policy for me")
        plan = self._planner().plan(a)
        assert plan.needs_clarification
        assert "age" in (plan.clarification_question or "").lower()

    def test_multi_step_decomposition(self):
        a = analyzer_with().analyze("Is AYUSH covered and what is the waiting period?")
        plan = self._planner().plan(a)
        assert len(plan.steps) >= 2
        assert not plan.needs_comparison

    def test_out_of_domain_has_no_steps(self):
        a = analyzer_with().analyze("What will the stock market do tomorrow?")
        plan = self._planner().plan(a)
        assert plan.steps == []
        assert a.query_type is QueryType.OUT_OF_DOMAIN

    def test_max_steps_cap(self):
        planner = RetrievalPlanner(products=PRODUCTS, max_products=4, max_steps=3)
        a = analyzer_with().analyze("Compare Digit and Elevate for coverage, waiting period and exclusions")
        plan = planner.plan(a)
        assert len(plan.steps) == 3


# ---------------------------------------------------------------------------
# Plan Executor
# ---------------------------------------------------------------------------

class TestPlanExecutor:
    def test_executes_each_step_with_filters(self):
        retriever = FakeRetriever(per_call=2)
        executor = PlanExecutor(retriever, rerank=False)
        a = analyzer_with().analyze("Compare Digit and Elevate for waiting period and exclusions")
        plan = RetrievalPlanner(products=PRODUCTS).plan(a, k=2)
        bundles = executor.execute(plan)
        assert len(bundles) == len(plan.steps)
        assert len(retriever.calls) == len(plan.steps)
        assert {c["product"] for c in retriever.calls} == {"Digit Health Insurance Policy", "Elevate"}

    def test_flatten_dedupes(self):
        class DupRetriever(FakeRetriever):
            def retrieve(self, *a, **k):
                return [make_chunk(chunk_id="X::c0"), make_chunk(chunk_id="X::c0")]

        executor = PlanExecutor(DupRetriever(), rerank=False)
        plan = RetrievalPlan(query="q", steps=[], needs_comparison=False)
        from src.agent.schema import PlanStep

        plan.steps = [PlanStep(1, "s", "q1", top_k=2), PlanStep(2, "s", "q2", top_k=2)]
        bundles = executor.execute(plan)
        assert len(executor.flatten(bundles)) == 1

    def test_max_chunks_budget(self):
        retriever = FakeRetriever(per_call=5)
        executor = PlanExecutor(retriever, rerank=False, max_chunks=3)
        from src.agent.schema import PlanStep

        plan = RetrievalPlan(query="q")
        plan.steps = [PlanStep(i, "s", f"q{i}", top_k=5) for i in range(1, 5)]
        bundles = executor.execute(plan)
        assert len(executor.flatten(bundles)) <= 3

    def test_group_by_product(self):
        retriever = FakeRetriever(per_call=1)
        executor = PlanExecutor(retriever, rerank=False)
        a = analyzer_with().analyze("Compare Digit and Elevate for coverage")
        plan = RetrievalPlanner(products=PRODUCTS).plan(a, k=1)
        grouped = executor.group_by_product(executor.execute(plan))
        assert set(grouped) == {"Digit Health Insurance Policy", "Elevate"}


# ---------------------------------------------------------------------------
# Comparison generator
# ---------------------------------------------------------------------------

class TestComparisonGenerator:
    def _chunks(self):
        return [
            make_chunk("D::c0", product="Digit Health Insurance Policy", text="Digit waiting is 30 days"),
            make_chunk("E::c0", product="Elevate", text="Elevate waiting is 24 months"),
        ]

    def test_prompt_groups_evidence_and_lists_criteria(self):
        gen = ComparisonGenerator(FakeLLM())
        prompt = gen.build_prompt("compare", ["Digit Health Insurance Policy", "Elevate"], ["waiting period"], self._chunks())
        assert "[1]" in prompt and "[2]" in prompt
        assert "Digit waiting is 30 days" in prompt
        assert "waiting period" in prompt

    def test_prompt_balances_evidence_across_products(self):
        # One product floods the pool; round-robin must still surface the other
        # policy's evidence before the character budget is exhausted.
        chunks = [
            make_chunk(f"A::{i}", product="A", text="padding " * 200) for i in range(10)
        ]
        chunks.append(make_chunk("B::0", product="B", text="B unique evidence clause"))
        gen = ComparisonGenerator(FakeLLM(), max_evidence_chars=1200, max_excerpt_chars=200)
        prompt = gen.build_prompt("compare", ["A", "B"], ["coverage"], chunks)
        assert "B unique evidence clause" in prompt

    def test_answer_resolves_citations(self):
        llm = FakeLLM("Digit waits 30 days [1] while Elevate waits 24 months [2].")
        gen = ComparisonGenerator(llm)
        result = gen.answer("compare", ["Digit Health Insurance Policy", "Elevate"], ["waiting period"], self._chunks())
        assert [c.marker for c in result.citations] == [1, 2]
        assert result.citations[0].product == "Digit Health Insurance Policy"
        assert not result.abstained
        assert "ONLY" in llm.last_system

    def test_citations_match_interleaved_prompt_numbering(self):
        # Input order differs from the round-robin order: interleaved is
        # [A0, B0, A1], so marker [2] must resolve to B0, not A1.
        chunks = [
            make_chunk("A::c0", product="A", text="A first"),
            make_chunk("A::c1", product="A", text="A second"),
            make_chunk("B::c0", product="B", text="B only"),
        ]
        llm = FakeLLM("the B clause is at [2] and A at [1][3].")
        gen = ComparisonGenerator(llm)
        result = gen.answer("compare", ["A", "B"], ["coverage"], chunks)
        by_marker = {c.marker: c for c in result.citations}
        assert by_marker[1].chunk_id == "A::c0"
        assert by_marker[2].chunk_id == "B::c0"
        assert by_marker[3].chunk_id == "A::c1"

    def test_abstains_without_evidence(self):
        llm = FakeLLM()
        gen = ComparisonGenerator(llm)
        result = gen.answer("compare", ["A", "B"], ["coverage"], [])
        assert result.abstained
        assert result.citations == []
        assert llm.calls == 0

    def test_refusal_detected(self):
        gen = ComparisonGenerator(FakeLLM("The policy documents provided do not contain this information."))
        result = gen.answer("compare", ["A"], ["coverage"], self._chunks())
        assert result.abstained


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TestPolicyRAGAgent:
    def _agent(self, llm=None, per_call=2):
        llm = llm or FakeLLM()
        executor = PlanExecutor(FakeRetriever(per_call=per_call), rerank=False)
        return PolicyRAGAgent(
            analyzer=analyzer_with(),
            planner=RetrievalPlanner(products=PRODUCTS),
            executor=executor,
            generator=AnswerGenerator(llm),
            comparison=ComparisonGenerator(llm),
        )

    def test_simple_query_uses_answer_generator(self):
        agent = self._agent(FakeLLM("The waiting period is 30 days [1]."))
        resp = agent.run("What is the waiting period for cataract surgery?")
        assert resp.query_type == "waiting_period"
        assert resp.answer
        assert resp.citations
        assert resp.evidence_count >= 1
        assert set(resp.latency_ms) >= {"analysis", "planning", "retrieval", "generation", "total"}

    def test_comparison_uses_comparison_generator(self):
        llm = FakeLLM("Digit waits 30 days [1].")
        agent = self._agent(llm)
        resp = agent.run("Compare Digit and Elevate for waiting period")
        assert resp.query_type == "product_comparison"
        assert resp.products == ["Digit Health Insurance Policy", "Elevate"]
        assert len(resp.steps) == 2
        assert llm.last_system and "compare insurance policies" in llm.last_system.lower()

    def test_out_of_domain_short_circuits(self):
        llm = FakeLLM()
        agent = self._agent(llm)
        resp = agent.run("What will the stock market do tomorrow?")
        assert resp.abstained
        assert resp.answer == OUT_OF_DOMAIN_ANSWER
        assert resp.evidence_count == 0
        assert llm.calls == 0

    def test_clarification_short_circuits(self):
        llm = FakeLLM()
        agent = self._agent(llm)
        resp = agent.run("Compare Policy A and Policy B for coverage")
        assert resp.clarification
        assert resp.abstained
        assert resp.evidence_count == 0
        assert llm.calls == 0
        assert resp.answer == resp.clarification

    def test_plan_only_helper(self):
        agent = self._agent()
        analysis, plan = agent.plan("What are the exclusions?")
        assert analysis.query_type is QueryType.EXCLUSION
        assert len(plan.steps) == 1
