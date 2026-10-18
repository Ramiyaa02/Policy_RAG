"""Tests for Phase 6 — Evaluation (failure taxonomy, pipeline eval, runner shape)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.evaluation.failure_taxonomy import (
    CODES,
    QueryFailureReport,
    aggregate,
    classify_query,
    render_markdown,
)
from src.evaluation.pipeline_eval import PipelineEvaluator, PipelineQueryResult, mean


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _result(**over):
    base = {
        "query_type": "policy_information",
        "abstained": False,
        "clarification": None,
        "evidence_chunk_ids": ["DOC-001::c0002"],
        "retrieval_top_chunk_ids": ["DOC-001::c0002"],
        "relevant_chunk_ids": ["DOC-001::c0002"],
        "claims": 3,
        "supported": 3,
        "contradicted": 0,
        "insufficient": 0,
        "unresolved_citations": 0,
        "unsupported_stripped": 0,
        "contradicted_kept": 0,
        "evidence_products": ["Digit Health Insurance Policy"],
        "steps": [{"step_id": 1}],
        "degenerate": False,
    }
    base.update(over)
    return base


def _classify(category="factual", answerable=True, qid="q1", result=None):
    result = result or _result()
    return classify_query(qid, category, answerable, result)


# ----------------------------------------------------------------------
# Taxonomy — clean runs
# ----------------------------------------------------------------------

class TestTaxonomyCleanRun:
    def test_good_run_has_only_positive_codes(self):
        report = _classify(result=_result(unsupported_stripped=1))
        assert report.codes == ["F15"]
        assert not report.failed

    def test_ood_correctly_abstained(self):
        report = _classify(category="ood", answerable=False, result=_result(abstained=True))
        assert report.codes == ["F16"]
        assert not report.failed


class TestTaxonomyFailures:
    def test_f1_no_relevant_chunk_in_evidence(self):
        report = _classify(result=_result(evidence_chunk_ids=["DOC-002::c0001"]))
        assert "F1" in report.codes
        assert report.notes["F1"]

    def test_f2_partial_evidence(self):
        report = _classify(
            result=_result(
                relevant_chunk_ids=["DOC-001::c0002", "DOC-006::c0026"],
                evidence_chunk_ids=["DOC-001::c0002"],
            )
        )
        assert "F2" in report.codes
        assert "F1" not in report.codes

    def test_f3_retrieved_but_not_selected(self):
        report = _classify(
            result=_result(
                relevant_chunk_ids=["DOC-006::c0026"],
                evidence_chunk_ids=["DOC-001::c0002"],
                retrieval_top_chunk_ids=["DOC-006::c0026", "DOC-001::c0002"],
            )
        )
        # relevant chunk was retrieved upstream but never used as evidence →
        # F1 (relevant evidence missing from the used set) + F3 (selection)
        assert "F1" in report.codes and "F3" in report.codes

    def test_f4_answerable_classified_ood(self):
        report = _classify(
            result=_result(query_type="out_of_domain", abstained=True, claims=0, supported=0, degenerate=False)
        )
        assert "F4" in report.codes
        assert "F10" in report.codes  # abstained on answerable

    def test_f7_unsupported_generation(self):
        report = _classify(result=_result(claims=2, supported=1, insufficient=1))
        assert "F7" in report.codes
        assert "F15" not in report.codes  # nothing stripped recorded

    def test_f6_degenerate_generation(self):
        # Generation itself failed (degenerate output), even though the Phase 5
        # handler turned it into an abstention — F6 records the reasoning failure.
        report = _classify(result=_result(claims=0, supported=0, degenerate=True, abstained=True))
        assert "F6" in report.codes
        assert "F10" in report.codes

    def test_no_f6_for_plain_refusal(self):
        # A clean generator refusal (OOD light-abstention) is not a reasoning failure.
        report = _classify(
            category="ood",
            answerable=False,
            result=_result(claims=0, supported=0, abstained=True, degenerate=False),
        )
        assert "F6" not in report.codes

    def test_f8_unresolved_citations(self):
        report = _classify(result=_result(unresolved_citations=2))
        assert "F8" in report.codes

    def test_f9_contradicted_kept(self):
        report = _classify(result=_result(contradicted_kept=1))
        assert "F9" in report.codes

    def test_f10_incorrect_abstention(self):
        report = _classify(result=_result(abstained=True, claims=0, supported=0))
        assert "F10" in report.codes

    def test_f11_unnecessary_abstention_not_triggered_by_grounding(self):
        report = _classify(
            result=_result(abstained=True, claims=0, supported=0, contradicted=0, insufficient=0, degenerate=False)
        )
        assert "F11" in report.codes

    def test_f12_comparison_from_single_product(self):
        report = _classify(
            category="comparison",
            result=_result(query_type="product_comparison", evidence_products=["One"]),
        )
        assert "F12" in report.codes

    def test_f13_contradicted_claim_present(self):
        report = _classify(result=_result(claims=3, supported=2, contradicted=1))
        assert "F13" in report.codes

    def test_f14_ood_answered(self):
        report = _classify(category="ood", answerable=False, result=_result())
        assert "F14" in report.codes
        assert not report.codes == []

    def test_codes_sorted_in_srs_order(self):
        report = _classify(
            result=_result(
                claims=3,
                supported=1,
                contradicted=1,
                insufficient=1,
                unresolved_citations=1,
                abstained=False,
            )
        )
        assert report.codes == sorted(report.codes, key=lambda c: list(CODES).index(c))


class TestTaxonomyAggregate:
    def test_aggregate_counts_and_rate(self):
        reports = [
            _classify(result=_result()),
            _classify(qid="q2", result=_result(unsupported_stripped=2)),
            _classify(
                qid="q3", category="ood", answerable=False, result=_result(abstained=True)
            ),
        ]
        agg = aggregate(reports)
        assert agg["num_queries"] == 3
        assert agg["num_failed"] == 0
        assert agg["codes"]["F15"] == 1
        assert agg["codes"]["F16"] == 1

    def test_render_markdown_lists_codes(self):
        agg = aggregate([
            _classify(),
            _classify(qid="q2", result=_result(unsupported_stripped=1)),
        ])
        md = render_markdown(agg)
        assert "F15" in md and "grounding_corrected" in md
        assert "| code | label | queries |" in md


# ----------------------------------------------------------------------
# Pipeline evaluator
# ----------------------------------------------------------------------

def _mk_result(**over):
    base = dict(
        label="full",
        qid="q1",
        query="q",
        category="factual",
        answerable=True,
        query_type="policy_information",
        abstained=False,
        final_answer="answer [1]",
        claims=3,
        supported=3,
        contradicted=0,
        insufficient=0,
        supported_rate=1.0,
        unsupported_stripped=0,
        contradicted_kept=0,
        unresolved_citations=0,
        relevant_chunk_ids=["DOC-001::c0002"],
        evidence_chunk_ids=["DOC-001::c0002"],
        retrieval_top_chunk_ids=["DOC-001::c0002"],
        evidence_products=["Digit Health Insurance Policy"],
        steps=[{"step_id": 1}],
        latency_ms={"retrieval": 100.0, "generation": 50.0, "total": 200.0},
        usage={"calls": 2, "prompt_tokens": 100, "completion_tokens": 50},
        answered_with_relevant_evidence=True,
    )
    base.update(over)
    return PipelineQueryResult(**base)


class TestPipelineAggregate:
    def test_aggregate_s14_metrics(self):
        results = [
            _mk_result(),
            _mk_result(qid="q2", abstained=True, answered_with_relevant_evidence=False, claims=0,
                       supported=0, supported_rate=0.0),
            _mk_result(qid="q3", claims=2, supported=1, insufficient=1, supported_rate=0.5,
                       answered_with_relevant_evidence=True),
        ]
        agg = PipelineEvaluator.aggregate(results)
        assert agg["answerable_queries"] == 3
        assert agg["answered"] == 2
        assert agg["abstentions_on_answerable"] == 1
        assert agg["answered_correct"] == 2
        assert agg["answered_correct_rate"] == 1.0
        # §14.4: (0 + 0 + 1) / (3 + 0 + 2)
        assert agg["total_fact_claims"] == 5
        assert agg["unsupported_claim_rate"] == 0.2
        # §14.5 mean supported rate over NOT-abstained answers (1.0 + 0.5) / 2
        assert agg["mean_supported_rate"] == 0.75
        assert agg["latency"]["total_ms"]["mean"] == 200.0
        assert agg["latency"]["components"]["retrieval"]["mean"] == 100.0
        assert agg["tokens"]["prompt_tokens"] == 300
        assert agg["tokens"]["completion_tokens"] == 150
        assert agg["cost_estimate_usd"] == 0.0

    def test_aggregate_empty(self):
        agg = PipelineEvaluator.aggregate([])
        assert agg["answered"] == 0
        assert agg["unsupported_claim_rate"] == 0.0

    def test_by_category_splits(self):
        results = [
            _mk_result(),
            _mk_result(qid="q2", category="ood_probes", answerable=False, claims=0,
                       supported=0, supported_rate=0.0, abstained=True,
                       answered_with_relevant_evidence=False),
        ]
        by_cat = PipelineEvaluator.aggregate_by_category(results)
        assert by_cat["factual"]["count"] == 1
        assert by_cat["ood_probes"]["answered"] == 0

    def test_mean_helper(self):
        assert mean([1.0, 3.0]) == 2.0
        assert mean([]) == 0.0


# ----------------------------------------------------------------------
# run_phase6 helpers
# ----------------------------------------------------------------------

class TestRunPhase6Helpers:
    def test_usage_recorder_diff(self):
        import run_phase6

        rec = run_phase6.UsageRecorder()
        rec.totals["calls"] = 2
        rec.totals["prompt_tokens"] = 10
        rec.totals["completion_tokens"] = 5
        snap = rec.snapshot()
        rec.totals["calls"] += 1
        d = rec.diff(snap, rec.snapshot())
        assert d == {"calls": 1, "prompt_tokens": 0, "completion_tokens": 0}

    def test_ablations_registry_known_codes(self):
        import run_phase6

        assert "no_rerank" in run_phase6.ABLATIONS
        assert "keyword_only" in run_phase6.ABLATIONS
        assert "no_strip" in run_phase6.ABLATIONS
        overlay = run_phase6._overlay({"retrieval": {"mode": "hybrid", "k": 5}}, {"retrieval": {"mode": "keyword"}})
        assert overlay["retrieval"]["mode"] == "keyword"
        assert overlay["retrieval"]["k"] == 5
