"""Phase 7 tests — application layer (API + UI), no LLM, no vector store.

The app is exercised with an injected stub agent so the HTTP/UI contract is
unit-testable offline, exactly like the phases before it.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.service import _serialize, create_app
from src.grounding.schema import Claim, ClaimVerification, VerifiedResponse, Verdict


# ----------------------------------------------------------------------
# Stub agent — a VerifiedResponse producer with configurable outcomes
# ----------------------------------------------------------------------


class StubAgent:
    def __init__(self, response: VerifiedResponse) -> None:
        self.response = response

    def run(self, query: str, k=None):  # noqa: ANN001, ANN002 — mirrors VerifiedAgent.run
        if k is not None and not (1 <= int(k) <= 30):
            raise ValueError("k out of range")
        return self.response


def _cited_response() -> VerifiedResponse:
    claim = Claim(text="The waiting period is 30 days", markers=[1], kind="fact", index=0)
    verdict = ClaimVerification(
        claim=claim, verdict=Verdict.SUPPORTED, support_score=0.9, evidence_chunk_ids=["DOC-006::c0028"]
    )
    return VerifiedResponse(
        query="What is the waiting period for cataract surgery?",
        answer="It is 30 days [1].",
        final_answer="It is 30 days.",
        query_type="policy_information",
        method="rules",
        model="llama3.2:3b",
        claims=[claim],
        verifications=[verdict],
        citations=[
            {
                "document_id": "DOC-006",
                "product": "MediCare",
                "pages": "13-13",
                "sections": "XII",
                "chunk_ids": ["DOC-006::c0028"],
            }
        ],
        products=["MediCare"],
        criteria=[],
        evidence_count=5,
        latency_ms={"total": 123.0, "retrieval": 40.0},
    )


def _abstained_response() -> VerifiedResponse:
    return VerifiedResponse(
        query="What is the current GST rate?",
        answer="",
        final_answer="I could not verify…",
        query_type="out_of_domain",
        abstained=True,
        clarification="Could you name the policy area?",
        evidence_count=0,
    )


@pytest.fixture()
def client() -> TestClient:
    app = create_app(agent=StubAgent(_cited_response()), llm_mode="verified (LLM)")
    return TestClient(app)


# ----------------------------------------------------------------------
# UI + health
# ----------------------------------------------------------------------


def test_root_serves_the_chat_ui(client: TestClient):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Verified RAG" in res.text
    assert "/api/stream" in res.text


def test_health_reports_pipeline(client: TestClient):
    data = client.get("/api/health").json()
    assert data["status"] == "ok"
    assert data["pipeline"] == "phase5-verified"
    assert "llm_mode" in data


# ----------------------------------------------------------------------
# /api/query
# ----------------------------------------------------------------------


def test_query_returns_verified_contract(client: TestClient):
    data = client.post("/api/query", json={"query": "What is the waiting period for cataract surgery?"}).json()
    assert data["query_type"] == "policy_information"
    assert data["method"] == "rules"
    assert data["abstained"] is False
    assert data["answer"] == "It is 30 days."
    assert data["model"] == "llama3.2:3b"
    g = data["grounding"]
    assert g["supported"] == 1 and g["total"] == 1
    assert g["contradicted"] == 0 and g["insufficient"] == 0
    assert data["products"] == ["MediCare"]
    assert data["evidence_count"] == 5
    assert data["latency_ms"]["total"] == 123.0
    citations = data["citations"]
    assert citations[0]["n"] == 1
    assert citations[0]["document_id"] == "DOC-006"
    assert citations[0]["chunk_ids"] == ["DOC-006::c0028"]


def test_query_estimates_abstained_flow():
    app = create_app(agent=StubAgent(_abstained_response()), llm_mode="verified (LLM)")
    with TestClient(app) as c:
        data = c.post("/api/query", json={"query": "What is the current GST rate?"}).json()
    assert data["abstained"] is True
    assert data["clarification"] == "Could you name the policy area?"
    assert data["grounding"]["total"] == 0


def test_query_rejects_blank_queries(client: TestClient):
    res = client.post("/api/query", json={"query": "   "})
    assert res.status_code == 422


def test_query_rejects_out_of_range_k(client: TestClient):
    res = client.post("/api/query", json={"query": "q", "k": 99})
    assert res.status_code == 422


# ----------------------------------------------------------------------
# /api/stream — SSE
# ----------------------------------------------------------------------


def test_stream_emits_working_then_result(client: TestClient):
    with client.stream("GET", "/api/stream", params={"q": "waiting period?"}) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        body = "".join(res.iter_text())
    events = [json.loads(line[6:]) for line in body.split("\n") if line.startswith("data: ")]
    assert events[0]["status"] == "working"
    assert events[-1]["status"] == "done"
    assert events[-1]["result"]["query_type"] == "policy_information"


def test_stream_reports_pipeline_errors():
    class BrokenAgent:
        def run(self, query, k=None):  # noqa: ANN001, ANN002
            raise RuntimeError("boom")

    app = create_app(agent=BrokenAgent(), llm_mode="verified (LLM)")
    with TestClient(app) as c:
        with c.stream("GET", "/api/stream", params={"q": "x"}) as res:
            body = "".join(res.iter_text())
    events = [json.loads(line[6:]) for line in body.split("\n") if line.startswith("data: ")]
    assert events[-1]["status"] == "error"


# ----------------------------------------------------------------------
# Reports viewer (phase 1-6 artifacts)
# ----------------------------------------------------------------------


def test_reports_index_lists_artifacts(client: TestClient):
    data = client.get("/api/reports").json()
    names = [r["name"] for r in data["reports"]]
    assert "phase6_report.md" in names


def test_report_file_serves_markdown(client: TestClient):
    res = client.get("/api/reports/phase6_report.md")
    assert res.status_code == 200
    assert "text/markdown" in res.headers["content-type"]


def test_report_file_blocks_traversal(client: TestClient):
    res = client.get("/api/reports/..%2FREADME.md")
    assert res.status_code in (404, 400)


def test_report_file_unknown_name(client: TestClient):
    assert client.get("/api/reports/does_not_exist.md").status_code == 404


# ----------------------------------------------------------------------
# Access control (optional API key)
# ----------------------------------------------------------------------


def test_api_key_sets_requirements(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POLICY_RAG_API_KEY", "s3cret")
    app = create_app(agent=StubAgent(_cited_response()), llm_mode="verified (LLM)")
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 401
        assert c.get("/api/health", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/api/health", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/").status_code == 200  # UI stays reachable


def test_root_stays_open_without_key(client: TestClient):
    assert client.get("/").status_code == 200


# ----------------------------------------------------------------------
# Serialize helper
# ----------------------------------------------------------------------


def test_serialize_falls_back_to_answer_when_no_final_answer():
    agent = StubAgent(_cited_response())
    data = _serialize(agent.response, "verified (LLM)")
    assert data["llm_mode"] == "verified (LLM)"
    assert data["answer"]
