"""Phase 5 — Grounding schema (SRS §8, Agent 5, FR-015/FR-016, O6/O7/O8).

Dataclasses for the grounding/verification layer:

    Generated answer
       │
       ▼  ClaimExtractor          -> list[Claim]        (claims.py)
    Claim text · citation markers · fact/conclusion kind
       │
       ▼  GroundingValidator (Agent 5) -> list[ClaimVerification] (verifier.py)
    supported | contradicted | insufficient
       │
       ▼  GroundingHandler        -> VerifiedResponse   (handler.py)
    Final answer · notes · abstention/clarification

Verification honours the SRS §8 three-way distinction — absence of evidence
is not evidence of absence, so a claim with no supporting excerpt is reported
``insufficient``, never silently dropped or relabelled as false.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """SRS §8 claim-support verdicts."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    NOT_VERIFIED = "not_verified"  # conclusions / non-factual segments


def _clean(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


@dataclass
class Claim:
    """One verify-able statement extracted from a generated answer."""

    text: str
    markers: list[int] = field(default_factory=list)  # [n] citation markers
    kind: str = "fact"  # "fact" | "conclusion"
    section: str = ""  # heading it appeared under, e.g. "Conclusion"
    index: int = 0  # order of appearance
    line: int = 0  # answer line the claim came from (for rewriting)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimVerification:
    """Agent 5 output for one :class:`Claim` (FR-015)."""

    claim: Claim
    verdict: Verdict = Verdict.NOT_VERIFIED
    support_score: float = 0.0
    evidence_chunk_ids: list[str] = field(default_factory=list)
    reason: str = ""
    method: str = "rules"  # "rules" | "llm"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["claim"] = self.claim.to_dict()
        return d


@dataclass
class VerifiedResponse:
    """Verified end-to-end response (agent trace + grounding report)."""

    query: str = ""
    query_type: str = ""
    method: str = "rules"
    answer: str = ""
    final_answer: str = ""  # after unsupported-claim handling
    claims: list[Claim] = field(default_factory=list)
    verifications: list[ClaimVerification] = field(default_factory=list)
    supported_rate: float = 0.0
    unsupported_claims: list[str] = field(default_factory=list)
    contradicted_claims: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    evidence_count: int = 0
    abstained: bool = False
    clarification: str | None = None
    model: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def supported_count(self) -> int:
        return sum(1 for v in self.verifications if v.verdict is Verdict.SUPPORTED)

    @property
    def contradicted_count(self) -> int:
        return sum(1 for v in self.verifications if v.verdict is Verdict.CONTRADICTED)

    @property
    def insufficient_count(self) -> int:
        return sum(1 for v in self.verifications if v.verdict is Verdict.INSUFFICIENT)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verifications"] = [v.to_dict() for v in self.verifications]
        d["claims"] = [c.to_dict() for c in self.claims]
        d["counts"] = {
            "claims": len(self.claims),
            "supported": self.supported_count,
            "contradicted": self.contradicted_count,
            "insufficient": self.insufficient_count,
        }
        return d
