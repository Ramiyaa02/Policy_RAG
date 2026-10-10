"""Phase 4 — Agent 1: Query Analyzer (SRS FR-007, FR-013, §11).

Classifies a user query, extracts structured requirements, detects which
products/criteria it refers to, and flags out-of-domain questions.

Design (chosen with the project owner): **rules first, LLM fallback**. Deterministic
cue scoring handles the common, well-formed cases and can be unit-tested with no
server; only when the rule confidence is low (or the query looks out-of-domain yet
mentions insurance) is the LLM asked to classify. Every result records
``method`` = ``"rules"`` or ``"llm"`` so the two paths stay auditable (NFR-002,
NFR-009).

The modules are thin on purpose: the analyzer produces structured facts, the
planner turns them into retrieval operations, and the agent executes them.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Sequence

from .schema import (
    COMPARISON_CRITERIA,
    QueryAnalysis,
    QueryType,
    UserRequirements,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Product matching
# ----------------------------------------------------------------------

_ALIAS_STOPWORDS = frozenset(
    {"policy", "insurance", "health", "general", "the", "and", "of", "ltd", "limited", "company"}
)


class ProductMatcher:
    """Resolves product names mentioned in free text to canonical corpus names.

    Aliases are derived from each product name: the full name plus its
    significant tokens ("Digit Health Insurance Policy" -> "digit",
    "my:Optima Secure" -> "optima"/"secure"). Token aliases are matched on word
    boundaries, and any alias that would map to more than one product is dropped
    to avoid ambiguous matches.
    """

    def __init__(
        self,
        products: Iterable[str],
        extra_aliases: dict[str, str] | None = None,
    ) -> None:
        self.products = [p for p in products if p]
        alias_to_product: dict[str, str] = {}
        ambiguous: set[str] = set()
        for product in self.products:
            for alias in self._aliases_for(product):
                existing = alias_to_product.get(alias)
                if existing is not None and existing != product:
                    ambiguous.add(alias)
                else:
                    alias_to_product[alias] = product
        for alias in ambiguous:
            alias_to_product.pop(alias, None)
        for alias, product in (extra_aliases or {}).items():
            alias_to_product[alias.lower().strip()] = product
        self.alias_to_product = alias_to_product
        self._patterns = [
            (re.compile(rf"\b{re.escape(alias)}\b"), product)
            for alias, product in sorted(alias_to_product.items(), key=lambda kv: -len(kv[0]))
        ]

    @staticmethod
    def _aliases_for(name: str) -> set[str]:
        base = re.sub(r"\s+", " ", name.lower().strip())
        tokens = re.findall(r"[a-z0-9]+", base)
        significant = [
            t for t in tokens if len(t) >= 4 and t not in _ALIAS_STOPWORDS
        ]
        aliases = {base}
        aliases.update(significant)
        if len(significant) >= 2:
            aliases.add(f"{significant[0]} {significant[1]}")
        return {a for a in aliases if a}

    def find(self, text: str) -> list[str]:
        """Canonical product names mentioned in ``text`` (corpus order, unique)."""
        lowered = (text or "").lower()
        found: set[str] = set()
        for pattern, product in self._patterns:
            if pattern.search(lowered):
                found.add(product)
        return [p for p in self.products if p in found]

    @classmethod
    def from_chunks(cls, chunks: Sequence[dict[str, Any]]) -> "ProductMatcher":
        """Build from the Phase 2 chunk corpus (``data/chunks/chunks.jsonl``)."""
        products: list[str] = []
        for row in chunks:
            product = row.get("product")
            if product and product not in products:
                products.append(product)
        return cls(products)


# ----------------------------------------------------------------------
# Cue tables (rules)
# ----------------------------------------------------------------------

_WAITING_CUES = ("waiting period", "wait period", "waiting time", "moratorium", "pre-existing", "pre existing")
_EXCLUSION_CUES = (
    "exclusion", "exclude", "excluded", "not covered", "does not cover",
    "will not cover", "is not covered", "not payable", "not reimbursable",
)
_ELIGIBILITY_CUES = (
    "eligib", "entry age", "age limit", "who can buy", "can i buy",
    "minimum age", "maximum age", "renewal age", "age criteria",
)
_COVERAGE_CUES = (
    "cover", "coverage", "benefit", "payable", "reimburse", "sum insured",
    "room rent", "ambulance", "maternity", "ayush", "day care", "critical illness",
    "organ donor", "domiciliary", "consumable", "deductible", "restoration",
    "cashless", "pre-hospitalization", "post-hospitalization",
)
_COMPARISON_CUES = (
    "compare", "comparison", "versus", " vs ", "which is better",
    "which one is better", "difference between", "better than",
)
_RECOMMENDATION_CUES = (
    "recommend", "suggest", "best policy", "best plan", "which policy should",
    "which plan should", "best for me", "suitable for", "should i buy",
)
_INFORMATION_CUES = (
    "what is", "how do", "how does", "how many", "when ", "where ", "who ",
    "definition", "means", "policy wording",
)

_DOMAIN_TERMS = (
    "insurance", "policy", "premium", "claim", "insured", "insurer", "hospital",
    "hospitalisation", "hospitalization", "network", "cashless", "tpa", "irdai",
    "sum insured", "cover", "coverage", "benefit", "waiting", "exclusion",
    "exclude", "deductible", "co-payment", "copayment", "renewal", "portability",
    "migration", "nomination", "grievance", "ombudsman", "scheme", "free look",
    "grace period", "moratorium", "room rent", "sub-limit", "sub limit", "maternity",
    "ayush", "cataract", "pre-existing", "pre existing", "restoration", "recharge",
    "bonus", "treatment", "medical", "expense", "ambulance", "organ", "disease",
    "illness", "injury", "surgery", "eligib", "disclosure", "fraud", "day care",
)

_OUT_OF_DOMAIN_PATTERNS = (
    "stock market", "share price", "cricket", "weather forecast", "recipe",
    "capital of", "who won", "python code", "write a function", "javascript",
    "football", "movie", "cryptocurrency", "bitcoin",
)

_CRITERION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "coverage": ("cover", "coverage", "benefit"),
    "eligibility": ("eligib", "age"),
    "exclusions": ("exclusion", "exclude", "excluded"),
    "waiting period": ("waiting", "moratorium", "pre-existing"),
    "benefits": ("benefit", "benefits"),
    "limitations": ("limit", "sub-limit", "sub limit", "cap"),
}


def _count_hits(text: str, cues: Sequence[str]) -> int:
    return sum(1 for cue in cues if cue in text)


def _match_criteria(text: str) -> list[str]:
    return [c for c in COMPARISON_CRITERIA if any(s in text for s in _CRITERION_SYNONYMS[c])]


# Requirement extraction patterns (FR-013).
_AGE_PATTERNS = (
    re.compile(r"(\d{1,3})\s*[-\s]?year[s]?[-\s]?old"),
    re.compile(r"\bage[d]?\s*(?:of|:)?\s*(\d{1,3})\b"),
    re.compile(r"\bage\s*(?:of|:)?\s*(\d{1,3})\b"),
)
_SUM_INSURED_RE = re.compile(
    r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(lakh|lac|lakhs|crore|cr)\b", re.IGNORECASE
)
_COVERAGE_OBJECTIVES = (
    "hospitalization", "maternity", "critical illness", "day care", "ayush",
    "organ donor", "ambulance", "domiciliary", "opd",
)
_PRIORITY_PHRASES = {
    "low waiting period": ("low waiting period", "shorter waiting", "short waiting", "low waiting"),
    "cheaper premium": ("cheaper premium", "low premium", "affordable premium", "lower premium"),
    "higher sum insured": ("higher sum insured", "high sum insured", "larger cover"),
    "fewer exclusions": ("fewer exclusions", "less exclusions", "minimal exclusions"),
}


def extract_requirements(query: str) -> UserRequirements:
    """Deterministic requirement extraction (FR-013). LLM may refine this later."""
    text = query.lower()
    age: int | None = None
    for pattern in _AGE_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = int(match.group(1))
            if 0 < candidate < 120:
                age = candidate
                break

    sum_insured = None
    match = _SUM_INSURED_RE.search(query)
    if match:
        amount, unit = match.group(1), match.group(2).lower()
        unit = "crore" if unit.startswith("cr") else "lakh"
        sum_insured = f"{amount} {unit}"

    coverage_objective = next((o for o in _COVERAGE_OBJECTIVES if o in text), None)

    priorities = [
        label for label, phrases in _PRIORITY_PHRASES.items() if any(p in text for p in phrases)
    ]

    tenure = None
    tenure_match = re.search(r"(\d{1,2})\s*(?:year|yr)s?\s*(?:term|tenure|plan)", text)
    if tenure_match:
        tenure = f"{tenure_match.group(1)} years"

    return UserRequirements(
        age=age,
        sum_insured=sum_insured,
        coverage_objective=coverage_objective,
        priorities=priorities,
        tenure=tenure,
        raw={"query": query},
    )


# ----------------------------------------------------------------------
# Query Analyzer
# ----------------------------------------------------------------------

_LLM_SYSTEM = (
    "You classify insurance-document questions. Reply with a single JSON object "
    "and nothing else. Keys: query_type (one of: policy_information, coverage, "
    "eligibility, exclusion, waiting_period, product_comparison, recommendation, "
    "multi_step, out_of_domain), criteria (array of strings), products (array of "
    "strings), requirements (object with optional keys age, sum_insured, "
    "coverage_objective, priorities)."
)


class QueryAnalyzer:
    """Agent 1 — classify queries and extract requirements (rules + LLM)."""

    _PRIORITY_ORDER = (
        QueryType.WAITING_PERIOD,
        QueryType.EXCLUSION,
        QueryType.ELIGIBILITY,
        QueryType.COVERAGE,
        QueryType.POLICY_INFORMATION,
    )

    def __init__(
        self,
        llm: Any | None = None,
        matcher: ProductMatcher | None = None,
        use_llm: bool = True,
        llm_confidence_threshold: float = 0.55,
    ) -> None:
        self.llm = llm
        self.matcher = matcher or ProductMatcher([])
        self.use_llm = use_llm and llm is not None
        self.llm_confidence_threshold = llm_confidence_threshold

    # ------------------------------------------------------------------

    def analyze(self, query: str) -> QueryAnalysis:
        analysis = self._analyze_rules(query)
        if (
            self.use_llm
            and analysis.confidence < self.llm_confidence_threshold
        ):
            refined = self._analyze_llm(query, analysis)
            if refined is not None:
                return refined
        return analysis

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    def _analyze_rules(self, query: str) -> QueryAnalysis:
        text = f" {query.lower()} "
        products = self.matcher.find(query)
        criteria = _match_criteria(text)

        waiting = _count_hits(text, _WAITING_CUES)
        exclusion = _count_hits(text, _EXCLUSION_CUES)
        eligibility = _count_hits(text, _ELIGIBILITY_CUES)
        coverage = _count_hits(text, _COVERAGE_CUES)
        information = _count_hits(text, _INFORMATION_CUES)
        comparison = _count_hits(text, _COMPARISON_CUES)
        recommendation = _count_hits(text, _RECOMMENDATION_CUES)
        if len(products) >= 2:
            comparison += 1

        requirements = extract_requirements(query)
        detected: list[str] = []
        matched: list[str] = []

        def note(kind: QueryType, score: int) -> None:
            if score:
                detected.append(kind.value)
                matched.append(f"{kind.value}:{score}")

        note(QueryType.WAITING_PERIOD, waiting)
        note(QueryType.EXCLUSION, exclusion)
        note(QueryType.ELIGIBILITY, eligibility)
        note(QueryType.COVERAGE, coverage)
        note(QueryType.PRODUCT_COMPARISON, comparison)
        note(QueryType.RECOMMENDATION, recommendation)

        insurance_cues = waiting + exclusion + eligibility + coverage + comparison + recommendation

        if self._is_out_of_domain(text, products, insurance_cues):
            return QueryAnalysis(
                query=query,
                query_type=QueryType.OUT_OF_DOMAIN,
                confidence=0.6 if not insurance_cues else 0.5,
                method="rules",
                is_complex=False,
                requirements=requirements,
                criteria=[],
                products=products,
                matched_cues=matched,
                detected_types=[],
            )

        if comparison:
            confidence = min(1.0, 0.55 + 0.15 * comparison)
            return QueryAnalysis(
                query=query,
                query_type=QueryType.PRODUCT_COMPARISON,
                confidence=confidence,
                method="rules",
                is_complex=True,
                requirements=requirements,
                criteria=criteria,
                products=products,
                matched_cues=matched,
                detected_types=detected,
            )

        if recommendation:
            confidence = min(1.0, 0.55 + 0.15 * recommendation)
            return QueryAnalysis(
                query=query,
                query_type=QueryType.RECOMMENDATION,
                confidence=confidence,
                method="rules",
                is_complex=True,
                requirements=requirements,
                criteria=criteria or ["coverage", "eligibility", "waiting period"],
                products=products,
                matched_cues=matched,
                detected_types=detected,
            )

        scores = {
            QueryType.WAITING_PERIOD: waiting,
            QueryType.EXCLUSION: exclusion,
            QueryType.ELIGIBILITY: eligibility,
            QueryType.COVERAGE: coverage,
            QueryType.POLICY_INFORMATION: information,
        }
        strong = [qt for qt in self._PRIORITY_ORDER if scores[qt] > 0]
        positive = [qt for qt in self._PRIORITY_ORDER if scores[qt] > 0 and qt != QueryType.POLICY_INFORMATION]
        if len(positive) >= 2:
            return QueryAnalysis(
                query=query,
                query_type=QueryType.MULTI_STEP,
                confidence=min(1.0, 0.5 + 0.1 * sum(scores[qt] for qt in strong)),
                method="rules",
                is_complex=True,
                requirements=requirements,
                criteria=criteria,
                products=products,
                matched_cues=matched,
                detected_types=[qt.value for qt in positive],
            )

        primary = self._PRIORITY_ORDER[0]
        top_score = 0
        for qt in self._PRIORITY_ORDER:
            if scores[qt] > top_score:
                primary, top_score = qt, scores[qt]
        confidence = min(1.0, 0.45 + 0.15 * top_score) if top_score else 0.3
        return QueryAnalysis(
            query=query,
            query_type=primary,
            confidence=confidence,
            method="rules",
            is_complex=False,
            requirements=requirements,
            criteria=criteria,
            products=products,
            matched_cues=matched,
            detected_types=[primary.value] if top_score else [],
        )

    @staticmethod
    def _is_out_of_domain(text: str, products: Sequence[str], insurance_cues: int) -> bool:
        """True when the query has no insurance footing.

        Generic question words ("what is", "how many") are deliberately not
        treated as domain evidence — otherwise every question would look
        in-domain.
        """
        if products or insurance_cues:
            return False
        if any(pattern in text for pattern in _OUT_OF_DOMAIN_PATTERNS):
            return True
        return not any(term in text for term in _DOMAIN_TERMS)

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _analyze_llm(self, query: str, fallback: QueryAnalysis) -> QueryAnalysis | None:
        prompt = (
            f"Question: {query}\n"
            "Classify it. If it is not about insurance policies, use out_of_domain."
        )
        try:
            raw = self.llm.generate(prompt, system=_LLM_SYSTEM)
        except Exception as exc:  # noqa: BLE001 - never fail the pipeline on the LLM
            logger.warning("LLM classification failed (%s); using rules.", exc)
            return None

        payload = _extract_json(raw)
        if not payload:
            logger.warning("LLM classification returned unparseable output; using rules.")
            return None

        try:
            query_type = QueryType(str(payload.get("query_type", "")).strip().lower())
        except ValueError:
            logger.warning("LLM returned unknown query_type %r; using rules.", payload.get("query_type"))
            return None

        requirements = self._merge_requirements(fallback.requirements, payload.get("requirements"))
        criteria = [c for c in (payload.get("criteria") or []) if c in COMPARISON_CRITERIA]
        products = payload.get("products") or fallback.products

        return QueryAnalysis(
            query=query,
            query_type=query_type,
            confidence=0.7,
            method="llm",
            is_complex=query_type in (QueryType.PRODUCT_COMPARISON, QueryType.RECOMMENDATION, QueryType.MULTI_STEP),
            requirements=requirements,
            criteria=criteria or fallback.criteria,
            products=[str(p) for p in products],
            matched_cues=fallback.matched_cues + ["llm"],
            detected_types=fallback.detected_types,
        )

    @staticmethod
    def _merge_requirements(base: UserRequirements, payload: Any) -> UserRequirements:
        if not isinstance(payload, dict):
            return base
        merged = UserRequirements(
            age=base.age,
            sum_insured=base.sum_insured,
            coverage_objective=base.coverage_objective,
            priorities=list(base.priorities),
            tenure=base.tenure,
            raw=base.raw,
        )
        if merged.age is None and isinstance(payload.get("age"), (int, float)):
            merged.age = int(payload["age"])
        if not merged.sum_insured and payload.get("sum_insured"):
            merged.sum_insured = str(payload["sum_insured"])
        if not merged.coverage_objective and payload.get("coverage_objective"):
            merged.coverage_objective = str(payload["coverage_objective"])
        if not merged.priorities and isinstance(payload.get("priorities"), list):
            merged.priorities = [str(p) for p in payload["priorities"]]
        return merged


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of an LLM response."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
