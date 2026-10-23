"""Deterministic checks. The model never decides escalation, these functions do."""
import re

from .schemas import ATTRIBUTES, Comparison, Fact, Preferences, Recommendation

PERSONAL_DATA = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email
    re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -]?)?\d{5}[ -]?\d{5}(?!\d)"),  # 10-digit phone
    re.compile(r"\b(?:my name is|customer name is|name is)\s+\w+", re.I),
]
RETURN_RE = re.compile(r"\b(refund|return|exchange)\w*\b", re.I)
OPENED_RE = re.compile(r"\bopen(ed)?\b", re.I)
DEFECT_RE = re.compile(r"\b(defective|faulty|broken)\b", re.I)
NOT_DEFECT_RE = re.compile(r"\b(not|isn'?t|is not|no)\s+(\w+\s+)?(defective|faulty|broken)\b", re.I)
AVAILABILITY_RE = re.compile(r"\b(in stock|out of stock|stock|availab\w*|inventory)\b", re.I)
DISCOUNT_RE = re.compile(r"\b(discount\w*|promo\w*|offer\w*|sale|deal|coupon)\b|\d+\s*%\s*off", re.I)
UNSUPPORTED_CLAIM_RE = re.compile(
    r"\b(in stock|out of stock|available|availability|discount\w*|offer|sale)\b|\d+\s*%\s*off", re.I
)
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def has_personal_data(*texts: str) -> bool:
    joined = " ".join(t for t in texts if t)
    return any(p.search(joined) for p in PERSONAL_DATA)


def is_policy_exception(question: str) -> bool:
    """Opened item + refund/return + not defective => policy says a supervisor must decide."""
    if not (RETURN_RE.search(question) and OPENED_RE.search(question)):
        return False
    return bool(NOT_DEFECT_RE.search(question)) or not DEFECT_RE.search(question)


def check_facts(facts: list[Fact], chunks_by_citation: dict[str, dict]) -> list[str]:
    problems = []
    for f in facts:
        chunk = chunks_by_citation.get(f.citation)
        if chunk is None:
            problems.append(f"fact cites a source that was not retrieved: {f.citation}")
            continue
        if chunk["product_id"] != "ALL" and f.product_id != chunk["product_id"]:
            problems.append(f"fact for {f.product_id} cites {f.citation}, which belongs to {chunk['product_id']}")
            continue
        haystack = (chunk["text"] + " " + " ".join(map(str, chunk["attributes"].values()))).replace(",", "")
        for num in NUMBER_RE.findall(f.value.replace(",", "")):
            if num not in haystack:
                problems.append(f"value '{f.value}' for {f.product_id} not found in {f.citation}")
                break
    return problems


def check_recommendation(rec: Recommendation, prefs: Preferences, candidates: dict[str, dict]) -> list[str]:
    problems = []
    stated = prefs.stated()
    if not rec.based_on_preferences or not set(rec.based_on_preferences) <= set(stated):
        problems.append(f"recommendation must be based on stated preferences {stated}")
    chunk = candidates.get(rec.product_id)
    if chunk is None:
        return problems + [f"recommended product {rec.product_id} is not a retrieved candidate"]
    attrs = chunk["attributes"]
    if prefs.budget_max_inr is not None and int(attrs.get("price_inr", 10**9)) > prefs.budget_max_inr:
        problems.append(f"{rec.product_id} exceeds the budget of Rs {prefs.budget_max_inr}")
    for need in prefs.must_have:
        if "noise" in need.lower() and attrs.get("noise_cancellation", "none").lower() == "none":
            problems.append(f"{rec.product_id} lacks the must-have '{need}'")
    return problems


def check_claims(facts: list[Fact], rec: Recommendation | None) -> list[str]:
    texts = [f.value for f in facts] + ([rec.reasoning] if rec else [])
    return ["generated text makes an availability/discount claim"] if any(UNSUPPORTED_CLAIM_RE.search(t) for t in texts) else []


def check_comparison(c: Comparison | None, category: str | None) -> list[str]:
    if c and category and c.attributes != ATTRIBUTES[category]:
        return ["comparison attributes are not the fixed list"]
    return []
