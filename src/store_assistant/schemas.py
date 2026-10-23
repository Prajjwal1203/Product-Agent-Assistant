from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Fixed comparison layout per category: same attributes, same order, every time.
ATTRIBUTES = {
    "headphones": ["noise_cancellation", "battery_life", "weight", "price_inr"],
    "chargers": ["power_output", "ports", "connector", "price_inr"],
}


class Preferences(BaseModel):
    budget_max_inr: int | None = None
    must_have: list[str] = Field(default_factory=list)
    use_case: str | None = None

    def stated(self) -> list[str]:
        out = []
        if self.budget_max_inr is not None:
            out.append("budget_max_inr")
        if self.must_have:
            out.append("must_have")
        if self.use_case:
            out.append("use_case")
        return out


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    product_ids: list[str] = Field(default_factory=list)
    preferences: Preferences = Field(default_factory=Preferences)


class Fact(BaseModel):
    product_id: str
    attribute: str
    value: str
    citation: str


class Row(BaseModel):
    product_id: str
    values: list[str]
    citations: list[str]


class Comparison(BaseModel):
    attributes: list[str]
    rows: list[Row]


class Recommendation(BaseModel):
    product_id: str
    reasoning: str
    based_on_preferences: list[str]


class Conflict(BaseModel):
    product_id: str
    attribute: str
    sources: list[str]
    resolution: str


class Response(BaseModel):
    intent: Literal["answer", "compare", "recommend"]
    facts: list[Fact] = Field(default_factory=list)
    comparison: Comparison | None = None
    recommendation: Recommendation | None = None
    missing_preferences: list[str] = Field(default_factory=list)
    unsupported: list[str] = Field(default_factory=list)
    data_conflicts: list[Conflict] = Field(default_factory=list)
    escalate: bool = False
    escalation_reason: str | None = None

    @model_validator(mode="after")
    def _escalation_needs_reason(self):
        if self.escalate and not self.escalation_reason:
            raise ValueError("escalate=true requires escalation_reason")
        return self


class LLMOut(BaseModel):
    """What the model is allowed to produce. Everything else is built or verified in code."""

    facts: list[Fact] = Field(default_factory=list)
    recommendation: Recommendation | None = None
