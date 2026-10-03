from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema


NAME_PATTERN = r"^[a-z][a-z0-9-]{0,63}$"
SOURCE_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
NAMESPACE_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ExtractorName = Literal[
    "structured_field",
    "calculated_metric",
    "numeric_bucket",
    "date_bucket",
    "semantic_text",
]


class PropertySpec(ApiModel):
    name: str = Field(pattern=NAME_PATTERN)
    description: str = Field(min_length=1, max_length=1_000)
    extractor: ExtractorName
    field_paths: list[str] = Field(default_factory=list, max_length=32)
    allowed_values: list[str] = Field(default_factory=list, max_length=100)
    metric: Literal["length", "reading-time", "engagement"] | None = None
    required: bool = False
    include_in_pln: bool = True

    @field_validator("field_paths", "allowed_values")
    @classmethod
    def clean_strings(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("list entries cannot be blank")
            if len(value) > 256:
                raise ValueError("list entries cannot exceed 256 characters")
            if value not in cleaned:
                cleaned.append(value)
        return cleaned

    @model_validator(mode="after")
    def validate_metric(self):
        if self.extractor == "calculated_metric" and self.metric is None:
            raise ValueError("calculated_metric properties require metric")
        if self.extractor != "calculated_metric" and self.metric is not None:
            raise ValueError("metric is only valid for calculated_metric properties")
        return self


class ExtractionPlan(ApiModel):
    source_name: str = Field(pattern=SOURCE_PATTERN)
    entity_type: str = Field(default="item", pattern=NAME_PATTERN)
    id_fields: list[str] = Field(min_length=1, max_length=16)
    text_fields: list[str] = Field(default_factory=list, max_length=32)
    properties: list[PropertySpec] = Field(min_length=1, max_length=100)
    version: int = Field(default=1, ge=1)
    planner: Literal["heuristic", "bedrock", "pinned"] = "heuristic"
    fingerprint: str = Field(default="", pattern=r"^$|^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def unique_properties(self):
        names = [item.name for item in self.properties]
        if len(names) != len(set(names)):
            raise ValueError("property names must be unique")
        return self


class PlanDraft(ApiModel):
    entity_type: str = Field(default="item", pattern=NAME_PATTERN)
    id_fields: list[str] = Field(default_factory=lambda: ["id"], max_length=16)
    text_fields: list[str] = Field(default_factory=list, max_length=32)
    properties: list[PropertySpec] = Field(default_factory=list, max_length=100)


class SemanticValue(ApiModel):
    property_name: str = Field(pattern=NAME_PATTERN)
    value: str = Field(min_length=1, max_length=256)
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quote: str | None = Field(default=None, max_length=1_000)


class SemanticResult(ApiModel):
    values: list[SemanticValue] = Field(default_factory=list, max_length=100)


class SemanticRecordResult(ApiModel):
    record_index: int = Field(ge=0, le=99)
    values: list[SemanticValue] = Field(default_factory=list, max_length=100)
    errors: SkipJsonSchema[list[str]] = Field(default_factory=list, max_length=100)


class SemanticBatchResult(ApiModel):
    records: list[SemanticRecordResult] = Field(default_factory=list, max_length=100)


class Evidence(ApiModel):
    method: ExtractorName
    detail: str = Field(min_length=1, max_length=1_000)
    quote: str | None = Field(default=None, max_length=1_000)


class ExtractedProperty(ApiModel):
    name: str = Field(pattern=NAME_PATTERN)
    value: str = Field(min_length=1, max_length=2_000)
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Evidence


class CompiledFact(ApiModel):
    source: str
    atom: str
    strength: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    proof_id: str
    idempotency_key: str
    property_name: str


class RecordResult(ApiModel):
    source_id: str
    entity_id: str
    properties: list[ExtractedProperty]
    facts: list[CompiledFact]
    errors: list[str] = Field(default_factory=list)


class Usage(ApiModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class PlanRequest(ApiModel):
    source_name: str = Field(default="json", pattern=SOURCE_PATTERN)
    records: list[dict[str, Any]] = Field(min_length=1, max_length=20)
    required_properties: list[str] = Field(
        default_factory=list,
        max_length=20,
    )
    use_model: bool = True


class PlanResponse(ApiModel):
    plan: ExtractionPlan
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)


class ExtractRequest(ApiModel):
    namespace: str = Field(pattern=NAMESPACE_PATTERN)
    plan: ExtractionPlan
    records: list[dict[str, Any]] = Field(min_length=1, max_length=100)


class ExtractResponse(ApiModel):
    contract_version: str
    plan_fingerprint: str
    records: list[RecordResult]
    usage: Usage = Field(default_factory=Usage)


class RunRequest(PlanRequest):
    namespace: str = Field(pattern=NAMESPACE_PATTERN)


class RunResponse(ApiModel):
    plan: ExtractionPlan
    extraction: ExtractResponse
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)


class ValidatePlanRequest(ApiModel):
    plan: ExtractionPlan
    required_properties: list[str] = Field(default_factory=list, max_length=20)


class ValidatePlanResponse(ApiModel):
    valid: bool
    fingerprint: str
