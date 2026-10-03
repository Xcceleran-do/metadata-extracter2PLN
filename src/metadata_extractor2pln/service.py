from __future__ import annotations

from .backends import ModelBackend
from .extractors import extract_records
from .models import (
    ExtractRequest,
    ExtractResponse,
    PlanRequest,
    PlanResponse,
    RunRequest,
    RunResponse,
    Usage,
    ValidatePlanRequest,
    ValidatePlanResponse,
)
from .planner import discover_plan, validate_and_pin_plan


class MetadataService:
    def __init__(self, backend: ModelBackend | None):
        self.backend = backend

    def plan(self, request: PlanRequest) -> PlanResponse:
        plan, model, usage = discover_plan(
            source_name=request.source_name,
            records=request.records,
            required_properties=request.required_properties,
            backend=self.backend,
            use_model=request.use_model,
        )
        return PlanResponse(plan=plan, model=model, usage=usage)

    def validate_plan(self, request: ValidatePlanRequest) -> ValidatePlanResponse:
        plan = validate_and_pin_plan(
            request.plan,
            request.required_properties,
            self.backend.provider if self.backend is not None else None,
        )
        return ValidatePlanResponse(valid=True, fingerprint=plan.fingerprint)

    def extract(self, request: ExtractRequest) -> ExtractResponse:
        if not request.plan.fingerprint:
            raise ValueError(
                "/v1/extract requires a fingerprinted plan; validate the plan first"
            )
        plan = validate_and_pin_plan(
            request.plan,
            backend_provider=self.backend.provider if self.backend is not None else None,
        )
        return extract_records(
            namespace=request.namespace,
            plan=plan,
            records=request.records,
            backend=self.backend,
        )

    def run(self, request: RunRequest) -> RunResponse:
        plan_response = self.plan(
            PlanRequest.model_validate(request.model_dump(exclude={"namespace"}))
        )
        extraction = self.extract(
            ExtractRequest(
                namespace=request.namespace,
                plan=plan_response.plan,
                records=request.records,
            )
        )
        usage = Usage(
            input_tokens=plan_response.usage.input_tokens
            + extraction.usage.input_tokens,
            output_tokens=plan_response.usage.output_tokens
            + extraction.usage.output_tokens,
        )
        return RunResponse(
            plan=plan_response.plan,
            extraction=extraction,
            model=plan_response.model,
            usage=usage,
        )
