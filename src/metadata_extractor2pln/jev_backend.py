from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Sequence

import httpx
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy

from .backends import BackendUnavailable
from .models import (
    PropertySpec,
    SemanticBatchResult,
    SemanticRecordResult,
    SemanticValue,
    Usage,
)
from .request_context import (
    get_request_deadline,
    request_cancelled,
    watch_request_cancellation,
)
from .structured_backend import StructuredBackend


logger = logging.getLogger(__name__)


class JEVBackend(StructuredBackend):
    """JEV backend for semantic classification."""

    name = "jev"
    provider = "jev"

    def __init__(
        self,
        *,
        model: str = "jev-latest",
        api_key: str | None = None,
        timeout_seconds: float = 45.0,
        transport: str = "typesafe",
        openrouter_api_key: str | None = None,
        openrouter_model: str = "typesafe/jev-1.13",
        openrouter_base_url: str = "https://openrouter.ai/api/alpha/decisions",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model = openrouter_model
        self.openrouter_base_url = openrouter_base_url

    @property
    def ready(self) -> bool:
        if self.transport == "openrouter":
            return bool(self.openrouter_api_key)
        return bool(self.api_key)

    def discover_plan(
        self,
        *,
        source_name: str,
        records: Sequence[dict[str, Any]],
        required_properties: Sequence[str],
    ) -> tuple[Any, Usage]:
        raise BackendUnavailable("JEV is configured for semantic classification only")

    def extract_semantics(
        self,
        *,
        texts: Sequence[str],
        properties: Sequence[PropertySpec],
    ) -> tuple[SemanticBatchResult, Usage]:
        if not self.ready:
            key = "OPENROUTER_API_KEY" if self.transport == "openrouter" else "TYPESAFE_API_KEY"
            raise BackendUnavailable(f"JEV is not configured; set {key}")
        if not properties:
            return SemanticBatchResult(records=[]), Usage()
        for prop in properties:
            if not prop.allowed_values:
                raise ValueError(
                    f"JEV classification requires allowed_values for property {prop.name!r}"
                )

        # The service calls synchronous backends in a worker thread. Each batch
        # owns its event loop and clients so cancellation closes in-flight I/O.
        return asyncio.run(self._extract_batch(texts, properties))

    async def _extract_batch(
        self,
        texts: Sequence[str],
        properties: Sequence[PropertySpec],
    ) -> tuple[SemanticBatchResult, Usage]:
        if self.transport == "openrouter":
            client = httpx.AsyncClient(timeout=self.timeout_seconds)
        else:
            client = AsyncTypeSafeClient(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                retry=RetryPolicy(max_retries=0),
            )

        records: list[SemanticRecordResult] = []
        total_usage = Usage()
        deadline = get_request_deadline()
        async with client:
            for record_index, text in enumerate(texts):
                remaining = None if deadline is None else deadline - time.monotonic()
                if request_cancelled() or (remaining is not None and remaining <= 0):
                    records.extend(
                        SemanticRecordResult(
                            record_index=index,
                            errors=["semantic extraction stopped at the request deadline or cancellation"],
                        )
                        for index in range(record_index, len(texts))
                    )
                    break

                timeout = self.timeout_seconds
                if remaining is not None:
                    timeout = min(timeout, remaining)
                try:
                    # Unlike HTTP phase timeouts, this bounds the entire call,
                    # including a response that arrives slowly in many chunks.
                    async with asyncio.timeout(timeout):
                        call = asyncio.create_task(self._classify(client, text, properties))
                        with watch_request_cancellation(call):
                            try:
                                answers, usage = await call
                            except asyncio.CancelledError:
                                if not request_cancelled():
                                    raise
                                raise BackendUnavailable("JEV request was cancelled") from None
                    total_usage.input_tokens += usage.input_tokens
                    total_usage.output_tokens += usage.output_tokens
                    values = self._convert_answers(properties=properties, answers=answers)
                    record = SemanticRecordResult(record_index=record_index, values=values)
                except Exception as exc:
                    logger.warning(
                        "JEV classification failed for record %s: %s",
                        record_index,
                        type(exc).__name__,
                    )
                    message = (
                        "semantic classification exceeded its time limit"
                        if isinstance(exc, TimeoutError)
                        else "semantic classification failed because the model request or response was invalid"
                    )
                    record = SemanticRecordResult(record_index=record_index, errors=[message])
                records.append(record)

        return SemanticBatchResult(records=records), total_usage

    async def _classify(
        self,
        client: Any,
        text: str,
        properties: Sequence[PropertySpec],
    ) -> tuple[dict[str, Any], Usage]:
        if self.transport != "openrouter":
            result = await client.system_one(
                state=text[:12000],
                questions=self._build_questions(properties),
                model=self.model,
            )
            return result.answers, Usage(
                input_tokens=result.usage.input_tokens or 0,
                output_tokens=result.usage.output_tokens or 0,
            )

        response = await client.post(
            self.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.openrouter_model,
                "state": text[:12000],
                "questions": self._build_openrouter_questions(properties),
            },
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise BackendUnavailable("OpenRouter JEV response did not contain a valid answers object")
        usage = result.get("usage") or {}
        if not isinstance(usage, dict):
            raise BackendUnavailable("OpenRouter JEV response contained invalid usage")
        return result["answers"], Usage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )

    @staticmethod
    def _build_questions(properties: Sequence[PropertySpec]) -> dict[str, Choice]:
        return {
            prop.name: Choice(
                instructions=prop.description,
                criteria={value: None for value in prop.allowed_values},
            )
            for prop in properties
        }

    @staticmethod
    def _build_openrouter_questions(
        properties: Sequence[PropertySpec],
    ) -> dict[str, dict[str, Any]]:
        return {
            prop.name: {
                "type": "choice",
                "instructions": prop.description,
                "criteria": {value: None for value in prop.allowed_values},
            }
            for prop in properties
        }

    @staticmethod
    def _convert_answers(
        *,
        properties: Sequence[PropertySpec],
        answers: dict[str, Any],
    ) -> list[SemanticValue]:
        if not isinstance(answers, dict):
            raise BackendUnavailable("JEV returned an invalid answers object")
        values: list[SemanticValue] = []
        for prop in properties:
            answer = answers.get(prop.name)
            if answer is None:
                raise BackendUnavailable(f"JEV response is missing answer for property {prop.name!r}")
            if isinstance(answer, dict):
                choice = answer.get("choice")
                confidence = answer.get("confidence", 0.0)
                probabilities = answer.get("probabilities", {})
            else:
                choice = answer.choice
                confidence = answer.confidence
                probabilities = answer.probabilities
            if not isinstance(choice, str) or not isinstance(probabilities, dict):
                raise BackendUnavailable(f"JEV returned an invalid choice for property {prop.name!r}")
            values.append(
                SemanticValue(
                    property_name=prop.name,
                    value=choice,
                    strength=float(probabilities.get(choice, 0.0)),
                    confidence=float(confidence),
                    evidence_quote=None,
                )
            )
        return values

    def _request(self, prompt: str, schema: type) -> tuple[Any, Usage]:
        raise BackendUnavailable("JEV does not support generic structured generation")
