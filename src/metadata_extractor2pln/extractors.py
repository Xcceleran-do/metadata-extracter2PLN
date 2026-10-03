from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from .backends import BackendUnavailable, ModelBackend
from .compiler import CONTRACT_VERSION, compile_fact
from .models import (
    Evidence,
    ExtractedProperty,
    ExtractionPlan,
    ExtractResponse,
    PropertySpec,
    RecordResult,
    SemanticValue,
    Usage,
)
from .utils import coerce_float, get_path, parse_datetime, stable_hash, text_from_value


@dataclass(frozen=True)
class Corpus:
    numeric: dict[str, list[float]]


logger = logging.getLogger(__name__)


def extract_records(
    *,
    namespace: str,
    plan: ExtractionPlan,
    records: Sequence[dict[str, Any]],
    backend: ModelBackend | None,
) -> ExtractResponse:
    corpus = _build_corpus(plan, records)
    source_ids = [
        _source_id(record, plan, index) for index, record in enumerate(records)
    ]
    properties_by_record: list[list[ExtractedProperty]] = [[] for _ in records]
    errors_by_record: list[list[str]] = [[] for _ in records]
    semantic_specs = [
        spec for spec in plan.properties if spec.extractor == "semantic_text"
    ]
    texts = [_record_text(record, plan) for record in records]
    usage = Usage()

    for index, record in enumerate(records):
        for spec in plan.properties:
            if spec.extractor == "semantic_text":
                continue
            try:
                extracted = _extract_deterministic(spec, record, corpus)
                if extracted is not None:
                    properties_by_record[index].append(extracted)
                elif spec.required:
                    errors_by_record[index].append(
                        f"required property {spec.name!r} could not be extracted"
                    )
            except ValueError as exc:
                errors_by_record[index].append(f"{spec.name}: {exc}")

    if semantic_specs:
        if backend is None or not backend.ready:
            for errors in errors_by_record:
                errors.extend(
                    f"required property {spec.name!r} needs a configured model backend"
                    for spec in semantic_specs
                    if spec.required
                )
        else:
            try:
                batch, usage = backend.extract_semantics(
                    texts=texts,
                    properties=semantic_specs,
                )
                returned: set[int] = set()
                for semantic_record in batch.records:
                    index = semantic_record.record_index
                    if index >= len(records):
                        for errors in errors_by_record:
                            errors.append(
                                f"model returned out-of-range record_index {index}"
                            )
                        continue
                    if index in returned:
                        errors_by_record[index].append(
                            f"model returned duplicate record_index {index}"
                        )
                        continue
                    returned.add(index)
                    errors_by_record[index].extend(semantic_record.errors)
                    extracted, semantic_errors = _validate_semantics(
                        semantic_record.values, semantic_specs, texts[index]
                    )
                    properties_by_record[index].extend(extracted)
                    errors_by_record[index].extend(semantic_errors)
                for index in set(range(len(records))) - returned:
                    errors_by_record[index].extend(
                        f"required property {spec.name!r} was not returned by the model"
                        for spec in semantic_specs
                        if spec.required
                    )
            except (BackendUnavailable, RuntimeError, ValueError):
                logger.exception("Semantic batch extraction failed")
                for errors in errors_by_record:
                    errors.append(
                        "semantic extraction failed because the model backend is unavailable"
                    )

    results: list[RecordResult] = []
    for index, source_id in enumerate(source_ids):
        properties = properties_by_record[index]
        facts = [
            compile_fact(namespace=namespace, entity_id=source_id, extracted=item)
            for item in properties
            if next(
                spec for spec in plan.properties if spec.name == item.name
            ).include_in_pln
        ]
        results.append(
            RecordResult(
                source_id=source_id,
                entity_id=f"{namespace}_{source_id}",
                properties=properties,
                facts=facts,
                errors=errors_by_record[index],
            )
        )
    return ExtractResponse(
        contract_version=CONTRACT_VERSION,
        plan_fingerprint=plan.fingerprint,
        records=results,
        usage=usage,
    )


def _extract_deterministic(
    spec: PropertySpec, record: dict[str, Any], corpus: Corpus
) -> ExtractedProperty | None:
    values = [get_path(record, path) for path in spec.field_paths]
    if spec.extractor == "structured_field":
        raw = next((value for value in values if value not in (None, "")), None)
        if raw is None:
            return None
        value = text_from_value(raw).strip()
        value = _canonical_allowed(spec, value)
        return _property(spec, value, 1.0, 1.0, f"copied from {spec.field_paths[0]}")
    if spec.extractor == "numeric_bucket":
        raw = next(
            (
                coerce_float(value)
                for value in values
                if coerce_float(value) is not None
            ),
            None,
        )
        if raw is None:
            return None
        population = corpus.numeric.get(spec.name, [])
        percentile = _percentile(raw, population)
        value = (
            "low" if percentile < 1 / 3 else "medium" if percentile < 2 / 3 else "high"
        )
        value = _canonical_allowed(spec, value)
        return _property(
            spec,
            value,
            0.15 + 0.8 * percentile,
            0.9,
            f"relative rank of numeric value {raw:g}",
        )
    if spec.extractor == "date_bucket":
        parsed = next(
            (parse_datetime(value) for value in values if parse_datetime(value)), None
        )
        if parsed is None:
            return None
        value = _canonical_allowed(spec, parsed.strftime("%Y-%m"))
        return _property(
            spec, value, 1.0, 1.0, f"calendar month derived from {parsed.isoformat()}"
        )
    if spec.extractor == "calculated_metric":
        if spec.metric in {"length", "reading-time"}:
            text = " ".join(text_from_value(value) for value in values).strip()
            if not text:
                return None
            words = len(text.split())
            if spec.metric == "length":
                value = (
                    "short" if words < 300 else "medium" if words < 1_000 else "long"
                )
                detail = f"classified from {words} words"
            else:
                minutes = max(1, math.ceil(words / 220))
                value = (
                    "quick" if minutes <= 2 else "medium" if minutes <= 5 else "long"
                )
                detail = f"estimated {minutes} minute reading time from {words} words"
            return _property(spec, _canonical_allowed(spec, value), 1.0, 0.98, detail)
        if spec.metric == "engagement":
            result = _engagement(record, spec.field_paths)
            if result is None:
                return None
            value, strength, detail = result
            return _property(
                spec,
                _canonical_allowed(spec, value),
                strength,
                0.9,
                detail,
            )
    return None


def _validate_semantics(
    values: Sequence[SemanticValue], specs: Sequence[PropertySpec], text: str
) -> tuple[list[ExtractedProperty], list[str]]:
    by_name = {spec.name: spec for spec in specs}
    seen: set[str] = set()
    extracted: list[ExtractedProperty] = []
    errors: list[str] = []
    for item in values:
        spec = by_name.get(item.property_name)
        if spec is None:
            logger.warning(
                "Ignoring unrequested semantic property %r", item.property_name
            )
            continue
        if item.property_name in seen:
            logger.warning(
                "Ignoring duplicate semantic property %r", item.property_name
            )
            continue
        seen.add(item.property_name)
        try:
            value = _canonical_allowed(spec, item.value)
        except ValueError as exc:
            if spec.required:
                errors.append(f"{spec.name}: {exc}")
            else:
                logger.warning(
                    "Ignoring optional semantic value for %s: %s", spec.name, exc
                )
            continue
        quote = item.evidence_quote
        confidence = item.confidence
        detail = "model classification constrained by the extraction plan"
        if quote and quote not in text:
            quote = None
            confidence = min(confidence, 0.6)
            detail = (
                "model classification without a verified source quote"
            )
        extracted.append(
            ExtractedProperty(
                name=spec.name,
                value=value,
                strength=item.strength,
                confidence=confidence,
                evidence=Evidence(
                    method="semantic_text",
                    detail=detail,
                    quote=quote,
                ),
            )
        )
    for spec in specs:
        if spec.required and spec.name not in seen:
            errors.append(
                f"required property {spec.name!r} was not returned by the model"
            )
    return extracted, errors


def _build_corpus(plan: ExtractionPlan, records: Sequence[dict[str, Any]]) -> Corpus:
    numeric: dict[str, list[float]] = {}
    for spec in plan.properties:
        if spec.extractor == "numeric_bucket":
            numeric[spec.name] = [
                number
                for record in records
                for path in spec.field_paths
                if (number := coerce_float(get_path(record, path))) is not None
            ]
    return Corpus(numeric=numeric)


def _property(
    spec: PropertySpec, value: str, strength: float, confidence: float, detail: str
) -> ExtractedProperty:
    return ExtractedProperty(
        name=spec.name,
        value=value,
        strength=max(0.0, min(1.0, strength)),
        confidence=max(0.0, min(1.0, confidence)),
        evidence=Evidence(method=spec.extractor, detail=detail),
    )


def _canonical_allowed(spec: PropertySpec, value: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError("extracted value is blank")
    if not spec.allowed_values:
        return value
    allowed = {item.casefold(): item for item in spec.allowed_values}
    try:
        return allowed[value.casefold()]
    except KeyError as exc:
        raise ValueError(f"value {value!r} is not in allowed_values") from exc


def _percentile(value: float, population: Sequence[float]) -> float:
    if not population:
        return 0.5
    below = sum(item < value for item in population)
    equal = sum(item == value for item in population)
    return (below + 0.5 * equal) / len(population)


def _engagement(
    record: dict[str, Any], field_paths: Sequence[str]
) -> tuple[str, float, str] | None:
    interactions = 0.0
    views: float | None = None
    found = False
    weights = {"comment": 2.0, "share": 3.0}
    for path in field_paths:
        number = coerce_float(get_path(record, path))
        if number is None:
            continue
        found = True
        lowered = path.lower()
        if "view" in lowered or "impression" in lowered:
            views = max(0.0, number) if views is None else views + max(0.0, number)
            continue
        weight = next(
            (value for name, value in weights.items() if name in lowered), 1.0
        )
        interactions += max(0.0, number) * weight
    if not found:
        return None

    if views is not None and views > 0:
        score = interactions / views
        if score < 0.01:
            label = "Low"
        elif score < 0.03:
            label = "Medium"
        elif score < 0.07:
            label = "High"
        else:
            label = "Very_High"
        strength = min(1.0, score / 0.1)
        detail = (
            f"weighted interactions={interactions:g}; views={views:g}; "
            f"engagement rate={score:.4f}"
        )
    else:
        score = interactions
        if score < 3:
            label = "Low"
        elif score < 10:
            label = "Medium"
        elif score < 30:
            label = "High"
        else:
            label = "Very_High"
        strength = min(1.0, math.log1p(score) / math.log(31))
        detail = f"weighted interactions={interactions:g}; views unavailable"
    return label, round(strength, 3), detail


def _record_text(record: dict[str, Any], plan: ExtractionPlan) -> str:
    return "\n".join(
        text_from_value(get_path(record, path)).strip()
        for path in plan.text_fields
        if text_from_value(get_path(record, path)).strip()
    )[:100_000]


def _source_id(record: dict[str, Any], plan: ExtractionPlan, index: int) -> str:
    for path in plan.id_fields:
        value = get_path(record, path)
        if value not in (None, ""):
            return str(value)[:256]
    return f"record-{index}-{stable_hash(record)[:12]}"
