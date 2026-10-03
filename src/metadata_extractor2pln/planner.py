from __future__ import annotations

from collections import Counter
import logging
from typing import Any, Iterable, Sequence

from .backends import BackendUnavailable, ModelBackend
from .models import ExtractionPlan, PlanDraft, PropertySpec, Usage
from .utils import get_path, normalize_name, schema_paths, stable_hash, text_from_value


ID_HINTS = ("id", "uuid", "slug", "url", "uri")
TEXT_HINTS = ("content", "text", "body", "description", "summary", "article")
DATE_HINTS = ("date", "time", "created", "published", "updated")
IDENTITY_NAMES = {"id", "identifier", "slug", "url", "uri", "name", "title"}
CONTENT_TAXONOMY = (
    (
        "audience-expertise",
        "Expertise expected from the intended reader",
        ["beginner", "intermediate", "advanced", "expert"],
    ),
    (
        "topic",
        "Dominant subject of the article",
        [
            "technology",
            "science",
            "business",
            "politics",
            "culture",
            "society",
            "health",
            "environment",
            "education",
            "entertainment",
            "philosophy",
            "other",
        ],
    ),
    (
        "tone",
        "Dominant communication tone",
        [
            "analytical",
            "educational",
            "conversational",
            "persuasive",
            "critical",
            "inspirational",
            "humorous",
            "neutral",
        ],
    ),
    (
        "content-type",
        "Editorial format of the article",
        [
            "news",
            "analysis",
            "opinion",
            "tutorial",
            "interview",
            "review",
            "research",
            "announcement",
            "narrative",
            "other",
        ],
    ),
    (
        "primary-goal",
        "Primary purpose of the article",
        ["inform", "explain", "persuade", "teach", "entertain", "critique", "promote", "discuss"],
    ),
    (
        "sentiment",
        "Overall evaluative sentiment",
        ["positive", "neutral", "negative", "mixed"],
    ),
    (
        "complexity",
        "Conceptual complexity of the subject matter",
        ["low", "medium", "high"],
    ),
    (
        "actionability",
        "How directly a reader can act on the article",
        ["low", "medium", "high"],
    ),
)
logger = logging.getLogger(__name__)


def discover_plan(
    *,
    source_name: str,
    records: Sequence[dict[str, Any]],
    required_properties: Sequence[str],
    backend: ModelBackend | None = None,
    use_model: bool = True,
) -> tuple[ExtractionPlan, str | None, Usage]:
    required = _required_names(required_properties)
    if use_model and backend is not None and backend.ready:
        try:
            draft, usage = backend.discover_plan(
                source_name=source_name,
                records=records,
                required_properties=required,
            )
            planner = backend.provider
            model = backend.name
        except BackendUnavailable as exc:
            logger.warning(
                "Model planning failed; using the deterministic planner: %s", exc
            )
            draft = heuristic_plan(records)
            usage = Usage()
            planner = "heuristic"
            model = None
    else:
        draft = heuristic_plan(records)
        usage = Usage()
        planner = "heuristic"
        model = None
    plan = sanitize_plan(
        source_name=source_name,
        draft=draft,
        records=records,
        required_properties=required,
        planner=planner,
    )

    validate_backend_compatibility(
        plan,
        getattr(backend, "provider", None) if backend is not None else None,
    )

    return plan, model, usage


def heuristic_plan(records: Sequence[dict[str, Any]]) -> PlanDraft:
    paths = schema_paths(records)
    leaf_paths = [
        path
        for path in paths
        if not any(other.startswith(f"{path}.") for other in paths)
    ]
    id_fields = _ranked_paths(leaf_paths, ID_HINTS)[:4] or [
        leaf_paths[0] if leaf_paths else "id"
    ]
    text_fields = _ranked_paths(leaf_paths, TEXT_HINTS)[:4]
    properties: list[PropertySpec] = []

    if text_fields:
        properties.extend(
            [
                PropertySpec(
                    name="length-bucket",
                    description="Length of the main text",
                    extractor="calculated_metric",
                    field_paths=text_fields,
                    metric="length",
                ),
                PropertySpec(
                    name="reading-time",
                    description="Estimated time required to read the main text",
                    extractor="calculated_metric",
                    field_paths=text_fields,
                    metric="reading-time",
                ),
            ]
        )

    for path in leaf_paths:
        name = normalize_name(path.rsplit(".", 1)[-1].replace("[]", ""))
        lowered = path.lower()
        if (
            name in IDENTITY_NAMES
            or path in text_fields
            or any(hint in lowered for hint in ID_HINTS)
        ):
            continue
        values = [get_path(record, path) for record in records]
        present = [value for value in values if value not in (None, "")]
        if not present:
            continue
        if any(hint in lowered for hint in DATE_HINTS):
            properties.append(
                PropertySpec(
                    name=f"{name}-period",
                    description=f"Calendar period derived from {path}",
                    extractor="date_bucket",
                    field_paths=[path],
                )
            )
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in present
        ):
            properties.append(
                PropertySpec(
                    name=name,
                    description=f"Relative numeric level of {path}",
                    extractor="numeric_bucket",
                    field_paths=[path],
                    allowed_values=["low", "medium", "high"],
                )
            )
        else:
            distinct = _distinct_strings(present)
            if 1 < len(distinct) <= 12:
                properties.append(
                    PropertySpec(
                        name=name,
                        description=f"Structured value from {path}",
                        extractor="structured_field",
                        field_paths=[path],
                        allowed_values=distinct,
                    )
                )
        if len(properties) >= 20:
            break

    return PlanDraft(
        id_fields=id_fields, text_fields=text_fields, properties=properties
    )


def sanitize_plan(
    *,
    source_name: str,
    draft: PlanDraft,
    records: Sequence[dict[str, Any]],
    required_properties: Sequence[str],
    planner: str,
) -> ExtractionPlan:
    available = set(schema_paths(records))
    ids = _valid_paths(draft.id_fields, available)
    texts = _valid_paths(draft.text_fields, available)
    if not ids:
        ids = _ranked_paths(available, ID_HINTS)[:4] or [
            sorted(available)[0] if available else "id"
        ]
    if not texts:
        texts = _ranked_paths(available, TEXT_HINTS)[:4]

    by_name: dict[str, PropertySpec] = {}
    for raw in draft.properties:
        name = normalize_name(raw.name)
        if name in IDENTITY_NAMES:
            continue
        if raw.extractor == "semantic_text" and not raw.allowed_values:
            continue
        paths = _valid_paths(raw.field_paths, available)
        if (
            raw.extractor in {"structured_field", "numeric_bucket", "date_bucket"}
            and not paths
        ):
            continue
        spec = raw.model_copy(update={"name": name, "field_paths": paths})
        by_name.setdefault(name, spec)

    if texts:
        by_name.update(
            {spec.name: spec for spec in _canonical_content_specs(texts)}
        )

    for required in _required_names(required_properties):
        by_name[required] = _required_spec(
            required, texts, available, by_name.get(required)
        )

    plan = ExtractionPlan(
        source_name=source_name,
        entity_type=normalize_name(draft.entity_type, "item"),
        id_fields=ids,
        text_fields=texts,
        properties=list(by_name.values()),
        planner=planner,
    )
    return plan.model_copy(update={"fingerprint": plan_fingerprint(plan)})


def validate_and_pin_plan(
    plan: ExtractionPlan,
    required_properties: Sequence[str] = (),
    backend_provider: str | None = None,
) -> ExtractionPlan:
    names = {item.name for item in plan.properties}
    missing = set(_required_names(required_properties)) - names

    if missing:
        raise ValueError(
            f"plan is missing required properties: {', '.join(sorted(missing))}"
        )

    validate_backend_compatibility(plan, backend_provider)

    expected = plan_fingerprint(plan)

    if plan.fingerprint and plan.fingerprint != expected:
        raise ValueError("plan fingerprint does not match its contents")

    return plan.model_copy(update={"planner": "pinned", "fingerprint": expected})

def plan_fingerprint(plan: ExtractionPlan) -> str:
    return stable_hash(plan.model_dump(exclude={"fingerprint", "planner"}, mode="json"))


def _required_spec(
    name: str,
    text_fields: list[str],
    available: set[str],
    proposed: PropertySpec | None,
) -> PropertySpec:
    if name == "engagement":
        hints = (
            "engagement",
            "like",
            "comment",
            "share",
            "view",
            "upvote",
            "reaction",
            "score",
        )
        paths = _ranked_paths(available, hints)[:16]
        return PropertySpec(
            name=name,
            description="Engagement level calculated from available interaction metrics",
            extractor="calculated_metric",
            field_paths=paths,
            allowed_values=["Low", "Medium", "High", "Very_High"],
            metric="engagement",
            required=True,
        )
    if name == "audience-expertise":
        return PropertySpec(
            name=name,
            description="Expertise expected from the intended reader",
            extractor="semantic_text",
            field_paths=text_fields,
            allowed_values=["beginner", "intermediate", "advanced", "expert"],
            required=True,
        )
    if proposed is not None:
        return proposed.model_copy(update={"name": name, "required": True})
    return PropertySpec(
        name=name,
        description=f"Semantic classification for {name}",
        extractor="semantic_text",
        field_paths=text_fields,
        required=True,
    )


def _canonical_content_specs(text_fields: list[str]) -> list[PropertySpec]:
    specs = [
        PropertySpec(
            name="length-bucket",
            description="Length of the main article text",
            extractor="calculated_metric",
            field_paths=text_fields,
            allowed_values=["short", "medium", "long"],
            metric="length",
        ),
        PropertySpec(
            name="reading-time",
            description="Estimated time required to read the article",
            extractor="calculated_metric",
            field_paths=text_fields,
            allowed_values=["quick", "medium", "long"],
            metric="reading-time",
        ),
    ]
    specs.extend(
        PropertySpec(
            name=name,
            description=description,
            extractor="semantic_text",
            field_paths=text_fields,
            allowed_values=allowed_values,
        )
        for name, description, allowed_values in CONTENT_TAXONOMY
    )
    return specs


def _required_names(values: Iterable[str]) -> list[str]:
    return list(
        dict.fromkeys(normalize_name(value) for value in values if str(value).strip())
    )


def _ranked_paths(paths: Iterable[str], hints: Sequence[str]) -> list[str]:
    ranked = []
    for path in paths:
        lowered = path.lower()
        scores = [index for index, hint in enumerate(hints) if hint in lowered]
        if scores:
            ranked.append((min(scores), len(path), path))
    return [item[2] for item in sorted(ranked)]


def _valid_paths(paths: Iterable[str], available: set[str]) -> list[str]:
    return list(dict.fromkeys(path for path in paths if path in available))


def _distinct_strings(values: Iterable[Any]) -> list[str]:
    counter = Counter(text_from_value(value).strip() for value in values)
    return [value for value, _ in counter.most_common() if value][:100]
def validate_backend_compatibility(
    plan: ExtractionPlan,
    backend_provider: str | None,
) -> None:
    if backend_provider != "jev":
        return

    incompatible = [
        item.name
        for item in plan.properties
        if item.extractor == "semantic_text"
        and not item.allowed_values
    ]

    if incompatible:
        raise ValueError(
            "JEV semantic classification requires allowed_values for "
            f"these properties: {', '.join(sorted(incompatible))}. "
            "Define explicit allowed values before using JEV."
        )