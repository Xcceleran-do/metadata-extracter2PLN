from metadata_extractor2pln.extractors import (
    _engagement,
    _validate_semantics,
    extract_records,
)
from metadata_extractor2pln.models import (
    PlanDraft,
    PropertySpec,
    SemanticBatchResult,
    SemanticRecordResult,
    SemanticValue,
    Usage,
)
from metadata_extractor2pln.planner import sanitize_plan
from metadata_extractor2pln.structured_backend import _coerce_semantic_batch


class FakeBackend:
    name = "fake-model"
    ready = True
    calls = 0

    def extract_semantics(self, *, texts, properties):
        self.calls += 1
        return (
            SemanticBatchResult(
                records=[
                    SemanticRecordResult(
                        record_index=index,
                        values=[
                            SemanticValue(
                                property_name="audience-expertise",
                                value="Expert",
                                strength=0.8,
                                confidence=0.9,
                                evidence_quote="experienced engineers",
                            )
                        ],
                    )
                    for index in range(len(texts))
                ]
            ),
            Usage(input_tokens=10, output_tokens=4),
        )


RECORDS = [
    {
        "id": "a",
        "content": "For experienced engineers.",
        "likes": 1,
        "comments": 0,
        "shares": 0,
    },
    {
        "id": "b",
        "content": "For experienced engineers.",
        "likes": 40,
        "comments": 20,
        "shares": 10,
    },
]


def _plan():
    draft = PlanDraft(
        id_fields=["id"],
        text_fields=["content"],
        properties=[
            PropertySpec(
                name="audience-expertise",
                description="Intended reader expertise",
                extractor="semantic_text",
                field_paths=["content"],
                allowed_values=["Beginner", "Expert"],
                required=True,
            )
        ],
    )
    return sanitize_plan(
        source_name="articles",
        draft=draft,
        records=RECORDS,
        required_properties=["engagement", "audience-expertise"],
        planner="heuristic",
    )


def test_extracts_semantics_engagement_and_compiles_petta_facts():
    backend = FakeBackend()
    response = extract_records(
        namespace="news",
        plan=_plan(),
        records=RECORDS,
        backend=backend,
    )

    first, second = response.records
    first_props = {item.name: item for item in first.properties}
    second_props = {item.name: item for item in second.properties}
    assert first_props["engagement"].value == "Low"
    assert second_props["engagement"].value == "Very_High"
    assert "weighted interactions=110" in second_props["engagement"].evidence.detail
    assert first_props["audience-expertise"].value == "expert"
    assert first_props["audience-expertise"].evidence.quote == "experienced engineers"
    assert all(fact.source.startswith("(: news_") for fact in first.facts)
    assert all("(STV " in fact.source for fact in first.facts)
    assert all(fact.atom.startswith("(") for fact in first.facts)
    assert response.usage.input_tokens == 10
    assert backend.calls == 1


def test_missing_model_is_an_explicit_record_error():
    response = extract_records(
        namespace="news", plan=_plan(), records=RECORDS[:1], backend=None
    )
    assert any(
        "needs a configured model backend" in error
        for error in response.records[0].errors
    )


def test_engagement_uses_interaction_rate_when_views_are_available():
    paths = ["views", "likes", "comments", "shares"]

    low = _engagement(
        {"views": 1_000, "likes": 2, "comments": 1, "shares": 0}, paths
    )
    high = _engagement(
        {"views": 100, "likes": 10, "comments": 10, "shares": 10}, paths
    )

    assert low is not None and low[0] == "Low"
    assert high is not None and high[0] == "Very_High"
    assert "engagement rate=" in high[2]


def test_oversized_semantic_value_is_isolated_to_its_record():
    result = _coerce_semantic_batch(
        {
            "records": [
                {
                    "record_index": 0,
                    "values": [
                        {
                            "property_name": "audience-expertise",
                            "value": "beginner",
                            "strength": 0.8,
                            "confidence": 0.9,
                        }
                    ],
                },
                {
                    "record_index": 1,
                    "values": [
                        {
                            "property_name": "audience-expertise",
                            "value": "copied article " * 100,
                            "strength": 0.8,
                            "confidence": 0.9,
                        }
                    ],
                },
            ]
        }
    )

    assert result.records[0].values[0].value == "beginner"
    assert result.records[1].values == []


def test_unverified_evidence_quote_does_not_reject_classification():
    spec = PropertySpec(
        name="audience-expertise",
        description="Expertise expected from the intended reader",
        extractor="semantic_text",
        allowed_values=["beginner", "intermediate", "advanced", "expert"],
        required=True,
    )
    properties, errors = _validate_semantics(
        [
            SemanticValue(
                property_name="audience-expertise",
                value="expert",
                strength=0.9,
                confidence=0.95,
                evidence_quote="a paraphrase that is not in the article",
            )
        ],
        [spec],
        "A technical article written for experienced engineers.",
    )

    assert errors == []
    assert properties[0].value == "expert"
    assert properties[0].evidence.quote is None
    assert properties[0].confidence == 0.6


def test_invalid_optional_semantic_value_is_not_fatal():
    spec = PropertySpec(
        name="tone",
        description="Dominant communication tone",
        extractor="semantic_text",
        allowed_values=["analytical", "conversational", "neutral"],
    )

    properties, errors = _validate_semantics(
        [
            SemanticValue(
                property_name="tone",
                value="narrative",
                strength=0.8,
                confidence=0.9,
            )
        ],
        [spec],
        "A narrative article.",
    )

    assert properties == []
    assert errors == []


def test_backend_errors_do_not_change_the_model_response_schema():
    legacy = SemanticBatchResult.model_validate(
        {"records": [{"record_index": 0, "values": []}]}
    )
    assert legacy.records[0].errors == []
    schema = SemanticBatchResult.model_json_schema()
    assert "errors" not in schema["$defs"]["SemanticRecordResult"]["properties"]
    failed = SemanticRecordResult(record_index=0, errors=["provider unavailable"])
    assert failed.model_dump()["errors"] == ["provider unavailable"]
