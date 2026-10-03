import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import pytest

from metadata_extractor2pln.backends import BackendUnavailable
from metadata_extractor2pln.jev_backend import JEVBackend
from metadata_extractor2pln.models import ExtractionPlan, PropertySpec, ExtractRequest
from metadata_extractor2pln.planner import validate_and_pin_plan
from metadata_extractor2pln.request_context import (
    RequestCancellation,
    reset_request_deadline,
    set_request_deadline,
    reset_request_cancelled,
    set_request_cancelled,
)
from metadata_extractor2pln.service import MetadataService


def make_sentiment_property():
    return PropertySpec(
        name="sentiment",
        description="Classify the sentiment of the text.",
        extractor="semantic_text",
        allowed_values=["positive", "negative"],
    )


def answer_payload():
    return {
        "model": "jev-test",
        "answers": {
            "sentiment": {
                "type": "choice",
                "choice": "positive",
                "confidence": 0.9,
                "probabilities": {"positive": 0.8, "negative": 0.2},
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def mock_client(monkeypatch, transport, responses):
    calls = []
    options = {}

    class Client:
        def __init__(self, **kwargs):
            options.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def next_response(self, call):
            calls.append(call)
            result = responses[len(calls) - 1]
            if isinstance(result, Exception):
                raise result
            return result

        async def system_one(self, **kwargs):
            result = await self.next_response(kwargs)
            return SimpleNamespace(
                answers=result["answers"],
                usage=SimpleNamespace(**result["usage"]),
            )

        async def post(self, url, **kwargs):
            result = await self.next_response({"url": url, **kwargs})
            return httpx.Response(200, json=result, request=httpx.Request("POST", url))

    if transport == "typesafe":
        monkeypatch.setattr("metadata_extractor2pln.jev_backend.AsyncTypeSafeClient", Client)
    else:
        monkeypatch.setattr("metadata_extractor2pln.jev_backend.httpx.AsyncClient", Client)
    backend = JEVBackend(
        transport=transport,
        model="jev-test",
        api_key="test-key",
        openrouter_api_key="test-openrouter-key",
    )
    return backend, calls, options


@pytest.mark.parametrize("transport", ["typesafe", "openrouter"])
def test_jev_extract_semantics(monkeypatch, transport):
    backend, calls, options = mock_client(monkeypatch, transport, [answer_payload()])
    result, usage = backend.extract_semantics(
        texts=["Excellent article."], properties=[make_sentiment_property()]
    )
    value = result.records[0].values[0]
    assert value.property_name == "sentiment"
    assert value.value == "positive"
    assert value.strength == 0.8
    assert value.confidence == 0.9
    assert value.evidence_quote is None
    assert result.records[0].errors == []
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)
    assert options["timeout"] == 45.0
    if transport == "typesafe":
        assert options["retry"].max_retries == 0
        assert calls[0]["model"] == "jev-test"
        assert calls[0]["state"] == "Excellent article."
        assert calls[0]["questions"]["sentiment"].criteria == {"positive": None, "negative": None}
    else:
        assert calls[0]["url"] == "https://openrouter.ai/api/alpha/decisions"
        assert calls[0]["headers"]["Authorization"] == "Bearer test-openrouter-key"
        assert calls[0]["json"]["model"] == "typesafe/jev-1.13"
        assert calls[0]["json"]["questions"]["sentiment"]["criteria"] == {"positive": None, "negative": None}


@pytest.mark.parametrize("transport", ["typesafe", "openrouter"])
def test_jev_requires_api_key(transport):
    backend = JEVBackend(transport=transport)
    assert not backend.ready
    with pytest.raises(BackendUnavailable, match="API_KEY"):
        backend.extract_semantics(texts=["text"], properties=[make_sentiment_property()])


def test_jev_does_not_discover_plans():
    with pytest.raises(BackendUnavailable, match="classification only"):
        JEVBackend(api_key="test-key").discover_plan(
            source_name="test", records=[{"text": "example"}], required_properties=[]
        )


def test_jev_requires_allowed_values():
    prop = make_sentiment_property().model_copy(update={"allowed_values": []})
    with pytest.raises(ValueError, match="allowed_values"):
        JEVBackend(api_key="test-key").extract_semantics(texts=["text"], properties=[prop])


@pytest.mark.parametrize("transport", ["typesafe", "openrouter"])
@pytest.mark.parametrize("failure", ["network", "missing_answer", "invalid_probability"])
def test_jev_preserves_successes_and_reports_failed_records(monkeypatch, transport, failure):
    bad = answer_payload()
    if failure == "network":
        bad = RuntimeError("secret provider detail")
    elif failure == "missing_answer":
        bad["answers"] = {}
    else:
        bad["answers"]["sentiment"]["probabilities"]["positive"] = 2.0
    backend, calls, _ = mock_client(
        monkeypatch, transport, [answer_payload(), bad, answer_payload()]
    )
    result, usage = backend.extract_semantics(
        texts=["first", "second", "third"], properties=[make_sentiment_property()]
    )
    assert len(calls) == 3
    assert [record.record_index for record in result.records] == [0, 1, 2]
    assert result.records[0].values[0].value == "positive"
    assert result.records[2].values[0].value == "positive"
    assert result.records[1].values == []
    assert result.records[1].errors
    assert "secret" not in result.records[1].errors[0]
    assert usage.input_tokens == (20 if failure == "network" else 30)


@pytest.mark.parametrize("invalid", [[], None, {"answers": []}, {"answers": {}, "usage": "bad"}])
def test_openrouter_invalid_response_is_isolated(monkeypatch, invalid):
    backend, _, _ = mock_client(monkeypatch, "openrouter", [answer_payload(), invalid])
    result, _ = backend.extract_semantics(
        texts=["first", "second"], properties=[make_sentiment_property()]
    )
    assert result.records[0].values
    assert result.records[1].errors


@pytest.mark.parametrize("transport", ["typesafe", "openrouter"])
def test_expired_deadline_starts_no_provider_call(monkeypatch, transport):
    backend, calls, _ = mock_client(monkeypatch, transport, [])
    token = set_request_deadline(time.monotonic() - 1)
    try:
        result, usage = backend.extract_semantics(
            texts=["first", "second"], properties=[make_sentiment_property()]
        )
    finally:
        reset_request_deadline(token)
    assert not calls
    assert len(result.records) == 2
    assert all(record.errors for record in result.records)
    assert usage.input_tokens == 0


@pytest.mark.parametrize("transport", ["typesafe", "openrouter"])
def test_absolute_deadline_cancels_inflight_call(monkeypatch, transport):
    backend, _, _ = mock_client(monkeypatch, transport, [])
    cancelled = []
    calls = []

    async def blocked_call(*args):
        calls.append(True)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(backend, "_classify", blocked_call)
    token = set_request_deadline(time.monotonic() + 0.05)
    try:
        result, _ = backend.extract_semantics(
            texts=["first", "second"], properties=[make_sentiment_property()]
        )
    finally:
        reset_request_deadline(token)
    assert calls == [True]
    assert cancelled == [True]
    assert all(record.errors for record in result.records)


@pytest.mark.parametrize("transport", ["typesafe", "openrouter"])
def test_per_call_deadline_cancels_slow_response(monkeypatch, transport):
    backend, _, _ = mock_client(monkeypatch, transport, [])
    backend.timeout_seconds = 0.01
    cancelled = []

    async def blocked_call(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr(backend, "_classify", blocked_call)
    result, _ = backend.extract_semantics(
        texts=["first", "second"], properties=[make_sentiment_property()]
    )
    assert cancelled == [True, True]
    assert all(record.errors for record in result.records)


def test_cancellation_stops_later_records(monkeypatch):
    backend, calls, _ = mock_client(monkeypatch, "typesafe", [answer_payload()])
    cancelled = RequestCancellation()
    original = backend._classify

    async def classify_then_cancel(*args):
        result = await original(*args)
        cancelled.set()
        return result

    monkeypatch.setattr(backend, "_classify", classify_then_cancel)
    token = set_request_cancelled(cancelled)
    try:
        result, _ = backend.extract_semantics(
            texts=["first", "second"], properties=[make_sentiment_property()]
        )
    finally:
        reset_request_cancelled(token)
    assert len(calls) == 1
    assert result.records[0].values
    assert result.records[1].errors


@pytest.mark.parametrize("transport", ["typesafe", "openrouter"])
@pytest.mark.parametrize("expired", [False, True])
def test_service_reports_optional_semantic_failure(monkeypatch, transport, expired):
    backend, calls, _ = mock_client(
        monkeypatch, transport, [answer_payload(), RuntimeError("failed")]
    )
    plan = validate_and_pin_plan(ExtractionPlan(
        source_name="articles",
        id_fields=["id"],
        text_fields=["content"],
        properties=[make_sentiment_property()],
    ))
    token = set_request_deadline(time.monotonic() - 1 if expired else None)
    try:
        response = MetadataService(backend).extract(ExtractRequest(
            namespace="demo",
            plan=plan,
            records=[{"id": "1", "content": "first"}, {"id": "2", "content": "second"}],
        ))
    finally:
        reset_request_deadline(token)
    if expired:
        assert not calls
        assert all(record.errors for record in response.records)
    else:
        assert response.records[0].facts
        assert response.records[0].errors == []
    assert response.records[1].errors
    assert response.records[1].facts == []


def test_openrouter_slow_stream_is_cancelled(monkeypatch):
    closed = []

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps(answer_payload()).encode()[:10]
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append(True)

    async def handler(request):
        return httpx.Response(200, stream=SlowBody())

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "metadata_extractor2pln.jev_backend.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    backend = JEVBackend(
        transport="openrouter", openrouter_api_key="test-key", timeout_seconds=0.01
    )
    result, _ = backend.extract_semantics(texts=["text"], properties=[make_sentiment_property()])
    assert closed
    assert result.records[0].errors


@pytest.mark.parametrize("transport", ["typesafe", "openrouter"])
def test_cancellation_interrupts_active_provider_call(monkeypatch, transport):
    backend, _, _ = mock_client(monkeypatch, transport, [])
    cancelled = RequestCancellation()
    stopped = []
    calls = []

    async def blocked_call(*args):
        calls.append(True)
        asyncio.get_running_loop().call_soon(cancelled.set)
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(True)

    monkeypatch.setattr(backend, "_classify", blocked_call)
    token = set_request_cancelled(cancelled)
    try:
        result, _ = backend.extract_semantics(
            texts=["first", "second"], properties=[make_sentiment_property()]
        )
    finally:
        reset_request_cancelled(token)
    assert calls == [True]
    assert stopped == [True]
    assert all(record.errors for record in result.records)


def test_cancellation_before_registration_starts_no_work():
    async def scenario():
        cancelled = RequestCancellation()
        entered = []

        async def classify():
            entered.append(True)

        task = asyncio.create_task(classify())
        cancelled.set()
        with cancelled.watch(task):
            with pytest.raises(asyncio.CancelledError):
                await task
        assert entered == []

    asyncio.run(scenario())


def test_repeated_cancellation_preserves_async_cleanup():
    async def scenario():
        cancelled = RequestCancellation()
        started = asyncio.Event()
        cleaning = asyncio.Event()
        finish_cleanup = asyncio.Event()

        async def classify():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish_cleanup.wait()

        task = asyncio.create_task(classify())
        with cancelled.watch(task):
            await started.wait()
            cancelled.set()
            await cleaning.wait()
            cancelled.set()
            finish_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert task.cancelling() == 1

    asyncio.run(scenario())


def real_typesafe_backend(monkeypatch, handler, *, timeout_seconds=45.0):
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

    def client_factory(**kwargs):
        assert isinstance(kwargs["retry"], RetryPolicy)
        assert kwargs["retry"].max_retries == 0
        return AsyncTypeSafeClient(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(
        "metadata_extractor2pln.jev_backend.AsyncTypeSafeClient", client_factory
    )
    return JEVBackend(
        transport="typesafe",
        model="jev-test",
        api_key="test-key",
        timeout_seconds=timeout_seconds,
    )


def test_typesafe_real_sdk_success_contract(monkeypatch):
    import httpx2

    requests = []

    async def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=answer_payload())

    backend = real_typesafe_backend(monkeypatch, handler)
    result, usage = backend.extract_semantics(
        texts=["Excellent article."], properties=[make_sentiment_property()]
    )

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert body["model"] == "jev-test"
    assert body["state"] == "Excellent article."
    assert body["questions"]["sentiment"]["type"] == "choice"
    assert body["questions"]["sentiment"]["criteria"] == {
        "positive": None, "negative": None
    }
    assert result.records[0].errors == []
    value = result.records[0].values[0]
    assert value.property_name == "sentiment"
    assert value.value == "positive"
    assert value.strength == 0.8
    assert value.confidence == 0.9
    assert (usage.input_tokens, usage.output_tokens) == (10, 5)


def test_typesafe_real_sdk_retryable_500_is_not_retried(monkeypatch):
    import httpx2

    requests = []

    async def handler(request):
        requests.append(request)
        return httpx2.Response(500, json={"error": "secret provider detail"})

    backend = real_typesafe_backend(monkeypatch, handler)
    result, usage = backend.extract_semantics(
        texts=["text"], properties=[make_sentiment_property()]
    )

    assert len(requests) == 1
    assert result.records[0].values == []
    assert result.records[0].errors
    assert "secret" not in result.records[0].errors[0]
    assert (usage.input_tokens, usage.output_tokens) == (0, 0)


def test_typesafe_real_sdk_stalled_stream_is_cancelled_and_closed(monkeypatch):
    import httpx2

    requests = []
    cancelled = []
    closed = []

    class StalledBody(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps(answer_payload()).encode()[:10]
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def aclose(self):
            closed.append(True)

    async def handler(request):
        requests.append(request)
        return httpx2.Response(200, stream=StalledBody())

    backend = real_typesafe_backend(monkeypatch, handler, timeout_seconds=0.05)
    result, usage = backend.extract_semantics(
        texts=["text"], properties=[make_sentiment_property()]
    )

    assert len(requests) == 1
    assert cancelled == [True]
    assert closed == [True]
    assert result.records[0].values == []
    assert result.records[0].errors == ["semantic classification exceeded its time limit"]
    assert (usage.input_tokens, usage.output_tokens) == (0, 0)
