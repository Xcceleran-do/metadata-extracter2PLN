import asyncio
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient

from metadata_extractor2pln import api
from metadata_extractor2pln.request_context import (
    RequestCancellation,
    get_request_deadline,
    request_cancelled,
    reset_request_cancelled,
    reset_request_deadline,
    set_request_cancelled,
    set_request_deadline,
)

from metadata_extractor2pln.api import create_app
from metadata_extractor2pln.config import Settings


def _client():
    settings = Settings(
        environment="test",
        api_keys=("tester:secret",),
        allowed_hosts=("testserver",),
        rate_limit_per_minute=20,
    )
    return TestClient(create_app(settings))


def test_health_is_public_and_api_is_authenticated():
    with _client() as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/run", json={}).status_code == 401


def test_offline_run_returns_deterministic_facts():
    payload = {
        "namespace": "demo",
        "source_name": "articles",
        "use_model": False,
        "required_properties": ["engagement"],
        "records": [
            {"id": "one", "content": "A short article.", "likes": 4, "comments": 1},
            {
                "id": "two",
                "content": "Another short article.",
                "likes": 20,
                "comments": 8,
            },
        ],
    }
    with _client() as client:
        response = client.post(
            "/v1/run",
            headers={"Authorization": "Bearer secret"},
            json=payload,
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["plan"]["fingerprint"]
    assert len(body["extraction"]["records"]) == 2
    engagement_fact = next(
        fact
        for fact in body["extraction"]["records"][0]["facts"]
        if fact["property_name"] == "engagement"
    )
    assert engagement_fact["atom"].startswith("(engagement ")


def test_body_size_limit_applies_before_validation():
    settings = Settings(
        environment="test",
        api_keys=("tester:secret",),
        allowed_hosts=("testserver",),
        max_request_bytes=20,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/run",
            headers={"Authorization": "Bearer secret"},
            content=b"x" * 21,
        )
    assert response.status_code == 413


def test_body_size_limit_cannot_be_bypassed_with_chunked_input():
    settings = Settings(
        environment="test",
        api_keys=("tester:secret",),
        allowed_hosts=("testserver",),
        max_request_bytes=10,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/run",
            headers={"Authorization": "Bearer secret"},
            content=iter([b"123456", b"789012"]),
        )
    assert response.status_code == 413
def test_jev_rejects_custom_semantic_property_without_allowed_values():
    settings = Settings(
        environment="test",
        model_provider="jev",
        jev_api_key="test-key",
        api_keys=("tester:secret",),
        allowed_hosts=("testserver",),
        rate_limit_per_minute=20,
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/v1/run",
            headers={"Authorization": "Bearer secret"},
            json={
                "namespace": "demo",
                "source_name": "articles",
                "use_model": False,
                "required_properties": ["stance"],
                "records": [
                    {
                        "id": "one",
                        "content": "A short article about the policy.",
                    }
                ],
            },
        )

    assert response.status_code == 422
    assert "allowed_values" in response.text
    assert "stance" in response.text


class _BlockingService:
    backend = None

    def __init__(self, loop):
        self.loop = loop
        self.started = asyncio.Event()
        self.release = Event()
        self.calls = 0
        self.cancelled = None
        self.deadline = None

    def run(self, request):
        self.calls += 1
        if self.calls == 1:
            self.deadline = get_request_deadline()
            self.loop.call_soon_threadsafe(self.started.set)
            assert self.release.wait(5), "worker was not released"
            self.cancelled = request_cancelled()
            raise RuntimeError("late worker failure")
        assert not request_cancelled()
        assert get_request_deadline() is not None
        # A response bypasses response-model validation for this injected service.
        return api.JSONResponse({"ok": True})

    plan = validate_plan = extract = run


def _blocking_app(service):
    return create_app(
        Settings(
            environment="test",
            api_keys=("tester:secret",),
            allowed_hosts=("testserver",),
            max_concurrent_requests=1,
            request_timeout_seconds=30,
        ),
        service=service,
    )


def test_timeout_includes_queue_and_retains_worker_permit(monkeypatch):
    async def scenario():
        loop = asyncio.get_running_loop()
        service = _BlockingService(loop)
        app = _blocking_app(service)
        timeouts = asyncio.Queue()
        original_timeout = asyncio.timeout

        def controlled_timeout(seconds):
            timeout = original_timeout(seconds)
            timeouts.put_nowait(timeout)
            return timeout

        monkeypatch.setattr(api.asyncio, "timeout", controlled_timeout)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            async def post():
                return await client.post(
                    "/v1/run",
                    headers={"Authorization": "Bearer secret"},
                    json={"namespace": "demo", "records": [{"id": "one"}]},
                )

            first = asyncio.create_task(post())
            try:
                first_timeout = await asyncio.wait_for(timeouts.get(), 5)
                await asyncio.wait_for(service.started.wait(), 5)
                assert service.deadline is not None
                first_timeout.reschedule(loop.time())
                response = await asyncio.wait_for(first, 5)
                assert response.status_code == 504
                assert response.json()["error"]["code"] == "request_timeout"

                # Each queued timeout must leave the original permit occupied.
                for _ in range(2):
                    queued = asyncio.create_task(post())
                    queued_timeout = await asyncio.wait_for(timeouts.get(), 5)
                    queued_timeout.reschedule(loop.time())
                    response = await asyncio.wait_for(queued, 5)
                    assert response.status_code == 504
                    assert service.calls == 1

                service.release.set()
                response = await asyncio.wait_for(post(), 5)
                assert response.status_code == 200
                assert service.calls == 2
                assert service.cancelled is True
            finally:
                service.release.set()
                if not first.done():
                    first.cancel()
                    await asyncio.gather(first, return_exceptions=True)

    asyncio.run(scenario())


def test_cancellation_signals_worker_and_retains_permit():
    async def scenario():
        service = _BlockingService(asyncio.get_running_loop())
        app = _blocking_app(service)
        # Call the route directly so middleware task groups do not mask cancellation.
        endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/run")
        first = asyncio.create_task(endpoint(None))
        try:
            await asyncio.wait_for(service.started.wait(), 5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(endpoint(None))
            # Event-loop barrier: the second request has reached semaphore admission.
            barrier = asyncio.Event()
            asyncio.get_running_loop().call_soon(barrier.set)
            await asyncio.wait_for(barrier.wait(), 5)
            assert service.calls == 1
            assert not second.done()
            service.release.set()
            response = await asyncio.wait_for(second, 5)
            assert response.status_code == 200
            assert service.cancelled is True
        finally:
            service.release.set()
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)

    asyncio.run(scenario())


def test_worker_resets_request_context(monkeypatch):
    async def scenario():
        original_to_thread = asyncio.to_thread
        observed = []
        outer_event = RequestCancellation()
        outer_event.set()

        async def inspect_to_thread(function):
            def inspect():
                deadline_token = set_request_deadline(123.0)
                cancelled_token = set_request_cancelled(outer_event)
                try:
                    try:
                        return function()
                    finally:
                        observed.append((get_request_deadline(), request_cancelled()))
                finally:
                    reset_request_cancelled(cancelled_token)
                    reset_request_deadline(deadline_token)

            return await original_to_thread(inspect)

        monkeypatch.setattr(api.asyncio, "to_thread", inspect_to_thread)
        service = _BlockingService(asyncio.get_running_loop())
        service.release.set()
        app = _blocking_app(service)
        endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/run")
        with pytest.raises(RuntimeError, match="late worker failure"):
            await endpoint(None)
        assert service.cancelled is False
        assert observed == [(123.0, True)]
        assert get_request_deadline() is None
        assert not request_cancelled()

    asyncio.run(scenario())
