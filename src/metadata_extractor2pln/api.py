from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings, get_settings
from .request_context import (
    RequestCancellation,
    reset_request_cancelled,
    reset_request_deadline,
    set_request_cancelled,
    set_request_deadline,
)
from .jev_backend import JEVBackend
from .backends import BackendUnavailable
from .bedrock import BedrockBackend
from .models import (
    ExtractRequest,
    ExtractResponse,
    PlanRequest,
    PlanResponse,
    RunRequest,
    RunResponse,
    ValidatePlanRequest,
    ValidatePlanResponse,
)
from .service import MetadataService


PUBLIC_PATHS = {"/health", "/ready"}
logger = logging.getLogger(__name__)


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        received = 0
        too_large = False
        replacement_sent = False

        async def limited_receive() -> Message:
            nonlocal received, too_large
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    too_large = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def limited_send(message: Message) -> None:
            nonlocal replacement_sent
            if not too_large:
                await send(message)
                return
            if replacement_sent:
                return
            replacement_sent = True
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "request_too_large",
                        "message": "request body exceeds configured limit",
                    }
                },
            )
            await response(scope, receive, send)

        await self.app(scope, limited_receive, limited_send)


def create_app(
    settings: Settings | None = None,
    *,
    service: MetadataService | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    settings.validate()
    if settings.model_provider == "jev":
        backend = JEVBackend(
            model=settings.jev_model,
            api_key=settings.jev_api_key,
            timeout_seconds=settings.model_timeout_seconds,
            transport=settings.jev_transport,
            openrouter_api_key=settings.openrouter_api_key,
            openrouter_model=settings.openrouter_model,
        )
    else:
        backend = BedrockBackend(
            model=settings.bedrock_model,
            region=settings.bedrock_region,
            access_key=settings.bedrock_access_key,
            secret_key=settings.bedrock_secret_key,
            timeout_seconds=settings.model_timeout_seconds,
            max_tokens=settings.bedrock_max_tokens,
        )
    service = service or MetadataService(backend)
    app = FastAPI(
        title="metadata-extractor2PLN",
        version="0.1.0",
        docs_url="/docs" if settings.expose_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.expose_docs else None,
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts)
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
    limiter: dict[str, deque[float]] = defaultdict(deque)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))[:128]
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.max_request_bytes:
                    return _error(
                        413,
                        "request_too_large",
                        "request body exceeds configured limit",
                        request_id,
                    )
            except ValueError:
                return _error(
                    400,
                    "invalid_content_length",
                    "invalid Content-Length header",
                    request_id,
                )

        if request.url.path not in PUBLIC_PATHS:
            token = _bearer_token(request.headers.get("authorization", ""))
            owner = _authenticate(token, settings.api_keys)
            if owner is None:
                return _error(401, "unauthorized", "invalid bearer token", request_id)
            now = time.monotonic()
            entries = limiter[owner]
            while entries and entries[0] <= now - 60:
                entries.popleft()
            if len(entries) >= settings.rate_limit_per_minute:
                return _error(
                    429, "rate_limited", "request rate limit exceeded", request_id
                )
            entries.append(now)
            request.state.owner = owner
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        details = [
            {"location": list(error["loc"]), "message": error["msg"]}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "request validation failed",
                    "details": details,
                }
            },
        )

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_contract", "message": str(exc)}},
        )

    @app.exception_handler(BackendUnavailable)
    async def backend_unavailable(request: Request, exc: BackendUnavailable):
        logger.exception("Model backend request failed", exc_info=exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "code": "model_backend_unavailable",
                    "message": "the configured model backend could not complete the request",
                }
            },
        )

    async def execute(function, request):
        deadline = time.monotonic() + settings.request_timeout_seconds
        cancelled = RequestCancellation()

        def run():
            deadline_token = set_request_deadline(deadline)
            cancelled_token = set_request_cancelled(cancelled)
            try:
                return function(request)
            finally:
                reset_request_cancelled(cancelled_token)
                reset_request_deadline(deadline_token)

        def worker_done(task):
            semaphore.release()
            # Observe exceptions even if the request stopped awaiting this worker.
            if not task.cancelled():
                task.exception()

        try:
            async with asyncio.timeout(settings.request_timeout_seconds):
                await semaphore.acquire()
                try:
                    task = asyncio.create_task(asyncio.to_thread(run))
                except BaseException:
                    semaphore.release()
                    raise
                task.add_done_callback(worker_done)
                return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        except TimeoutError:
            cancelled.set()
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "code": "request_timeout",
                        "message": "processing exceeded its time limit",
                    }
                },
            )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        model_ready = bool(service.backend and service.backend.ready)
        return {
            "status": "ready",
            "model_ready": model_ready,
            "model": service.backend.name if model_ready else None,
        }

    @app.post("/v1/plans/discover", response_model=PlanResponse)
    async def plan(request: PlanRequest):
        return await execute(service.plan, request)

    @app.post("/v1/plans/validate", response_model=ValidatePlanResponse)
    async def validate_plan(request: ValidatePlanRequest):
        return await execute(service.validate_plan, request)

    @app.post("/v1/extract", response_model=ExtractResponse)
    async def extract(request: ExtractRequest):
        return await execute(service.extract, request)

    @app.post("/v1/run", response_model=RunResponse)
    async def run(request: RunRequest):
        return await execute(service.run, request)

    return app


def _bearer_token(header: str) -> str | None:
    scheme, separator, token = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token:
        return None
    return token


def _authenticate(token: str | None, configured: tuple[str, ...]) -> str | None:
    if token is None:
        return None
    for entry in configured:
        owner, _separator, secret = entry.partition(":")
        if hmac.compare_digest(token, secret):
            return owner
    return None


def _error(status: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={"X-Request-ID": request_id},
    )


app = create_app()
