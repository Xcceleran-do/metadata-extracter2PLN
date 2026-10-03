from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, Lock


class RequestCancellation(Event):
    """Signal cancellation to the request's active async provider call."""

    def __init__(self):
        super().__init__()
        self._lock = Lock()
        self._task: asyncio.Task | None = None

    def set(self) -> None:
        with self._lock:
            if self.is_set():
                return
            super().set()
            if self._task is not None:
                self._task.get_loop().call_soon_threadsafe(self._task.cancel)

    @contextmanager
    def watch(self, task: asyncio.Task):
        with self._lock:
            self._task = task
            if self.is_set():
                task.cancel()
        try:
            yield
        finally:
            with self._lock:
                self._task = None


_request_deadline: ContextVar[float | None] = ContextVar(
    "request_deadline",
    default=None,
)
_request_cancelled: ContextVar[RequestCancellation | None] = ContextVar(
    "request_cancelled",
    default=None,
)


def set_request_deadline(deadline: float | None):
    return _request_deadline.set(deadline)


def reset_request_deadline(token) -> None:
    _request_deadline.reset(token)


def get_request_deadline() -> float | None:
    return _request_deadline.get()


def set_request_cancelled(event: RequestCancellation | None):
    return _request_cancelled.set(event)


def reset_request_cancelled(token) -> None:
    _request_cancelled.reset(token)


def request_cancelled() -> bool:
    event = _request_cancelled.get()
    return event is not None and event.is_set()


@contextmanager
def watch_request_cancellation(task: asyncio.Task):
    event = _request_cancelled.get()
    if event is None:
        yield
    else:
        with event.watch(task):
            yield
