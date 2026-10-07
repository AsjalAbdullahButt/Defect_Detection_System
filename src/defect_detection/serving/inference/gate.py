"""Bounded, time-limited execution of blocking inference work.

* A dedicated thread pool sized to ``max_concurrent`` keeps decode + inference off the event
  loop, so health checks stay responsive under load.
* Admission control: if every slot is busy the request fails fast with 503 instead of
  queueing without bound (queues hide overload until latency explodes).
* Timeout: the caller gets 504 after ``timeout_s``. A Python thread cannot be killed, so the
  slot is only released when the work really finishes; otherwise a stuck model could make
  the server accept more work than it has CPU for.
"""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from defect_detection.serving.errors import InferenceTimeoutError, OverloadedError

T = TypeVar("T")


class InferenceGate:
    """Run callables in a bounded pool with fail-fast admission and a timeout."""

    def __init__(self, max_concurrent: int, timeout_s: float) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="infer")
        self._max = max_concurrent
        self._timeout = timeout_s
        self._active = 0  # only touched from the event-loop thread

    @property
    def active(self) -> int:
        """Slots currently in use."""
        return self._active

    async def run(self, fn: Callable[[], T]) -> T:
        """Execute ``fn`` in the pool; raise OverloadedError or InferenceTimeoutError."""
        if self._active >= self._max:
            raise OverloadedError("The server is at capacity; retry shortly.")
        loop = asyncio.get_running_loop()
        self._active += 1
        future = loop.run_in_executor(self._executor, fn)
        future.add_done_callback(self._release)
        try:
            return await asyncio.wait_for(asyncio.shield(future), self._timeout)
        except TimeoutError as exc:
            raise InferenceTimeoutError("Inference took too long.") from exc

    def _release(self, _: "asyncio.Future[T]") -> None:
        self._active -= 1

    def shutdown(self) -> None:
        """Finish running work and stop the pool (called on graceful shutdown)."""
        self._executor.shutdown(wait=True, cancel_futures=True)
