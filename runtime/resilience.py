"""runtime/resilience.py — provider-edge failure handling (M8).

Purpose
    One vendor blip degrades the call instead of silencing or killing it
    (redesign §7). Resilience lives in *wrappers* implementing the same
    provider Protocols, composed at the composition root — conversation
    logic never learns that failures exist.

Responsibilities
    - CircuitBreaker: after N consecutive failures, skip the provider for
      a cooldown instead of stacking doomed retries onto the hot path.
      Pure and clock-injected — unit-testable without sleeping.
    - ResilientTTS: per-attempt timeout + ONE retry, and only if no audio
      was yielded yet — a mid-clause failure never replays audio. Budget
      is strict: a slow retry is worse than the fallback line.
    - ResilientLLM: first-token timeout (a dead socket hangs exactly
      there) + breaker. Mid-stream errors still propagate so
      runtime/clauses.py keeps its speak-what-we-have contract.

What happens when everything fails: the wrappers yield nothing, and the
session speaks the agent's scripted fallback line instead of dead air.
STT resilience is different in kind (a stateful connection, not a
request) and lives in the adapter itself: providers/stt/deepgram.py
reconnects with backoff and reports "dead" when the budget is exhausted.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable

from runtime.interfaces import LLM, TTS
from runtime.types import AudioFormat, AudioFrame, LLMDelta

log = logging.getLogger("runtime.resilience")


class CircuitBreaker:
    """Circuit-breaker-lite: consecutive failures open it; after the
    cooldown one probe call is allowed through (half-open); success
    closes it again."""

    def __init__(self, *, failure_threshold: int = 3, cooldown_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._threshold = failure_threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._failures = 0
        self._open_until: float | None = None

    def allow(self) -> bool:
        if self._open_until is None:
            return True
        if self._clock() >= self._open_until:
            # Half-open: let one probe through; failure re-opens below.
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._open_until = self._clock() + self._cooldown_s

    @property
    def is_open(self) -> bool:
        return self._open_until is not None and self._clock() < self._open_until


class ResilientTTS:
    """Implements runtime.interfaces.TTS around any inner TTS."""

    def __init__(self, inner: TTS, *, breaker: CircuitBreaker | None = None,
                 attempt_timeout_s: float = 3.0) -> None:
        self._inner = inner
        self._breaker = breaker if breaker is not None else CircuitBreaker()
        self._attempt_timeout_s = attempt_timeout_s
        self.supports_streaming_input = inner.supports_streaming_input

    async def healthy(self) -> bool:
        probe = getattr(self._inner, "healthy", None)
        return await probe() if probe is not None else True

    def stream_text(self, text_stream, fmt):
        """Stream a live reply without buffering or replaying partial audio."""
        stream = getattr(self._inner, "stream_text", None)
        if stream is None:
            raise AttributeError("inner TTS does not support streaming input")

        async def _wrapped():
            if not self._breaker.allow():
                log.warning("TTS breaker open; skipping live synthesis")
                return
            yielded = False
            try:
                async for frame in stream(text_stream, fmt):
                    yielded = True
                    yield frame
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self._breaker.record_failure()
                log.warning("Live TTS failed: %s", str(e) or type(e).__name__)
                raise
            else:
                if yielded:
                    self._breaker.record_success()

        return _wrapped()

    async def synthesize(self, text: str,
                         fmt: AudioFormat) -> AsyncIterator[AudioFrame]:
        if not self._breaker.allow():
            log.warning("TTS breaker open; skipping synthesis")
            return
        sentinel = object()
        for attempt in (1, 2):
            yielded = False
            stream = aiter(self._inner.synthesize(text, fmt))
            try:
                while True:
                    # The timeout bounds the *producer's* time to the next
                    # frame only. Wrapping the whole iteration would also
                    # count time spent suspended at `yield` while the
                    # transport paces audio out — long clauses would "time
                    # out" mid-playback.
                    frame = await asyncio.wait_for(
                        anext(stream, sentinel), self._attempt_timeout_s)
                    if not isinstance(frame, AudioFrame):
                        break  # sentinel: the inner stream is exhausted
                    yielded = True
                    yield frame
            except Exception as e:  # noqa: BLE001
                self._breaker.record_failure()
                log.warning("TTS attempt %d failed: %s",
                            attempt, str(e) or type(e).__name__)
                if yielded:
                    return  # partial audio went out; a retry would replay it
            else:
                self._breaker.record_success()
                return
        return


class ResilientLLM:
    """Implements runtime.interfaces.LLM around any inner LLM."""

    def __init__(self, inner: LLM, *, breaker: CircuitBreaker | None = None,
                 first_token_timeout_s: float = 10.0) -> None:
        self._inner = inner
        self._breaker = breaker if breaker is not None else CircuitBreaker()
        self._first_token_timeout_s = first_token_timeout_s

    async def healthy(self) -> bool:
        probe = getattr(self._inner, "healthy", None)
        return await probe() if probe is not None else True

    async def stream(self, messages: list[Any],
                     tools: list[dict] | None = None) -> AsyncIterator[LLMDelta]:
        if not self._breaker.allow():
            log.warning("LLM breaker open; skipping generation")
            return
        stream = aiter(self._inner.stream(messages, tools))
        try:
            # A dead endpoint hangs exactly at the first token; once tokens
            # flow, an overall deadline would only truncate long replies.
            async with asyncio.timeout(self._first_token_timeout_s):
                first = await anext(stream, None)
        except Exception as e:  # noqa: BLE001
            self._breaker.record_failure()
            log.error("LLM first token failed: %s", str(e) or type(e).__name__)
            raise
        if first is None:
            self._breaker.record_success()
            return
        yield first
        try:
            async for delta in stream:
                yield delta
        except Exception:
            self._breaker.record_failure()
            raise  # clauses.py speaks what it already has
        self._breaker.record_success()