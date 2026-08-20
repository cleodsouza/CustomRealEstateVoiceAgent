"""runtime/tts_cache.py — replay cache for synthesized utterances (S6).

Purpose
    The runtime speaks the same static lines on every call: the agent's
    greeting, the thinking filler, the fallback line, hold lines. Paying a
    full TTS round trip (observed 1.8–5.5 s on a REST provider) for text
    we synthesized seconds ago is pure latency. This wrapper implements
    the TTS Protocol around any inner TTS and replays remembered frames.

Design
    - LRU keyed by (text, format). Static lines stay hot forever; dynamic
      replies that never repeat age out. A cached clause is ~100 KB of
      mu-law frames, so the default budget is a few MB.
    - Only COMPLETE, non-empty syntheses are stored: a stream the consumer
      abandoned (barge-in) or that failed/yielded nothing (breaker open)
      must not poison the cache — the resilience wrapper's contract that
      "nothing yielded" means failure is preserved.
    - Composed OUTSIDE the resilience wrapper at the composition root:
      a cache hit never touches the breaker; misses inherit the full
      retry/timeout discipline.

This is runtime logic (latency engineering), not provider logic: it works
identically over any TTS adapter.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import AsyncIterator

from runtime.interfaces import TTS
from runtime.types import AudioFormat, AudioFrame

log = logging.getLogger("runtime.tts_cache")

DEFAULT_MAX_ENTRIES = 64


class CachedTTS:
    """Implements runtime.interfaces.TTS around any inner TTS."""

    def __init__(self, inner: TTS, *, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._inner = inner
        self._max = max_entries
        self._cache: OrderedDict[tuple[str, AudioFormat], list[AudioFrame]] = (
            OrderedDict())
        self.supports_streaming_input = inner.supports_streaming_input

    async def healthy(self) -> bool:
        probe = getattr(self._inner, "healthy", None)
        return await probe() if probe is not None else True

    def stream_text(self, text_stream, fmt):
        """Live replies bypass the replay cache and stream through directly."""
        stream = getattr(self._inner, "stream_text", None)
        if stream is None:
            raise AttributeError("inner TTS does not support streaming input")
        return stream(text_stream, fmt)

    async def prewarm(self, texts: list[str], fmt: AudioFormat) -> None:
        """Synthesize static lines into the cache ahead of need (e.g. the
        greeting at provider build, so even call #1 starts instantly).
        Failures are logged, never raised — prewarming is opportunistic."""
        for text in texts:
            if not text.strip() or (text, fmt) in self._cache:
                continue
            try:
                async for _ in self.synthesize(text, fmt):
                    pass
            except Exception as e:  # noqa: BLE001
                log.warning("TTS prewarm failed for %r: %s", text[:40], e)

    def synthesize(self, text: str, fmt: AudioFormat) -> AsyncIterator[AudioFrame]:
        return self._synthesize(text, fmt)

    async def _synthesize(self, text: str,
                          fmt: AudioFormat) -> AsyncIterator[AudioFrame]:
        key = (text, fmt)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            for frame in hit:
                yield frame
            return
        frames: list[AudioFrame] = []
        async for frame in self._inner.synthesize(text, fmt):
            frames.append(frame)
            yield frame
        # Reached only when the inner stream finished normally AND the
        # consumer didn't abandon us (a thrown GeneratorExit / cancel never
        # gets here). Empty output = failure upstream — never cached.
        if frames:
            self._cache[key] = frames
            self._cache.move_to_end(key)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)