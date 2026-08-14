"""CachedTTS (S6): replay cache for synthesized utterances."""
import asyncio

from runtime.tts_cache import CachedTTS
from runtime.types import MULAW_8K, AudioFrame


def frame(b=b"\x00"):
    return AudioFrame(payload=b * 160, format=MULAW_8K)


class CountingTTS:
    supports_streaming_input = False

    def __init__(self, frames_per_call=3, fail=False):
        self.calls = []
        self._n = frames_per_call
        self._fail = fail

    async def synthesize(self, text, fmt):
        self.calls.append(text)
        if self._fail:
            return  # resilience contract: failure = nothing yielded
        for _ in range(self._n):
            yield frame()


async def test_repeat_synthesis_hits_cache():
    inner = CountingTTS()
    tts = CachedTTS(inner)
    first = [f async for f in tts.synthesize("नमस्ते", MULAW_8K)]
    second = [f async for f in tts.synthesize("नमस्ते", MULAW_8K)]
    assert first == second and len(first) == 3
    assert inner.calls == ["नमस्ते"]          # inner hit exactly once


async def test_empty_synthesis_is_never_cached():
    inner = CountingTTS(fail=True)
    tts = CachedTTS(inner)
    assert [f async for f in tts.synthesize("x", MULAW_8K)] == []
    assert [f async for f in tts.synthesize("x", MULAW_8K)] == []
    assert inner.calls == ["x", "x"]          # retried against the inner


async def test_abandoned_stream_is_not_cached():
    """Barge-in abandons the generator mid-flight; a partial synthesis
    must not poison the cache."""
    inner = CountingTTS(frames_per_call=5)
    tts = CachedTTS(inner)
    agen = tts.synthesize("लंबा वाक्य", MULAW_8K)
    await anext(agen)                          # one frame, then abandon
    await agen.aclose()
    full = [f async for f in tts.synthesize("लंबा वाक्य", MULAW_8K)]
    assert len(full) == 5                      # re-synthesized, complete
    assert inner.calls == ["लंबा वाक्य", "लंबा वाक्य"]


async def test_lru_evicts_oldest():
    inner = CountingTTS()
    tts = CachedTTS(inner, max_entries=2)
    for text in ("a", "b", "c"):               # "a" falls out
        [f async for f in tts.synthesize(text, MULAW_8K)]
    [f async for f in tts.synthesize("a", MULAW_8K)]
    assert inner.calls == ["a", "b", "c", "a"]


async def test_prewarm_fills_cache_and_skips_blanks():
    inner = CountingTTS()
    tts = CachedTTS(inner)
    await tts.prewarm(["नमस्ते!", "", "हम्म"], MULAW_8K)
    assert inner.calls == ["नमस्ते!", "हम्म"]
    [f async for f in tts.synthesize("नमस्ते!", MULAW_8K)]
    assert inner.calls == ["नमस्ते!", "हम्म"]  # served from cache
