"""runtime/clauses.py — Clause chunking: the runtime's core latency trick.

Re-chunks any LLM adapter's delta stream into speakable clauses, flushing
the first clause early so TTS starts long before the reply finishes.
This is runtime logic, not provider logic — it applies identically to
every LLM adapter.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from runtime.markers import MarkerGuard
from runtime.turn_events import Token, ToolCallDetected
from runtime.types import LLMDelta

log = logging.getLogger("runtime.clauses")

# Flush a chunk to TTS at these boundaries (Devanagari danda included).
HARD_BREAKS = "।?!.\n"
# S6: 120 → 60; S7: 60 → 10. Live calls showed models opening with a short
# acknowledgement ("ठीक है Clon जी।", ~15 chars) that a 60-char floor glued
# to the long sentence after it — one fat clause, one 3-4.5 s TTS round
# trip before any sound. A 10-char floor ships that opener to TTS
# immediately (a sub-second synthesis); synth-ahead hides the rest under
# its playback. MIN_CHUNK keeps later clauses long for prosody.
MIN_FIRST_CHUNK = 10     # start speaking fast: flush the first clause early
# S7.2: 180 → 100. The one-go pitch produced ~230-char middle clauses that
# blew the 8 s TTS attempt budget (observed: timeout + full retry = 14 s of
# dead air). ~100 chars synthesizes in 2-4 s — under the budget — while its
# predecessor's playback (6-9 s) still covers it, so the chain stays gapless.
MIN_CHUNK = 100          # later chunks: long enough for prosody, small enough for TTS


async def stream_clauses(deltas: AsyncIterator[LLMDelta]) -> AsyncIterator[str]:
    """Yield speakable chunks as they form.

    If the delta stream errors mid-reply, whatever is buffered is still
    yielded — the agent speaks what it has instead of going silent.
    """
    buf = ""
    first = True
    try:
        async for delta in deltas:
            if not delta.text:
                continue
            buf += delta.text
            threshold = MIN_FIRST_CHUNK if first else MIN_CHUNK
            # Flush only on a hard sentence break, never mid-sentence.
            while True:
                idx = _breakpoint(buf, threshold)
                if idx is None:
                    break
                chunk, buf = buf[: idx + 1], buf[idx + 1 :]
                chunk = chunk.strip()
                if chunk:
                    first = False
                    yield chunk
        if buf.strip():
            yield buf.strip()
    except Exception as e:  # noqa: BLE001
        log.error("LLM stream error: %s", e)
        if buf.strip():
            yield buf.strip()


async def stream_turn_text(
    deltas: AsyncIterator[LLMDelta], guard: MarkerGuard,
) -> AsyncIterator[Token | ToolCallDetected | str]:
    """Two views of one delta stream, interleaved in a single pass (S2).

    - Token: delta-granular text, filtered through `guard` so no fragment
      of a potential tool marker ever reaches a token consumer.
    - str: a raw speakable chunk — same buffer, thresholds, breakpoints,
      strip, and error fallback as stream_clauses, so clause-consuming
      behavior stays byte-identical. Marker text is still IN these chunks;
      extraction/dispatch remains the strategy's per-clause job.
    - ToolCallDetected: a native tool-call delta, passed through in stream
      order. Dispatch policy belongs to the strategy, not here.

    If the delta stream errors mid-reply, held tokens and the buffered
    chunk are still yielded — the agent speaks what it has.
    """
    buf = ""
    first = True

    def _flush_buf() -> str:
        return buf.strip()

    try:
        async for delta in deltas:
            if delta.tool_call is not None:
                yield ToolCallDetected(name=delta.tool_call.name,
                                       args=delta.tool_call.args,
                                       call_id=delta.tool_call.id)
            if not delta.text:
                continue
            safe = guard.feed(delta.text)
            if safe:
                yield Token(text=safe)
            buf += delta.text
            threshold = MIN_FIRST_CHUNK if first else MIN_CHUNK
            while True:
                idx = _breakpoint(buf, threshold)
                if idx is None:
                    break
                chunk, buf = buf[: idx + 1], buf[idx + 1 :]
                chunk = chunk.strip()
                if chunk:
                    first = False
                    yield chunk
        tail = guard.flush()
        if tail:
            yield Token(text=tail)
        if _flush_buf():
            yield _flush_buf()
    except Exception as e:  # noqa: BLE001
        log.error("LLM stream error: %s", e)
        tail = guard.flush()
        if tail:
            yield Token(text=tail)
        if _flush_buf():
            yield _flush_buf()


def _breakpoint(text: str, min_len: int) -> int | None:
    """Index of a good place to cut `text`, or None.

    Known limitation (pinned in tests): '.' inside "4.30pm" counts as a
    sentence break. Smarter breaking is a Turn Engine (M4) concern.
    """
    for i, ch in enumerate(text):
        if ch in HARD_BREAKS and i >= min_len:
            return i
    return None
