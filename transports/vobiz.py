"""transports/vobiz.py — Vobiz bidirectional media-stream adapter.

Owns the whole Vobiz WS dialect: inbound event JSON (start / media /
playedStream / clearedAudio / stop) is normalized into TransportEvents, and
play/clear/checkpoint become playAudio/clearAudio/checkpoint messages.

Two long-standing defects are closed here rather than in the session:

D2 (concurrent writers): the speak task, barge-in clear, and checkpoint all
write to the same WebSocket. A single asyncio.Lock around every send
guarantees frames are never interleaved mid-message. A queue + drain task
would give the same guarantee with more moving parts (queue lifetime,
drain-task shutdown); the lock is the smaller mechanism and lives behind the
Transport protocol, so it can become a queue later without touching callers.

D7 (pacing drift): the old `sleep(0.02)` per frame accumulated lag — each
iteration paid 20 ms *plus* send/encode time, so long replies slipped audibly
behind real time. play() now paces against a monotonic deadline: each frame's
send time is scheduled exactly FRAME_SECONDS after the previous one. If the
producer falls behind (slow TTS), the deadline re-anchors to "now" instead of
bursting a backlog at the carrier — late audio plays late, it never plays
fast. clear() drops the anchor so the next utterance starts fresh.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncIterator

from fastapi import WebSocket, WebSocketDisconnect

from runtime.types import (
    MULAW_8K,
    AudioFormat,
    AudioFrame,
    CallEnded,
    CallStarted,
    MediaReceived,
    OutputCleared,
    PlaybackFinished,
    TransportEvent,
)

log = logging.getLogger("transport.vobiz")

FRAME_SECONDS = 0.02  # Vobiz speaks 20 ms mu-law frames


class VobizTransport:
    """Transport implementation over a Vobiz bidirectional stream WebSocket."""

    def __init__(self, websocket: WebSocket, caller: str | None = None):
        self._ws = websocket
        self._send_lock = asyncio.Lock()
        self._stream_id: str | None = None
        self._next_send: float | None = None
        # Vobiz's WS "start" frame carries no caller number at all — it must
        # come from the /answer webhook's From param, threaded through the
        # /ws query string by the caller of this constructor.
        self._caller = caller

    @property
    def audio_format(self) -> AudioFormat:
        return MULAW_8K

    # ------------------------------------------------------------- inbound
    async def events(self) -> AsyncIterator[TransportEvent]:
        while True:
            try:
                raw = await self._ws.receive_text()
            except WebSocketDisconnect:
                log.info("Vobiz WebSocket disconnected")
                yield CallEnded()
                return
            except Exception as e:  # noqa: BLE001
                log.error("Vobiz WS receive error: %s", e)
                yield CallEnded()
                return

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = data.get("event")
            if event == "start":
                log.info("Vobiz start frame: %s", data)
                # Vobiz nests the call/stream identifiers under a "start"
                # object (confirmed against production captures), not at
                # the top level — e.g. {"event": "start", "start":
                # {"callId": ..., "streamId": ..., ...}}. Top-level keys are
                # kept as a fallback for any carrier variant that flattens
                # them (and for our own test fixtures).
                start = data.get("start") or {}
                self._stream_id = (start.get("streamId") or data.get("streamId")
                                   or data.get("StreamID")
                                   or data.get("stream_id") or "default")
                call_id = (start.get("callId") or data.get("callId")
                           or data.get("CallID") or data.get("call_id")
                           or "unknown")
                # The start frame itself never carries a caller number —
                # that has to be injected at construction time from the
                # /answer webhook's From param (see server.py's /ws route).
                caller = (self._caller or data.get("from")
                          or data.get("From") or "unknown")
                yield CallStarted(stream_id=self._stream_id, call_id=call_id,
                                  caller=caller)
            elif event == "media":
                payload = (data.get("media") or {}).get("payload")
                if not payload:
                    continue
                yield MediaReceived(
                    frame=AudioFrame(payload=base64.b64decode(payload),
                                     format=MULAW_8K))
            elif event == "playedStream":
                yield PlaybackFinished()
            elif event == "clearedAudio":
                yield OutputCleared()
            elif event == "stop":
                yield CallEnded()
                return

    # ------------------------------------------------------------ outbound
    async def play(self, frame: AudioFrame) -> None:
        now = asyncio.get_running_loop().time()
        if self._next_send is None or now >= self._next_send:
            # First frame, or the producer fell behind: re-anchor rather than
            # burst-send the backlog.
            self._next_send = now
        else:
            # Sleep outside the lock so a barge-in clear() is never stuck
            # behind a pacing wait.
            await asyncio.sleep(self._next_send - now)
        await self._send({
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
                "payload": base64.b64encode(frame.payload).decode("ascii"),
            },
        })
        self._next_send += FRAME_SECONDS

    async def clear(self) -> None:
        if self._stream_id is None:
            return
        self._next_send = None
        await self._send({"event": "clearAudio", "streamId": self._stream_id})

    async def checkpoint(self, name: str) -> None:
        if self._stream_id is None:
            return
        await self._send({
            "event": "checkpoint",
            "streamId": self._stream_id,
            "name": name,
        })

    async def _send(self, obj: dict) -> None:
        async with self._send_lock:
            await self._ws.send_text(json.dumps(obj))
