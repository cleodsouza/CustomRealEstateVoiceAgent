"""VobizTransport unit tests: wire-protocol normalization, single-writer
sends, and deadline-based frame pacing (D2/D7 regression coverage)."""
import asyncio
import base64
import json
from collections import deque

from fastapi import WebSocketDisconnect

from runtime.types import (
    MULAW_8K,
    AudioFrame,
    CallEnded,
    CallStarted,
    MediaReceived,
    OutputCleared,
    PlaybackFinished,
)
from transports.vobiz import VobizTransport


class FakeWS:
    """Minimal stand-in for a FastAPI WebSocket."""

    def __init__(self, incoming: list[str] | None = None):
        self.incoming = deque(incoming or [])
        self.sent_texts: list[str] = []

    async def receive_text(self) -> str:
        if not self.incoming:
            raise WebSocketDisconnect(1000)
        return self.incoming.popleft()

    async def send_text(self, text: str) -> None:
        self.sent_texts.append(text)


async def collect_events(transport):
    return [ev async for ev in transport.events()]


# ------------------------------------------------------------- normalization
async def test_events_normalizes_full_call():
    ulaw = b"\x7f" * 160
    ws = FakeWS([
        json.dumps({"event": "start", "streamId": "s1", "callId": "c1",
                    "from": "+911234567890"}),
        json.dumps({"event": "media",
                    "media": {"payload": base64.b64encode(ulaw).decode("ascii")}}),
        json.dumps({"event": "playedStream"}),
        json.dumps({"event": "clearedAudio"}),
        json.dumps({"event": "stop"}),
    ])
    events = await collect_events(VobizTransport(ws))

    assert events[0] == CallStarted(stream_id="s1", call_id="c1",
                                    caller="+911234567890")
    assert events[1] == MediaReceived(frame=AudioFrame(payload=ulaw, format=MULAW_8K))
    assert events[2] == PlaybackFinished()
    assert events[3] == OutputCleared()
    assert events[4] == CallEnded()
    assert len(events) == 5


async def test_events_skips_garbage_and_empty_media():
    ws = FakeWS([
        "this is not json",
        json.dumps({"event": "media", "media": {}}),
        json.dumps({"event": "unknown-future-event"}),
    ])
    # Disconnect after the queue drains → a single CallEnded.
    assert await collect_events(VobizTransport(ws)) == [CallEnded()]


async def test_start_key_spelling_fallbacks():
    ws = FakeWS([json.dumps({"event": "start", "StreamID": "S9",
                             "CallID": "C9", "From": "+9199"})])
    events = await collect_events(VobizTransport(ws))
    assert events[0] == CallStarted(stream_id="S9", call_id="C9", caller="+9199")


async def test_start_nests_call_and_stream_id_in_production():
    # Real Vobiz "start" frames nest callId/streamId under a "start" object
    # rather than at the top level, and carry no caller number at all — the
    # top-level fallback keys above only cover carrier variants/fixtures
    # that flatten the shape.
    ws = FakeWS([json.dumps({
        "sequenceNumber": 0,
        "event": "start",
        "start": {
            "callId": "5401fd2e-6344-40df-a22c-c8ffea7a92e7",
            "streamId": "c4dfd815-a92a-4140-ab85-5ff28c004116",
            "accountId": "500025",
            "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-l16", "sampleRate": 16000},
        },
        "extra_headers": "{}",
    })])
    events = await collect_events(VobizTransport(ws))
    assert events[0] == CallStarted(
        stream_id="c4dfd815-a92a-4140-ab85-5ff28c004116",
        call_id="5401fd2e-6344-40df-a22c-c8ffea7a92e7",
        caller="unknown")


async def test_caller_injected_at_construction_wins_over_start_frame():
    # The caller's number never appears in the WS start frame; server.py
    # threads it in from the /answer webhook's From param via the
    # constructor instead.
    ws = FakeWS([json.dumps({"event": "start", "start": {
        "callId": "c1", "streamId": "s1"}})])
    events = await collect_events(VobizTransport(ws, caller="+911234567890"))
    assert events[0] == CallStarted(stream_id="s1", call_id="c1",
                                    caller="+911234567890")


# ------------------------------------------------------------------ outbound
async def test_play_sends_exact_vobiz_shape():
    ws = FakeWS()
    t = VobizTransport(ws)
    payload = b"\xff" * 160
    await t.play(AudioFrame(payload=payload, format=MULAW_8K))

    assert len(ws.sent_texts) == 1
    msg = json.loads(ws.sent_texts[0])
    assert msg == {
        "event": "playAudio",
        "media": {
            "contentType": "audio/x-mulaw",
            "sampleRate": 8000,
            "payload": base64.b64encode(payload).decode("ascii"),
        },
    }
    assert base64.b64decode(msg["media"]["payload"]) == payload


async def test_clear_and_checkpoint_noop_before_start():
    ws = FakeWS()
    t = VobizTransport(ws)
    await t.clear()
    await t.checkpoint("turn-0")
    assert ws.sent_texts == []


async def test_clear_and_checkpoint_carry_stream_id():
    ws = FakeWS([json.dumps({"event": "start", "streamId": "s1"}),
                 json.dumps({"event": "stop"})])
    t = VobizTransport(ws)
    await collect_events(t)

    await t.clear()
    await t.checkpoint("turn-3")
    assert json.loads(ws.sent_texts[0]) == {"event": "clearAudio", "streamId": "s1"}
    assert json.loads(ws.sent_texts[1]) == {
        "event": "checkpoint", "streamId": "s1", "name": "turn-3"}


# -------------------------------------------------------------------- pacing
async def test_play_paces_frames_to_realtime():
    ws = FakeWS()
    t = VobizTransport(ws)
    frame = AudioFrame(payload=b"\x00" * 160, format=MULAW_8K)

    start = asyncio.get_running_loop().time()
    for _ in range(3):
        await t.play(frame)
    elapsed = asyncio.get_running_loop().time() - start

    # Frame 1 is immediate; frames 2–3 wait for their 20 ms deadlines.
    # Bounds are loose for Windows timer resolution.
    assert elapsed >= 0.03
    assert elapsed < 0.5
    assert len(ws.sent_texts) == 3


async def test_clear_resets_pacing_anchor():
    ws = FakeWS([json.dumps({"event": "start", "streamId": "s1"}),
                 json.dumps({"event": "stop"})])
    t = VobizTransport(ws)
    await collect_events(t)
    frame = AudioFrame(payload=b"\x00" * 160, format=MULAW_8K)

    await t.play(frame)
    await t.clear()
    # After a clear, the next frame re-anchors: it must not sleep out the
    # deadline of the aborted utterance.
    start = asyncio.get_running_loop().time()
    await t.play(frame)
    assert asyncio.get_running_loop().time() - start < 0.015
