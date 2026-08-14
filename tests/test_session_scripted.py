"""Scripted-call characterization tests for CallSession.

Drives the orchestrator end-to-end through a LocalTransport and asserts the
outbound play/clear/checkpoint operation sequence. The turn-taking *rules*
are tested in test_turn_engine.py; here we test that the session wires
engine intents to real provider calls correctly — including the M4
behavior fixes that live in the wiring (D4 history truncation, filler).
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import audio
import config
from runtime import agent_registry
from runtime.events import NULL_BUS
from runtime.tools import MarkerToolStrategy, ToolExecutor, ToolRegistry, ToolSpec
from runtime.types import (
    MULAW_8K,
    AudioFrame,
    CallEnded,
    CallStarted,
    LLMDelta,
    MediaReceived,
    PlaybackFinished,
    STTEvent,
)
from session import CallSession
from transports.local import LocalTransport


# ---------------------------------------------------------------------- fakes
class FakeSTT:
    emits_endpoint = True

    def __init__(self, on_event):
        self.on_event = on_event
        self.frames = []
        self.started = False
        self.closed = False

    async def start(self):
        self.started = True

    async def send_audio(self, frame):
        self.frames.append(frame.payload)

    async def close(self):
        self.closed = True


class FakeTTS:
    supports_streaming_input = False
    N_FRAMES = 3

    def __init__(self):
        self.texts = []

    async def synthesize(self, text, fmt):
        assert fmt == MULAW_8K
        self.texts.append(text)
        for _ in range(self.N_FRAMES):
            yield AudioFrame(payload=b"\xff" * audio.FRAME_BYTES, format=MULAW_8K)


class GatedTTS(FakeTTS):
    """Blocks synthesis for chosen call indices until .gate is set — lets a
    test freeze the pipeline mid-reply at a deterministic point."""

    def __init__(self, block_indices):
        super().__init__()
        self.block_indices = block_indices
        self.gate = asyncio.Event()
        self._calls = 0

    async def synthesize(self, text, fmt):
        idx = self._calls
        self._calls += 1
        if idx in self.block_indices:
            await self.gate.wait()
        async for frame in super().synthesize(text, fmt):
            yield frame


class FakeLLM:
    """Yields scripted delta streams, one list per user turn."""

    def __init__(self):
        self.replies: list[list[str]] = []

    async def stream(self, messages):
        for part in self.replies.pop(0):
            yield LLMDelta(text=part)


# -------------------------------------------------------------------- helpers
START = CallStarted(stream_id="s1", call_id="c1", caller="+911234567890")

LOUD_ULAW = audio.pcm16_to_ulaw(np.full(160, 8000, dtype=np.int16))


def media(ulaw: bytes) -> MediaReceived:
    return MediaReceived(frame=AudioFrame(payload=ulaw, format=MULAW_8K))


def op_kinds(transport):
    return [op[0] for op in transport.ops]


def make_sess(monkeypatch, *, filler="", tts=None, tool_registry=None, bus=None):
    # These are engine defaults the Priya agent inherits, so patching config
    # before resolve() flows them into the resolved AgentConfig.
    monkeypatch.setattr(config, "ENDPOINT_SILENCE_MS", 20)
    monkeypatch.setattr(config, "BARGEIN_MIN_FRAMES", 3)
    monkeypatch.setattr(config, "THINKING_FILLER", filler)
    monkeypatch.setattr(config, "SPECULATIVE_REPLY", False)
    agent = agent_registry.resolve()
    transport = LocalTransport()
    strategy = executor = None
    if tool_registry is not None:
        specs = tool_registry.resolve(list(agent.tools))
        strategy = MarkerToolStrategy(specs)
        executor = ToolExecutor(tool_registry, bus if bus is not None else NULL_BUS)
    s = CallSession(transport, agent=agent, stt_factory=FakeSTT,
                    tts=tts or FakeTTS(), llm=FakeLLM(), bus=bus,
                    tool_strategy=strategy, tool_executor=executor)
    # Deterministic VAD: always classify audio as speech
    s._vad = SimpleNamespace(is_speech=lambda pcm_bytes, rate: True)
    s.transport = transport
    return s


def fake_tool_registry(book_handler, brochure_handler):
    """Priya-shaped registry with injected handlers — the test seam the
    old booking-module monkeypatching becomes under the tool registry."""
    reg = ToolRegistry()
    reg.register(ToolSpec(
        name="book_site_visit", description="book", parameters={"type": "object"},
        handler=book_handler, owner="test", marker="BOOK"))
    reg.register(ToolSpec(
        name="send_brochure", description="brochure", parameters={"type": "object"},
        handler=brochure_handler, owner="test", marker="BROCHURE"))
    return reg


@pytest.fixture
def sess(monkeypatch):
    return make_sess(monkeypatch)


async def start_call(s):
    """Dispatch CallStarted and play the greeting through to LISTENING."""
    await s._dispatch(START)
    await s._speak_task
    await s._dispatch(PlaybackFinished())


async def run_user_turn(s, text, finish_playback=True):
    """Deliver an STT final and wait for the endpoint + full reply."""
    await s.stt.on_event(STTEvent(kind="final", text=text))
    await asyncio.sleep(0.1)  # > patched ENDPOINT_SILENCE_MS
    assert s._speak_task is not None
    await s._speak_task
    if finish_playback:
        await s._dispatch(PlaybackFinished())


# ---------------------------------------------------------------------- tests
async def test_greeting_plays_on_start(sess):
    await sess._dispatch(START)
    await sess._speak_task

    assert op_kinds(sess.transport) == ["play"] * FakeTTS.N_FRAMES + ["checkpoint"]
    assert sess.transport.ops[-1] == ("checkpoint", "turn-0")
    # Frames are mu-law 8k
    frame = sess.transport.ops[0][1]
    assert len(frame.payload) == audio.FRAME_BYTES
    assert frame.format == MULAW_8K
    # Greeting is recorded in history; STT is running
    assert sess.messages[1] == {"role": "assistant", "content": sess.agent.greeting}
    assert sess.stt.started


async def test_normal_turn_event_sequence(sess):
    # First delta ends with a danda past MIN_FIRST_CHUNK so chunking splits
    # the reply into two spoken clauses, exactly as with a real token stream.
    first_clause = "प" * 125 + "।"
    sess._llm.replies = [[first_clause, " दूसरा वाक्य।"]]
    await start_call(sess)
    n_greeting_ops = len(sess.transport.ops)

    await run_user_turn(sess, "mujhe 2BHK chahiye")

    turn_ops = sess.transport.ops[n_greeting_ops:]
    # Each clause is spoken separately, each followed by its own checkpoint.
    assert [op[0] for op in turn_ops] == (
        ["play"] * 3 + ["checkpoint"] + ["play"] * 3 + ["checkpoint"]
    )
    assert turn_ops[3] == ("checkpoint", "turn-1")
    assert turn_ops[7] == ("checkpoint", "turn-1")
    assert sess.tts.texts[1:] == [first_clause, "दूसरा वाक्य।"]
    # History gained the user turn and the joined assistant reply
    assert sess.messages[-2] == {"role": "user", "content": "mujhe 2BHK chahiye"}
    assert sess.messages[-1] == {
        "role": "assistant",
        "content": first_clause + " दूसरा वाक्य।",
    }


async def test_media_during_greeting_never_barges_in(sess):
    await sess._dispatch(START)
    # Greeting task is scheduled; loud speech arrives while it plays
    for _ in range(5):
        await sess._dispatch(media(LOUD_ULAW))
    await sess._speak_task

    assert ("clear",) not in sess.transport.ops
    # Audio was still forwarded to STT continuously
    assert len(sess.stt.frames) == 5


async def test_d3_partial_during_greeting_tail_never_interrupts(sess):
    await sess._dispatch(START)
    await sess._speak_task  # greeting sent; carrier still playing the tail
    await sess.stt.on_event(STTEvent(kind="partial", text="hello"))
    assert ("clear",) not in sess.transport.ops


async def test_sustained_loud_speech_triggers_clear(sess):
    sess._llm.replies = [["एक वाक्य।"]]
    await start_call(sess)
    # Leave playback unfinished: the draining tail (D6) keeps barge-in armed.
    await run_user_turn(sess, "haan boliye", finish_playback=False)

    for _ in range(3):
        await sess._dispatch(media(LOUD_ULAW))

    assert op_kinds(sess.transport).count("clear") == 1


async def test_d1_d4_new_turn_cancels_output_and_history_keeps_only_played(
        monkeypatch):
    # Call 0 = greeting, 1 = clause one, 2 = clause two (frozen), 3 = reply 2.
    tts = GatedTTS(block_indices={2})
    sess = make_sess(monkeypatch, tts=tts)
    first_clause = "प" * 125 + "।"
    sess._llm.replies = [[first_clause, " दूसरा वाक्य।"], ["theek hai."]]
    await start_call(sess)

    # Turn 1: clause one plays; clause two freezes inside TTS.
    await sess.stt.on_event(STTEvent(kind="final", text="pehla sawaal"))
    await asyncio.sleep(0.1)
    assert first_clause in tts.texts

    # Caller speaks again — the commit must cancel turn 1's audio (D1).
    await sess.stt.on_event(STTEvent(kind="final", text="naya sawaal"))
    await asyncio.sleep(0.1)
    await sess._speak_task  # turn 2's reply

    assert op_kinds(sess.transport).count("clear") == 1
    # D4: turn 1's history holds only the clause that actually played.
    assert sess.messages[-4] == {"role": "user", "content": "pehla sawaal"}
    assert sess.messages[-3] == {"role": "assistant", "content": first_clause}
    assert sess.messages[-2] == {"role": "user", "content": "naya sawaal"}
    assert sess.messages[-1] == {"role": "assistant", "content": "theek hai."}


async def test_filler_masks_thinking_and_stays_out_of_history(monkeypatch):
    sess = make_sess(monkeypatch, filler="हम्म")
    sess._llm.replies = [["एक वाक्य।"]]
    await start_call(sess)
    await run_user_turn(sess, "haan")

    # Filler is spoken before the reply clause, but never enters history.
    assert sess.tts.texts == [sess.agent.greeting, "हम्म", "एक वाक्य।"]
    assert sess.messages[-1] == {"role": "assistant", "content": "एक वाक्य।"}
    assert all(m["content"] != "हम्म" for m in sess.messages)


async def test_booking_and_brochure_markers_dispatch(monkeypatch):
    saved, brochures = [], []

    async def fake_book(ctx, args):
        saved.append((ctx.call_id, ctx.caller_name, args))

    async def fake_brochure(ctx, args):
        brochures.append(ctx.caller_number)

    sess = make_sess(monkeypatch,
                     tool_registry=fake_tool_registry(fake_book, fake_brochure))
    sess._llm.replies = [
        ["ठीक है, book कर देती हूं। [[BOOK day=Sunday time=4pm name=Rahul]] [[BROCHURE]]"]
    ]

    await start_call(sess)
    await run_user_turn(sess, "Sunday 4 baje aaunga, Rahul bol raha hoon")
    await asyncio.sleep(0.05)  # let the executor's fire-and-forget tasks run

    assert saved == [("c1", "unknown", {"day": "Sunday", "time": "4pm", "name": "Rahul"})]
    assert brochures == ["+911234567890"]
    # Markers never reach TTS
    assert all("[[" not in t for t in sess.tts.texts)


async def test_fallback_line_spoken_when_reply_produces_nothing(sess):
    # LLM yields no deltas at all (dead endpoint / open breaker): the
    # caller hears the scripted apology, not dead air (M8).
    sess._llm.replies = [[]]
    await start_call(sess)
    await run_user_turn(sess, "hello?")

    assert sess.tts.texts[-1] == sess.agent.turn.fallback_line
    # The model never said it: history holds the user turn, no assistant.
    assert sess.messages[-1] == {"role": "user", "content": "hello?"}


async def test_no_fallback_for_marker_only_reply(monkeypatch):
    # A reply that is pure tool call (no speech) is a completed reply,
    # not a failure — apologizing would be wrong.
    brochures = []

    async def fake_book(ctx, args):
        return None

    async def fake_brochure(ctx, args):
        brochures.append(ctx.caller_number)

    sess = make_sess(monkeypatch,
                     tool_registry=fake_tool_registry(fake_book, fake_brochure))
    sess._llm.replies = [["[[BROCHURE]]"]]
    await start_call(sess)
    await run_user_turn(sess, "brochure bhejo")
    await asyncio.sleep(0.05)

    assert brochures == ["+911234567890"]
    assert sess.agent.turn.fallback_line not in sess.tts.texts


async def test_history_is_capped_before_the_llm_sees_it(monkeypatch):
    monkeypatch.setattr(config, "HISTORY_MAX_MESSAGES", 4)
    sess = make_sess(monkeypatch)
    sess._llm.replies = [["ठीक है।"]]
    await start_call(sess)
    # Simulate a long call: pad history far beyond the budget.
    for i in range(20):
        sess.messages.append({"role": "user", "content": f"u{i}"})
        sess.messages.append({"role": "assistant", "content": f"a{i}"})

    await run_user_turn(sess, "latest question")

    # system + trimmed tail (ending in the new user turn) + new reply
    assert sess.messages[0]["role"] == "system"
    assert len(sess.messages) == 1 + 4 + 1
    assert sess.messages[-2] == {"role": "user", "content": "latest question"}
    assert sess.messages[-1] == {"role": "assistant", "content": "ठीक है।"}


async def test_run_loop_plays_greeting_and_cleans_up(sess):
    """Full lifecycle through run(): events in via the transport queue."""
    sess.transport.feed(START)
    sess.transport.feed(CallEnded())

    await sess.run()
    # run() dispatches CallStarted (greeting task starts) then exits on
    # CallEnded; cleanup cancels the in-flight greeting and closes STT.
    # S7.3: STT connects CONCURRENTLY with the greeting (the greeting no
    # longer waits ~1.2 s for the recognizer socket), so an immediate
    # hangup rightly cancels the still-pending connect — `started` is no
    # longer guaranteed here, `closed` is.
    assert sess.stt.closed
