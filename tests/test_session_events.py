"""M6: the session announces the call as typed events on the bus.

Drives scripted calls (same fakes as test_session_scripted) with a real
EventBus and asserts the emitted event sequence and latency payloads.
"""
import asyncio
from types import SimpleNamespace

import pytest

import config
from runtime import agent_registry, events
from runtime.events import EventBus
from runtime.tools import MarkerToolStrategy, ToolExecutor
from runtime.types import CallEnded, PlaybackFinished, STTEvent
from session import CallSession
from test_session_scripted import (
    LOUD_ULAW,
    START,
    FakeLLM,
    FakeSTT,
    FakeTTS,
    GatedTTS,
    fake_tool_registry,
    media,
)
from transports.local import LocalTransport


class Recorder:
    def __init__(self):
        self.seen = []

    async def __call__(self, event):
        self.seen.append(event)

    def kinds(self):
        return [type(e).__name__ for e in self.seen]

    def only(self, cls):
        return [e for e in self.seen if isinstance(e, cls)]


@pytest.fixture
def rig(monkeypatch):
    def make(*, filler="", tts=None, tool_registry=None):
        monkeypatch.setattr(config, "ENDPOINT_SILENCE_MS", 20)
        monkeypatch.setattr(config, "BARGEIN_MIN_FRAMES", 3)
        monkeypatch.setattr(config, "THINKING_FILLER", filler)
        monkeypatch.setattr(config, "SPECULATIVE_REPLY", False)
        agent = agent_registry.resolve()
        transport = LocalTransport()
        bus = EventBus()
        rec = Recorder()
        bus.subscribe(rec)
        strategy = executor = None
        if tool_registry is not None:
            strategy = MarkerToolStrategy(tool_registry.resolve(list(agent.tools)))
            executor = ToolExecutor(tool_registry, bus)
        s = CallSession(transport, agent=agent, stt_factory=FakeSTT,
                        tts=tts or FakeTTS(), llm=FakeLLM(), bus=bus,
                        tool_strategy=strategy, tool_executor=executor)
        s._vad = SimpleNamespace(is_speech=lambda pcm_bytes, rate: True)
        s.transport = transport
        return s, bus, rec
    made = []

    def tracked(**kw):
        r = make(**kw)
        made.append(r[1])
        return r
    yield tracked
    for bus in made:
        bus.close()


async def _user_turn(sess, text):
    await sess.stt.on_event(STTEvent(kind="final", text=text))
    await asyncio.sleep(0.1)
    await sess._speak_task


async def test_happy_path_event_sequence_and_latencies(rig):
    sess, bus, rec = rig()
    sess._llm.replies = [["एक वाक्य।"]]

    await sess._dispatch(START)
    await sess._speak_task
    await sess._dispatch(PlaybackFinished())      # greeting drains
    await _user_turn(sess, "haan boliye")
    await sess._dispatch(PlaybackFinished())      # reply drains
    await bus.flush()

    assert rec.kinds() == [
        "CallStarted",
        "SpeechStarted",     # greeting (turn 0)
        "SpeechEnded",
        "ThinkingStarted",   # user turn committed
        "PhaseChanged",      # S3 mirror: pipeline entered "generating"
        "ThinkingFinished",  # first clause available
        "SpeechStarted",     # first reply audio
        "PhaseChanged",      # S3 mirror: pipeline entered "done"
        "TokenStreamEnded",  # S3 mirror: per-turn token aggregate
        "TurnCompleted",     # pipeline done (audio still draining)
        "SpeechEnded",       # playback drained at the carrier
    ]
    started = rec.only(events.CallStarted)[0]
    assert started.call_id == "c1"
    assert started.agent_id == "priya"

    turn = rec.only(events.TurnCompleted)[0]
    assert turn.turn_seq == 1
    assert turn.user_text == "haan boliye"
    assert turn.agent_text == "एक वाक्य।"
    assert not turn.interrupted
    assert turn.thinking_s is not None and turn.thinking_s >= 0
    assert turn.first_audio_s is not None and turn.first_audio_s >= 0
    # Greeting speech is turn 0; reply speech is turn 1
    assert [e.turn_seq for e in rec.only(events.SpeechStarted)] == [0, 1]


async def test_stream_mirrors_carry_phases_and_token_aggregate(rig):
    """S3: PhaseChanged mirrors the pipeline's phases in order; the token
    stream settles into exactly one TokenStreamEnded with a real count and
    a raw TTFT no larger than the clause-level thinking latency."""
    sess, bus, rec = rig()
    sess._llm.replies = [["एक वाक्य।"]]

    await sess._dispatch(START)
    await sess._speak_task
    await sess._dispatch(PlaybackFinished())
    await _user_turn(sess, "haan boliye")
    await bus.flush()

    phases = rec.only(events.PhaseChanged)
    assert [(p.phase, p.turn_seq) for p in phases] == [
        ("generating", 1), ("done", 1)]
    ended = rec.only(events.TokenStreamEnded)
    assert len(ended) == 1
    assert ended[0].turn_seq == 1
    assert ended[0].tokens >= 1
    turn = rec.only(events.TurnCompleted)[0]
    assert ended[0].first_token_s is not None
    assert 0 <= ended[0].first_token_s <= turn.thinking_s


async def test_bargein_emits_agent_interrupted(rig):
    sess, bus, rec = rig()
    sess._llm.replies = [["एक वाक्य।"]]
    await sess._dispatch(START)
    await sess._speak_task
    await sess._dispatch(PlaybackFinished())
    await _user_turn(sess, "haan")
    # Playback unfinished: draining tail keeps barge-in armed (D6).
    for _ in range(3):
        await sess._dispatch(media(LOUD_ULAW))
    await bus.flush()

    hits = rec.only(events.AgentInterrupted)
    assert len(hits) == 1
    assert hits[0].turn_seq == 1
    assert hits[0].reaction_s >= 0


async def test_superseding_turn_marks_first_turn_interrupted(rig):
    # Same D1 scenario as the scripted test: clause two frozen inside TTS
    # when the next user turn commits and cancels the pipeline.
    tts = GatedTTS(block_indices={2})
    sess, bus, rec = rig(tts=tts)
    first_clause = "प" * 125 + "।"
    sess._llm.replies = [[first_clause, " दूसरा वाक्य।"], ["theek hai."]]

    await sess._dispatch(START)
    await sess._speak_task
    await sess._dispatch(PlaybackFinished())
    await sess.stt.on_event(STTEvent(kind="final", text="pehla sawaal"))
    await asyncio.sleep(0.1)
    await sess.stt.on_event(STTEvent(kind="final", text="naya sawaal"))
    await asyncio.sleep(0.1)
    await sess._speak_task
    await bus.flush()

    turns = rec.only(events.TurnCompleted)
    assert [(t.turn_seq, t.interrupted) for t in turns] == [(1, True), (2, False)]
    assert turns[0].agent_text == first_clause  # D4: only what was heard
    assert len(rec.only(events.AgentInterrupted)) == 1


async def test_tool_audit_events(rig):
    async def fake_book(ctx, args):
        return None

    async def fake_brochure(ctx, args):
        raise RuntimeError("whatsapp api down")

    sess, bus, rec = rig(tool_registry=fake_tool_registry(fake_book, fake_brochure))
    sess._llm.replies = [
        ["ठीक है। [[BOOK day=Sunday time=4pm name=Rahul]] [[BROCHURE]]"]
    ]

    await sess._dispatch(START)
    await sess._speak_task
    await sess._dispatch(PlaybackFinished())
    await _user_turn(sess, "Sunday aaunga")
    await asyncio.sleep(0.05)  # let the executor's fire-and-forget tasks run
    await bus.flush()

    assert {e.tool for e in rec.only(events.ToolCalled)} == {
        "book_site_visit", "send_brochure"}
    assert [e.tool for e in rec.only(events.ToolSucceeded)] == ["book_site_visit"]
    failed = rec.only(events.ToolFailed)
    assert [(e.tool, e.error) for e in failed] == [
        ("send_brochure", "whatsapp api down")]

async def test_empty_reply_emits_fallback_spoken(rig):
    sess, bus, rec = rig()
    sess._llm.replies = [[]]
    await sess._dispatch(START)
    await sess._speak_task
    await sess._dispatch(PlaybackFinished())
    await _user_turn(sess, "hello?")
    await bus.flush()

    fallback = rec.only(events.FallbackSpoken)
    assert [(e.call_id, e.turn_seq) for e in fallback] == [("c1", 1)]
    turn = rec.only(events.TurnCompleted)[0]
    assert turn.agent_text == "" and not turn.interrupted


async def test_stt_dead_emits_provider_failed_alarm(rig):
    sess, bus, rec = rig()
    await sess._dispatch(START)
    await sess._speak_task

    await sess.stt.on_event(STTEvent(kind="dead", text=""))
    await bus.flush()

    alarms = rec.only(events.ProviderFailed)
    assert len(alarms) == 1
    assert alarms[0].provider == "stt"
    assert alarms[0].call_id == "c1"


async def test_run_lifecycle_emits_call_ended_and_session_closed(rig):
    sess, bus, rec = rig()
    sess.transport.feed(START)
    sess.transport.feed(CallEnded())

    await sess.run()
    await bus.flush()

    kinds = rec.kinds()
    assert kinds[0] == "CallStarted"
    assert kinds[-2:] == ["CallEnded", "SessionClosed"]
