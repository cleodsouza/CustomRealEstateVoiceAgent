"""S6: synth-ahead — the next clause synthesizes WHILE the current one
plays, so REST-TTS latency hides under playback instead of gapping the
reply mid-sentence."""
import asyncio

from runtime import agent_registry
from runtime.types import MULAW_8K, AudioFrame, PlaybackFinished, STTEvent
from session import CallSession
from test_session_scripted import START, FakeLLM, FakeSTT

import audio
import config


class JournalingTTS:
    supports_streaming_input = False

    def __init__(self, journal):
        self.journal = journal

    async def synthesize(self, text, fmt):
        self.journal.append(("synth", text[:12]))
        for _ in range(3):
            yield AudioFrame(payload=b"\xff" * audio.FRAME_BYTES, format=MULAW_8K)


class JournalingTransport:
    """Minimal Transport: records play/checkpoint into the shared journal,
    yielding to the loop on each play like a real paced carrier."""

    audio_format = MULAW_8K

    def __init__(self, journal):
        self.journal = journal

    def events(self):  # pragma: no cover — tests drive _dispatch directly
        raise NotImplementedError

    async def play(self, frame):
        self.journal.append(("play", ""))
        await asyncio.sleep(0)      # give background tasks the floor

    async def clear(self):
        self.journal.append(("clear", ""))

    async def checkpoint(self, name):
        self.journal.append(("checkpoint", name))


async def test_next_clause_synthesizes_during_playback(monkeypatch):
    monkeypatch.setattr(config, "ENDPOINT_SILENCE_MS", 20)
    monkeypatch.setattr(config, "THINKING_FILLER", "")
    monkeypatch.setattr(config, "SPECULATIVE_REPLY", False)
    journal: list = []
    llm = FakeLLM()
    first_clause = "क" * 61 + "।"               # > MIN_FIRST_CHUNK → splits
    llm.replies = [[first_clause, " दूसरा वाक्य।"]]

    sess = CallSession(JournalingTransport(journal),
                       agent=agent_registry.resolve(),
                       stt_factory=FakeSTT, tts=JournalingTTS(journal),
                       llm=llm)
    await sess._dispatch(START)
    await sess._speak_task
    await sess._dispatch(PlaybackFinished())
    await sess.stt.on_event(STTEvent(kind="final", text="बताइए"))
    await asyncio.sleep(0.1)
    await sess._speak_task

    # journal: greeting synth+plays+checkpoint, clause1 synth, ...
    synths = [i for i, (op, _) in enumerate(journal) if op == "synth"]
    assert len(synths) == 3                    # greeting, clause 1, clause 2
    c1_synth, c2_synth = synths[1], synths[2]
    # The point of S6: clause 2's synthesis began BEFORE clause 1 finished
    # playing (its checkpoint) — TTS latency hides under playback.
    c1_checkpoint = next(i for i, (op, name) in enumerate(journal)
                         if op == "checkpoint" and i > c1_synth)
    assert c2_synth < c1_checkpoint

    # And both clauses were spoken, in order, into history.
    assert sess.messages[-1]["content"] == first_clause + " दूसरा वाक्य।"


class SlowSTT:
    """Recognizer whose connect takes real time — the greeting must not
    wait for it (S7.3)."""

    emits_endpoint = True

    def __init__(self, on_event):
        self.on_event = on_event
        self.started = False
        self.closed = False

    async def start(self):
        await asyncio.sleep(0.2)
        self.started = True

    async def send_audio(self, frame):
        pass

    async def close(self):
        self.closed = True


async def test_greeting_does_not_wait_for_stt_connect(monkeypatch):
    from runtime.types import CallStarted
    from test_session_scripted import FakeLLM, FakeTTS

    monkeypatch.setattr(config, "SPECULATIVE_REPLY", False)
    journal: list = []
    sess = CallSession(JournalingTransport(journal),
                       agent=agent_registry.resolve(),
                       stt_factory=SlowSTT, tts=FakeTTS(), llm=FakeLLM())
    await sess._dispatch(CallStarted(stream_id="s1", call_id="c1",
                                     caller="+91"))
    await sess._speak_task                     # greeting fully played...
    assert sess.tts.texts[0] == sess.agent.greeting
    assert not sess.stt.started                # ...while STT still connecting
    await asyncio.sleep(0.25)
    assert sess.stt.started                    # and STT arrives on its own
