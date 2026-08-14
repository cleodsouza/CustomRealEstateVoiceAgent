"""S7: speculative reply — generation starts at STT-final, inside the
endpoint silence window; the commit adopts or abandons it. Invariants:
adopted turns re-use the ALREADY-RUNNING generation (one LLM call), skip
the filler when the first clause is ready, and never let a speculative
tool execute before the commit."""
import asyncio

import config
from runtime import agent_registry
from runtime.events import NULL_BUS
from runtime.tools import MarkerToolStrategy, ToolExecutor, ToolRegistry, ToolSpec
from runtime.types import LLMDelta, PlaybackFinished, STTEvent
from session import CallSession
from test_session_scripted import START, FakeSTT, FakeTTS
from transports.local import LocalTransport


class CountingLLM:
    """Same scripted reply for every call; counts calls — the speculation
    ledger. (Popping fakes would miscount abandoned speculations.)"""

    def __init__(self, parts):
        self.parts = parts
        self.calls = 0

    async def stream(self, messages, tools=None):
        self.calls += 1
        await asyncio.sleep(0.01)          # a whiff of TTFT
        for p in self.parts:
            yield LLMDelta(text=p)


def make_sess(monkeypatch, llm, *, silence_ms=60, filler="हम्म",
              tool_registry=None, strategy=None):
    monkeypatch.setattr(config, "ENDPOINT_SILENCE_MS", silence_ms)
    monkeypatch.setattr(config, "THINKING_FILLER", filler)
    monkeypatch.setattr(config, "SPECULATIVE_REPLY", True)
    agent = agent_registry.resolve()
    executor = ToolExecutor(tool_registry, NULL_BUS) if tool_registry else None
    sess = CallSession(LocalTransport(), agent=agent, stt_factory=FakeSTT,
                       tts=FakeTTS(), llm=llm,
                       tool_strategy=strategy, tool_executor=executor)
    return sess


async def _start(sess):
    await sess._dispatch(START)
    await sess._speak_task
    await sess._dispatch(PlaybackFinished())


async def test_adopted_speculation_single_llm_call_and_no_filler(monkeypatch):
    llm = CountingLLM(["नमस्ते जी, बताइए।"])
    sess = make_sess(monkeypatch, llm)
    await _start(sess)

    await sess.stt.on_event(STTEvent(kind="final", text="hello"))
    await asyncio.sleep(0.03)              # inside the silence window
    assert llm.calls == 1                  # speculation already running
    await asyncio.sleep(0.1)               # endpoint fires, turn commits
    await sess._speak_task

    assert llm.calls == 1                  # adopted, never re-generated
    # First clause was synthesized during speculation → filler skipped.
    assert "हम्म" not in sess.tts.texts
    assert sess.tts.texts[-1] == "नमस्ते जी, बताइए।"
    assert sess.messages[-1] == {"role": "assistant",
                                 "content": "नमस्ते जी, बताइए।"}


async def test_more_speech_abandons_and_respeculates(monkeypatch):
    llm = CountingLLM(["ठीक है।"])
    sess = make_sess(monkeypatch, llm, silence_ms=80)
    await _start(sess)

    await sess.stt.on_event(STTEvent(kind="final", text="मुझे दो"))
    await asyncio.sleep(0.03)
    await sess.stt.on_event(STTEvent(kind="final", text="BHK चाहिए"))
    await asyncio.sleep(0.15)              # endpoint fires on accumulated text
    await sess._speak_task

    assert llm.calls == 2                  # spec1 abandoned, spec2 adopted
    # Committed turn carries the full accumulated utterance.
    user = [m for m in sess.messages if m["role"] == "user"]
    assert user[-1]["content"] == "मुझे दो BHK चाहिए"
    assert sess.messages[-1]["content"] == "ठीक है।"


async def test_speculative_tools_never_fire_before_commit(monkeypatch):
    fired = []

    async def book(ctx, args):
        fired.append(args)

    spec = ToolSpec(name="book_site_visit", description="d",
                    parameters={"type": "object"}, handler=book,
                    owner="test", marker="BOOK")
    reg = ToolRegistry()
    reg.register(spec)
    llm = CountingLLM(["पक्का कर दिया। [[BOOK day=Sun]]"])
    sess = make_sess(monkeypatch, llm, silence_ms=150,
                     tool_registry=reg, strategy=MarkerToolStrategy([spec]))
    await _start(sess)

    await sess.stt.on_event(STTEvent(kind="final", text="book kar do"))
    await asyncio.sleep(0.08)              # speculation done; commit hasn't
    assert llm.calls == 1
    assert fired == []                     # queued, not executed
    await asyncio.sleep(0.15)              # commit adopts
    await sess._speak_task
    await asyncio.sleep(0.05)              # executor task runs
    assert fired == [{"day": "Sun"}]       # released exactly once


async def test_abandoned_speculation_drops_queued_tools(monkeypatch):
    fired = []

    async def book(ctx, args):
        fired.append(args)

    spec = ToolSpec(name="book_site_visit", description="d",
                    parameters={"type": "object"}, handler=book,
                    owner="test", marker="BOOK")
    reg = ToolRegistry()
    reg.register(spec)

    class TwoReplyLLM:
        """Reply 1 (speculated, then abandoned) books; reply 2 doesn't."""

        def __init__(self):
            self.calls = 0

        async def stream(self, messages, tools=None):
            self.calls += 1
            yield LLMDelta(text="बुक करती हूं। [[BOOK day=Sun]]"
                           if self.calls == 1 else "और बताइए।")

    llm = TwoReplyLLM()
    sess = make_sess(monkeypatch, llm, silence_ms=80,
                     tool_registry=reg, strategy=MarkerToolStrategy([spec]))
    await _start(sess)

    await sess.stt.on_event(STTEvent(kind="final", text="book"))
    await asyncio.sleep(0.03)              # spec1 runs, queues the BOOK
    await sess.stt.on_event(STTEvent(kind="final", text="mat karo"))
    await asyncio.sleep(0.15)
    await sess._speak_task
    await asyncio.sleep(0.05)

    assert llm.calls == 2
    assert fired == []                     # spec1's queued call died with it
    assert sess.messages[-1]["content"] == "और बताइए।"


async def test_commit_beating_prefetch_plays_filler_but_still_adopts(monkeypatch):
    """Instant provider endpointing means the commit can arrive before the
    speculative prefetch produces anything. The filler must play (there IS
    a gap to mask) while the SAME generation continues — never a second
    LLM call, never silent dead air."""
    llm = CountingLLM(["नमस्ते जी, बताइए।"])

    async def slow_stream(messages, tools=None):
        llm.calls += 1
        await asyncio.sleep(0.2)               # TTFT far beyond the window
        from runtime.types import LLMDelta
        yield LLMDelta(text="नमस्ते जी, बताइए।")

    llm.stream = slow_stream
    sess = make_sess(monkeypatch, llm, silence_ms=30)
    await _start(sess)

    await sess.stt.on_event(STTEvent(kind="final", text="hello"))
    await asyncio.sleep(0.4)                   # commit at 30ms, LLM at 200ms
    await sess._speak_task

    assert llm.calls == 1                      # adopted mid-flight
    assert "हम्म" in sess.tts.texts            # gap was masked
    assert sess.tts.texts[-1] == "नमस्ते जी, बताइए।"
    assert sess.messages[-1]["content"] == "नमस्ते जी, बताइए।"


async def test_adopted_turn_dispatches_tools_found_after_first_clause(monkeypatch):
    """A [[BOOK]] in the LAST clause of an adopted turn must still fire —
    the queue redirects to direct dispatch once adoption happens."""
    fired = []

    async def book(ctx, args):
        fired.append(args)

    spec = ToolSpec(name="book_site_visit", description="d",
                    parameters={"type": "object"}, handler=book,
                    owner="test", marker="BOOK")
    reg = ToolRegistry()
    reg.register(spec)
    # Clause 1 flushes early (danda past the 10-char floor); the marker
    # arrives in a later chunk, well after the prefetch stopped.
    llm = CountingLLM(["ठीक है, पक्का बुक कर देती हूं। ",
                       "आपको confirmation मिलेगा। [[BOOK day=Sat time=2pm]]"])
    sess = make_sess(monkeypatch, llm, silence_ms=40,
                     tool_registry=reg, strategy=MarkerToolStrategy([spec]))
    await _start(sess)

    await sess.stt.on_event(STTEvent(kind="final", text="book kar do"))
    await asyncio.sleep(0.15)
    await sess._speak_task
    await asyncio.sleep(0.05)

    assert llm.calls == 1
    assert fired == [{"day": "Sat", "time": "2pm"}]


async def test_partial_transcript_speculation_adopted_at_final(monkeypatch):
    """S7.2: speculation starts on the INTERIM transcript (no punctuation)
    and the final ("हां जी.") adopts it via normalized matching — one LLM
    call, started well before the endpoint."""
    llm = CountingLLM(["बिल्कुल जी।"])
    sess = make_sess(monkeypatch, llm, silence_ms=80)
    await _start(sess)

    await sess.stt.on_event(STTEvent(kind="partial", text="हां जी"))
    await asyncio.sleep(0.05)
    assert llm.calls == 1                  # speculating from the interim
    await sess.stt.on_event(STTEvent(kind="final", text="हां जी."))
    await asyncio.sleep(0.02)
    assert llm.calls == 1                  # final kept the running spec
    await asyncio.sleep(0.15)
    await sess._speak_task

    assert llm.calls == 1                  # adopted; never re-generated
    assert sess.tts.texts[-1] == "बिल्कुल जी।"
    user = [m for m in sess.messages if m["role"] == "user"]
    assert user[-1]["content"] == "हां जी."   # history carries the FINAL


async def test_changed_partial_restarts_speculation(monkeypatch):
    llm = CountingLLM(["ठीक।"])
    sess = make_sess(monkeypatch, llm, silence_ms=80)
    await _start(sess)

    await sess.stt.on_event(STTEvent(kind="partial", text="मुझे"))
    await asyncio.sleep(0.03)
    await sess.stt.on_event(STTEvent(kind="partial", text="मुझे दो BHK"))
    await asyncio.sleep(0.03)
    assert llm.calls == 2                  # utterance grew: re-speculated
    await sess.stt.on_event(STTEvent(kind="final", text="मुझे दो BHK"))
    await asyncio.sleep(0.15)
    await sess._speak_task

    assert llm.calls == 2                  # final matched the second spec
    assert sess.messages[-1]["content"] == "ठीक।"
