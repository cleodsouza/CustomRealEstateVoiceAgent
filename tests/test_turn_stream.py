"""S2: the typed TurnEvent stream — marker guard, strategy event sequences.

The regression gate for clause behavior is the untouched pre-S2 suite
(test_tools.py, test_clauses.py, test_session_scripted.py). This file pins
what is NEW: token granularity, marker guarding, event ordering, Done.
"""
from runtime.markers import MarkerGuard
from runtime.tools import MarkerToolStrategy, NativeToolStrategy, ToolSpec
from runtime.turn_events import (
    Clause,
    Done,
    Phase,
    Token,
    ToolCallDetected,
    TurnPhase,
)
from runtime.types import LLMDelta, ToolCallRequest


async def _h(ctx, args):
    return None


def spec_of(name, marker=None):
    return ToolSpec(name=name, description="d", parameters={"type": "object"},
                    handler=_h, owner="test", marker=marker)


class ScriptedLLM:
    def __init__(self, deltas):
        self._deltas = deltas
        self.seen_tools = "UNSET"

    async def stream(self, messages, tools=None):
        self.seen_tools = tools
        for d in self._deltas:
            yield d


def text_deltas(*parts):
    return [LLMDelta(text=p) for p in parts]


# ------------------------------------------------------------- MarkerGuard
def test_guard_passes_plain_text_immediately():
    g = MarkerGuard({"BOOK": "book"})
    assert g.feed("नमस्ते, ") == "नमस्ते, "
    assert g.feed("कैसे हैं?") == "कैसे हैं?"
    assert g.flush() == ""


def test_guard_strips_recognized_marker_split_across_deltas():
    g = MarkerGuard({"BOOK": "book"})
    out = g.feed("ठीक है। [[BO")
    assert out == "ठीक है। "          # nothing marker-shaped leaks
    assert g.feed("OK day=Sun") == ""  # still inside the candidate
    assert g.feed("]] पक्का।") == " पक्का।"
    assert g.flush() == ""


def test_guard_releases_unrecognized_marker_verbatim():
    g = MarkerGuard({"BOOK": "book"})
    assert g.feed("देखिए [[OTHER x=1]] हां") == "देखिए [[OTHER x=1]] हां"


def test_guard_releases_non_marker_bracket_text():
    g = MarkerGuard({"BOOK": "book"})
    # "[[ 5 ]]" fails the marker shape (\w+ required immediately after [[ )
    assert g.feed("array [[ 5 ]] end") == "array [[ 5 ]] end"


def test_guard_holds_trailing_single_bracket_then_releases():
    g = MarkerGuard({"BOOK": "book"})
    assert g.feed("पहले [") == "पहले "
    assert g.feed("अभी नहीं") == "[अभी नहीं"


def test_guard_flush_releases_unclosed_candidate():
    g = MarkerGuard({"BOOK": "book"})
    assert g.feed("अंत [[BOOK day=") == "अंत "
    assert g.flush() == "[[BOOK day="


def test_guard_length_cap_releases_runaway_hold():
    g = MarkerGuard({"BOOK": "book"}, max_hold=10)
    out = g.feed("x [[BOOKaaaaaaaaaaaaaaaa y")
    assert out.startswith("x [[BOOK")   # cap released it, nothing swallowed
    assert "aaaa" in out + g.flush()


def test_guard_overlapping_candidates():
    g = MarkerGuard({"BOOK": "book"})
    # First [[ disproven by a lone ] — the later [[BOOK]] must still strip.
    assert g.feed("a [[x] b [[BOOK]] c") == "a [[x] b  c"


# --------------------------------------------------- MarkerToolStrategy.run
async def test_marker_run_event_sequence_and_token_guarding():
    strategy = MarkerToolStrategy([spec_of("book_site_visit", marker="BOOK")])
    llm = ScriptedLLM(text_deltas(
        "ठीक है, कर देती हूं। ", "[[BOOK day=Sun", " time=4pm]]"))
    calls = []

    events = [e async for e in strategy.run(
        llm, [], lambda n, a: calls.append((n, a)))]

    assert events[0] == Phase(phase=TurnPhase.GENERATING)
    # No token contains any fragment of the marker.
    token_text = "".join(e.text for e in events if isinstance(e, Token))
    assert "[[" not in token_text and "BOOK" not in token_text
    assert token_text.strip() == "ठीक है, कर देती हूं।"
    # Dispatch happened exactly once, via the clause path.
    assert calls == [("book_site_visit", {"day": "Sun", "time": "4pm"})]
    assert [e for e in events if isinstance(e, ToolCallDetected)] == [
        ToolCallDetected(name="book_site_visit",
                         args={"day": "Sun", "time": "4pm"})]
    # Clause text is the clean speech, ids ascend from 1.
    clauses = [e for e in events if isinstance(e, Clause)]
    assert [(c.text, c.clause_id) for c in clauses] == [("ठीक है, कर देती हूं।", 1)]
    # Terminal pair, in order: Phase(DONE) then Done.
    assert events[-2] == Phase(phase=TurnPhase.DONE)
    assert events[-1] == Done(full_text="ठीक है, कर देती हूं।")
    assert llm.seen_tools is None


async def test_marker_run_clause_view_matches_old_clauses_surface():
    strategy = MarkerToolStrategy([spec_of("send_brochure", marker="BROCHURE")])
    llm = ScriptedLLM(text_deltas("ठीक है। [[BROCHURE]]"))
    calls = []
    out = [c async for c in strategy.clauses(
        llm, [], lambda n, a: calls.append((n, a)))]
    assert out == ["ठीक है।"]
    assert calls == [("send_brochure", {})]


# --------------------------------------------------- NativeToolStrategy.run
async def test_native_run_interleaves_tool_event_in_stream_order():
    strategy = NativeToolStrategy([spec_of("check_slots")])
    llm = ScriptedLLM([
        LLMDelta(tool_call=ToolCallRequest(name="check_slots",
                                           args={"day": "Sun"})),
        LLMDelta(text="देख रही हूं।"),
    ])
    calls = []

    events = [e async for e in strategy.run(
        llm, [], lambda n, a: calls.append((n, a)))]

    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "Phase"
    assert kinds.index("ToolCallDetected") < kinds.index("Token")
    assert calls == [("check_slots", {"day": "Sun"})]
    assert events[-1] == Done(full_text="देख रही हूं।")
    assert llm.seen_tools is not None


async def test_native_run_leaves_foreign_markers_in_tokens_and_clauses():
    strategy = NativeToolStrategy([spec_of("t")])
    llm = ScriptedLLM(text_deltas("देखो [[NOTE x]] ठीक।"))
    events = [e async for e in strategy.run(llm, [], lambda n, a: None)]
    token_text = "".join(e.text for e in events if isinstance(e, Token))
    assert "[[NOTE x]]" in token_text
    clause = next(e for e in events if isinstance(e, Clause))
    assert "[[NOTE x]]" in clause.text


# ----------------------------------------------------------- error fallback
class DyingLLM:
    async def stream(self, messages, tools=None):
        yield LLMDelta(text="आधा जवाब")
        raise RuntimeError("connection reset")


async def test_stream_error_still_yields_buffered_text_and_done():
    strategy = MarkerToolStrategy(())
    events = [e async for e in strategy.run(DyingLLM(), [], lambda n, a: None)]
    clauses = [e.text for e in events if isinstance(e, Clause)]
    assert clauses == ["आधा जवाब"]          # speak-what-we-have preserved
    assert events[-1] == Done(full_text="आधा जवाब")


# ============================================================ S4: feedback
import asyncio
import dataclasses
import json

import pytest

from runtime import agent_registry
from runtime.events import NULL_BUS
from runtime.tools import ToolContext, ToolExecutor, ToolOutcome, ToolRegistry


class MultiRoundLLM:
    """Yields one scripted delta list per stream() call and records every
    request's messages — the feedback loop's wire truth."""

    def __init__(self, rounds):
        self._rounds = [list(r) for r in rounds]
        self.requests: list[list] = []

    async def stream(self, messages, tools=None):
        self.requests.append(list(messages))
        for d in self._rounds.pop(0):
            yield d


def feedback_spec(name, marker=None, **kw):
    return ToolSpec(name=name, description="d", parameters={"type": "object"},
                    handler=_h, owner="test", marker=marker, feedback=True, **kw)


class AwaitRecorder:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    async def __call__(self, name, args):
        self.calls.append((name, args))
        return self.outcome


async def test_native_feedback_halt_execute_resume():
    spec = feedback_spec("check_slots")
    strategy = NativeToolStrategy([spec])
    llm = MultiRoundLLM([
        [LLMDelta(text="देखती हूं। "),
         LLMDelta(tool_call=ToolCallRequest(name="check_slots",
                                            args={"day": "Sun"}, id="abc"))],
        [LLMDelta(text="Sunday 4 बजे free है।")],
    ])
    awaiter = AwaitRecorder(ToolOutcome(ok=True, result={"slots": ["4pm"]}))
    messages = [{"role": "system", "content": "S"},
                {"role": "user", "content": "slot?"}]
    fire_and_forget = []

    events = [e async for e in strategy.run(
        llm, messages, lambda n, a: fire_and_forget.append(n), awaiter)]

    kinds = [type(e).__name__ for e in events]
    # halt → execute → resume, in order, visible in-band.
    i_call = kinds.index("ToolCallDetected")
    i_tool = kinds.index("ToolSettled")
    phases = [(e.phase, e.detail) for e in events if isinstance(e, Phase)]
    assert (TurnPhase.TOOL, "check_slots") in phases
    assert (TurnPhase.RESUMING, "check_slots") in phases
    assert i_call < i_tool
    assert awaiter.calls == [("check_slots", {"day": "Sun"})]
    assert fire_and_forget == []        # feedback path, not dispatch path

    # The model heard back: request 2 = request 1 + the OpenAI exchange.
    assert len(llm.requests) == 2
    added = llm.requests[1][len(messages):]
    assert added[0]["role"] == "assistant"
    assert added[0]["tool_calls"][0]["id"] == "abc"
    assert added[1]["role"] == "tool"
    assert added[1]["tool_call_id"] == "abc"
    payload = json.loads(added[1]["content"])
    assert payload["ok"] is True and payload["result"] == {"slots": ["4pm"]}

    # Caller's history object was never touched (copy-on-write).
    assert len(messages) == 2

    done = events[-1]
    assert isinstance(done, Done)
    assert done.context == tuple(added)
    assert "देखती हूं।" in done.full_text and "free है।" in done.full_text
    # Post-tool answer was speakable.
    assert [e.text for e in events if isinstance(e, Clause)][-1] == \
        "Sunday 4 बजे free है।"


async def test_marker_feedback_uses_marker_protocol_messages():
    spec = feedback_spec("check_slots", marker="CHECK")
    strategy = MarkerToolStrategy([spec])
    llm = MultiRoundLLM([
        text_deltas("देखती हूं। [[CHECK day=Sun]]"),
        text_deltas("4 बजे free है।"),
    ])
    awaiter = AwaitRecorder(ToolOutcome(ok=True, result=["4pm"]))
    messages = [{"role": "system", "content": "S"}]

    events = [e async for e in strategy.run(llm, messages,
                                            lambda n, a: None, awaiter)]

    added = llm.requests[1][len(messages):]
    assert added[0] == {"role": "assistant", "content": "[[CHECK day=Sun]]"}
    assert added[1]["role"] == "user"
    assert added[1]["content"].startswith("[[TOOL_RESULT check_slots]] ")
    assert json.loads(added[1]["content"].split("]] ", 1)[1])["ok"] is True
    # Tokens never leaked the marker, across both rounds.
    token_text = "".join(e.text for e in events if isinstance(e, Token))
    assert "CHECK" not in token_text and "[[" not in token_text


async def test_round_budget_degrades_to_fire_and_forget():
    spec = feedback_spec("check_slots")
    call = LLMDelta(tool_call=ToolCallRequest(name="check_slots", args={}))
    strategy = NativeToolStrategy([spec], max_tool_rounds=2)
    llm = MultiRoundLLM([[call], [call], [call]])   # model loops forever
    awaiter = AwaitRecorder(ToolOutcome(ok=True))
    dispatched = []

    [e async for e in strategy.run(llm, [{"role": "system", "content": "S"}],
                                   lambda n, a: dispatched.append(n), awaiter)]

    assert len(llm.requests) == 3          # initial + 2 resume rounds
    assert len(awaiter.calls) == 2         # the budget
    assert dispatched == ["check_slots"]   # 3rd call still EXECUTES, F&F


async def test_no_await_tool_keeps_feedback_tools_fire_and_forget():
    """A consumer that lends no executor (tests, old wiring) gets exactly
    the pre-S4 contract, feedback flag or not."""
    spec = feedback_spec("check_slots")
    call = LLMDelta(tool_call=ToolCallRequest(name="check_slots", args={}))
    strategy = NativeToolStrategy([spec])
    llm = MultiRoundLLM([[call, LLMDelta(text="ठीक।")]])
    dispatched = []

    [e async for e in strategy.run(llm, [], lambda n, a: dispatched.append(n))]

    assert len(llm.requests) == 1
    assert dispatched == ["check_slots"]


# ------------------------------------------------------- executor awaiting
def _ctx():
    return ToolContext(call_id="c1", caller_number="+91", caller_name="R",
                       agent=agent_registry.resolve())


class Emitted:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)

    def kinds(self):
        return [type(e).__name__ for e in self.events]


async def test_run_and_wait_returns_outcome_with_audit_events():
    reg, bus = ToolRegistry(), Emitted()

    async def h(ctx, args):
        return {"n": args["n"] + 1}

    reg.register(ToolSpec(name="inc", description="d", parameters={},
                          handler=h, owner="t"))
    out = await ToolExecutor(reg, bus).run_and_wait("inc", {"n": 1}, _ctx())
    assert out == ToolOutcome(ok=True, result={"n": 2})
    assert bus.kinds() == ["ToolCalled", "ToolSucceeded"]


async def test_run_and_wait_failure_becomes_outcome_not_exception():
    reg, bus = ToolRegistry(), Emitted()

    async def h(ctx, args):
        raise RuntimeError("api down")

    reg.register(ToolSpec(name="broken", description="d", parameters={},
                          handler=h, owner="t"))
    out = await ToolExecutor(reg, bus).run_and_wait("broken", {}, _ctx())
    assert out.ok is False and out.error == "api down"
    assert bus.kinds() == ["ToolCalled", "ToolFailed"]


async def test_run_and_wait_unregistered_tool():
    bus = Emitted()
    out = await ToolExecutor(ToolRegistry(), bus).run_and_wait("ghost", {}, _ctx())
    assert out.ok is False and "unregistered" in out.error
    assert bus.kinds() == ["ToolFailed"]


async def test_detach_on_cancel_lets_side_effect_finish():
    reg, bus = ToolRegistry(), Emitted()
    started, finished = asyncio.Event(), asyncio.Event()

    async def slow(ctx, args):
        started.set()
        await asyncio.sleep(0.05)
        finished.set()

    reg.register(ToolSpec(name="slow", description="d", parameters={},
                          handler=slow, owner="t", detach_on_cancel=True))
    ex = ToolExecutor(reg, bus)
    consumer = asyncio.create_task(ex.run_and_wait("slow", {}, _ctx()))
    await started.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await asyncio.sleep(0.1)
    assert finished.is_set()            # the booking still happened
    assert bus.kinds() == ["ToolCalled", "ToolSucceeded"]


async def test_cancel_without_detach_kills_the_tool():
    reg, bus = ToolRegistry(), Emitted()
    started, finished = asyncio.Event(), asyncio.Event()

    async def lookup(ctx, args):
        started.set()
        await asyncio.sleep(0.05)
        finished.set()

    reg.register(ToolSpec(name="lookup", description="d", parameters={},
                          handler=lookup, owner="t", detach_on_cancel=False))
    ex = ToolExecutor(reg, bus)
    consumer = asyncio.create_task(ex.run_and_wait("lookup", {}, _ctx()))
    await started.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await asyncio.sleep(0.1)
    assert not finished.is_set()        # pure lookup died with the turn


# ------------------------------------------------- session integration (S4)
async def test_session_feedback_tool_speaks_holds_and_commits(monkeypatch):
    """Full cycle through CallSession: pre-tool clause spoken, hold line
    spoken during the tool, answer resumed, exchange + spoken reply
    committed to durable history — hold line in none of it."""
    import config
    from runtime.types import PlaybackFinished, STTEvent
    from session import CallSession
    from test_session_scripted import START, FakeLLM, FakeSTT, FakeTTS
    from transports.local import LocalTransport

    monkeypatch.setattr(config, "ENDPOINT_SILENCE_MS", 20)
    monkeypatch.setattr(config, "THINKING_FILLER", "")
    agent = agent_registry.resolve()
    agent = dataclasses.replace(
        agent, tool_config={**agent.tool_config, "hold_line": "एक मिनट रुकिए।"})

    async def check_slots(ctx, args):
        return {"slots": ["4pm"], "day": args.get("day")}

    spec = ToolSpec(name="check_slots", description="d",
                    parameters={"type": "object"}, handler=check_slots,
                    owner="test", marker="CHECK", feedback=True)
    reg = ToolRegistry()
    reg.register(spec)
    tts, llm = FakeTTS(), FakeLLM()
    llm.replies = [["ठीक है, देखती हूं। [[CHECK day=Sunday]]"],
                   ["Sunday को 4 बजे free है।"]]
    sess = CallSession(LocalTransport(), agent=agent, stt_factory=FakeSTT,
                       tts=tts, llm=llm,
                       tool_strategy=MarkerToolStrategy([spec]),
                       tool_executor=ToolExecutor(reg, NULL_BUS))

    await sess._dispatch(START)
    await sess._speak_task
    await sess._dispatch(PlaybackFinished())
    await sess.stt.on_event(STTEvent(kind="final", text="Sunday slot hai?"))
    await asyncio.sleep(0.1)
    await sess._speak_task

    # Heard, in order: greeting, pre-tool clause, hold line, resumed answer.
    assert tts.texts == [agent.greeting, "ठीक है, देखती हूं।",
                         "एक मिनट रुकिए।", "Sunday को 4 बजे free है।"]

    # Durable history: exchange pair, then the full spoken reply — and the
    # hold line nowhere (not the model's words, D4).
    assert sess.messages[-3]["content"] == "[[CHECK day=Sunday]]"
    assert sess.messages[-2]["content"].startswith("[[TOOL_RESULT check_slots]]")
    assert sess.messages[-1] == {
        "role": "assistant",
        "content": "ठीक है, देखती हूं। Sunday को 4 बजे free है।"}
    assert all("एक मिनट" not in (m.get("content") or "") for m in sess.messages)
