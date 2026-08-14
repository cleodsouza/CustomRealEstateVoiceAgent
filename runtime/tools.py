"""runtime/tools.py — tool registry, executor, and dispatch strategies.

Purpose
    Tools are registered capabilities; the runtime executes them without
    knowing what they do (CLAUDE.md tool rules). An agent lists tool names
    in its record; the composition root resolves them against the registry
    and wires the per-agent dispatch strategy.

Responsibilities
    - ToolSpec: name / description / JSON-schema parameters / handler /
      owner / timeout / retry — the legacy marker token that triggers it
      under the marker strategy — and, since S4, the feedback contract:
      `feedback=True` means the tool's result is awaited and fed back to
      the model mid-turn; `detach_on_cancel` decides what barge-in does to
      an in-flight feedback tool (detach and let the side effect finish —
      the default, because implicit rollback is worse — or cancel it).
    - ToolRegistry: name → spec. Populated at the composition root by the
      agents' tool modules (dynamic loading from agent specs is the P3
      plugin SDK, not built until demanded).
    - ToolExecutor: two paths.
        * dispatch() — fire-and-forget: its own task, per-spec timeout and
          bounded retries, ToolCalled/Succeeded/Failed audit events. NEVER
          blocks the reply pipeline; a crash becomes an event, not dead air.
        * run_and_wait() (S4) — same execution and audit contract, but
          awaited: returns a ToolOutcome the strategy can hand back to the
          model. Cancellation honors the spec's detach_on_cancel.
    - Dispatch strategies (one interface, two implementations, per-agent
      choice via LLMPolicy.tool_dispatch): both now yield the typed
      TurnEvent stream (S2) and share the bounded feedback-round loop (S4):
        * MarkerToolStrategy — the model writes [[MARKER k=v]] tokens in
          its text; we strip and dispatch them. DEFAULT: the observed
          production reality (small Hinglish-capable models, OpenAI-compat
          quirks) is exactly where native tool-calling is unreliable
          (ROADMAP §1.5 amendment 5). Feedback rounds speak the marker
          protocol: the call is echoed as assistant text, the result comes
          back as a `[[TOOL_RESULT name]] {json}` user message — agents
          using feedback marker tools must prompt for that convention.
        * NativeToolStrategy — passes OpenAI-format tool schemas to the
          LLM and routes the assembled tool_call deltas the adapter
          yields. Feedback rounds speak the OpenAI protocol: an assistant
          tool_calls message plus a role="tool" result message.

    The feedback loop is bounded by max_tool_rounds (LLMPolicy). When the
    budget is exhausted, further feedback-marked calls still EXECUTE —
    they just degrade to fire-and-forget, so a looping model can't spin
    the turn forever. Tools without `feedback` keep the fire-and-forget
    contract the markers always had, untouched.

    Round messages are TURN-LOCAL: the strategy copies the history before
    extending it (copy-on-write), so a half-finished turn can never leak
    tool exchanges into the session's durable history. The settled turn's
    exchanges travel out on Done.context for the consumer to commit.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Protocol, Sequence

from runtime.agent import AgentConfig
from runtime.clauses import stream_turn_text
from runtime.events import EventEmitter, ToolCalled, ToolFailed, ToolSucceeded
from runtime.interfaces import LLM
from runtime.markers import MarkerGuard, extract_tool_calls
from runtime.turn_events import (
    Clause,
    Done,
    Phase,
    Token,
    ToolCallDetected,
    ToolSettled,
    TurnEvent,
    TurnPhase,
)

log = logging.getLogger("runtime.tools")

DEFAULT_MAX_TOOL_ROUNDS = 3


@dataclass(frozen=True)
class ToolContext:
    """What the runtime knows about the call a tool runs inside."""

    call_id: str
    caller_number: str
    caller_name: str
    agent: AgentConfig


@dataclass(frozen=True)
class ToolOutcome:
    """What awaiting a tool produced (S4). `result` is the handler's return
    value — business data the runtime carries opaquely back to the model."""

    ok: bool
    result: Any = None
    error: str = ""


ToolHandler = Callable[[ToolContext, dict], Awaitable[Any]]
OnToolCall = Callable[[str, dict], None]
# The awaitable execution path a consumer lends the strategy for feedback
# tools: async (name, args) -> ToolOutcome.
AwaitTool = Callable[[str, dict], Awaitable[ToolOutcome]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict            # JSON Schema, used by the native strategy
    handler: ToolHandler
    owner: str                  # who owns the business logic
    marker: str | None = None   # UPPERCASE token for the marker strategy
    timeout_s: float = 10.0
    retries: int = 0            # additional attempts after the first
    # S4: await the result and feed it back to the model mid-turn. False
    # (default) keeps the original fire-and-forget contract exactly.
    feedback: bool = False
    # S4: barge-in policy for an in-flight feedback tool. True (default):
    # the pipeline detaches and the tool finishes in the background — a
    # side effect is never implicitly rolled back. False: cancel with the
    # pipeline (right for pure lookups).
    detach_on_cancel: bool = True


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool {spec.name!r} already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def resolve(self, names: Iterable[str]) -> list[ToolSpec]:
        """Agent tool names → specs. Unknown names log and are skipped —
        a misconfigured agent loses a capability, not the call."""
        specs = []
        for name in names:
            spec = self._specs.get(name)
            if spec is None:
                log.warning("Agent references unregistered tool %r", name)
            else:
                specs.append(spec)
        return specs


class ToolExecutor:
    """Runs tools off the voice loop with audit events."""

    def __init__(self, registry: ToolRegistry, bus: EventEmitter) -> None:
        self._registry = registry
        self._bus = bus

    def dispatch(self, name: str, args: dict,
                 ctx: ToolContext) -> asyncio.Task | None:
        """Fire and forget: synchronous, returns immediately."""
        spec = self._registry.get(name)
        if spec is None:
            log.warning("Dispatch of unregistered tool %r dropped", name)
            self._bus.emit(ToolFailed(call_id=ctx.call_id, tool=name,
                                      error="unregistered tool"))
            return None
        return asyncio.create_task(self._run(spec, args, ctx),
                                   name=f"tool-{spec.name}")

    async def run_and_wait(self, name: str, args: dict,
                           ctx: ToolContext) -> ToolOutcome:
        """Awaitable path (S4): same execution and audit contract as
        dispatch(), but the outcome comes back to the caller. Runs in its
        own task so the spec's detach_on_cancel can hold: if the awaiting
        pipeline is cancelled (barge-in), a detaching tool keeps running
        to completion — audit events still land from its own task."""
        spec = self._registry.get(name)
        if spec is None:
            log.warning("Awaited call to unregistered tool %r dropped", name)
            self._bus.emit(ToolFailed(call_id=ctx.call_id, tool=name,
                                      error="unregistered tool"))
            return ToolOutcome(ok=False, error="unregistered tool")
        task = asyncio.create_task(self._run(spec, args, ctx),
                                   name=f"tool-{spec.name}")
        if spec.detach_on_cancel:
            # shield: cancelling the awaiter leaves the task running.
            return await asyncio.shield(task)
        try:
            return await task
        except asyncio.CancelledError:
            task.cancel()
            raise

    async def _run(self, spec: ToolSpec, args: dict,
                   ctx: ToolContext) -> ToolOutcome:
        self._bus.emit(ToolCalled(call_id=ctx.call_id, tool=spec.name))
        attempts = spec.retries + 1
        last_error = ""
        for attempt in range(1, attempts + 1):
            try:
                result = await asyncio.wait_for(spec.handler(ctx, args),
                                                timeout=spec.timeout_s)
            except asyncio.CancelledError:
                raise  # a cancelled tool is cancelled, not failed
            except Exception as e:  # noqa: BLE001 — a tool may fail arbitrarily
                last_error = str(e) or type(e).__name__
                log.warning("Tool %s attempt %d/%d failed: %s",
                            spec.name, attempt, attempts, last_error)
            else:
                self._bus.emit(ToolSucceeded(call_id=ctx.call_id,
                                             tool=spec.name))
                return ToolOutcome(ok=True, result=result)
        log.error("Tool %s failed after %d attempt(s): %s",
                  spec.name, attempts, last_error)
        self._bus.emit(ToolFailed(call_id=ctx.call_id, tool=spec.name,
                                  error=last_error))
        return ToolOutcome(ok=False, error=last_error)


# -------------------------------------------------------------- strategies
class ToolDispatchStrategy(Protocol):
    """Turns an LLM reply stream into a typed TurnEvent stream (S2):
    marker-guarded Tokens, speakable Clauses, Phase markers, tool events,
    and a final Done. Fire-and-forget tool calls route to `on_tool`;
    feedback tools (S4) run through `await_tool` and their results are fed
    back to the model in bounded resume rounds. Consumers pull the
    generator and take only the event kinds they need."""

    def run(self, llm: LLM, messages: list[Any], on_tool: OnToolCall,
            await_tool: AwaitTool | None = None) -> AsyncIterator[TurnEvent]: ...


def _outcome_payload(outcome: ToolOutcome) -> str:
    return json.dumps(
        {"ok": outcome.ok, "result": outcome.result, "error": outcome.error},
        ensure_ascii=False, default=str)


class _StrategyBase:
    """The shared turn loop: stream a round, classify tool calls, run
    feedback rounds (bounded), finish with Done. Subclasses supply the
    wire specifics via three hooks: how a round's stream opens, how tool
    calls are extracted from clause text, and what a feedback exchange
    looks like as messages."""

    def __init__(self, specs: Sequence[ToolSpec],
                 max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS) -> None:
        self._specs = {s.name: s for s in specs}
        self._max_rounds = max_tool_rounds

    # ------------------------------------------------------------- hooks
    def _open_stream(self, llm: LLM, messages: list[Any]) -> AsyncIterator[Any]:
        raise NotImplementedError

    def _extract(self, chunk: str) -> tuple[str, list[tuple[str, dict]]]:
        raise NotImplementedError

    def _feedback_messages(self, call: ToolCallDetected, outcome: ToolOutcome,
                           call_n: int) -> list[dict]:
        raise NotImplementedError

    # ------------------------------------------------------ compat view
    async def clauses(self, llm: LLM, messages: list[Any],
                      on_tool: OnToolCall) -> AsyncIterator[str]:
        """The pre-S2 text-only surface: clause strings, tools all
        fire-and-forget (no await_tool → no feedback rounds)."""
        async for ev in self.run(llm, messages, on_tool):
            if isinstance(ev, Clause):
                yield ev.text

    # ------------------------------------------------------------- loop
    async def run(self, llm: LLM, messages: list[Any], on_tool: OnToolCall,
                  await_tool: AwaitTool | None = None) -> AsyncIterator[TurnEvent]:
        yield Phase(phase=TurnPhase.GENERATING)
        msgs = messages
        extra: list[dict] = []      # feedback exchanges → Done.context
        full: list[str] = []
        clause_id = 0
        rounds = 0
        call_n = 0
        while True:
            pending: list[ToolCallDetected] = []

            def classify(ev: ToolCallDetected) -> None:
                """Feedback tools (with budget and an executor lent to us)
                are deferred to the round step; everything else keeps the
                original fire-and-forget contract — including feedback
                tools once the round budget is spent: they still EXECUTE,
                the model just doesn't hear back."""
                spec = self._specs.get(ev.name)
                if (spec is not None and spec.feedback
                        and await_tool is not None
                        and rounds < self._max_rounds):
                    pending.append(ev)
                else:
                    on_tool(ev.name, ev.args)

            if full and not full[-1].endswith((" ", "\n")):
                full.append(" ")    # round boundary: keep words apart
            async for item in self._open_stream(llm, msgs):
                if isinstance(item, Token):
                    full.append(item.text)
                    yield item
                elif isinstance(item, ToolCallDetected):
                    yield item
                    classify(item)
                else:  # raw clause text
                    clean, calls = self._extract(item)
                    for name, args in calls:
                        ev = ToolCallDetected(name=name, args=args)
                        yield ev
                        classify(ev)
                    if clean:
                        clause_id += 1
                        yield Clause(text=clean, clause_id=clause_id)
            if not pending:
                break
            # ------------------------------------------- feedback round
            rounds += 1
            if msgs is messages:
                msgs = list(messages)   # copy-on-write: history stays ours
            for call in pending:
                yield Phase(phase=TurnPhase.TOOL, detail=call.name)
                assert await_tool is not None  # classify() guarantees it
                outcome = await await_tool(call.name, call.args)
                yield ToolSettled(name=call.name, ok=outcome.ok)
                call_n += 1
                exchange = self._feedback_messages(call, outcome, call_n)
                msgs.extend(exchange)
                extra.extend(exchange)
            yield Phase(phase=TurnPhase.RESUMING, detail=pending[-1].name)
        yield Phase(phase=TurnPhase.DONE)
        yield Done(full_text="".join(full).strip(), context=tuple(extra))


class MarkerToolStrategy(_StrategyBase):
    """Default/fallback: [[MARKER k=v]] tokens parsed out of clause text.

    Tokens are guarded against recognized markers (nothing tool-shaped is
    ever surfaced); dispatch stays on the clause path via
    extract_tool_calls, exactly as before S2, so a tool fires once."""

    def __init__(self, specs: Sequence[ToolSpec],
                 max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS) -> None:
        super().__init__(specs, max_tool_rounds)
        self._markers = {s.marker: s.name for s in specs if s.marker}
        self._marker_of = {s.name: s.marker for s in specs if s.marker}

    def _open_stream(self, llm: LLM, messages: list[Any]) -> AsyncIterator[Any]:
        return stream_turn_text(llm.stream(messages), MarkerGuard(self._markers))

    def _extract(self, chunk: str) -> tuple[str, list[tuple[str, dict]]]:
        return extract_tool_calls(chunk, self._markers)

    def _feedback_messages(self, call: ToolCallDetected, outcome: ToolOutcome,
                           call_n: int) -> list[dict]:
        """Marker-protocol exchange: echo the call as assistant text, hand
        the result back as a [[TOOL_RESULT name]] user message. Agents
        with feedback marker tools must prompt the model for this shape."""
        marker = self._marker_of.get(call.name) or call.name.upper()
        kv = " ".join(f"{k}={v}" for k, v in call.args.items())
        return [
            {"role": "assistant",
             "content": f"[[{marker}{' ' + kv if kv else ''}]]"},
            {"role": "user",
             "content": f"[[TOOL_RESULT {call.name}]] {_outcome_payload(outcome)}"},
        ]


class NativeToolStrategy(_StrategyBase):
    """Native LLM tool-calls: schemas go up with the request; the adapter
    yields assembled ToolCallRequest deltas alongside the text stream.

    The token guard runs with an empty marker set: foreign [[...]] text
    passes through untouched, mirroring the clause path, which never
    stripped markers under this strategy."""

    def __init__(self, specs: Sequence[ToolSpec],
                 max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS) -> None:
        super().__init__(specs, max_tool_rounds)
        self._payload: list[dict] | None = [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in specs
        ] or None

    def _open_stream(self, llm: LLM, messages: list[Any]) -> AsyncIterator[Any]:
        return stream_turn_text(llm.stream(messages, tools=self._payload),
                                MarkerGuard({}))

    def _extract(self, chunk: str) -> tuple[str, list[tuple[str, dict]]]:
        return chunk, []

    def _feedback_messages(self, call: ToolCallDetected, outcome: ToolOutcome,
                           call_n: int) -> list[dict]:
        """OpenAI-protocol exchange. Adapters that don't surface call ids
        get a synthesized one — correlation only has to hold within the
        turn's own context window."""
        cid = call.call_id or f"call_{call_n}"
        return [
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": cid, "type": "function",
                "function": {"name": call.name,
                             "arguments": json.dumps(call.args,
                                                     ensure_ascii=False)},
            }]},
            {"role": "tool", "tool_call_id": cid,
             "content": _outcome_payload(outcome)},
        ]
