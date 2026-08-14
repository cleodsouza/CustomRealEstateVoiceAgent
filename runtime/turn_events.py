"""runtime/turn_events.py — the typed vocabulary of a streamed reply turn.

Purpose
    The reply pipeline (a tool-dispatch strategy consuming an LLM delta
    stream) yields these IN-BAND to its one consumer, as a pull-based
    async generator. This is control flow — distinct from runtime/events.py,
    whose bus carries observational FACTS to any number of subscribers
    (Article VII). The pipeline yields TurnEvents; the session may mirror
    some of them onto the bus for observers.

Responsibilities
    - Define the event union a turn's generator yields. Frozen dataclasses,
      like engine intents and bus events, so replay tests compare by value.
    - Define TurnPhase, the pipeline's coarse position: which phase the
      generator is in between yields.

Consumers take only what they need:
    - The voice session consumes Clause (speak via TTS) — byte-identical
      semantics to the old AsyncIterator[str].
    - A text transport (chat, SMS, WhatsApp) consumes Token for live
      rendering and Done for the final message.
    - Phase/ToolCallDetected/ToolSettled let any consumer show status
      ("checking availability…") without knowing tool internals.

Invariants
    - Token text is marker-guarded: no fragment of a recognized tool marker
      or native tool syntax ever appears in a Token.
    - Every Clause's text also appeared as Tokens (tokens are the superset);
      consumers pick ONE granularity for delivery and use the other for
      display, never both.
    - Done is yielded exactly once, last, unless the generator is cancelled
      (barge-in) — cancellation unwinds without Done, and the consumer's
      finally-commit owns history (D4).

Extension points
    New event kinds join the union; consumers switch on type and ignore
    kinds they don't know, so adding one is not a breaking change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TurnPhase(Enum):
    """Where the pipeline is between yields."""

    GENERATING = "generating"   # pulling LLM deltas, emitting tokens/clauses
    TOOL = "tool"               # halted: awaiting a feedback tool's result
    RESUMING = "resuming"       # tool result injected; reopening the stream
    DONE = "done"               # stream settled; Done event follows


@dataclass(frozen=True)
class Token:
    """One marker-guarded increment of raw model text."""

    text: str


@dataclass(frozen=True)
class Clause:
    """One speakable chunk (existing stream_clauses semantics). clause_id
    is unique and ascending within the turn — the delivery ledger's key."""

    text: str
    clause_id: int


@dataclass(frozen=True)
class Phase:
    """The pipeline entered a new phase. detail names the cause where one
    exists (e.g. the tool name for TOOL/RESUMING)."""

    phase: TurnPhase
    detail: str = ""


@dataclass(frozen=True)
class ToolCallDetected:
    """The model requested a tool. Informational: dispatch/await policy is
    the pipeline's job; consumers use this for status display only.
    call_id is the provider's native call id where one exists ("" for
    marker calls); feedback rounds correlate the result message with it."""

    name: str
    args: dict = field(default_factory=dict)
    call_id: str = ""


@dataclass(frozen=True)
class ToolSettled:
    """A feedback tool's round completed (ok or not); generation resumes."""

    name: str
    ok: bool


@dataclass(frozen=True)
class Done:
    """The turn's stream settled normally. full_text is everything the
    model said (markers stripped), regardless of what was delivered —
    delivery-based history commits stay the consumer's job (D4).

    context (S4) carries the turn's feedback-tool exchanges (assistant
    tool-call + tool-result message pairs, in order) so the consumer can
    commit them to durable history alongside the spoken reply. Empty for
    turns without feedback rounds. A cancelled turn never yields Done, so
    its exchanges are dropped with it — by design."""

    full_text: str
    context: tuple = ()


TurnEvent = Token | Clause | Phase | ToolCallDetected | ToolSettled | Done
