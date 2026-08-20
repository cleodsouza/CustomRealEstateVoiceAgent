"""runtime/events.py — typed conversation events + the in-process event bus.

Purpose
    Everything important that happens during a call is announced as a typed
    event. Logs, metrics, transcripts, analytics, dashboards subscribe to
    the bus instead of being welded into runtime modules (CLAUDE.md event
    rules). The runtime emits facts; it never knows who is listening.

Responsibilities
    - Define the event vocabulary (frozen dataclasses, correlated by
      call_id and, where meaningful, turn_seq).
    - EventBus: non-blocking fan-out. `emit()` is synchronous — a queue
      put, nothing more — so the audio hot path never awaits a subscriber.
      A drain task delivers events to subscribers in order; a slow or
      crashing subscriber delays/loses only its own delivery, never the
      call.

Events are FACTS, not commands. Nothing subscribed to this bus may be the
sole executor of a business action — that is the tool executor's job (M7).
A subscriber crashing or the bus being replaced by a NullBus must never
change conversational behavior.

Scope notes
    - SpeechStarted/SpeechEnded describe *agent* speech (first audio frame
      handed to the transport → playback fully drained). User-speech events
      wait for a signal worth trusting (semantic VAD, M12); today's RMS+VAD
      verdict is too noisy to publish as fact.
    - An interrupted turn emits AgentInterrupted instead of SpeechEnded.

Lifecycle
    One EventBus per process, created at the composition root. The drain
    task starts lazily on first emit and re-binds if the running loop
    changes (test suites create a fresh loop per test; in production the
    loop never changes).

Extension points
    subscribe() any `async (event) -> None`. For sessions that need no
    observability (unit tests), pass NULL_BUS — emit() drops.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

log = logging.getLogger("runtime.events")


# ------------------------------------------------------------------- events
@dataclass(frozen=True)
class CallStarted:
    call_id: str
    caller: str
    agent_id: str


@dataclass(frozen=True)
class CallEnded:
    call_id: str


@dataclass(frozen=True)
class ThinkingStarted:
    """The caller's turn committed; the reply pipeline is starting."""

    call_id: str
    turn_seq: int


@dataclass(frozen=True)
class ThinkingFinished:
    """First speakable clause is available. thinking_s = commit → now
    (LLM time-to-first-token + first-clause accumulation)."""

    call_id: str
    turn_seq: int
    thinking_s: float


@dataclass(frozen=True)
class SpeechStarted:
    """First audio frame of the turn handed to the transport. Includes the
    thinking filler when enabled — this is perceived response start."""

    call_id: str
    turn_seq: int


@dataclass(frozen=True)
class SpeechEnded:
    """The turn's audio fully played out at the carrier (post-drain, D6)."""

    call_id: str
    turn_seq: int


@dataclass(frozen=True)
class AgentInterrupted:
    """In-flight agent output was cancelled. reaction_s = time from the
    engine's CancelOutput intent to the pipeline unwound + buffer cleared."""

    call_id: str
    turn_seq: int
    reaction_s: float


@dataclass(frozen=True)
class TurnCompleted:
    """One user turn's reply pipeline finished.

    ``thinking_s`` and ``first_audio_s`` are retained for compatibility.
    The additional fields give the real-time latency breakdown.
    """

    call_id: str
    turn_seq: int
    user_text: str
    agent_text: str
    thinking_s: float | None
    first_audio_s: float | None
    interrupted: bool
    endpoint_s: float | None = None
    llm_ttft_s: float | None = None
    first_text_s: float | None = None
    tts_to_audio_s: float | None = None
    llm_done_s: float | None = None
    generation_s: float | None = None

@dataclass(frozen=True)
class ToolCalled:
    call_id: str
    tool: str


@dataclass(frozen=True)
class ToolSucceeded:
    call_id: str
    tool: str


@dataclass(frozen=True)
class ToolFailed:
    call_id: str
    tool: str
    error: str


@dataclass(frozen=True)
class SessionClosed:
    call_id: str


@dataclass(frozen=True)
class ProviderFailed:
    """Alarm-grade: a provider is gone beyond its retry budget (e.g. the
    recognizer could not be reconnected — the call is deaf, M8/D5)."""

    call_id: str
    provider: str  # "stt" | "tts" | "llm"
    error: str


@dataclass(frozen=True)
class FallbackSpoken:
    """The reply pipeline produced nothing (LLM/TTS failure or open
    breaker) and the scripted fallback line was spoken instead of silence."""

    call_id: str
    turn_seq: int


@dataclass(frozen=True)
class PhaseChanged:
    """Bus mirror of the reply pipeline's in-band Phase event (S3): the
    turn's generator entered a new phase ("generating", "tool", "resuming",
    "done"). Observational — the pipeline's consumer acts on the in-band
    event; this exists for dashboards and analytics (Article VII). phase is
    the TurnPhase value as a string so subscribers stay serialization-flat."""

    call_id: str
    turn_seq: int
    phase: str
    detail: str = ""


@dataclass(frozen=True)
class TokenStreamEnded:
    """Per-turn token-stream aggregate, emitted when the pipeline yields
    Done (S3). Deliberately ONE event per turn, not one per token — token-
    granular bus traffic stays off unless a consumer demonstrates the need.
    tokens counts delta-granular Token events (not model tokenizer tokens);
    first_token_s is commit → first Token, the rawest time-to-first-token
    the runtime can observe (turn_thinking_seconds includes clause
    accumulation on top of it). A turn cancelled mid-stream never settles,
    so it emits AgentInterrupted instead of this."""

    call_id: str
    turn_seq: int
    tokens: int
    first_token_s: float | None


Event = (
    CallStarted | CallEnded | ThinkingStarted | ThinkingFinished
    | SpeechStarted | SpeechEnded | AgentInterrupted | TurnCompleted
    | ToolCalled | ToolSucceeded | ToolFailed | SessionClosed
    | ProviderFailed | FallbackSpoken | PhaseChanged | TokenStreamEnded
)

Subscriber = Callable[[Event], Awaitable[None]]


# --------------------------------------------------------------------- bus
class EventEmitter(Protocol):
    """What the session types against: something you can hand a fact to."""

    def emit(self, event: Event) -> None: ...


class EventBus:
    """In-process async pub/sub. No broker — a broker arrives only when a
    second process needs these events (P3), behind this same interface."""

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []
        self._queue: asyncio.Queue[Event] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def emit(self, event: Event) -> None:
        """Non-blocking, hot-path safe: enqueue and return. Never raises
        into the caller — observability must not be able to break a call."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop to deliver into (sync context) — drop
        if loop is not self._loop:
            # First emit, or a new loop (each test runs its own): re-bind.
            self._loop = loop
            self._queue = asyncio.Queue()
            self._task = loop.create_task(self._drain(self._queue),
                                          name="event-bus-drain")
        assert self._queue is not None
        self._queue.put_nowait(event)

    async def _drain(self, queue: asyncio.Queue[Event]) -> None:
        while True:
            event = await queue.get()
            for subscriber in list(self._subscribers):
                try:
                    await subscriber(event)
                except Exception:  # noqa: BLE001 — isolate subscriber crashes
                    log.exception("subscriber %r failed on %s",
                                  subscriber, type(event).__name__)
            queue.task_done()

    async def flush(self) -> None:
        """Wait until every emitted event has been delivered (tests)."""
        if self._queue is not None:
            await self._queue.join()

    def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
        self._task = None
        self._loop = None
        self._queue = None


class NullBus:
    """emit() drops. For sessions constructed without observability."""

    def emit(self, event: Event) -> None:  # noqa: ARG002
        return


NULL_BUS = NullBus()