"""runtime/types.py — Normalized data types the runtime speaks internally.

Provider adapters translate vendor wire formats into these and back;
orchestration code never sees a vendor payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class AudioFormat:
    """Encoding + rate of an audio stream; adapters negotiate with this."""

    encoding: Literal["mulaw", "pcm16"]
    sample_rate: int
    channels: int = 1


# The only format the runtime speaks today (Vobiz telephony). PCM-native
# transports become possible once M3 negotiates formats per transport.
MULAW_8K = AudioFormat(encoding="mulaw", sample_rate=8000)


@dataclass(frozen=True)
class AudioFrame:
    """One chunk of audio on its way through the runtime."""

    payload: bytes
    format: AudioFormat


@dataclass(frozen=True)
class STTEvent:
    """Normalized recognizer event.

    "endpoint" = the provider believes the utterance is over (Deepgram
    UtteranceEnd). Only consulted when the Turn Engine's endpointer is
    provider-trusting; text is empty for this kind.

    "dead" = the recognizer is permanently lost (reconnect budget
    exhausted, M8/D5). The session raises the deaf-call alarm; the call
    itself continues — a one-way agent beats a dropped line.
    """

    kind: Literal["partial", "final", "endpoint", "dead"]
    text: str


@dataclass(frozen=True)
class ToolCallRequest:
    """A fully-assembled native tool call from the model. Adapters
    accumulate the provider's streaming fragments and emit one of these;
    the runtime never sees half a tool call. `id` is the provider's call
    id when it surfaces one ("" otherwise) — feedback rounds (S4) need it
    to correlate the tool-result message; a missing id is synthesized."""

    name: str
    args: dict
    id: str = ""


@dataclass(frozen=True)
class LLMDelta:
    """One increment of a streamed model reply: text, or (since M7) a
    complete native tool-call request."""

    text: str = ""
    tool_call: ToolCallRequest | None = None


# ---------------------------------------------------------------------------
# Transport events — the single internal contract every carrier adapter
# normalizes to. The session never sees carrier JSON.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CallStarted:
    stream_id: str
    call_id: str
    caller: str


@dataclass(frozen=True)
class MediaReceived:
    frame: AudioFrame


@dataclass(frozen=True)
class PlaybackFinished:
    """The transport finished playing everything sent so far."""


@dataclass(frozen=True)
class OutputCleared:
    """Buffered output was dropped (barge-in acknowledgement)."""


@dataclass(frozen=True)
class CallEnded:
    pass


TransportEvent = CallStarted | MediaReceived | PlaybackFinished | OutputCleared | CallEnded
