"""runtime/agent.py — the agent as a data record, not code.

The single largest structural change of the redesign (§6): the process is
an engine; an agent is a row. `AgentConfig` is that row — persona, voice,
and the provider/turn *policies* a call runs under. It carries no secrets:
API keys resolve separately (per-tenant, eventually), so the same record
is safe to ship to a control plane or store in a database.

This module is pure data. It imports no vendor, no I/O, and no `config`.
`from_dict` builds a record from a parsed JSON spec, filling any section
the spec omits from caller-supplied engine defaults — so an agent file can
pin only what makes it distinctive (persona, voice) and inherit the
deployment's latency/model defaults for the rest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from runtime.turn_engine import TurnPolicy


@dataclass(frozen=True)
class VoiceConfig:
    """TTS voice policy — how the agent sounds."""

    model: str
    speaker: str
    language: str
    pace: float


@dataclass(frozen=True)
class STTPolicy:
    """Recognizer policy. `endpointer` = who decides end-of-turn:
    "fixed" (our silence timer) or "provider" (trust the STT's own
    endpoint events, capability-gated on STT.emits_endpoint)."""

    model: str
    language: str
    endpointer: str


@dataclass(frozen=True)
class LLMPolicy:
    """Inference policy. `base_url` selects the OpenAI-compatible server;
    the api key is injected at the composition root, never stored here.
    `tool_dispatch` picks the tool strategy: "marker" (default — reliable
    with small models) or "native" (OpenAI-protocol tool calls)."""

    base_url: str
    model: str
    temperature: float
    max_tokens: int
    tool_dispatch: str
    # Conversation-history budget (M8): oldest-turn eviction keeps a long
    # call from inflating context without bound. LLM policy because the
    # budget is really about this model's latency/cost envelope.
    history_max_messages: int
    history_max_chars: int
    # Feedback-tool round budget per turn (S4): how many times a turn may
    # halt, execute a feedback tool, and resume with the result before
    # remaining calls degrade to fire-and-forget. LLM policy because the
    # cost is model round trips.
    max_tool_rounds: int = 3


@dataclass(frozen=True)
class TurnSettings:
    """Turn-taking feel. A superset of the engine's TurnPolicy: it also
    holds the knobs the *session* applies before the engine sees a frame
    (RMS gate, VAD aggressiveness) and the endpoint silence window."""

    endpoint_silence_ms: int
    bargein_rms_threshold: float
    bargein_min_frames: int
    vad_aggressiveness: int
    partial_interrupt_after_s: float
    filler: str
    # Spoken when a reply turn yields no audio (provider failure) — heard
    # degradation instead of dead air (M8). "" disables.
    fallback_line: str

    def engine_policy(self) -> TurnPolicy:
        """The subset the pure Turn Engine consumes."""
        return TurnPolicy(
            bargein_min_frames=self.bargein_min_frames,
            partial_interrupt_after_s=self.partial_interrupt_after_s,
            filler=self.filler,
        )


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    tenant_id: str
    system_prompt: str
    greeting: str
    # Carried with the agent; not yet injected into the prompt — that lands
    # with the Context Compiler subsystem. Kept here so an agent's knowledge
    # travels with its record instead of rotting in a global constant.
    knowledge: str
    voice: VoiceConfig
    stt: STTPolicy
    llm: LLMPolicy
    turn: TurnSettings
    # Names of registered tools this agent may call (M7). The composition
    # root resolves them against the ToolRegistry; the record stays pure
    # data — no handlers, no imports.
    tools: tuple[str, ...]
    # Freeform per-agent tool configuration (e.g. brochure URL), passed to
    # handlers via ToolContext.agent. Business config, not runtime config.
    tool_config: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any], defaults: dict[str, Any]) -> AgentConfig:
        """Build from a parsed spec. Persona fields are required; each
        policy section is merged over `defaults[section]`, so an omitted or
        partial section inherits the deployment's engine defaults."""

        def section(name: str) -> dict[str, Any]:
            return {**defaults[name], **data.get(name, {})}

        return cls(
            agent_id=data["agent_id"],
            tenant_id=data.get("tenant_id", "default"),
            system_prompt=data["system_prompt"],
            greeting=data["greeting"],
            knowledge=data.get("knowledge", ""),
            voice=VoiceConfig(**section("voice")),
            stt=STTPolicy(**section("stt")),
            llm=LLMPolicy(**section("llm")),
            turn=TurnSettings(**section("turn")),
            tools=tuple(data.get("tools", ())),
            tool_config=dict(data.get("tool_config", {})),
        )
