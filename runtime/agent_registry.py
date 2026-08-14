"""runtime/agent_registry.py — file-backed resolution of AgentConfig.

The control-plane seam: a call arrives with an `agent_id` (or a called
number) and this returns the `AgentConfig` to run it under, defaulting to
Priya — the reference agent, no longer the runtime. Today the store is a
directory of JSON files; the resolve() contract is what a database- or
API-backed registry would implement later without touching callers.

Persona and voice come from the agent's JSON; any policy section the file
omits inherits the deployment's engine defaults from `config.py`. Defaults
are read live on each resolve() so an operator's env overrides (and tests'
monkeypatching) still take effect — the JSON specs are cached, the defaults
are not.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import config
from runtime.agent import AgentConfig

log = logging.getLogger("agent.registry")

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
DEFAULT_AGENT_ID = "priya"

# Parsed JSON specs by agent_id, plus a called-number → agent_id index.
# Cached (the files don't change mid-process); the *defaults* they merge
# against are recomputed per resolve().
_specs: dict[str, dict[str, Any]] | None = None
_by_number: dict[str, str] = {}


def _load_specs() -> dict[str, dict[str, Any]]:
    global _specs
    if _specs is None:
        specs: dict[str, dict[str, Any]] = {}
        for path in sorted(AGENTS_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            agent_id = data["agent_id"]
            specs[agent_id] = data
            for number in data.get("numbers", []):
                _by_number[str(number)] = agent_id
        _specs = specs
    return _specs


def _defaults() -> dict[str, dict[str, Any]]:
    """Engine defaults an agent file inherits for any policy it omits.
    Read live so env overrides / test monkeypatching are honored."""
    return {
        "voice": {
            "model": config.TTS_MODEL,
            "speaker": config.TTS_SPEAKER,
            "language": config.TTS_LANGUAGE,
            "pace": config.TTS_PACE,
        },
        "stt": {
            "model": config.STT_MODEL,
            "language": config.STT_LANGUAGE,
            "endpointer": config.ENDPOINTER,
        },
        "llm": {
            "base_url": config.LLM_BASE_URL,
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
            "tool_dispatch": config.LLM_TOOL_DISPATCH,
            "history_max_messages": config.HISTORY_MAX_MESSAGES,
            "history_max_chars": config.HISTORY_MAX_CHARS,
            "max_tool_rounds": config.LLM_MAX_TOOL_ROUNDS,
        },
        "turn": {
            "endpoint_silence_ms": config.ENDPOINT_SILENCE_MS,
            "bargein_rms_threshold": config.BARGEIN_RMS_THRESHOLD,
            "bargein_min_frames": config.BARGEIN_MIN_FRAMES,
            "vad_aggressiveness": config.VAD_AGGRESSIVENESS,
            "partial_interrupt_after_s": config.PARTIAL_INTERRUPT_AFTER_S,
            "filler": config.THINKING_FILLER,
            "fallback_line": config.FALLBACK_LINE,
        },
    }


def resolve(agent_id: str | None = None,
            called_number: str | None = None) -> AgentConfig:
    """Resolve a call to its agent. Precedence: explicit agent_id, then
    called-number mapping, then the default agent. An unknown agent_id
    logs and falls back to the default rather than dropping the call."""
    specs = _load_specs()

    key: str | None = None
    if agent_id is not None:
        if agent_id in specs:
            key = agent_id
        else:
            log.warning("Unknown agent_id %r; falling back to %r",
                        agent_id, DEFAULT_AGENT_ID)
    if key is None and called_number is not None:
        key = _by_number.get(str(called_number))
    if key is None:
        key = DEFAULT_AGENT_ID

    return AgentConfig.from_dict(specs[key], _defaults())
