"""
config.py — Deployment engine defaults + credentials.

Since M5 this is *not* the agent. Priya's persona (system prompt, greeting,
knowledge, voice) lives in agents/priya.json and is resolved per call by
runtime.agent_registry. What remains here are the credentials and the
engine defaults an agent inherits for any policy its file doesn't pin —
the tunables of the deployment, not of any one agent.
"""
from __future__ import annotations
import json
import os
from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


# ---------------------------------------------------------------------------
# Credentials / endpoints
# ---------------------------------------------------------------------------
# Required — startup fails immediately if missing, instead of limping along
# and sending empty-string auth headers to providers.
SARVAM_API_KEY = _get("SARVAM_API_KEY")
DEEPGRAM_API_KEY = _get("DEEPGRAM_API_KEY")
VOBIZ_AUTH_ID = _get("VOBIZ_AUTH_ID")
VOBIZ_AUTH_TOKEN = _get("VOBIZ_AUTH_TOKEN")
VOBIZ_API_BASE = os.getenv("VOBIZ_API_BASE", "https://api.vobiz.ai/api/v1")
# WhatsApp channel UUID (Vobiz Console -> WhatsApp -> Channels -> Create),
# not the same thing as VOBIZ_AUTH_ID. WhatsApp messaging lives under its own
# /messaging resource tree, scoped by channel_id in the request body rather
# than by /Account/{auth_id}/ like the voice Call API. Only required if an
# agent's tool_config sets brochure_url (agents/priya_tools.send_brochure).
VOBIZ_WHATSAPP_CHANNEL_ID = os.getenv("VOBIZ_WHATSAPP_CHANNEL_ID", "")

# Shared secret embedded in the wss:// URL that /answer hands to Vobiz and
# validated on every /ws connect. Any long random string:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"
WS_AUTH_TOKEN = _get("WS_AUTH_TOKEN")

# Public hostname Vobiz will reach (no scheme). e.g. agent.mycompany.com
# In local dev this is your ngrok host, e.g. abc123.ngrok-free.app
PUBLIC_HOST = os.getenv("PUBLIC_HOST", "localhost:8000")

# LLM (Ollama, OpenAI-compatible endpoint). Swap base_url to point at vLLM,
# Sarvam, or any OpenAI-compatible server without changing pipeline code.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")          # ignored by Ollama
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2:7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "160"))
# How agents dispatch tools by default: "marker" ([[TOKEN k=v]] in the
# reply text — reliable with small models) or "native" (OpenAI-protocol
# tool calls). An agent can pin its own in its llm section.
LLM_TOOL_DISPATCH = os.getenv("LLM_TOOL_DISPATCH", "marker")
# Feedback-tool round budget per turn (S4): how many halt-execute-resume
# rounds a turn may run before feedback tools degrade to fire-and-forget.
LLM_MAX_TOOL_ROUNDS = int(os.getenv("LLM_MAX_TOOL_ROUNDS", "3"))
# Extra JSON merged into every chat-completions request body (S7) — e.g.
# {"reasoning_effort": "none"} switches Gemini 2.5's thinking off, cutting
# time-to-first-token. The adapter self-heals if the endpoint rejects it.
_raw_extra = os.getenv("LLM_EXTRA_BODY", "").strip()
LLM_EXTRA_BODY = json.loads(_raw_extra) if _raw_extra else None

# Default STT model/language for agents that don't pin their own. The rest
# of the Deepgram connection string is fixed in providers/stt/deepgram.py.
STT_MODEL = os.getenv("STT_MODEL", "nova-2")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "hi")

# Default TTS voice for agents that don't pin their own (Priya pins hers in
# agents/priya.json).
TTS_MODEL = os.getenv("TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "shubh")           # v3 default voice
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "hi-IN")
TTS_PACE = float(os.getenv("TTS_PACE", "1.05"))           # only pace works on v3
# Per-attempt budget for ResilientTTS (M8). SarvamTTS is a single REST round
# trip (no partial frames until the whole clause is synthesized), so this
# must cover real Bulbul v3 latency, not just a dead-socket timeout.
TTS_ATTEMPT_TIMEOUT_S = float(os.getenv("TTS_ATTEMPT_TIMEOUT_S", "8.0"))
# Ask the TTS provider to normalize numerals / mixed-language text before
# synthesis ("600" spoken as a number, not digit-by-digit). The Sarvam
# adapter self-heals if the endpoint rejects the flag.
TTS_PREPROCESSING = os.getenv("TTS_PREPROCESSING", "true").strip().lower() == "true"

# ---------------------------------------------------------------------------
# Turn-taking / latency tuning
# ---------------------------------------------------------------------------
# Silence after last final transcript before we treat the turn as finished.
ENDPOINT_SILENCE_MS = int(os.getenv("ENDPOINT_SILENCE_MS", "550"))
# End-of-turn strategy: "fixed" = the silence window above; "provider" =
# trust the STT's endpoint events (Deepgram UtteranceEnd) with the silence
# window kept armed as a safety fallback. Capability-gated: falls back to
# fixed when the STT doesn't emit endpoints.
ENDPOINTER = os.getenv("ENDPOINTER", "fixed")
# Barge-in: energy threshold + consecutive 20 ms frames of speech that must
# arrive while the agent is talking before we cut its audio.
BARGEIN_RMS_THRESHOLD = float(os.getenv("BARGEIN_RMS_THRESHOLD", "650"))
BARGEIN_MIN_FRAMES = int(os.getenv("BARGEIN_MIN_FRAMES", "25"))  # ~500 ms
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "2"))   # 0–3; 2 = balanced
# A mid-speech partial only interrupts the agent after it has held the floor
# this long — below it, the partial is likely the agent's own audio echoing.
PARTIAL_INTERRUPT_AFTER_S = float(os.getenv("PARTIAL_INTERRUPT_AFTER_S", "0.5"))
# S7: start the reply pipeline (LLM + first-clause TTS) speculatively at
# each STT final, DURING the endpoint silence window; the commit adopts the
# running generation when the text matches, or discards it. Costs the odd
# wasted LLM call when the caller keeps talking; buys the whole endpoint
# window back as thinking time. Tools never fire before the commit.
SPECULATIVE_REPLY = os.getenv("SPECULATIVE_REPLY", "true").strip().lower() == "true"
# Tiny acknowledgement ("हम्म…") spoken the instant the caller's turn commits,
# masking LLM time-to-first-token (wired via the Turn Engine's THINKING
# state since M4). Set to "" to disable.
THINKING_FILLER = os.getenv("THINKING_FILLER", "हम्म")
# Spoken when a reply turn produces no audio at all (LLM/TTS failure or an
# open circuit breaker) — degradation the caller can hear instead of dead
# air (M8). Set to "" to disable.
FALLBACK_LINE = os.getenv(
    "FALLBACK_LINE",
    "माफ़ कीजिए, एक छोटी सी technical दिक्कत आ गई। क्या आप दोबारा बोल सकते हैं?")

# Conversation-history budget per call (M8): oldest turns are evicted once
# either cap is exceeded, so a long call can't inflate LLM latency/cost
# without bound. The system prompt never counts against these.
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "24"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "6000"))

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
# "text" (dev default) or "json" — one JSON object per log line for prod
# log pipelines. Structured conversation events are always logged as JSON
# payloads by the event-log subscriber regardless of this setting.
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
# Where the transcript subscriber appends per-call JSONL records.
TRANSCRIPTS_PATH = os.getenv("TRANSCRIPTS_PATH", "transcripts.jsonl")
# Where call recordings (WAV files) are stored, one per call.
RECORDINGS_PATH = os.getenv("RECORDINGS_PATH", "recordings")
# Where the booking tool appends site-visit records; the dashboard reads it.
# (Must match the agent's tool_config bookings_path if that is overridden.)
BOOKINGS_PATH = os.getenv("BOOKINGS_PATH", "bookings.jsonl")
