"""
server.py — Single production server.

Exposes:
  POST /answer        -> returns <Stream> XML telling Vobiz to open a WS to /ws
  POST /hangup        -> call-ended webhook (logging / CRM hook)
  POST /stream-status -> stream lifecycle events
  GET  /health        -> liveness check; ?deep=true probes provider health
  GET  /metrics       -> Prometheus text exposition (per-turn latency, counts)
  WS   /ws            -> the bidirectional audio stream; one CallSession each.
                         Requires ?token=<WS_AUTH_TOKEN>, which /answer embeds
                         in the URL it hands to Vobiz.

Production note: this is ONE server with a native WebSocket route — no internal
localhost proxy hop (the dev reference uses two servers + ngrok; that extra hop
adds latency you don't want in prod). Put this behind TLS on a real domain and
set PUBLIC_HOST to that domain. For local testing, run ngrok and set
PUBLIC_HOST to the ngrok host.

Run:  uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
(scale by running more single-worker processes behind a load balancer; each
call pins one event loop)
"""
from __future__ import annotations
import asyncio
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from xml.sax.saxutils import escape, quoteattr

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

import config
import dashboard
from agents import priya_tools
from providers.llm.openai_compat import OpenAICompatLLM
from providers.stt.deepgram import DeepgramSTT
from providers.tts.sarvam import SarvamTTS
from runtime.agent import AgentConfig
from runtime.agent_registry import resolve as resolve_agent
from runtime.events import EventBus
from runtime.interfaces import LLM, TTS, OnSTTEvent, STTFactory, SupportsHealth
from runtime.metrics import MetricsRegistry, TurnMetrics
from runtime.tts_cache import CachedTTS
from runtime.resilience import CircuitBreaker, ResilientLLM, ResilientTTS
from runtime.sinks import EventLogSubscriber, JsonFormatter, TranscriptWriter
from runtime.tools import (
    MarkerToolStrategy,
    NativeToolStrategy,
    ToolDispatchStrategy,
    ToolExecutor,
    ToolRegistry,
)
from runtime.types import MULAW_8K, STTEvent
from session import CallSession
from transports.vobiz import VobizTransport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
if config.LOG_FORMAT == "json":
    for _handler in logging.getLogger().handlers:
        _handler.setFormatter(JsonFormatter())
log = logging.getLogger("server")

app = FastAPI(title="Northern Heights Voice Agent")

# Mount brochures and other static files at /static
app.mount("/static", StaticFiles(directory="brochures"), name="static")

# ---------------------------------------------------------------------------
# Observability wiring (M6): one process-wide bus; sinks subscribe here at
# the composition root, never inside runtime modules. Every subscriber is
# a pure observer — removing any of them changes nothing about a call.
# ---------------------------------------------------------------------------
BUS = EventBus()
METRICS = MetricsRegistry()
BUS.subscribe(EventLogSubscriber())
BUS.subscribe(TurnMetrics(METRICS))
BUS.subscribe(TranscriptWriter(config.TRANSCRIPTS_PATH))
BUS.subscribe(priya_tools.PostCallBrochureSender())  # Auto-send brochure (M7)

# Tool wiring (M7): agents' tool modules register their specs here; agents
# reference tools by name in their records. Dynamic loading of tool modules
# from agent specs is the P3 plugin SDK — until then registration is an
# explicit line at this composition root.
TOOL_REGISTRY = ToolRegistry()
priya_tools.register(TOOL_REGISTRY)
TOOL_EXECUTOR = ToolExecutor(TOOL_REGISTRY, BUS)


# ---------------------------------------------------------------------------
# Composition root — the only place vendor names, agent policy, and secrets
# meet. Providers are built from the resolved AgentConfig's policy; secrets
# come from config (per-tenant secret resolution arrives with tenancy, P3).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Providers:
    stt_factory: STTFactory
    tts: TTS
    llm: LLM
    tool_strategy: ToolDispatchStrategy


# One provider set per agent, built lazily and reused across that agent's
# calls (LLM/TTS hold connection pools worth keeping warm; STT is still
# per-call via the factory). In tenancy this key becomes (tenant, agent).
_provider_cache: dict[str, _Providers] = {}


def _build_providers(agent: AgentConfig) -> _Providers:
    cached = _provider_cache.get(agent.agent_id)
    if cached is not None:
        return cached
    # M8: vendors are wrapped in resilience decorators here, at the edge —
    # bounded retry, first-token timeout, circuit-breaker-lite. Breakers
    # are per agent (this cache's granularity), shared across its calls.
    llm: LLM = ResilientLLM(OpenAICompatLLM(
        base_url=agent.llm.base_url,
        api_key=config.LLM_API_KEY,
        model=agent.llm.model,
        temperature=agent.llm.temperature,
        max_tokens=agent.llm.max_tokens,
        extra_body=config.LLM_EXTRA_BODY,
    ), breaker=CircuitBreaker())
    # Cache OUTSIDE resilience (S6): a hit replays remembered frames and
    # never touches the breaker; misses inherit retry/timeout discipline.
    tts: TTS = CachedTTS(ResilientTTS(SarvamTTS(
        api_key=config.SARVAM_API_KEY,
        model=agent.voice.model,
        speaker=agent.voice.speaker,
        language=agent.voice.language,
        pace=agent.voice.pace,
        preprocessing=config.TTS_PREPROCESSING,
        streaming_min_buffer_size=config.TTS_STREAM_MIN_BUFFER_SIZE,
        streaming_max_chunk_length=config.TTS_STREAM_MAX_CHUNK_LENGTH,
        streaming_sample_rate=config.TTS_STREAM_SAMPLE_RATE,
        streaming_audio_queue=config.TTS_STREAM_AUDIO_QUEUE,
    ), breaker=CircuitBreaker(), attempt_timeout_s=config.TTS_ATTEMPT_TIMEOUT_S))
    # Opportunistic prewarm of the agent's static lines so even the first
    # call's greeting starts instantly; fire-and-forget off the hot path.
    # (No running loop — e.g. a sync test building providers — skips it.)
    try:
        asyncio.create_task(tts.prewarm(
            [agent.greeting, agent.turn.filler, agent.turn.fallback_line],
            MULAW_8K))
    except RuntimeError:
        pass

    def stt_factory(on_event: OnSTTEvent) -> DeepgramSTT:
        return DeepgramSTT(
            api_key=config.DEEPGRAM_API_KEY,
            on_event=on_event,
            model=agent.stt.model,
            language=agent.stt.language,
        )

    # Per-agent tool dispatch: the agent's tool names resolve to specs, and
    # its llm.tool_dispatch policy picks the strategy.
    specs = TOOL_REGISTRY.resolve(agent.tools)
    strategy: ToolDispatchStrategy
    if agent.llm.tool_dispatch == "native":
        strategy = NativeToolStrategy(specs,
                                      max_tool_rounds=agent.llm.max_tool_rounds)
    else:
        strategy = MarkerToolStrategy(specs,
                                      max_tool_rounds=agent.llm.max_tool_rounds)

    providers = _Providers(stt_factory=stt_factory, tts=tts, llm=llm,
                           tool_strategy=strategy)
    _provider_cache[agent.agent_id] = providers
    return providers


@app.on_event("startup")
async def _warm_default_agent() -> None:
    """Build the default agent's providers while the server is idle: the
    TTS prewarm (greeting/filler/fallback) runs long before any call, so
    even call #1's greeting replays from cache instead of paying a live
    synthesis (observed 3-4 s). Warmup failure is logged, never fatal —
    the first call simply pays what it always paid."""
    try:
        _build_providers(resolve_agent())
    except Exception:  # noqa: BLE001 — warmup must never block startup
        log.exception("Provider warmup failed; first call pays the cost")


# M8 (D10 closed): every webhook route requires the shared secret in its
# URL — make_call.py embeds it in answer/hangup URLs, /answer embeds it in
# the status-callback URL, and inbound Answer URLs configured in the Vobiz
# dashboard must include ?token=<WS_AUTH_TOKEN>. Vobiz publishes no webhook
# signature scheme for us to verify, so a URL-borne secret is the strongest
# available check (same tradeoff as M0's WS token: it appears in carrier
# logs, which we accept).
def _webhook_authorized(request: Request) -> bool:
    token = request.query_params.get("token", "")
    if secrets.compare_digest(token, config.WS_AUTH_TOKEN):
        return True
    log.warning("Rejected webhook %s: bad or missing token", request.url.path)
    return False


@app.post("/answer")
async def answer(request: Request):
    if not _webhook_authorized(request):
        return PlainTextResponse("Forbidden", status_code=403)
    form = await request.form()
    log.info("answer webhook: From=%s To=%s Dir=%s",
             form.get("From"), form.get("To"), form.get("Direction"))
    # Agent selection rides through our own URLs, so it doesn't depend on the
    # carrier echoing custom params: make_call adds ?agent=, we forward it to
    # /ws, which resolves it. Absent → the default agent.
    agent_id = request.query_params.get("agent")
    ws_url = f"wss://{config.PUBLIC_HOST}/ws?token={config.WS_AUTH_TOKEN}"
    if agent_id:
        ws_url += f"&agent={agent_id}"
    # Same trick for the caller's number: Vobiz's WS "start" frame carries
    # no caller field at all, so it has to ride through here from the one
    # place Vobiz does hand it to us — this webhook's From param.
    caller = form.get("From", "")
    if caller:
        ws_url += f"&from={quote(caller)}"
    status_url = (f"https://{config.PUBLIC_HOST}/stream-status"
                  f"?token={config.WS_AUTH_TOKEN}")
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Stream bidirectional="true" keepCallAlive="true" '
        f'contentType="audio/x-mulaw;rate=8000" '
        f'statusCallbackUrl={quoteattr(status_url)} statusCallbackMethod="POST">'
        f"{escape(ws_url)}"
        "</Stream>"
        "</Response>"
    )
    return Response(content=xml, media_type="application/xml")

@app.post("/hangup")
async def hangup(request: Request):
    if not _webhook_authorized(request):
        return PlainTextResponse("Forbidden", status_code=403)
    form = await request.form()
    log.info("hangup: UUID=%s Duration=%ss Cause=%s",
             form.get("CallUUID"), form.get("Duration"), form.get("HangupCause"))
    return PlainTextResponse("OK")


@app.post("/stream-status")
async def stream_status(request: Request):
    if not _webhook_authorized(request):
        return PlainTextResponse("Forbidden", status_code=403)
    form = await request.form()
    log.info("stream-status: %s", dict(form))
    return PlainTextResponse("OK")


@app.get("/health")
async def health(deep: bool = False):
    # The tokened WS URL is a secret; only Vobiz (via /answer) gets it.
    if not deep:
        return {"status": "ok"}
    # Deep mode probes provider reachability with the default agent's
    # provider set. Probes are concurrent and individually time-boxed so a
    # hung vendor can't hang the health check.
    providers = _build_providers(resolve_agent())

    async def _ignore(ev: STTEvent) -> None:
        return

    probes: dict[str, object] = {
        "llm": providers.llm,
        "tts": providers.tts,
        "stt": providers.stt_factory(_ignore),
    }

    async def _probe(obj: object) -> bool | None:
        if not isinstance(obj, SupportsHealth):
            return None  # adapter offers no probe
        try:
            return await asyncio.wait_for(obj.healthy(), timeout=5.0)
        except Exception:  # noqa: BLE001
            return False

    results = dict(zip(probes, await asyncio.gather(*map(_probe, probes.values()))))
    ok = all(v is not False for v in results.values())
    return {"status": "ok" if ok else "degraded", "providers": results}


# ---------------------------------------------------------------------------
# Operations dashboard — an observability *consumer*: it reads the JSONL
# files the sinks already write and renders them. It subscribes to nothing
# and holds no state; deleting these two routes changes nothing about a
# call. Token-gated like the webhooks (transcript-derived data is PII):
# open /dashboard?token=<WS_AUTH_TOKEN>.
# ---------------------------------------------------------------------------
@app.get("/dashboard")
async def dashboard_page(request: Request):
    if not _webhook_authorized(request):
        return PlainTextResponse("Forbidden", status_code=403)
    return FileResponse(Path(__file__).parent / "dashboard.html",
                        media_type="text/html")


@app.get("/dashboard/data")
async def dashboard_data(request: Request):
    if not _webhook_authorized(request):
        return PlainTextResponse("Forbidden", status_code=403)
    # File parsing runs off the event loop; a live call never waits on it.
    return await asyncio.to_thread(
        dashboard.build_snapshot, config.TRANSCRIPTS_PATH, config.BOOKINGS_PATH)


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(
        METRICS.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    token = websocket.query_params.get("token", "")
    if not secrets.compare_digest(token, config.WS_AUTH_TOKEN):
        log.warning("Rejected /ws connect: bad or missing token")
        # Closing before accept() makes the ASGI server reject the handshake.
        await websocket.close(code=1008)
        return
    await websocket.accept()

    agent = resolve_agent(agent_id=websocket.query_params.get("agent"))
    providers = _build_providers(agent)
    log.info("Vobiz WebSocket connected (agent=%s)", agent.agent_id)

    caller = websocket.query_params.get("from") or None
    transport = VobizTransport(websocket, caller=caller)
    session = CallSession(
        transport, agent=agent,
        stt_factory=providers.stt_factory, tts=providers.tts, llm=providers.llm,
        bus=BUS,
        tool_strategy=providers.tool_strategy, tool_executor=TOOL_EXECUTOR,
    )
    try:
        await session.run()
    except Exception as e:  # noqa: BLE001
        log.error("WS session error: %s", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")