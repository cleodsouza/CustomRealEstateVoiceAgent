# Production Readiness Assessment & AWS Hosting Plan

Written 2026-07-16, after the S1–S7.3 streaming/latency work. Companion to
CONSTITUTION.md (philosophy), RUNTIME_REDESIGN.md (target architecture), and
STREAMING_PIPELINE_DESIGN.md (what landed this week).

---

## Part 1 — What is already professional-grade

These are not aspirations; each is implemented, wired, and test-pinned
(215 tests passing).

### Architecture

The separation the Constitution demands actually exists in the code. The
Turn Engine is a pure, I/O-free state machine — every turn-taking rule
(barge-in thresholds, greeting immunity, D1 commit-kills-audio, D6 drain
semantics) is a named transition that replays deterministically in tests.
Transports are plugs: the session consumes normalized TransportEvents and
never sees Vobiz JSON; the chat-consumer test proves the same reply pipeline
serves a text channel with zero changes. Providers live behind Protocols —
Deepgram, Sarvam, and the OpenAI-compatible LLM each sit in one adapter, and
no vendor name appears in orchestration code. Agents are data: Priya is a
JSON record (persona, voice, tools, policies) resolved per call; the engine
carries none of her identity. Tools are registered capabilities with schema,
timeout, retries, and owner; the runtime executes them without knowing what
they do, and the feedback-tool loop (halt → execute → resume with the
result) is bounded and side-effect-safe.

### The streaming pipeline

Typed TurnEvent stream (tokens, clauses, phases, tool events, Done) with a
delivery-truth history rule: the transcript records only what the caller
actually heard. Latency is engineered in layers that compose: clause
chunking (10-char first flush / 100-char middle), synthesis-ahead (clause
N+1 synthesizes while N plays), a TTS replay cache with startup prewarm
(greeting/filler/fallback are free), speculative generation from interim
transcripts (adopted or abandoned at commit, tools quarantined until
adoption), provider endpointing (~300 ms end-of-turn), and an adaptive
filler that only plays when there is actually a gap to mask. Measured, not
asserted: per-turn TTFT/first-audio/reaction histograms at /metrics.

### Resilience

Circuit-breaker-lite per provider; TTS retry with a no-replay guarantee;
LLM first-token timeout; STT reconnect with exponential backoff and a loud
"call is deaf" alarm when the budget is exhausted; a scripted fallback line
instead of dead air; speak-what-we-have on mid-stream LLM errors;
self-healing optional provider parameters (preprocessing, extra_body) that
degrade instead of breaking calls; placeholder-credential detection that
turns config mistakes into one clear log line.

### Observability & security floor

Typed event bus with pure-observer sinks: structured logs (JSON mode for
log pipelines), Prometheus metrics, JSONL transcripts, per-call WAV
recordings, an operator dashboard. Secrets load from env with fail-fast
startup; every externally reachable route (webhooks and the WebSocket)
authenticates via shared token; .env is gitignored with an .env.example
template; CI runs on GitHub Actions.

---

## Part 2 — What is left to get to production

Ordered by risk, not effort.

### P0 — must fix before real traffic

1. **Durable storage.** Bookings, transcripts, and recordings are local
   files (bookings.jsonl, transcripts.jsonl, recordings/*.wav). A process
   restart keeps them; a host loss destroys them; two hosts can't share
   them. Move: bookings + transcripts → a database (DynamoDB or Postgres),
   recordings → S3. The seams exist (TranscriptWriter is a bus subscriber;
   the booking tool takes a path from tool_config) — this is adapter work,
   not surgery.
2. **Secret rotation and management.** The current .env keys (Sarvam,
   Deepgram, Vobiz, Gemini) have been pasted into logs and terminals during
   debugging — rotate all of them before go-live, and move them into a
   secrets manager rather than a file on disk.
3. **Real credentials for the two dead integrations.** Google Calendar
   still has the placeholder service-account JSON; WhatsApp brochure needs
   VOBIZ_WHATSAPP_CHANNEL_ID. Both fail soft today, but they are advertised
   features of the agent.
4. **A deployment artifact.** There is no Dockerfile. Production needs a
   reproducible image (pinned Python 3.12, pinned deps) — also the unit of
   rollback.

### P1 — needed within the first weeks of traffic

5. **Streaming TTS adapter.** The remaining latency wall is Sarvam's REST
   round trip (1.5–4.5 s per clause; one observed >8 s timeout). Their
   WebSocket streaming API would put first audio at a few hundred ms. The
   TTS Protocol already carries `supports_streaming_input` for exactly this.
6. **Alerting.** Metrics exist but nothing pages anyone. Alarms on:
   provider_failures_total (the deaf-call alarm), fallbacks_total spikes,
   p95 turn_first_audio_seconds, and process health.
7. **Load and soak testing.** The runtime has never held N concurrent
   calls. Each call is one WS + one Deepgram WS + LLM/TTS streams; find the
   per-process ceiling (likely tens of calls per vCPU), pin it, and set the
   autoscaling knee below it. Also a soak test for slow leaks (tasks,
   sockets, cache growth).
8. **Error tracking.** Structured logs are not error triage. Sentry (or
   CloudWatch alarms on ERROR-level JSON logs) with release tagging.
9. **Compliance for recorded calls.** Recordings and transcripts contain
   PII. Needed: a retention policy (S3 lifecycle rules), a disclosure line
   in the greeting where required, and a data-deletion path. For outbound
   campaigns in India: DND/TRAI compliance checks before dialing.

### P2 — platform maturity

10. **Multi-process event bus.** The bus, metrics registry, and provider
    cache are in-process. Fine for N processes behind a load balancer
    (each observes its own calls), but cross-process analytics needs the
    P3 broker seam (SQS/SNS or Redis streams) the redesign already names.
11. **Context summarization (M12).** trim_history evicts; long calls
    should summarize instead.
12. **Semantic VAD / smarter endpointing.** RMS+VAD barge-in is workable;
    echo-robust semantic endpointing is the next quality jump.
13. **Tenant management.** Agents are already data; per-tenant secrets,
    per-tenant provider policy, and agent hot-reload complete the
    platform story.
14. **Outbound campaign hardening.** Answering-machine detection, retry
    policies, calling-hours windows, per-trunk concurrency caps.
15. **Call-quality regression corpus.** Recorded production calls replayed
    through the Turn Engine in CI (the replay seam exists; the corpus
    doesn't yet).

---

## Part 3 — Hosting on AWS

Yes, this runs well on AWS. The one architectural fact that shapes
everything: a call is a long-lived WebSocket carrying 20 ms audio frames —
this is a latency-sensitive, stateful-connection workload, not a stateless
web app. Two stages:

### Stage 1 — one EC2 box (right for now: one agent, pilot traffic)

Cheapest, simplest, lowest latency jitter; replaces ngrok with a real
domain.

- **Region: ap-south-1 (Mumbai).** Callers, Vobiz, and Sarvam are all in
  India — round trips off the audio path matter.
- **Instance:** t3.small/t3.medium (₹~1.2–2.5k/mo). One `uvicorn --workers 1`
  process per the server's own guidance.
- **TLS + domain:** Route 53 domain → Caddy (or nginx + certbot) in front
  of uvicorn. Caddy auto-provisions certificates and proxies WebSockets
  with zero config. Set PUBLIC_HOST to the domain; point the Vobiz answer
  URL at it.
- **Process supervision:** systemd unit (restart=always) or Docker with
  `--restart unless-stopped`.
- **State off the box immediately:** recordings → S3 sync (or write
  directly), bookings/transcripts → DynamoDB (two tables, on-demand
  billing, effectively free at pilot volume).
- **Secrets:** SSM Parameter Store (free tier) → an EnvironmentFile the
  systemd unit loads at boot. Nothing long-lived on disk.
- **Logs/metrics:** CloudWatch agent ships journald logs (set
  LOG_FORMAT=json); scrape /metrics with the CloudWatch agent's Prometheus
  support; two alarms (process down, provider_failures_total > 0).

This is a weekend of work and is honestly the right platform until you
have concurrent-call volume.

### Stage 2 — ECS Fargate (when calls run concurrently or per-tenant)

- **Image:** Dockerfile (python:3.12-slim, uv/pip install, non-root user)
  → ECR. CI already exists — add a build-and-push job and an ECS deploy
  step.
- **Service:** ECS Fargate, 1 vCPU / 2 GB per task, N tasks. Each task is
  one single-worker uvicorn — the "scale by adding processes" model the
  server was designed for.
- **Load balancer:** ALB (it proxies WebSockets natively). Two things that
  bite voice specifically: raise the **idle timeout** to ≥ 300 s (calls
  are long; audio frames keep it alive, but tool-heavy silences may not),
  and use target-group **deregistration delay** ≥ max call length so a
  deploy drains live calls instead of cutting them off mid-sentence.
  No stickiness needed — a call is one WS connection, pinned to its task
  by nature; /answer webhooks can land anywhere.
- **Autoscaling:** on ActiveConnectionCount per target (from the load
  test's ceiling) with CPU as a backstop. Scale-in protection during
  drain.
- **State:** same as stage 1 (DynamoDB + S3) — that migration is what
  makes multi-task correct, since tasks share nothing.
- **Secrets:** Secrets Manager referenced directly in the task definition.
- **Observability:** logs → CloudWatch (JSON), metrics → Amazon Managed
  Prometheus + Grafana (the /metrics endpoint is already Prometheus
  format), alarms → SNS.
- **Networking:** tasks in private subnets, ALB public; NAT gateway for
  egress to Deepgram/Sarvam/Gemini/Vobiz. (Fargate cost ~$35/task/mo +
  ALB ~$25/mo + NAT ~$35/mo.)

### What NOT to use

- **Lambda / API Gateway WebSockets:** wrong shape — 20 ms frame cadence,
  long connections, and warm provider pools don't fit function invocation.
- **Multiple uvicorn workers per container:** the design scales by
  processes; keep 1 worker/task and add tasks.
- **Separate ngrok-style tunnels in prod:** the ALB/Caddy domain replaces
  /answer's public URL entirely.

### Migration order (each step independently shippable)

1. Dockerfile + image build in CI.
2. EC2 + Caddy + domain + SSM secrets; flip the Vobiz answer URL. (Pilot
   is production from this moment.)
3. Recordings → S3; bookings/transcripts → DynamoDB (adapters behind the
   existing seams; dashboard reads the new store).
4. CloudWatch logs/metrics/alarms.
5. Rotate every provider key.
6. When concurrency demands: ECR → ECS Fargate + ALB, same image, same
   env, autoscaling from load-test numbers.
