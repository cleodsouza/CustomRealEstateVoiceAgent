# Resume — Work Experience Entry

## Suggested header

**AI Full-Stack Developer Intern** — N Rose Developers *(or "Independent client project" if you'd rather not name them)*
June 2026 – July 2026

*(Adjust dates to whenever you actually worked on it — repo history runs June 2 to July 6, 2026.)*

---

## Bullet points (pick 4–6, mix and match)

- Designed and built a production-oriented real-time voice AI agent for inbound/outbound sales calls, architecting a full streaming pipeline (telephony → speech-to-text → LLM → text-to-speech → telephony) tuned for sub-second (~700–900ms) response latency using clause-level chunking and pipelined synthesis.

- Integrated multiple third-party AI/telephony APIs (Vobiz for WebSocket-based bidirectional audio streaming, Deepgram for streaming speech recognition, Sarvam for Hindi/Hinglish text-to-speech, and an OpenAI-compatible LLM endpoint), building a provider-agnostic adapter layer so any vendor could be swapped without touching orchestration logic.

- Architected a provider-agnostic runtime from the ground up: typed interfaces (Python Protocols) for STT/TTS/LLM/Transport, an explicit turn-taking state machine (listening/thinking/speaking/interrupted) replacing ad-hoc boolean flags, an in-process event bus (call/turn/tool lifecycle events), and a tool registry/executor supporting both native LLM tool-calling and a custom marker-based fallback protocol for less reliable models.

- Built an internal operations dashboard (FastAPI backend + vanilla JS/SVG frontend with light/dark theming) that reconstructs call history, latency metrics, and booking activity from append-only JSONL event logs, giving non-technical stakeholders visibility into live call quality without needing a database.

- Implemented production reliability patterns end-to-end: circuit breakers and bounded retries for LLM/TTS calls, automatic STT reconnect with jittered backoff, spoken fallback responses instead of dead air on provider failure, and a bounded conversation-history window to keep long calls from degrading latency.

- Instrumented the system for observability: Prometheus-compatible metrics endpoint (per-turn latency breakdowns, barge-in reaction time), structured JSON logging, and a deep health-check endpoint that probes all upstream providers concurrently.

- Wrote and maintained a 165+ test characterization suite (pytest, async) covering audio codec correctness, conversation state-machine replay, scripted end-to-end call flows, and provider-failure injection — used as a safety net for iterative refactors across an 8-milestone technical roadmap.

- Identified and remediated a live credential leak (API keys committed and pushed to a public repository), rotating all affected secrets and purging git history — then hardened the deployment with fail-fast credential validation and shared-secret authentication on every webhook/WebSocket route.

- Added multi-tenant support for a calendar-booking tool (per-agent Google Calendar service-account integration) so the same runtime could serve multiple client businesses in isolation, with graceful degradation when a tenant hadn't configured calendar access.

- Diagnosed and fixed a live production integration bug by reverse-engineering an undocumented/inconsistent third-party WhatsApp Business API, correcting the request schema and authentication flow so post-call brochure delivery worked end-to-end.

---

## Shorter version (4 bullets, if space-constrained)

- Built a production-oriented real-time voice AI agent (telephony → STT → LLM → TTS pipeline) achieving sub-second response latency, integrating Vobiz, Deepgram, and Sarvam APIs behind a provider-agnostic adapter layer.

- Architected the core runtime: an explicit turn-taking state machine, event-driven observability (Prometheus metrics, structured logs), and a tool-calling system supporting both native LLM function calls and a custom fallback protocol.

- Built a full-stack ops dashboard (FastAPI + hand-rolled JS/SVG charts) for live call and booking analytics, and added multi-tenant Google Calendar integration for client bookings.

- Drove reliability and testing discipline: circuit breakers, automatic reconnect, graceful degradation on provider failure, and a 165+ test suite used to safely execute an 8-milestone refactor roadmap — including remediating a leaked-credential security incident.

---

## Skills/tech line for your skills section

Python, FastAPI, WebSockets, asyncio, real-time audio streaming (G.711 µ-law), REST API integration, LLM orchestration, pytest, Prometheus, JavaScript/SVG (data viz), Google Calendar API, OAuth2/service accounts, git.

---

## Notes

- I framed this as an internship per your answer — swap the title/dates/company name freely.
- Bullets are ordered roughly by resume-impact; trim to whatever fits your format (usually 3–5 bullets per role).
- If you want, I can tailor a version that leads harder with either the AI/latency-engineering angle or the full-stack/dashboard angle depending on the specific job description.
