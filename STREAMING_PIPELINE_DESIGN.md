# Streaming Turn Pipeline — Design

Status: IMPLEMENTED (S1–S5 landed 2026-07-16; see §7 for deviations)
Scope: convert the reply pipeline's yield surface from clause-only to a typed,
token-capable event stream; add tool-result feedback with halt/resume.
Constitution check: Articles IV, VII, IX, XII — addressed inline.

---

## 0. Premise correction (what is and isn't batch today)

The runtime already streams end-to-end on the voice path:

    LLM deltas ──► stream_clauses (first flush ~120 chars) ──► TTS ──► transport

What the current architecture actually lacks, mapped to the refactor's goals:

| Ask | Reality today | Gap |
|---|---|---|
| Async-generator yield of tokens | Pipeline yields **clauses** (voice-tuned), only inside `CallSession._generate_and_speak` | No token granularity; no consumable surface outside the voice session |
| `node_change/status` events | Bus has ThinkingStarted/Finished, SpeechStarted/Ended, Tool* | No in-band phase signal in the reply stream itself; bus is observational only (Article VII) and must stay that way |
| Post-stream memory commit | Already correct: `finally` block commits spoken clauses only (D4) | Must be **preserved** under the new surface, and extended to multi-round (tool-feedback) turns |
| Mid-stream tool halt/resume | Marker strategy dispatches per-clause; native adapter yields tool calls **only at stream end**; both fire-and-forget, results never fed back | This is the one real contract change |

So this is not batch→stream. It is: (a) widen the yield surface from
`AsyncIterator[str]` to a typed turn-event stream, (b) make tool calls
awaitable with model feedback, behind a per-tool flag.

---

## 1. Subsystem ownership

**Owner: the reply-pipeline seam that `ToolDispatchStrategy` already occupies**
(`runtime/tools.py` + `runtime/clauses.py`). It is the one place that
interprets the LLM stream. It grows into the redesign's Conversation Engine
seat incrementally — we do not build a new "Conversation Engine" subsystem now
(Article XII: no speculative structure; build the smaller thing that preserves
the seam).

Explicitly **not** the owner:

- `TurnEngine` — stays pure and untouched. It decides when output lives or
  dies (`is_stale`, CancelOutput); it never sees clause/token content. The
  existing deviation note in turn_engine.py already establishes that content
  is data flow, not engine traffic. No new states: a tool gap mid-reply is
  still AGENT_SPEAKING (draining rules D6 already ignore mid-turn
  playback-finished) or THINKING if nothing has been spoken yet.
- The EventBus — events are facts, not commands (Article VII). Routing the
  reply stream through the bus would make observers load-bearing. Control
  flow stays in-band; the bus mirrors observations.
- Providers — the token tee and marker guard are runtime logic (same argument
  as clauses.py's header). Adapters stay dumb delta pumps.

---

## 2. The yield surface: `TurnEvent` async generator

### 2.1 Shape

The strategy interface changes from

    clauses(llm, messages, on_tool) -> AsyncIterator[str]

to a single async generator of a typed union (frozen dataclasses, like
intents and bus events):

    TurnEvent =
        Token(text)                      # raw delta text, marker-guarded
      | Clause(text, clause_id)          # speakable chunk (existing semantics)
      | Phase(name, detail)              # "generating" | "tool" | "resuming" | "done"
      | ToolCallDetected(name, args)     # informational; execution is below
      | ToolSettled(name, ok)            # feedback round completed
      | Done(full_text)                  # stream finished; accumulation complete

Consumers subscribe by *reading the generator*, and take only what they need:
the voice session consumes `Clause` (identical behavior to today), a future
chat transport consumes `Token` + `Done`, the dashboard gets bus mirrors.

### 2.2 Pull generator, not EventEmitter — decision and tradeoffs

Alternatives considered:

1. **Async generator (pull)** — CHOSEN.
   - Backpressure for free: TTS/transport pacing naturally throttles the pull;
     no unbounded queue between LLM and mouth.
   - Cancellation is the existing mechanism: barge-in cancels `_speak_task`,
     CancelledError propagates through the generator, `finally` commits
     history. D1/D4 semantics survive unmodified.
   - Deterministic: a recorded delta stream replays to an identical event
     sequence — replay-testable like the engine (Article VIII).
2. EventEmitter / callback push.
   - Pro: multiple simultaneous consumers without teeing.
   - Con: no backpressure; cancellation and completion lifecycles must be
     reinvented; ordering guarantees blur; drifts toward the bus becoming
     control flow. Rejected.
3. Two parallel generators (tokens and clauses).
   - Con: two cursors over one stream — teeing, buffering, and cancellation
     coordination for zero present need. Rejected (Article XII).

Fan-out to *observers* (not controllers) is what the bus is for: the pipeline
mirrors `Phase`, clause boundaries, and (sampled or gated) token counts onto
the bus. Token-per-event on the bus is high-volume; default mirror is
clause-level plus token counts, with a config gate for full token mirroring
when a chat transport actually needs server-side observation.

### 2.3 Interception points (the "diff logic", no code)

- `providers/llm/openai_compat.py` — yield each assembled `ToolCallRequest`
  **as soon as its fragments complete** (index change / finish_reason) instead
  of holding all until stream end. Timing-only change within the LLM
  Protocol's contract ("yields assembled deltas"); required for mid-stream
  halt.
- `runtime/tools.py` — strategies become the turn-event generator. The clause
  accumulator (`stream_clauses`) is consumed internally; each delta is teed:
  raw text → `Token` events, accumulated → `Clause` events. Marker guard
  below.
- `runtime/clauses.py` — unchanged.
- `session.py::_generate_and_speak` — switches on event type instead of
  iterating strings. `Clause` handling is byte-for-byte today's behavior
  (speak, count frames, D4 ledger). `Token`/`Phase` are ignored by the voice
  path (or mirrored to the bus). The filler/first-clause latency trick is
  unchanged: "first clause" becomes "first Clause event".
- `runtime/events.py` — new observational events: `PhaseChanged`,
  `ToolResultInjected`, optional gated `TokensStreamed(count)`.
- Untouched: `turn_engine.py`, transports, STT/TTS adapters, endpointing.

### 2.4 Marker guard (token path correctness)

Today markers are stripped per-clause, so a `[[BOOK ...]]` split across deltas
is safe because the clause buffer reassembles it. A raw token surface breaks
that: tokens inside a marker must never reach a consumer. The tee therefore
runs behind an incremental guard: on an unmatched `[[`, token emission holds
back from that point until `]]` closes it (strip if recognized, release if
not) or the possibility is disproven. Clause output already had this
property; the guard gives tokens the same guarantee. Worst case (model opens
`[[` and never closes) is bounded by a max-holdback length, then released.

---

## 3. Mid-stream tool triggers: halt, execute, resume

This is the only behavioral contract change, so it is flag-gated per tool.

### 3.1 Contract

`ToolSpec` gains `feedback: bool = False` (and the executor an awaitable path
returning a result payload; audit events ToolCalled/Succeeded/Failed
unchanged). Default False → every existing tool keeps fire-and-forget
semantics exactly (Article IX: behavior preserved unless named and tested).

### 3.2 The inner loop (lives in the strategy/pipeline, bounded)

For a turn, up to `max_tool_rounds` (LLMPolicy knob, default small, e.g. 3):

1. Stream deltas, yielding `Token`/`Clause` events as normal.
2. Tool call detected (assembled native delta, or recognized marker):
   - fire-and-forget tool → dispatch as today, keep streaming. No halt.
   - feedback tool → flush any complete buffered clause (what the model said
     before the call is still spoken — "let me check that for you" keeps
     working), yield `Phase("tool")`, **stop pulling the LLM stream**.
3. Await the executor with the spec's timeout/retries. Timeout or failure
   produces an error result payload — the model gets to recover verbally;
   the existing fallback-line rule remains the terminal safety net.
4. Append to the *turn-local* message buffer (not session history — §4):
   assistant tool-call message, then tool-result message.
5. Yield `Phase("resuming")`, reopen `llm.stream` with the extended messages,
   continue yielding events. Round counter increments; at the bound, the
   result is injected with an instruction-free final round (no further tools
   honored) so a looping model can't spin.

### 3.3 Turn Engine and interruption interplay

No engine changes. The invariants that make this safe already exist:

- Staleness: the pipeline checks `engine.is_stale(seq)` at every event
  boundary, exactly as the clause loop does today.
- Barge-in during a tool wait: CancelOutput → `_cancel_output` cancels the
  pipeline task → CancelledError unwinds the generator mid-await. The
  in-flight tool task: a feedback tool whose consumer died is cancelled with
  it *unless* the spec marks it `detach_on_cancel` (side-effecting tools like
  a booking write must not be half-cancelled; pure lookups should die).
  Default: detach — never roll back a side effect implicitly.
- D6 draining: a long tool gap after clause 1 leaves `_sending=True`, so a
  mid-turn PlaybackFinished is ignored — the engine already refuses to end
  the turn early. Verified by existing rules; add a replay test to pin it.
- Voice UX for slow tools: an optional per-tool hold line ("ek minute...")
  sourced from `tool_config` (business config, not engine config), spoken by
  the session on `Phase("tool")` if the gap exceeds a threshold. The engine
  never knows.

---

## 4. State & memory strategy

Principle (unchanged from D4): **history records what the caller actually
received, committed at exactly one point, after the stream settles.**

- Accumulation is turn-local. The pipeline owns a turn buffer: emitted text,
  tool-call/tool-result message pairs. Nothing touches `session.messages`
  mid-stream — a half-generated reply can never leak into the next turn's
  prompt.
- Commit happens where it happens today: the consumer's `finally`. Voice
  commits the D4 ledger (clauses whose audio actually went out — the session
  keeps building `spoken` from `Clause` events + frame counts). A chat
  transport, where emission is delivery, commits `Done.full_text`.
- Multi-round turns commit the tool-call/result message pairs *between* the
  spoken segments, so the next turn's context shows the model why it said
  what it said. `trim_history` must learn to evict a tool-call/result pair
  atomically (a dangling tool message corrupts OpenAI-format context) — small
  named change to `runtime/context.py`, with tests.
- Interrupted mid-tool: ledger commits whatever was spoken before the halt;
  the pending tool round's messages are dropped (the model never produced a
  post-tool answer; recording half a round would fabricate history).

---

## 5. Migration plan (each step deployable, tested, behavior-preserved)

1. **S1 — Types + adapter timing.** Define `TurnEvent`; adapter yields
   assembled tool calls eagerly. Replay tests pin that transcripts and
   dispatch behavior are byte-identical.
2. **S2 — Strategies emit TurnEvents.** `clauses()` → `run()` yielding
   events; session switches on type, consuming only `Clause`. Marker guard
   lands here (token path exists, voice ignores it). Existing session/replay
   tests must pass unmodified — that is the regression gate.
3. **S3 — Bus mirrors + metrics.** `PhaseChanged`, token counts,
   time-to-first-token as a first-class metric (Article X).
4. **S4 — Feedback tools.** `feedback` flag, awaitable executor path,
   bounded resume loop, `trim_history` pair-eviction, hold-line. One pilot
   tool flips the flag; every other tool's behavior is untouched by
   construction.
5. **S5 — Proof of transport independence.** A minimal chat consumer (local
   transport or test harness) that renders `Token` events and commits
   `Done.full_text`, demonstrating the same pipeline serves a non-voice
   transport with zero engine/session-rule changes.

Ordering rationale: S1–S2 are structural (no behavior change, protected by
replay tests before refactor — Article IX); S4 is the sole behavioral change,
isolated behind a per-tool flag; S5 proves the constitutional payoff
(Article II).

---

## 6. Risks / open questions for review

- `max_tool_rounds` default and the no-more-tools terminal round policy.
- Detach-vs-cancel default for in-flight feedback tools on barge-in
  (proposed: detach, because implicit rollback is worse than a completed
  side effect plus an AgentInterrupted event).
- Token mirroring on the bus: off by default? (proposed: yes, gated).
- Whether `Phase` names should be an enum now or strings until a second
  consumer exists (proposed: enum — cheap, and replay tests compare them).

---

## 7. Implementation notes (as landed)

All §6 proposals were adopted: `max_tool_rounds` defaults to 3
(`LLM_MAX_TOOL_ROUNDS`, per-agent via `llm.max_tool_rounds`); detach is the
default (`ToolSpec.detach_on_cancel=True`); token-granular bus traffic is
one `TokenStreamEnded` aggregate per settled turn; `TurnPhase` is an enum.

Landed shape, by subsystem:

- `runtime/turn_events.py` — the TurnEvent vocabulary (new).
- `runtime/markers.py::MarkerGuard` — incremental token-path marker guard;
  strips only, never dispatches (dispatch stays on the clause path).
- `runtime/clauses.py::stream_turn_text` — one pass, two views: guarded
  Tokens + byte-identical raw clause chunks; `stream_clauses` untouched.
- `runtime/tools.py` — `ToolOutcome`; `ToolExecutor.run_and_wait` (same
  audit contract as dispatch, shield-detach semantics); `_StrategyBase`
  owns the shared bounded round loop; both strategies keep a `clauses()`
  compatibility view.
- `session.py` — consumes `run()`; mirrors `PhaseChanged` +
  `TokenStreamEnded`; commits `Done.context` (tool exchanges) before the
  spoken reply in `finally`; optional `tool_config.hold_line` spoken on a
  TOOL phase, never entering history.
- `runtime/context.py::trim_history` — atomic tool-pair eviction; None
  content counted as zero.
- `providers/llm/openai_compat.py` — eager per-slot tool-call assembly;
  call ids captured for feedback correlation.
- Proof of Article II: `tests/test_chat_consumer.py` renders Tokens and
  commits `Done.full_text` from the same generator, no session involved.

Deviations from §3, named:

1. Round budget exhaustion degrades feedback calls to fire-and-forget
   (they still execute; the model doesn't hear back) instead of an
   "instruction-free final round" — simpler, same spin-proofing, and the
   audit trail is identical.
2. The pipeline does not abandon a round's stream mid-flight on tool
   detection: OpenAI-protocol calls complete at stream end anyway, and
   finishing the round preserves the buffered-tail flush ("speak what we
   have"). Detection is still eager at the adapter.
3. Marker-protocol feedback uses `[[TOOL_RESULT name]] {json}` as a user
   message; agents enabling feedback on marker tools must prompt for it.
4. Known limitation: a hold line can, in principle, interleave with the
   thinking filler when a feedback tool triggers during the filler; both
   are short scripted utterances and neither enters history. Revisit if
   heard in practice.
5. Interrupted turns drop ALL of the turn's feedback exchanges (not just
   the pending round) — no Done, no commit. The spoken ledger still
   commits per D4.

---

## 8. S6 — live-call latency & pronunciation follow-up (2026-07-16)

First production call measured: endpoint silence 1.5 s (env), no filler
(env), 1.8–5.5 s TTS per clause (fresh TLS handshake per request +
120-char first clause), greeting re-synthesized every call, digits and
Latin loanwords mispronounced. Landed, each independently testable:

- `.env`: ENDPOINT_SILENCE_MS 1500→700; THINKING_FILLER re-enabled.
- `providers/tts/sarvam.py`: ONE pooled httpx client per adapter;
  `enable_preprocessing` sent for numeral/mixed-language normalization,
  self-healing (dropped permanently after a 4xx, request retried once).
- `runtime/tts_cache.py` (new): CachedTTS — LRU replay of complete,
  non-empty syntheses; composed OUTSIDE ResilientTTS; static lines
  (greeting/filler/fallback) prewarmed at provider build.
- `session.py`: synth-ahead pipeline — `_synth` (full utterance into
  memory) + `_play` split; the NEXT utterance is pulled and synthesized
  while the current one plays; the FIRST clause synthesizes under the
  filler. Hold lines flow through the same path (no concurrent-speak
  race). Engine untouched; D4/barge-in semantics pinned by the existing
  suite.
- `runtime/clauses.py`: MIN_FIRST_CHUNK 120→60 — safe now that
  synth-ahead hides the follow-on clause's latency.
- `agents/priya.json`: speakable facts + speech-style rules (numbers,
  prices, units in words: 'छह सौ से सात सौ square feet', 'पौने दो करोड़',
  'एक एकड़') so digit-strings and 'acre'/'Cr' never reach the TTS.

Expected turn profile: endpoint 0.7 s + cached filler heard almost
immediately; first clause audio ≈ LLM TTFT + one short TTS round trip,
overlapped with the filler. Greeting instant from the prewarmed cache.
Note: `_synth` buffers a full utterance (~100 KB) — fine for REST TTS
(audio arrives whole anyway); revisit when a streaming-synthesis
provider lands.

---

## 9. S7 — speculative reply (2026-07-16)

Goal: make the filler unnecessary by removing the serial wait entirely.
The reply pipeline (LLM stream + first-clause synthesis) now STARTS at
each STT final — inside the endpoint silence the caller is already giving
us — and the engine's CommitUserTurn either ADOPTS the running generation
(same text, same history) or abandons it. The Turn Engine is untouched:
it still owns every commit; speculation is pipeline prefetching in the
session wiring.

Safety invariants (all test-pinned in tests/test_speculation.py):
- Nothing speculative has a side effect: fire-and-forget tool calls are
  queued and released only on adoption (dropped on abandon); feedback
  tools gate on adoption before executing.
- Adoption requires exact text match AND unchanged history; barge-in and
  multi-final utterances abandon cleanly and re-speculate on the
  accumulated text.
- An adopted turn with its first clause already synthesized SKIPS the
  filler — the filler self-retires whenever speculation wins, and still
  masks the gap when it doesn't (cold cache, slow model). Keep
  THINKING_FILLER set; it now only plays when actually needed.

Supporting latency cuts, all self-healing/gated:
- LLM_EXTRA_BODY (config) → adapter `extra_body`, set in .env to
  {"reasoning_effort": "none"} to switch Gemini 2.5's thinking off;
  dropped permanently after a 4xx.
- Deepgram `speech_final` now maps to the endpoint event (~300 ms);
  .env ENDPOINTER=provider trusts it, 700 ms timer stays as fallback.
  Revert to `fixed` if callers get cut off mid-sentence.
- Sarvam synthesizes at 8 kHz (telephony rate) directly; stripped with
  preprocessing if the endpoint rejects optional params.
- Cost note: speculation spends an extra LLM call whenever the caller
  keeps talking past a final. SPECULATIVE_REPLY=false turns it off.

Integrations note: calendar/WhatsApp failures in production were
credentials, not code — placeholder service_account_key in priya.json
(now detected and skipped with a clear log) and unset
VOBIZ_WHATSAPP_CHANNEL_ID.

S7.1 field fixes (same day, from the second live call):
- Deepgram delivers speech_final WITH the final → commits are instant and
  speculation gets no head start. Adoption no longer BLOCKS on the
  prefetch (that blocking both delayed the reply and defeated the
  filler-skip check): if the prefetch finished, first audio is immediate
  and the filler is skipped; if it is still in flight, the filler plays
  over it and the same generation is adopted mid-flight.
- Bug fix: an adopted turn's tool queue now redirects to direct dispatch
  after adoption — a [[BOOK]] in the last clause used to queue forever.
- MIN_FIRST_CHUNK 60 → 10: ship the model's short opening sentence
  ("ठीक है Clon जी।") to TTS alone — sub-second synthesis — instead of
  gluing it to the long sentence after it.
- priya.json: dropped the "1-2 short sentences" cap that caused
  drip-feeding; property questions now get one complete answer (size,
  price, amenities, location). LLM_MAX_TOKENS=320 to make room.
Remaining wall: Sarvam REST synthesis (~1.5–4.5 s per clause). The next
real step for sub-second replies end-to-end is the Sarvam streaming
(WebSocket) TTS adapter — the TTS Protocol seam already supports it via
supports_streaming_input.

S7.2 (third live call): first_audio_s hit 0.0 on every turn (filler path),
two issues remained —
- A ~230-char middle clause (one-go pitch × MIN_CHUNK=180) blew the 8 s
  TTS budget: timeout + full retry = 14 s of silence. MIN_CHUNK 180 → 100:
  every synthesis fits the budget and its predecessor's playback covers it.
- Every adoption was "(in flight)" — with instant provider endpointing,
  final-time speculation has no head start, so the filler played every
  turn. Speculation now starts from INTERIM transcripts (gated to
  LISTENING/USER_SPEAKING so echo can't trigger it), kept across
  partial→final via normalized text matching (punctuation/casing
  insensitive); durable history still records the exact final. Combined
  with the documented Gemini thinking_config (reasoning_effort was being
  silently ignored — thinking was still on), the prefetch should now beat
  the commit and the filler self-retires on typical turns.
