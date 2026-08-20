"""
session.py — Per-call wiring: transport + providers + Turn Engine.

Since M4 the session makes no turn-taking decisions. It normalizes raw
signals into engine events (speech verdicts, transcripts, timer firings,
playback progress), executes the intents the engine returns (greeting,
commit, cancel, timers), and runs the LLM→clauses→TTS→transport reply
pipeline. All the *rules* — barge-in thresholds, greeting policy,
endpointing, staleness — live in runtime/turn_engine.py where they are
pure and replay-testable.

Since M6 the session also announces what happens as typed events on the
bus (runtime/events.py). Emission is a synchronous enqueue — never awaited
on the audio path — and the session is correct with a NullBus: events are
observations, not control flow.
"""
from __future__ import annotations
import asyncio
import contextlib
import logging
from typing import AsyncIterator

import webrtcvad

import audio
import config
import recording
from runtime import events
from runtime.agent import AgentConfig
from runtime.context import trim_history
from runtime.endpointing import Endpointer, FixedSilenceEndpointer, ProviderEndpointer
from runtime.events import NULL_BUS, EventEmitter
from runtime.interfaces import LLM, TTS, STTFactory, Transport
from runtime.tools import (
    MarkerToolStrategy,
    ToolContext,
    ToolDispatchStrategy,
    ToolExecutor,
    ToolOutcome,
)
from runtime import turn_events
from runtime.turn_events import Clause as TurnClause
from runtime.turn_engine import (
    ArmEndpointTimer,
    CancelOutput,
    CommitUserTurn,
    Intent,
    PlayGreeting,
    TurnEngine,
    TurnState,
)
from runtime.types import (
    MULAW_8K,
    AudioFrame,
    CallEnded,
    CallStarted,
    MediaReceived,
    PlaybackFinished,
    STTEvent,
    TransportEvent,
)

log = logging.getLogger("session")


def _normalize_utterance(text: str) -> str:
    """Loose equality for speculation adoption (S7.2): interim transcripts
    differ from finals in punctuation/casing ("हां जी" vs "हां जी."), never
    in meaning — a speculation built on the interim is still the right
    reply for the final."""
    return "".join(ch for ch in text if ch not in ".,?!।;:-").strip().casefold()


class _Speculation:
    """One speculative reply run (S7), started at an STT final — inside the
    endpoint silence window, before the engine commits the turn. The engine
    still owns turn-taking; this is pipeline prefetching. Nothing
    speculative may have a side effect: fire-and-forget tool calls are
    QUEUED (released on adoption, dropped on abandon) and feedback tools
    are GATED on adoption before they execute."""

    def __init__(self, text: str, history_len: int) -> None:
        self.text = text
        self.history_len = history_len   # len(session.messages) at start
        self.gen = None                  # the strategy's TurnEvent generator
        self.events: list = []           # events pulled pre-adoption
        self.first_text: str | None = None
        self.first_frames: list = []
        self.queued_tools: list[tuple[str, dict]] = []
        self.adopted = asyncio.Event()
        self.task: asyncio.Task | None = None

    def abandon(self) -> None:
        """Not adopted: kill the prefetch, close the generator, drop the
        queued tools. Nothing executed, so there is nothing to undo."""
        if self.task is not None and not self.task.done():
            self.task.cancel()
        gen = self.gen
        if gen is not None:
            async def _close() -> None:
                with contextlib.suppress(Exception):
                    await gen.aclose()
            with contextlib.suppress(RuntimeError):  # no loop: nothing ran
                asyncio.get_running_loop().create_task(_close())


class CallSession:
    def __init__(self, transport: Transport, *, agent: AgentConfig,
                 stt_factory: STTFactory, tts: TTS, llm: LLM,
                 engine: TurnEngine | None = None,
                 bus: EventEmitter | None = None,
                 tool_strategy: ToolDispatchStrategy | None = None,
                 tool_executor: ToolExecutor | None = None):
        self._transport = transport
        self.agent = agent
        self._bus: EventEmitter = bus if bus is not None else NULL_BUS
        # No strategy wired (tests, toolless agents) → a marker strategy
        # with zero specs: plain clause streaming, nothing dispatched.
        self._tooling: ToolDispatchStrategy = (
            tool_strategy if tool_strategy is not None else MarkerToolStrategy(()))
        self._tool_executor = tool_executor
        self.stream_id: str | None = None
        self.call_id: str | None = None
        self.caller_number: str = "unknown"
        self.caller_name: str = "unknown"
        self._recorder: recording.CallRecorder | None = None

        self.messages: list[dict] = [{"role": "system", "content": agent.system_prompt}]
        self.stt = stt_factory(self._on_stt_event)
        self.tts = tts
        self._llm = llm
        self._engine = engine if engine is not None else TurnEngine(
            policy=agent.turn.engine_policy(),
            endpointer=self._pick_endpointer(),
        )

        self._endpoint_timer: asyncio.Task | None = None
        self._speak_task: asyncio.Task | None = None
        self._stt_start_task: asyncio.Task | None = None
        self._vad = webrtcvad.Vad(agent.turn.vad_aggressiveness)

        # Per-turn latency bookkeeping (reset at each CommitUserTurn) —
        # measured here in the wiring, published as event payload.
        self._turn_t0: float | None = None
        self._thinking_s: float | None = None
        self._first_audio_s: float | None = None
        self._speech_started = False
        self._last_user = ""
        self._turn_had_tools = False
        # S7 speculation: the run started at the last STT final, and the
        # session's mirror of the engine's pending-utterance accumulation
        # (so the speculative text matches what a commit would carry).
        self._speculation: _Speculation | None = None
        self._spec_accum = ""

    def _cid(self) -> str:
        return self.call_id or "unknown"

    def _pick_endpointer(self) -> Endpointer:
        delay_s = self.agent.turn.endpoint_silence_ms / 1000
        if self.agent.stt.endpointer == "provider" and self.stt.emits_endpoint:
            return ProviderEndpointer(fallback_delay_s=delay_s)
        return FixedSilenceEndpointer(delay_s=delay_s)

    @staticmethod
    def _now() -> float:
        return asyncio.get_event_loop().time()

    # ---------------------------------------------------------------- events
    async def run(self) -> None:
        """Pump transport events until the call ends. The one loop per call."""
        try:
            async for ev in self._transport.events():
                await self._dispatch(ev)
                if isinstance(ev, CallEnded):
                    break
        finally:
            await self.cleanup()

    async def _dispatch(self, ev: TransportEvent) -> None:
        if isinstance(ev, CallStarted):
            await self._on_start(ev)
        elif isinstance(ev, MediaReceived):
            await self._on_media(ev.frame)
        elif isinstance(ev, PlaybackFinished):
            # If the engine leaves AGENT_SPEAKING here, the turn's audio
            # truly finished at the carrier (post-drain, D6) — the moment
            # SpeechEnded describes.
            was_speaking = self._engine.state is TurnState.AGENT_SPEAKING
            seq = self._engine.turn_seq
            await self._execute(self._engine.playback_finished())
            if was_speaking and self._engine.state is not TurnState.AGENT_SPEAKING:
                self._bus.emit(events.SpeechEnded(call_id=self._cid(), turn_seq=seq))
        elif isinstance(ev, CallEnded):
            self._bus.emit(events.CallEnded(call_id=self._cid()))
        # OutputCleared: the engine already left AGENT_SPEAKING when it
        # emitted CancelOutput; the carrier ack carries no new information.

    async def _on_start(self, ev: CallStarted) -> None:
        self.stream_id = ev.stream_id
        self.call_id = ev.call_id
        self.caller_number = ev.caller
        log.info("Call start stream=%s call=%s", self.stream_id, self.call_id)
        self._bus.emit(events.CallStarted(
            call_id=self._cid(), caller=ev.caller, agent_id=self.agent.agent_id))
        self._recorder = recording.CallRecorder(self.call_id, config.RECORDINGS_PATH)
        # S7.3: the greeting must not wait for the recognizer's WebSocket
        # (observed ~1.2 s). The caller isn't speaking during the greeting,
        # and the adapter drops early frames gracefully, so STT connects
        # concurrently while the (cached) greeting is already playing.
        self._stt_start_task = asyncio.create_task(self.stt.start(),
                                                    name="stt-start")
        self._stt_start_task.add_done_callback(
            lambda t: t.cancelled() or t.exception() is None
            or log.error("STT start failed: %s", t.exception()))
        await self._execute(self._engine.call_started())

    async def _on_media(self, frame: AudioFrame) -> None:
        # Always forward audio to STT — the recognizer needs a continuous stream.
        await self.stt.send_audio(frame)

        # Record the frame (fire-and-forget; never blocks the audio path)
        if self._recorder:
            asyncio.create_task(self._recorder.record_caller_frame(frame.payload))

        # Normalize the frame to a speech verdict; the engine owns what the
        # verdict *means* (barge-in counting, greeting immunity).
        pcm = audio.ulaw_to_pcm16(frame.payload)
        rms_high = audio.rms(pcm) > self.agent.turn.bargein_rms_threshold
        # 20 ms frame at 8 kHz = 160 samples = 320 bytes — valid webrtcvad size
        try:
            vad_speech = self._vad.is_speech(pcm.tobytes(), 8000)
        except Exception:
            vad_speech = False
        await self._execute(self._engine.media_frame(rms_high and vad_speech))

    async def _on_stt_event(self, ev: STTEvent) -> None:
        if ev.kind == "partial":
            await self._execute(self._engine.stt_partial(self._now()))
        elif ev.kind == "final":
            log.info("STT FINAL: %s", ev.text)
            self._spec_accum = (self._spec_accum + " " + ev.text).strip()
            seq_before = self._engine.turn_seq
            await self._execute(self._engine.stt_final(ev.text))
            if self._engine.turn_seq == seq_before:
                # No commit yet — the endpoint window just opened. Spend it
                # generating the probable reply (S7).
                self._start_speculation(self._spec_accum)
        elif ev.kind == "endpoint":
            await self._execute(self._engine.stt_endpoint())
        elif ev.kind == "dead":
            # D5 alarm: the recognizer is gone beyond its reconnect budget.
            # The call continues one-way; operators get a loud fact.
            log.error("STT is dead for call %s — the call is deaf", self._cid())
            self._bus.emit(events.ProviderFailed(
                call_id=self._cid(), provider="stt",
                error="recognizer lost; reconnect budget exhausted"))

    # ------------------------------------------------------------- intents
    async def _execute(self, intents: list[Intent]) -> None:
        for intent in intents:
            if isinstance(intent, PlayGreeting):
                self.messages.append({"role": "assistant", "content": self.agent.greeting})
                self._speak_task = asyncio.create_task(self._speak_greeting())
            elif isinstance(intent, ArmEndpointTimer):
                if self._endpoint_timer and not self._endpoint_timer.done():
                    self._endpoint_timer.cancel()
                self._endpoint_timer = asyncio.create_task(
                    self._fire_endpoint(intent.generation, intent.delay_s))
            elif isinstance(intent, CancelOutput):
                await self._cancel_output(intent.turn_seq)
            elif isinstance(intent, CommitUserTurn):
                log.info("USER: %s", intent.text)
                self.messages.append({"role": "user", "content": intent.text})
                self._last_user = intent.text
                self._spec_accum = ""
                self._turn_t0 = self._now()
                self._thinking_s = None
                self._first_audio_s = None
                self._speech_started = False
                self._turn_had_tools = False
                self._bus.emit(events.ThinkingStarted(
                    call_id=self._cid(), turn_seq=intent.turn_seq))
                self._speak_task = asyncio.create_task(
                    self._generate_and_speak(intent.turn_seq, intent.play_filler))

    async def _fire_endpoint(self, generation: int, delay_s: float) -> None:
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return
        await self._execute(self._engine.endpoint_fired(generation))

    async def _cancel_output(self, turn_seq: int) -> None:
        t0 = self._now()
        task = self._speak_task
        if task and not task.done():
            task.cancel()
            # Wait for the pipeline to unwind so its history append (spoken
            # clauses only, D4) lands *before* any new turn's user message.
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._transport.clear()
        # reaction_s is the real mouth-shut latency: CancelOutput intent →
        # pipeline unwound + carrier buffer cleared.
        self._bus.emit(events.AgentInterrupted(
            call_id=self._cid(), turn_seq=turn_seq,
            reaction_s=self._now() - t0))
        log.info("Cancelled output for turn %s", turn_seq)

    # ------------------------------------------------------------ pipeline
    async def _speak_greeting(self) -> None:
        frames = 0
        try:
            frames = await self._speak(self.agent.greeting, 0)
        finally:
            await self._execute(
                self._engine.speaking_finished(0, any_audio=frames > 0))

    def _tool_ctx(self) -> ToolContext:
        return ToolContext(
            call_id=self._cid(),
            caller_number=self.caller_number,
            caller_name=self.caller_name,
            agent=self.agent,
        )

    def _dispatch_tool(self, name: str, args: dict) -> None:
        """Route a tool call from the dispatch strategy to the executor.
        Synchronous — the executor runs the tool in its own task."""
        self._turn_had_tools = True
        if self._tool_executor is None:
            log.warning("Tool call %r dropped: no executor wired", name)
            return
        self._tool_executor.dispatch(name, args, self._tool_ctx())

    async def _await_tool(self, name: str, args: dict) -> ToolOutcome:
        """The awaitable path a strategy uses for feedback tools (S4)."""
        self._turn_had_tools = True
        if self._tool_executor is None:
            log.warning("Awaited tool call %r dropped: no executor wired", name)
            return ToolOutcome(ok=False, error="no executor wired")
        return await self._tool_executor.run_and_wait(name, args, self._tool_ctx())

    # ---------------------------------------------------------- speculation
    def _start_speculation(self, text: str) -> None:
        """S7: begin generating the reply for `text` NOW, during the
        endpoint silence, so LLM time-to-first-token and first-clause
        synthesis run in time the caller is already giving us. Adoption
        (or abandonment) happens at CommitUserTurn."""
        if not config.SPECULATIVE_REPLY or not text.strip():
            return
        running = self._speculation
        if (running is not None
                and running.history_len == len(self.messages)
                and _normalize_utterance(running.text) == _normalize_utterance(text)):
            return  # same utterance (modulo punctuation): keep its head start
        self._cancel_speculation()
        spec = _Speculation(text=text, history_len=len(self.messages))
        msgs = trim_history(
            [*self.messages, {"role": "user", "content": text}],
            max_messages=self.agent.llm.history_max_messages,
            max_chars=self.agent.llm.history_max_chars)

        async def gated_await_tool(name: str, args: dict) -> ToolOutcome:
            await spec.adopted.wait()   # feedback tools wait for the commit
            return await self._await_tool(name, args)

        def on_tool(name: str, args: dict) -> None:
            # Queue while speculative; dispatch directly once adopted (the
            # generator keeps being consumed long after adoption — a
            # booking marker in the LAST clause must still fire).
            if spec.adopted.is_set():
                self._dispatch_tool(name, args)
            else:
                spec.queued_tools.append((name, args))

        spec.gen = self._tooling.run(
            self._llm, msgs, on_tool, await_tool=gated_await_tool)

        async def _prefetch() -> None:
            # Pull up to (and including) the first clause, then synthesize
            # it — the same work _first_speech would do after the commit.
            async for ev in spec.gen:
                spec.events.append(ev)
                if isinstance(ev, TurnClause):
                    spec.first_text = ev.text
                    # Live TTS has no reason to pre-synthesize a REST-style
                    # clause. We still prefetch the LLM head during the
                    # endpoint window, then feed that head into the live
                    # WebSocket once the turn commits.
                    if not self._supports_live_tts():
                        spec.first_frames = await self._synth(ev.text)
                    return

        spec.task = asyncio.create_task(_prefetch())
        self._speculation = spec

    def _cancel_speculation(self) -> None:
        if self._speculation is not None:
            self._speculation.abandon()
            self._speculation = None

    @staticmethod
    async def _adopted_stream(spec: _Speculation):
        """Settle the prefetch (it owns the generator until it finishes),
        replay everything it pulled, then continue the live generator."""
        if spec.task is not None:
            with contextlib.suppress(Exception):
                await spec.task
        for ev in spec.events:
            yield ev
        async for ev in spec.gen:
            yield ev

    def _supports_live_tts(self) -> bool:
        return callable(getattr(self.tts, "stream_text", None))

    async def _generate_and_speak(self, seq: int, play_filler: bool) -> None:
        # Live Sarvam path: LLM text and TTS audio overlap. All legacy TTS
        # adapters continue through the original clause+synth-ahead path.
        if self._supports_live_tts():
            await self._generate_and_speak_streaming(seq, play_filler)
            return

        spoken: list[str] = []
        frames = 0
        interrupted = False
        # Proto Context Compiler (M8): cap history before it reaches the
        # model, so hour-long calls can't inflate latency without bound.
        self.messages = trim_history(
            self.messages,
            max_messages=self.agent.llm.history_max_messages,
            max_chars=self.agent.llm.history_max_chars)
        # The strategy yields a typed TurnEvent stream (S2). The voice path
        # consumes Clause events and skips Token/Phase/Done for delivery:
        # delivery here is TTS audio, and the history commit stays the D4
        # spoken-clause ledger below.
        # S7: adopt the speculative run if it was for exactly this turn —
        # same committed text, same history it was built on. Otherwise the
        # speculation is abandoned (its queued tools drop, unexecuted).
        spec = self._speculation
        self._speculation = None
        adopted = (spec is not None
                   and _normalize_utterance(spec.text)
                   == _normalize_utterance(self._last_user)
                   and spec.history_len == len(self.messages) - 1)
        preloaded: dict[str, list] = {}
        if spec is not None and not adopted:
            spec.abandon()
        # spec_ready: the prefetch FINISHED before the commit — audio for
        # the first clause is in hand and the filler has nothing to mask.
        # When the commit beats the prefetch (fast endpointing), adoption
        # must NOT block on it: the filler plays while the prefetch keeps
        # running in the background.
        spec_ready = False
        if adopted:
            assert spec is not None
            spec.adopted.set()          # unblock any gated feedback round
            spec_ready = (spec.task is not None and spec.task.done()
                          and bool(spec.first_frames))
            for name, args in spec.queued_tools:
                self._dispatch_tool(name, args)
            spec.queued_tools.clear()
            gen = self._adopted_stream(spec)
            log.info("Adopted speculative reply for turn %d (%s)",
                     seq, "prefetched" if spec_ready else "in flight")
        else:
            gen = self._tooling.run(self._llm, self.messages,
                                    self._dispatch_tool,
                                    await_tool=self._await_tool)
        adopted_spec = spec if adopted else None

        # S3: while pulling clauses for TTS, mirror the stream's shape onto
        # the bus — phases as they happen, tokens as ONE aggregate at Done.
        # S4: Done.context carries the turn's feedback-tool exchanges for
        # the durable-history commit in `finally`; a TOOL phase queues the
        # agent's hold line (heard, never recorded).
        token_count = 0
        first_token_s: float | None = None
        done_context: tuple = ()

        async def _next_speech() -> tuple[str, bool] | None:
            """Next utterance to speak: (text, counts_for_history)."""
            nonlocal token_count, first_token_s, done_context
            async for ev in gen:
                if isinstance(ev, TurnClause):
                    return ev.text, True
                if isinstance(ev, turn_events.Token):
                    token_count += 1
                    if first_token_s is None and self._turn_t0 is not None:
                        first_token_s = self._now() - self._turn_t0
                elif isinstance(ev, turn_events.Phase):
                    self._bus.emit(events.PhaseChanged(
                        call_id=self._cid(), turn_seq=seq,
                        phase=ev.phase.value, detail=ev.detail))
                    if ev.phase is turn_events.TurnPhase.TOOL:
                        # Business config, not engine config: the agent may
                        # hold the floor during a slow feedback tool. The
                        # generator stays paused mid-round; the next pull
                        # resumes it, so the tool runs WHILE this plays.
                        hold = str(self.agent.tool_config.get("hold_line") or "")
                        if hold and not self._engine.is_stale(seq):
                            return hold, False
                elif isinstance(ev, turn_events.Done):
                    done_context = ev.context
                    self._bus.emit(events.TokenStreamEnded(
                        call_id=self._cid(), turn_seq=seq,
                        tokens=token_count, first_token_s=first_token_s))
            return None

        async def _advance() -> tuple[tuple[str, bool] | None, list]:
            """Pull the next utterance AND synthesize it — run while the
            previous utterance is still playing (S6 synth-ahead), so TTS
            latency hides under playback instead of gapping the reply."""
            item = await _next_speech()
            if item is None:
                return None, []
            return item, await self._synth(item[0])

        async def _first_speech() -> tuple[tuple[str, bool] | None, list]:
            # Timestamped at text availability, inside the task, so the
            # filler playing in the foreground can't pollute the
            # measurement — then synthesized here too, under the filler.
            if adopted_spec is not None and adopted_spec.task is not None:
                # Let the adopted prefetch finish (it already includes the
                # first clause's synthesis) while the filler holds the
                # floor; its output becomes ours.
                with contextlib.suppress(Exception):
                    await adopted_spec.task
                if adopted_spec.first_text and adopted_spec.first_frames:
                    preloaded[adopted_spec.first_text] = adopted_spec.first_frames
            item = await _next_speech()
            if item is not None and self._turn_t0 is not None:
                self._thinking_s = self._now() - self._turn_t0
                self._bus.emit(events.ThinkingFinished(
                    call_id=self._cid(), turn_seq=seq,
                    thinking_s=self._thinking_s))
            if item is None:
                return None, []
            cached = preloaded.pop(item[0], None)
            if cached is not None:
                return item, cached     # synthesized during speculation
            return item, await self._synth(item[0])

        # Start pulling (and synthesizing) the first clause *before* the
        # filler plays, so the filler genuinely masks LLM time-to-first-
        # token and first-clause synthesis instead of adding to them.
        first_task = asyncio.create_task(_first_speech())
        adv_task: asyncio.Task | None = None
        try:
            # The filler exists to mask thinking time; a PREFETCHED first
            # clause means there is nothing to mask — skip it and answer
            # immediately. If the prefetch is still in flight (instant
            # provider endpointing), the filler plays over it (S7).
            if play_filler and not spec_ready:
                frames += await self._speak(self.agent.turn.filler, seq)
            item, synth = await first_task
            while item is not None:
                if self._engine.is_stale(seq):
                    interrupted = True
                    return
                text, in_history = item
                # The dispatch strategy already stripped tool calls and
                # routed them; text arriving here is pure speech.
                log.info("LLM chunk: %s", text)
                # S6 synth-ahead: fetch + synthesize the NEXT utterance
                # while this one is playing.
                adv_task = asyncio.create_task(_advance())
                played = await self._play(text, synth, seq)
                frames += played
                # D4: history records what the caller actually heard — a
                # clause counts only if its audio went out (a TTS failure
                # yields zero frames and must not enter the transcript).
                if played and in_history:
                    spoken.append(text)
                item, synth = await adv_task
                adv_task = None
            # Nothing spoken and no tool fired: the pipeline failed (LLM
            # died, TTS breaker open, empty stream). Degrade audibly (M8) —
            # a scripted apology beats dead air. Never enters history: the
            # model didn't say it.
            if (not spoken and not self._turn_had_tools
                    and not self._engine.is_stale(seq)
                    and self.agent.turn.fallback_line):
                log.warning("Turn %d produced nothing; speaking fallback line", seq)
                self._bus.emit(events.FallbackSpoken(
                    call_id=self._cid(), turn_seq=seq))
                frames += await self._speak(self.agent.turn.fallback_line, seq)
        except asyncio.CancelledError:
            interrupted = True
            log.info("Reply turn %s cancelled (barge-in)", seq)
            raise
        finally:
            first_task.cancel()
            if adv_task is not None:
                adv_task.cancel()
            # S4: a settled turn's feedback-tool exchanges enter durable
            # history BEFORE the spoken reply, so the next turn's context
            # shows the model why it said what it said. Interrupted turns
            # never yield Done, so their exchanges drop with the turn.
            if done_context:
                self.messages.extend(dict(m) for m in done_context)
            if spoken:
                reply = " ".join(spoken)
                self.messages.append({"role": "assistant", "content": reply})
                log.info("PRIYA: %s", reply)
            self._bus.emit(events.TurnCompleted(
                call_id=self._cid(), turn_seq=seq,
                user_text=self._last_user, agent_text=" ".join(spoken),
                thinking_s=self._thinking_s,
                first_audio_s=self._first_audio_s,
                interrupted=interrupted))
            await self._execute(
                self._engine.speaking_finished(seq, any_audio=frames > 0))

    async def _generate_and_speak_streaming(self, seq: int, play_filler: bool) -> None:
        """True live reply path: LLM -> incremental text -> Sarvam WS -> Vobiz."""
        spoken: list[str] = []
        submitted: list[tuple[str, bool]] = []
        frames = 0
        interrupted = False
        done_context: tuple = ()
        token_count = 0
        first_token_s: float | None = None

        self.messages = trim_history(
            self.messages,
            max_messages=self.agent.llm.history_max_messages,
            max_chars=self.agent.llm.history_max_chars)

        spec = self._speculation
        self._speculation = None
        adopted = (
            spec is not None
            and _normalize_utterance(spec.text) == _normalize_utterance(self._last_user)
            and spec.history_len == len(self.messages) - 1
        )
        if spec is not None and not adopted:
            spec.abandon()

        if adopted:
            assert spec is not None
            spec.adopted.set()
            for name, args in spec.queued_tools:
                self._dispatch_tool(name, args)
            spec.queued_tools.clear()
            gen = self._adopted_stream(spec)
        else:
            gen = self._tooling.run(
                self._llm,
                self.messages,
                self._dispatch_tool,
                await_tool=self._await_tool,
            )

        async def text_source() -> AsyncIterator[str]:
            """Convert the typed turn stream into small live TTS chunks.

            We deliberately consume Token events here rather than waiting for
            Clause events. Token events are marker-guarded, so they are safe to
            speak incrementally. This is the key batch->realtime boundary:
            the Sarvam WebSocket gets roughly 30-50 characters at a time while
            the LLM is still generating the rest of the answer.
            """
            nonlocal token_count, first_token_s, done_context

            live_buf = ""
            TARGET_CHARS = 42

            def take_chunk(force: bool = False) -> str | None:
                nonlocal live_buf
                if not live_buf.strip():
                    return None
                if not force and len(live_buf) < TARGET_CHARS:
                    return None

                # Prefer punctuation or whitespace near the target so we don't
                # routinely cut in the middle of a word.
                boundary_chars = " ।?!.,;:\n"
                cut = None
                upper = min(len(live_buf), TARGET_CHARS + 20)
                for i in range(upper - 1, max(0, TARGET_CHARS - 20) - 1, -1):
                    if live_buf[i] in boundary_chars:
                        cut = i + 1
                        break

                if cut is None:
                    cut = TARGET_CHARS if len(live_buf) >= TARGET_CHARS else len(live_buf)

                chunk = live_buf[:cut].strip()
                live_buf = live_buf[cut:]
                return chunk or None

            async for ev in gen:
                if isinstance(ev, turn_events.Token):
                    token_count += 1
                    if first_token_s is None and self._turn_t0 is not None:
                        first_token_s = self._now() - self._turn_t0

                    live_buf += ev.text
                    # Drain one chunk per token event when enough text is ready.
                    while True:
                        chunk = take_chunk(force=False)
                        if chunk is None:
                            break
                        log.info("LLM live chunk: %s", chunk)
                        submitted.append((chunk, True))
                        yield chunk

                elif isinstance(ev, turn_events.Phase):
                    self._bus.emit(events.PhaseChanged(
                        call_id=self._cid(), turn_seq=seq,
                        phase=ev.phase.value, detail=ev.detail))
                    if ev.phase is turn_events.TurnPhase.TOOL:
                        # Flush normal text before the hold line so the hold
                        # doesn't leapfrog content already generated.
                        chunk = take_chunk(force=True)
                        if chunk:
                            submitted.append((chunk, True))
                            yield chunk
                        hold = str(self.agent.tool_config.get("hold_line") or "")
                        if hold and not self._engine.is_stale(seq):
                            submitted.append((hold, False))
                            yield hold

                elif isinstance(ev, turn_events.Done):
                    done_context = ev.context
                    chunk = take_chunk(force=True)
                    if chunk:
                        submitted.append((chunk, True))
                        yield chunk
                    self._bus.emit(events.TokenStreamEnded(
                        call_id=self._cid(), turn_seq=seq,
                        tokens=token_count, first_token_s=first_token_s))

        try:
            # Mark the engine as speaking before the first audio byte can
            # arrive. This arms barge-in while the TTS socket is generating.
            await self._execute(self._engine.speaking_started(seq, self._now()))

            # Do not serialize a legacy filler in front of the live TTS
            # stream. The whole point of this path is overlapping LLM→TTS→
            # carrier work; a filler here would reintroduce a blocking turn.
            audio_seen = False
            async for frame in self.tts.stream_text(text_source(), MULAW_8K):
                if self._engine.is_stale(seq):
                    interrupted = True
                    return
                if not audio_seen:
                    audio_seen = True
                    self._speech_started = True
                    if self._turn_t0 is not None:
                        self._first_audio_s = self._now() - self._turn_t0
                    self._bus.emit(events.SpeechStarted(
                        call_id=self._cid(), turn_seq=seq))
                await self._transport.play(frame)
                frames += 1
                if self._recorder:
                    asyncio.create_task(
                        self._recorder.record_agent_frame(frame.payload))

            if frames:
                await self._transport.checkpoint(f"turn-{seq}")

            if audio_seen:
                # Streaming audio is not addressable to individual LLM clauses
                # after the carrier boundary. For a normally completed turn,
                # everything the TTS stream accepted is considered spoken.
                spoken.extend(text for text, in_history in submitted if in_history)

            if (not spoken and not self._turn_had_tools
                    and not self._engine.is_stale(seq)
                    and self.agent.turn.fallback_line):
                log.warning("Turn %d live stream produced no audio; fallback", seq)
                self._bus.emit(events.FallbackSpoken(
                    call_id=self._cid(), turn_seq=seq))
                frames += await self._speak(self.agent.turn.fallback_line, seq)
        except asyncio.CancelledError:
            interrupted = True
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("live TTS pipeline failed: %s", exc)
            if (not self._turn_had_tools
                    and not self._engine.is_stale(seq)
                    and self.agent.turn.fallback_line):
                self._bus.emit(events.FallbackSpoken(
                    call_id=self._cid(), turn_seq=seq))
                frames += await self._speak(self.agent.turn.fallback_line, seq)
        finally:
            if done_context:
                self.messages.extend(dict(m) for m in done_context)
            if spoken:
                reply = " ".join(spoken)
                self.messages.append({"role": "assistant", "content": reply})
                log.info("PRIYA: %s", reply)
            self._bus.emit(events.TurnCompleted(
                call_id=self._cid(), turn_seq=seq,
                user_text=self._last_user, agent_text=" ".join(spoken),
                thinking_s=(self._now() - self._turn_t0) if self._turn_t0 else None,
                first_audio_s=self._first_audio_s,
                interrupted=interrupted))
            await self._execute(
                self._engine.speaking_finished(seq, any_audio=frames > 0))

    async def _synth(self, text: str) -> list[AudioFrame]:
        """Synthesize one utterance fully into memory (S6). Failures come
        back as an empty list — the caller's zero-frame handling (fallback
        line, D4 exclusion) is unchanged from the streaming days. A clause
        of frames is ~100 KB; buffering one ahead is the latency win.
        (Both current TTS providers are REST — audio arrives all at once
        anyway; revisit if a streaming-synthesis provider lands.)"""
        frames: list[AudioFrame] = []
        try:
            async for frame in self.tts.synthesize(text, MULAW_8K):
                frames.append(frame)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.error("synth error: %s", e)
        return frames

    async def _play(self, text: str, frames_in: list[AudioFrame],
                    seq: int) -> int:
        """Play pre-synthesized frames; returns frames actually sent."""
        log.info("SPEAK called: %s", text[:60])
        await self._execute(self._engine.speaking_started(seq, self._now()))
        frame_count = 0
        try:
            for frame in frames_in:
                frame_count += 1
                if frame_count == 1 and not self._speech_started:
                    # First audible output of this turn (filler counts —
                    # this is when the caller hears the agent respond).
                    self._speech_started = True
                    if self._turn_t0 is not None:
                        self._first_audio_s = self._now() - self._turn_t0
                    self._bus.emit(events.SpeechStarted(
                        call_id=self._cid(), turn_seq=seq))
                await self._transport.play(frame)
                if self._recorder:
                    asyncio.create_task(self._recorder.record_agent_frame(frame.payload))
            log.info("SPEAK done: %d frames sent", frame_count)
            # The transport no-ops this before the call starts.
            await self._transport.checkpoint(f"turn-{seq}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("speak error: %s", e)
        return frame_count

    async def _speak(self, text: str, seq: int) -> int:
        """Synthesize and play one utterance (greeting, filler, fallback —
        callers with nothing useful to overlap); returns frames sent."""
        return await self._play(text, await self._synth(text), seq)

    # --------------------------------------------------------------- teardown
    async def cleanup(self) -> None:
        log.info("Call cleanup stream=%s", self.stream_id)
        self._cancel_speculation()
        # A connect still in flight must not complete after hangup — a
        # recognizer socket nobody owns would leak (S7.3).
        for t in (self._endpoint_timer, self._speak_task, self._stt_start_task):
            if t and not t.done():
                t.cancel()
        await self.stt.close()
        # Finalize the recording (async, fire-and-forget)
        if self._recorder:
            asyncio.create_task(self._recorder.finalize())
        self._bus.emit(events.SessionClosed(call_id=self._cid()))