"""S5: transport independence, proven.

A chat consumer of the SAME TurnEvent generator the voice session pulls:
it renders Tokens as they stream (live typing), shows Phase as status,
and commits Done.full_text + Done.context as its history. No TTS, no
Turn Engine, no session — Article II made concrete: a new channel is a
new consumer of the pipeline, and nothing else.
"""
import json

from runtime.tools import MarkerToolStrategy, ToolOutcome, ToolSpec
from runtime.turn_events import Clause, Done, Phase, Token, TurnPhase
from runtime.types import LLMDelta


async def _h(ctx, args):
    return None


class ChatTurn:
    """Minimal text-channel consumer: token-granular delivery, Done-based
    commit. What a WhatsApp/browser-chat transport's turn loop would do."""

    def __init__(self):
        self.rendered: list[str] = []   # incremental UI updates
        self.status: list[str] = []     # "typing…", "checking check_slots…"
        self.history: list[dict] = []   # durable commit, once, at Done
        self.clauses_consumed = 0

    async def consume(self, gen) -> None:
        async for ev in gen:
            if isinstance(ev, Token):
                self.rendered.append(ev.text)
            elif isinstance(ev, Phase):
                self.status.append(ev.phase.value)
            elif isinstance(ev, Done):
                self.history.extend(ev.context)
                self.history.append(
                    {"role": "assistant", "content": ev.full_text})
            elif isinstance(ev, Clause):
                self.clauses_consumed += 1  # a chat channel never needs these


class MultiRoundLLM:
    def __init__(self, rounds):
        self._rounds = [list(r) for r in rounds]
        self.requests: list[list] = []

    async def stream(self, messages, tools=None):
        self.requests.append(list(messages))
        for d in self._rounds.pop(0):
            yield d


async def test_chat_consumer_streams_tokens_and_commits_on_done():
    spec = ToolSpec(name="check_slots", description="d",
                    parameters={"type": "object"}, handler=_h, owner="test",
                    marker="CHECK", feedback=True)
    strategy = MarkerToolStrategy([spec])
    llm = MultiRoundLLM([
        [LLMDelta(text="ठीक है, "), LLMDelta(text="देखती हूं। "),
         LLMDelta(text="[[CHECK day=Sun]]")],
        [LLMDelta(text="Sunday 4 बजे free है।")],
    ])

    async def await_tool(name, args):
        return ToolOutcome(ok=True, result={"slots": ["4pm"]})

    chat = ChatTurn()
    await chat.consume(strategy.run(
        llm, [{"role": "system", "content": "S"}],
        lambda n, a: None, await_tool))

    # Live rendering happened token-by-token, marker never visible.
    assert len(chat.rendered) >= 3
    assert "".join(chat.rendered).strip() == "ठीक है, देखती हूं। Sunday 4 बजे free है।"
    assert all("CHECK" not in t and "[[" not in t for t in chat.rendered)

    # Status line saw the full phase arc, tool round included.
    assert chat.status == ["generating", "tool", "resuming", "done"]

    # One commit at Done: the tool exchange, then the final message —
    # identical semantics to the voice session's durable history, derived
    # from the same events.
    assert chat.history[-1]["content"] == "".join(chat.rendered).strip()
    assert chat.history[0]["content"] == "[[CHECK day=Sun]]"
    assert json.loads(
        chat.history[1]["content"].split("]] ", 1)[1])["ok"] is True

    # The voice path's delivery unit flowed past unused — same generator,
    # different granularity per transport.
    assert chat.clauses_consumed >= 1
