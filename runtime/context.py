"""runtime/context.py — conversation-context budgeting (M8).

The proto Context Compiler: today it only caps history growth so a long
call can't inflate LLM latency and cost without bound (the unbounded-
growth hole from the redesign). Oldest turns are evicted whole; the
system prompt always survives. M12 replaces eviction with summarization
and templated prompt assembly behind this same seam.
"""
from __future__ import annotations


def trim_history(messages: list[dict], *, max_messages: int,
                 max_chars: int) -> list[dict]:
    """Return messages within budget: messages[0] (the system prompt) plus
    the newest tail. Both budgets apply to the tail only — the system
    prompt is never counted against them, never evicted.

    Tool exchanges evict atomically (S4): eviction runs oldest-first, so
    an assistant tool-call message always evicts before its result; a
    role="tool" message left at the head of the tail is an orphan whose
    call is gone, and OpenAI-format context rejects it — drop it too.
    (Content may be None on assistant tool-call messages; count it as 0.)"""
    if not messages:
        return messages
    system, tail = messages[0], list(messages[1:])

    def _over() -> bool:
        return bool(tail) and (
            len(tail) > max_messages
            or sum(len(m.get("content") or "") for m in tail) > max_chars
        )

    while _over():
        tail.pop(0)
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)   # orphaned result: its tool-call partner is gone
    while tail and tail[0].get("role") == "tool":
        tail.pop(0)       # never hand the model a headless tool result
    return [system, *tail]
