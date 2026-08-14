"""Tests for providers/llm/openai_compat.py with an injected fake client."""
from types import SimpleNamespace

from providers.llm.openai_compat import OpenAICompatLLM
from runtime.types import LLMDelta, ToolCallRequest


def _chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def _tc_chunk(index, name=None, arguments=""):
    """One streamed tool-call fragment, OpenAI wire shape."""
    tc = SimpleNamespace(index=index,
                         function=SimpleNamespace(name=name, arguments=arguments))
    delta = SimpleNamespace(content=None, tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class FakeStream:
    def __init__(self, parts):
        self._parts = list(parts)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._parts:
            part = self._parts.pop(0)
            # Strings/None are text deltas; prebuilt chunks pass through.
            return part if isinstance(part, SimpleNamespace) else _chunk(part)
        raise StopAsyncIteration


def make_adapter(parts, seen_kwargs):
    async def create(**kwargs):
        seen_kwargs.update(kwargs)
        return FakeStream(parts)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return OpenAICompatLLM(
        base_url="http://unused", api_key="unused", model="test-model",
        temperature=0.4, client=fake_client,
    )


async def test_yields_deltas_and_passes_request_params():
    seen: dict = {}
    adapter = make_adapter(["एक", " दो"], seen)
    messages = [{"role": "user", "content": "hi"}]

    out = [d async for d in adapter.stream(messages)]

    assert out == [LLMDelta(text="एक"), LLMDelta(text=" दो")]
    assert seen["model"] == "test-model"
    assert seen["temperature"] == 0.4
    assert seen["stream"] is True
    assert seen["max_tokens"] == 160
    assert seen["messages"] is messages


async def test_none_and_empty_deltas_are_skipped():
    adapter = make_adapter([None, "ठीक", None, ""], {})
    out = [d.text async for d in adapter.stream([])]
    assert out == ["ठीक"]


async def test_tools_kwarg_only_sent_when_provided():
    seen: dict = {}
    adapter = make_adapter(["ok"], seen)
    [d async for d in adapter.stream([])]
    assert "tools" not in seen

    seen.clear()
    payload = [{"type": "function", "function": {"name": "t"}}]
    adapter = make_adapter(["ok"], seen)
    [d async for d in adapter.stream([], tools=payload)]
    assert seen["tools"] is payload


async def test_fragmented_tool_call_is_assembled_and_yielded_once():
    # Name in the first fragment; argument JSON split across three.
    adapter = make_adapter([
        "ठीक है।",
        _tc_chunk(0, name="book_site_visit", arguments='{"da'),
        _tc_chunk(0, arguments='y": "Sun'),
        _tc_chunk(0, arguments='day"}'),
    ], {})

    out = [d async for d in adapter.stream([], tools=[{}])]

    assert out[0] == LLMDelta(text="ठीक है।")
    assert out[1:] == [LLMDelta(tool_call=ToolCallRequest(
        name="book_site_visit", args={"day": "Sunday"}))]


async def test_malformed_tool_call_arguments_are_dropped():
    adapter = make_adapter([
        "ok",
        _tc_chunk(0, name="book_site_visit", arguments='{"day": broken'),
    ], {})
    out = [d async for d in adapter.stream([], tools=[{}])]
    # Text still flows; the unparseable call never reaches the runtime.
    assert out == [LLMDelta(text="ok")]


async def test_empty_arguments_become_empty_dict():
    adapter = make_adapter([_tc_chunk(0, name="send_brochure")], {})
    out = [d async for d in adapter.stream([], tools=[{}])]
    assert out == [LLMDelta(tool_call=ToolCallRequest(name="send_brochure", args={}))]


# --------------------------------------------------- eager assembly (S1)
async def test_tool_call_yields_before_trailing_text():
    """A completed call is yielded the moment text resumes — the pipeline's
    halt/resume loop needs the call mid-stream, not at stream end."""
    adapter = make_adapter([
        _tc_chunk(0, name="check_slots", arguments='{"day": "Sunday"}'),
        "उस दिन",
        " समय है।",
    ], {})
    out = [d async for d in adapter.stream([], tools=[{}])]
    assert out == [
        LLMDelta(tool_call=ToolCallRequest(name="check_slots", args={"day": "Sunday"})),
        LLMDelta(text="उस दिन"),
        LLMDelta(text=" समय है।"),
    ]


async def test_second_index_closes_first_tool_call():
    adapter = make_adapter([
        _tc_chunk(0, name="check_slots", arguments='{"day": "Sun"}'),
        _tc_chunk(1, name="send_brochure", arguments='{}'),
    ], {})
    out = [d async for d in adapter.stream([], tools=[{}])]
    assert out == [
        LLMDelta(tool_call=ToolCallRequest(name="check_slots", args={"day": "Sun"})),
        LLMDelta(tool_call=ToolCallRequest(name="send_brochure", args={})),
    ]


async def test_fragmented_call_not_flushed_by_same_index_fragments():
    """Fragments of the SAME call must keep accumulating, never flush early."""
    adapter = make_adapter([
        _tc_chunk(0, name="book_site_visit", arguments='{"da'),
        _tc_chunk(0, arguments='y": "Sunday"}'),
    ], {})
    out = [d async for d in adapter.stream([], tools=[{}])]
    assert out == [LLMDelta(tool_call=ToolCallRequest(
        name="book_site_visit", args={"day": "Sunday"}))]


# ------------------------------------------------- extra_body self-heal (S7)
def make_adapter_kw(parts, seen_calls, *, reject_extra=False, **adapter_kw):
    class Rejected(Exception):
        status_code = 400

    async def create(**kwargs):
        seen_calls.append(kwargs)
        if reject_extra and "extra_body" in kwargs:
            raise Rejected("unknown field")
        return FakeStream(list(parts))

    fake_client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    return OpenAICompatLLM(
        base_url="http://unused", api_key="unused", model="test-model",
        temperature=0.4, client=fake_client, **adapter_kw)


async def test_extra_body_is_sent_when_configured():
    seen: list = []
    adapter = make_adapter_kw(["ok"], seen,
                              extra_body={"reasoning_effort": "none"})
    [d async for d in adapter.stream([])]
    assert seen[0]["extra_body"] == {"reasoning_effort": "none"}


async def test_rejected_extra_body_self_heals_and_stays_off():
    seen: list = []
    adapter = make_adapter_kw(["ok"], seen, reject_extra=True,
                              extra_body={"reasoning_effort": "none"})
    out = [d.text async for d in adapter.stream([])]
    assert out == ["ok"]                       # healed within the request
    assert "extra_body" in seen[0] and "extra_body" not in seen[1]

    [d async for d in adapter.stream([])]
    assert "extra_body" not in seen[2]         # remembered: off for good
    assert len(seen) == 3
