"""Priya's business tools (agents/priya_tools.py) — successor to the
deleted test_booking.py. Bookings write JSONL off the loop; brochure
payload comes from the agent's tool_config."""
import dataclasses
import json

import pytest

import config
from agents import priya_tools
from runtime import agent_registry
from runtime.tools import ToolContext, ToolRegistry


def make_ctx(tool_config=None):
    agent = agent_registry.resolve()
    if tool_config is not None:
        agent = dataclasses.replace(agent, tool_config=tool_config)
    return ToolContext(call_id="c1", caller_number="+911234567890",
                       caller_name="Rahul", agent=agent)


def test_register_installs_both_specs_with_markers():
    reg = ToolRegistry()
    priya_tools.register(reg)

    book = reg.get("book_site_visit")
    brochure = reg.get("send_brochure")
    assert book is not None and brochure is not None
    assert book.marker == "BOOK"
    assert brochure.marker == "BROCHURE"
    assert book.owner == "n-rose-developers"
    assert set(book.parameters["required"]) == {"day", "time", "name"}


async def test_book_site_visit_appends_jsonl(tmp_path):
    store = tmp_path / "bookings.jsonl"
    ctx = make_ctx(tool_config={"bookings_path": str(store)})

    await priya_tools.book_site_visit(ctx, {"day": "Sunday", "time": "4pm"})
    await priya_tools.book_site_visit(ctx, {"day": "Monday", "time": "11am"})

    records = [json.loads(line) for line in
               store.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["call_id"] == "c1"
    assert records[0]["caller"] == "Rahul"
    assert records[0]["day"] == "Sunday"
    assert "ts" in records[0]


async def test_book_site_visit_raises_on_write_failure(tmp_path):
    # A directory in place of the file: the error must propagate — the
    # ToolExecutor owns the catch and the ToolFailed event.
    ctx = make_ctx(tool_config={"bookings_path": str(tmp_path)})
    try:
        await priya_tools.book_site_visit(ctx, {"day": "Sunday"})
    except OSError:
        pass
    else:
        raise AssertionError("expected OSError")


def test_brochure_payload_comes_from_tool_config(monkeypatch):
    monkeypatch.setattr(config, "VOBIZ_WHATSAPP_CHANNEL_ID", "ch_123")
    payload = priya_tools._brochure_payload("+919999999999", {
        "brochure_url": "https://cdn.example.com/b.pdf",
        "brochure_filename": "NH.pdf",
        "brochure_caption": "Northern Heights",
    })
    # Vobiz's WhatsApp API scopes messages by channel_id (its own /messaging
    # resource, not the voice Account/{auth_id} path) and names the media
    # field "url", not "link".
    assert payload == {
        "channel_id": "ch_123",
        "to": "+919999999999",
        "type": "document",
        "document": {"url": "https://cdn.example.com/b.pdf",
                     "filename": "NH.pdf", "caption": "Northern Heights"},
    }


async def test_send_brochure_raises_without_channel_id(monkeypatch):
    monkeypatch.setattr(config, "VOBIZ_WHATSAPP_CHANNEL_ID", "")
    ctx = make_ctx(tool_config={"brochure_url": "https://cdn.example.com/b.pdf"})
    with pytest.raises(RuntimeError, match="VOBIZ_WHATSAPP_CHANNEL_ID"):
        await priya_tools.send_brochure(ctx, {})


async def test_send_brochure_posts_to_messaging_endpoint(monkeypatch):
    monkeypatch.setattr(config, "VOBIZ_WHATSAPP_CHANNEL_ID", "ch_123")
    ctx = make_ctx(tool_config={"brochure_url": "https://cdn.example.com/b.pdf"})

    calls = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

    class FakeAsyncClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json, headers):
            calls.append((url, json, headers))
            return FakeResponse()

    monkeypatch.setattr(priya_tools.httpx, "AsyncClient", FakeAsyncClient)
    await priya_tools.send_brochure(ctx, {})

    assert len(calls) == 1
    url, payload, headers = calls[0]
    # No /Account/{auth_id}/ prefix — WhatsApp is a sibling /messaging
    # resource on Vobiz, scoped by channel_id in the body instead.
    assert url == f"{config.VOBIZ_API_BASE}/messaging/messages"
    assert payload["channel_id"] == "ch_123"
    assert payload["to"] == "+911234567890"
    assert headers["X-Auth-ID"] == config.VOBIZ_AUTH_ID


def test_priya_record_lists_her_tools():
    agent = agent_registry.resolve()
    assert agent.tools == ("book_site_visit", "send_brochure")
    assert "brochure_url" in agent.tool_config
    assert agent.llm.tool_dispatch == "marker"  # inherited engine default


async def test_duplicate_booking_is_suppressed(tmp_path):
    store = tmp_path / "bookings.jsonl"
    ctx = make_ctx(tool_config={"bookings_path": str(store)})

    first = await priya_tools.book_site_visit(
        ctx, {"day": "Sunday", "time": "2pm", "name": "Rahul", "flat": "2 BHK"})
    second = await priya_tools.book_site_visit(
        ctx, {"day": "Sunday", "time": "2pm", "name": "Rahul", "flat": "2BHK"})

    records = [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert first["status"] == "created"
    assert second["status"] == "already_confirmed"


async def test_booking_records_normalized_flat_type(tmp_path):
    store = tmp_path / "bookings.jsonl"
    ctx = make_ctx(tool_config={"bookings_path": str(store)})

    await priya_tools.book_site_visit(
        ctx, {"day": "Sunday", "time": "4pm", "name": "Rahul", "flat": "2 BHK"})
    await priya_tools.book_site_visit(
        ctx, {"day": "Monday", "time": "11am", "name": "Meera", "flat": "दो BHK"})
    await priya_tools.book_site_visit(
        ctx, {"day": "Tuesday", "time": "1pm", "name": "Amit", "flat": "3bhk"})
    await priya_tools.book_site_visit(
        ctx, {"day": "Wednesday", "time": "2pm", "name": "Sara"})  # model omitted it

    records = [json.loads(line) for line in
               store.read_text(encoding="utf-8").splitlines()]
    assert [r["flat"] for r in records] == ["2BHK", "2BHK", "3BHK", ""]
    assert records[0]["visitor_name"] == "Rahul"


def test_flat_normalization_shapes():
    n = priya_tools._normalize_flat
    assert n("2BHK") == n("2 bhk") == n("दो BHK") == n("two bhk") == "2BHK"
    assert n("3 BHK") == n("तीन bhk") == "3BHK"
    assert n("") == ""
    assert n("penthouse") == "penthouse"   # unrecognized passes through