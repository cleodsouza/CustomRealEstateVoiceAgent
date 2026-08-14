"""agents/priya_tools.py — Priya's business tools (successor to booking.py).

Site-visit booking and WhatsApp brochure delivery are N Rose Developers'
business logic, not runtime code (M7). The runtime executes these through
the ToolRegistry/ToolExecutor and knows nothing about real estate; per-
agent knobs (bookings path, brochure URL/filename/caption) come from the
agent record's tool_config, so the same handlers serve any future agent.

Google Calendar integration (M9): when a booking is made, add the appointment
to the tenant's Google Calendar. Each agent stores its service_account_key
and calendar_id in tool_config for multi-tenant support.

PostCallBrochureSender is an event subscriber (Article VII) that listens
for CallEnded and sends the brochure automatically via WhatsApp.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

import config
from providers.calendar.google_calendar import GoogleCalendarClient
from runtime import agent_registry, events
from runtime.tools import ToolContext, ToolRegistry, ToolSpec

log = logging.getLogger("agents.priya")


# ------------------------------------------------------------------ booking
def _append_line(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_placeholder_key(key) -> bool:
    """True when tool_config still carries the sample service-account JSON
    (values like 'your-project-id' / '...' from the template)."""
    if not isinstance(key, dict):
        return False
    pk = str(key.get("private_key", ""))
    project = str(key.get("project_id", ""))
    return ("\n...\n" in pk or pk.strip().endswith("...")
            or "-----\n...\n-----" in pk or project.startswith("your-")
            or key.get("client_id") == "...")


async def _add_to_google_calendar(ctx: ToolContext, args: dict) -> None:
    """Add the site visit to the tenant's Google Calendar (M9).

    Expects tool_config to have:
    - service_account_key (dict): Full Google service account JSON
    - calendar_id (str): Calendar ID or "primary"

    Falls back silently if calendar is not configured (some tenants may not
    use Google Calendar yet).
    """
    tool_cfg = ctx.agent.tool_config
    if not tool_cfg.get("service_account_key") or not tool_cfg.get("calendar_id"):
        log.debug("Google Calendar not configured for agent %s", ctx.agent.agent_id)
        return
    if _is_placeholder_key(tool_cfg["service_account_key"]):
        # Template credentials from the sample agent file — treat exactly
        # like "not configured" instead of failing deep in PEM parsing on
        # every booking ("Unable to load PEM file...").
        log.warning("Google Calendar for agent %s still has placeholder "
                    "service_account_key; skipping calendar sync. Paste the "
                    "real service-account JSON into tool_config and share "
                    "the calendar with its client_email.",
                    ctx.agent.agent_id)
        return

    try:
        svc_acct_key = tool_cfg["service_account_key"]
        if isinstance(svc_acct_key, str):
            svc_acct_key = json.loads(svc_acct_key)

        client = GoogleCalendarClient(svc_acct_key, tool_cfg["calendar_id"])

        # Parse day and time from the booking args
        # day might be "Sat" (weekday) or "Saturday", time might be "4pm" or "16:00"
        # For now, we'll construct a reasonable datetime and let the calendar
        # UX handle any manual adjustments if needed.
        day_str = args.get("day", "")
        time_str = args.get("time", "")
        name = args.get("name", "Guest")

        # Simple parse: "Sat" -> index in week, "4pm" -> 16:00
        # This is a basic implementation; production would use dateutil.parser
        flat = _normalize_flat(args.get("flat", ""))
        event_title = (f"Site Visit – {ctx.agent.tenant_id} – {name}"
                       + (f" ({flat})" if flat else ""))
        event_desc = f"Caller: {name}\nPhone: {ctx.caller_number}\nCall ID: {ctx.call_id}"

        # Construct an approximate datetime. If we can't parse precisely, use
        # a placeholder in the description and let the agent handle it.
        try:
            # Attempt basic weekday parsing
            weekdays = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
            day_lower = day_str.lower()[:3]
            weekday_idx = weekdays.get(day_lower)

            if weekday_idx is not None:
                # Find the next occurrence of this weekday
                today = datetime.now()
                days_ahead = (weekday_idx - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7  # If it's today, book for next week
                target_date = today.replace(hour=0, minute=0, second=0,
                                          microsecond=0) + timedelta(days=days_ahead)

                # Parse time: "4pm", "4:00pm", "16:00"
                hour = 14  # default 2 PM
                if "pm" in time_str.lower():
                    if ":" in time_str:
                        h = int(time_str.split(":")[0])
                        hour = h if h == 12 else h + 12
                    else:
                        h = int("".join(c for c in time_str if c.isdigit()))
                        hour = h if h == 12 else h + 12
                elif "am" in time_str.lower():
                    if ":" in time_str:
                        h = int(time_str.split(":")[0])
                        hour = h % 12
                    else:
                        h = int("".join(c for c in time_str if c.isdigit()))
                        hour = h % 12

                start_time = target_date.replace(hour=hour, minute=0)
            else:
                # Couldn't parse day; use next Tuesday as default
                today = datetime.now()
                days_ahead = (1 - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                start_time = today.replace(hour=14, minute=0,
                                          second=0, microsecond=0) + timedelta(days=days_ahead)

        except (ValueError, AttributeError) as e:
            log.warning("Couldn't parse booking time %s/%s: %s; using default",
                       day_str, time_str, e)
            start_time = datetime.now().replace(hour=14, minute=0, second=0,
                                                microsecond=0) + timedelta(days=7)

        event = await client.create_event(
            title=event_title,
            start_time=start_time,
            duration_minutes=30,
            description=event_desc,
            attendee_email=""  # Caller's email not available; agent can update
        )
        log.info("Google Calendar event created: %s (event_id=%s)",
                event.get("id"), event.get("summary"))
    except Exception as e:  # noqa: BLE001
        log.error("Failed to add booking to Google Calendar: %s", e)
        # Don't raise — booking JSONL still gets saved, calendar is a nice-to-have


def _normalize_flat(raw: str) -> str:
    """Canonicalize the flat type for the bookings table: '2 BHK', '2bhk',
    'दो BHK' all become '2BHK' (same for 3). Anything unrecognized passes
    through stripped — the record keeps what the model said."""
    v = str(raw or "").strip()
    low = v.casefold().replace(" ", "")
    if low.startswith(("2", "दो", "two")) and "bhk" in low:
        return "2BHK"
    if low.startswith(("3", "तीन", "three")) and "bhk" in low:
        return "3BHK"
    return v


async def book_site_visit(ctx: ToolContext, args: dict) -> None:
    """Book a site visit: save to JSONL and add to Google Calendar (M9).

    The booking record includes the caller's actual name (from args), not
    the undefined caller_name field. The JSONL serves as the source of truth
    for bookings; Google Calendar is a read-model for the agent's team.
    """
    # Use the name extracted from the LLM's [[BOOK ...]] marker
    visitor_name = args.get("name", "Unknown")

    record = {
        "ts": datetime.now().isoformat(),
        "call_id": ctx.call_id,
        "caller_phone": ctx.caller_number,
        "caller": ctx.caller_name,
        "visitor_name": visitor_name,  # The name the LLM extracted
        **args,
        "flat": _normalize_flat(args.get("flat", "")),
    }
    path = Path(ctx.agent.tool_config.get("bookings_path", "bookings.jsonl"))
    await asyncio.to_thread(_append_line, path, record)
    log.info("Booking saved: %s", record)

    # Try to add to Google Calendar (multi-tenant, per-agent)
    await _add_to_google_calendar(ctx, args)


# ----------------------------------------------------------------- brochure
def _brochure_payload(to_number: str, cfg: dict) -> dict:
    return {
        "channel_id": config.VOBIZ_WHATSAPP_CHANNEL_ID,
        "to": to_number,
        "type": "document",
        "document": {
            "url": cfg["brochure_url"],
            "filename": cfg.get("brochure_filename", "brochure.pdf"),
            "caption": cfg.get("brochure_caption", ""),
        },
    }


async def send_brochure(ctx: ToolContext, args: dict) -> None:
    """Send the project brochure to the caller's WhatsApp via Vobiz.
    Failures raise — the ToolExecutor owns retry and the ToolFailed event.

    WhatsApp messaging is a separate Vobiz resource tree from voice (no
    /Account/{auth_id}/ prefix); it's scoped by channel_id instead."""
    if not config.VOBIZ_WHATSAPP_CHANNEL_ID:
        raise RuntimeError(
            "VOBIZ_WHATSAPP_CHANNEL_ID is not set — create/find the WhatsApp "
            "channel in the Vobiz console and set its id in .env")
    url = f"{config.VOBIZ_API_BASE}/messaging/messages"
    headers = {
        "Content-Type": "application/json",
        "X-Auth-ID": config.VOBIZ_AUTH_ID,
        "X-Auth-Token": config.VOBIZ_AUTH_TOKEN,
    }
    payload = _brochure_payload(ctx.caller_number, ctx.agent.tool_config)
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(url, json=payload, headers=headers)
        log.info("Brochure -> %s (HTTP %s)", ctx.caller_number, r.status_code)
        r.raise_for_status()


# ---------------------------------------------------- post-call workflow (M7)
@dataclass
class PostCallBrochureSender:
    """Event subscriber (Article VII) that sends the brochure automatically
    when a call ends. Maintains a cache of (call_id -> caller_number, agent_id)
    across CallStarted/CallEnded, then calls send_brochure directly (not via
    ToolExecutor, since the call session has ended and the tool registry context
    is no longer available)."""

    _calls: dict[str, tuple[str, str]] = field(default_factory=dict)

    async def __call__(self, event: events.Event) -> None:
        if isinstance(event, events.CallStarted):
            # Store the call info for later
            self._calls[event.call_id] = (event.caller, event.agent_id)
        elif isinstance(event, events.CallEnded):
            # Send brochure if this call has recorded info
            info = self._calls.pop(event.call_id, None)
            if not info:
                return
            caller_number, agent_id = info
            try:
                agent = agent_registry.resolve(agent_id)
                tool_config = agent.tool_config
                # Don't send if brochure_url is not configured
                if not tool_config.get("brochure_url"):
                    return
                # Create a minimal context to call send_brochure
                ctx = ToolContext(
                    call_id=event.call_id,
                    caller_number=caller_number,
                    caller_name="",  # not needed for WhatsApp send
                    agent=agent,
                )
                await send_brochure(ctx, {})
            except Exception as e:  # noqa: BLE001
                log.error("Failed to send post-call brochure for %s: %s",
                         event.call_id, e)


# ------------------------------------------------------------- registration
def register(registry: ToolRegistry) -> None:
    registry.register(ToolSpec(
        name="book_site_visit",
        description="Book a site-visit appointment for the caller at "
                    "Northern Heights.",
        parameters={
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "Day of the visit"},
                "time": {"type": "string", "description": "Time of the visit"},
                "name": {"type": "string", "description": "Caller's name"},
                "flat": {"type": "string",
                         "description": "Flat type chosen (2BHK or 3BHK)"},
            },
            "required": ["day", "time", "name"],
        },
        handler=book_site_visit,
        owner="n-rose-developers",
        marker="BOOK",
        timeout_s=5.0,
        retries=1,
    ))
    registry.register(ToolSpec(
        name="send_brochure",
        description="Send the Northern Heights brochure to the caller's "
                    "WhatsApp.",
        parameters={"type": "object", "properties": {}},
        handler=send_brochure,
        owner="n-rose-developers",
        marker="BROCHURE",
        timeout_s=10.0,
        retries=1,
    ))
