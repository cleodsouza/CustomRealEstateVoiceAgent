# Google Calendar Integration (M9)

## Overview

M9 adds automatic Google Calendar integration for site-visit bookings. When a caller agrees to a visit, the appointment is simultaneously:
1. **Persisted** in `bookings.jsonl` (the source of truth)
2. **Indexed** on the tenant's Google Calendar (read-model for the agent's team)

This is a **multi-tenant feature**: each company (tenant) maintains their own Google Calendar via credentials stored in the agent's `tool_config`.

---

## Architecture

### Multi-Tenant by Design

- Each agent record (e.g., `agents/priya.json`) carries its own `calendar_id` and `service_account_key` in `tool_config`
- Multiple agents (different companies) can run side-by-side with independent calendars
- Calendar integration is **optional**: if `service_account_key` is absent, bookings still save to JSONL

### Booking Flow (M9)

```
LLM emits [[BOOK day=Sat time=4pm name=Rohan]]
                    │
                    ▼
        _dispatch_tool("book_site_visit", args)
                    │
                    ▼
        book_site_visit(ctx, args)
                    │
                    ├─→ Save to JSONL (async, worker thread)
                    │   - ts: ISO datetime
                    │   - call_id: unique call identifier
                    │   - caller_phone: from ctx.caller_number
                    │   - visitor_name: from args["name"]
                    │
                    └─→ _add_to_google_calendar(ctx, args)
                        - Parse day/time from args
                        - Create calendar event
                        - Log event ID or graceful failure
```

### Fixed: The "unknown" Values Issue (M9)

**Problem (M8):** The booking record saved `"caller": "unknown"` because `session.caller_name` was never populated from the transport layer.

**Solution (M9):**
- Renamed booking field from `"caller"` to `"caller_phone"` (the phone number from Vobiz)
- Added `"visitor_name"` from the LLM's marker `[[BOOK ... name=Rohan ...]]`
- This separates *transport identity* (phone number) from *visitor identity* (name the caller told Priya)
- Both are now captured correctly

**Migration:** Existing bookings.jsonl records with `"caller": "unknown"` remain as-is; new bookings use the correct fields.

---

## Setup

### Step 1: Create a Google Cloud Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (or use an existing one)
3. Enable the **Google Calendar API**:
   - APIs & Services → Library
   - Search "Google Calendar API"
   - Click Enable
4. Create a service account:
   - APIs & Services → Credentials → Create Credentials → Service Account
   - Give it a name (e.g., "Northern Heights Booking Agent")
   - Grant the role **Editor** (or a custom role with `calendar.events.create`)
   - Create a key: JSON format → download
5. This JSON file is the `service_account_key` value

### Step 2: Configure the Agent

Edit your agent record (e.g., `agents/priya.json`):

```json
{
  "agent_id": "priya",
  "tenant_id": "n-rose-developers",
  ...
  "tool_config": {
    "bookings_path": "bookings.jsonl",
    "calendar_id": "primary",
    "service_account_key": {
      "type": "service_account",
      "project_id": "your-gcp-project-id",
      "private_key_id": "...",
      "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
      "client_email": "your-service-account@your-project-id.iam.gserviceaccount.com",
      ...
    }
  }
}
```

**Key fields:**
- `calendar_id`: Use `"primary"` for the service account's own calendar, or a specific calendar ID you've shared with the service account
- `service_account_key`: The full downloaded JSON object

### Step 3: Verify Access

The service account must have:
1. **Google Calendar API enabled** in the project
2. **Editing permissions** on the target calendar (if not "primary")
   - Share the calendar with the service account email and grant "Make changes to events"

### Step 4: Test

Make a test call and trigger a booking:
1. Start the runtime: `uvicorn server:app --host 0.0.0.0 --port 8000`
2. Place an inbound or outbound call
3. Say something like "Yes, I want to visit on Saturday at 4 PM" → Priya books it
4. Check the logs for:
   - `"Booking saved: ..."` (JSONL)
   - `"Google Calendar event created: ..."` (calendar)
5. Verify the event appears in Google Calendar

---

## Multi-Tenant Example

To run two agents with separate calendars on one runtime:

**`agents/priya.json`** (N Rose Developers):
```json
{
  "agent_id": "priya",
  "tenant_id": "n-rose-developers",
  "tool_config": {
    "calendar_id": "primary",
    "service_account_key": { "...": "N Rose service account" }
  }
}
```

**`agents/other_agent.json`** (Different company):
```json
{
  "agent_id": "other_agent",
  "tenant_id": "other-company",
  "tool_config": {
    "calendar_id": "primary",
    "service_account_key": { "...": "Other company service account" }
  }
}
```

Each call resolves its agent → uses that agent's calendar credentials. No cross-tenant leakage.

---

## Fallback & Graceful Degradation (M8)

Calendar integration is **best-effort**, never blocking:

- If `service_account_key` is absent → skip calendar, log debug message, booking still saves
- If Google Calendar API is down → log error, booking still saves, agent doesn't know the difference
- If time/day parsing fails → use a sensible default (next week, 2 PM) and log warning

The JSONL is the source of truth; Google Calendar is a convenience read-model.

---

## Troubleshooting

### Event Not Appearing in Calendar

1. **Check credentials:**
   ```python
   from providers.calendar.google_calendar import GoogleCalendarClient
   import json
   
   key = json.loads(open("path/to/service_account.json").read())
   client = GoogleCalendarClient(key, "primary")
   ```

2. **Verify service account has access:**
   - In Google Calendar settings, check that the service account email is listed as an editor

3. **Check logs:**
   ```
   tail -f server.log | grep "Google Calendar"
   ```

### Parse Errors on Day/Time

The current parser handles:
- Days: "Mon", "Tuesday", "Wed", etc. (case-insensitive, first 3 chars)
- Times: "4pm", "4:00pm", "16:00", "4am" (12/24 hour formats)

If Priya's NLG produces a format the parser doesn't recognize, update the parse logic in `agents/priya_tools.py` → `_add_to_google_calendar()`.

### "Already Exists" Error

If you see an error like "The event already exists", the service account tried to create an event twice (network retry, etc.). Google Calendar API is idempotent in most cases; a re-run should dedup. This is a benign edge case.

---

## Future Work (P2)

1. **Attendee email**: Capture caller's email (if available) and invite them to the event
2. **Timezone**: Make timezone configurable per-agent instead of hardcoded to Asia/Kolkata
3. **Fallback calendars**: If primary calendar is full, spill to a secondary
4. **Dual-write**: Also write bookings to a database instead of (or in addition to) JSONL
5. **Attendee confirmation**: Send the calendar invite to the caller's email for RSVP tracking

---

## Code Structure

- **`providers/calendar/google_calendar.py`**: GoogleCalendarClient (vendor adapter)
- **`agents/priya_tools.py`**: book_site_visit() and _add_to_google_calendar() (business logic)
- **`agents/priya.json`**: Agent record with calendar_id and service_account_key

Article III (CONSTITUTION.md): Providers are capabilities, not behavior. GoogleCalendarClient is a pure adapter; the decision to book (and when) lives in the business logic.
