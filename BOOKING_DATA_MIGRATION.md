# Booking Data: Before (M8) vs After (M9)

## The Problem (M8)

When a booking was made, the JSONL record contained:

```json
{
  "ts": "2026-07-13 14:30:00",
  "call_id": "vobiz-call-uuid-abc123",
  "caller": "unknown",
  "day": "Sat",
  "time": "4pm",
  "name": "Rohan"
}
```

**Issues:**
1. `"caller": "unknown"` — useless, the actual name is buried in `"name"`
2. The distinction between the phone number (from Vobiz) and the visitor's name (from the LLM) was lost
3. Any downstream system reading this JSONL couldn't reliably identify who the appointment is for

---

## The Root Cause

In `session.py`:
```python
# Line 80: initialized to "unknown"
self.caller_name: str = "unknown"

# Line 149-151: only caller_number (phone) was ever set from Vobiz
self.stream_id = ev.stream_id
self.call_id = ev.call_id
self.caller_number = ev.caller  # Vobiz sends phone number, not name
```

The transport (Vobiz) only provides:
- `call_id`: Unique call identifier
- `caller`: The caller's phone number

The caller's actual name (e.g., "Rohan") only comes later when the LLM extracts it from the `[[BOOK name=Rohan day=Sat time=4pm]]` marker.

So the booking function had:
- `ctx.call_id` ✅ Available
- `ctx.caller_number` ✅ Available (phone)
- `ctx.caller_name` ❌ Stays as "unknown"
- `args["name"]` ✅ Available (from LLM marker)

---

## The Fix (M9)

Renamed the fields to accurately reflect what they contain:

```json
{
  "ts": "2026-07-13T14:30:00.123456",
  "call_id": "vobiz-call-uuid-abc123",
  "caller_phone": "+91-98765-43210",
  "visitor_name": "Rohan",
  "day": "Sat",
  "time": "4pm"
}
```

**Improvements:**
1. `"caller_phone"` — the actual phone number from Vobiz (no longer "unknown")
2. `"visitor_name"` — the name the caller told Priya (extracted from the LLM marker)
3. Clear distinction between **transport identity** (phone) and **visitor identity** (name)
4. Downstream systems (CRM, calendar, notifications) now have both pieces of information
5. `"ts"` is now ISO format for better parsing and timezone handling

---

## Mapping Old → New

| Old Field | Old Value | New Field | New Value | Source |
|-----------|-----------|-----------|-----------|--------|
| `ts` | `"2026-07-13 14:30:00"` | `ts` | `"2026-07-13T14:30:00.123456"` | `datetime.now().isoformat()` |
| `call_id` | `"vobiz-call-uuid-abc123"` | `call_id` | `"vobiz-call-uuid-abc123"` | Vobiz (unchanged) |
| `caller` | `"unknown"` | `caller_phone` | `"+91-98765-43210"` | Vobiz transport (now captured) |
| `name` | `"Rohan"` | `visitor_name` | `"Rohan"` | LLM marker (preserved) |
| — | — | `day` | `"Sat"` | LLM marker (preserved) |
| — | — | `time` | `"4pm"` | LLM marker (preserved) |

---

## Code Changes

### Before (M8): `agents/priya_tools.py`

```python
async def book_site_visit(ctx: ToolContext, args: dict) -> None:
    """Append the appointment to the bookings JSONL."""
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "call_id": ctx.call_id,
        "caller": ctx.caller_name,  # ❌ Always "unknown"
        **args,  # {"day": "Sat", "time": "4pm", "name": "Rohan"}
    }
    path = Path(ctx.agent.tool_config.get("bookings_path", "bookings.jsonl"))
    await asyncio.to_thread(_append_line, path, record)
    log.info("Booking saved: %s", record)
```

### After (M9): `agents/priya_tools.py`

```python
async def book_site_visit(ctx: ToolContext, args: dict) -> None:
    """Book a site visit: save to JSONL and add to Google Calendar."""
    visitor_name = args.get("name", "Unknown")  # ✅ Use name from LLM
    
    record = {
        "ts": datetime.now().isoformat(),  # ✅ ISO format
        "call_id": ctx.call_id,
        "caller_phone": ctx.caller_number,  # ✅ The actual phone number
        "visitor_name": visitor_name,  # ✅ The actual visitor name
        **args,  # Preserves day, time, name
    }
    path = Path(ctx.agent.tool_config.get("bookings_path", "bookings.jsonl"))
    await asyncio.to_thread(_append_line, path, record)
    log.info("Booking saved: %s", record)
    
    # ✅ NEW: Also add to Google Calendar (M9)
    await _add_to_google_calendar(ctx, args)
```

---

## Real Example

### Scenario
- Inbound call from +91-98765-43210
- Caller tells Priya: "My name is Rohan Kumar, I'd like to visit on Saturday at 4 PM"
- Priya books it with: `[[BOOK day=Sat time=4pm name=Rohan Kumar]]`

### M8 Output (JSONL)
```json
{"ts": "2026-07-13 14:30:00", "call_id": "vobiz-98765-43210", "caller": "unknown", "day": "Sat", "time": "4pm", "name": "Rohan Kumar"}
```

**Problem for downstream:** Who is this appointment for? The `"caller"` field says "unknown". The agent reads the `"name"` field, but it's buried. Confusing.

### M9 Output (JSONL)
```json
{"ts": "2026-07-13T14:30:00.123456", "call_id": "vobiz-98765-43210", "caller_phone": "+91-98765-43210", "visitor_name": "Rohan Kumar", "day": "Sat", "time": "4pm"}
```

**Benefit for downstream:** Crystal clear. The appointment is for "Rohan Kumar" (visitor_name) who called from "+91-98765-43210" (caller_phone). Google Calendar gets populated automatically. CRM gets a clean record. Notifications go to the right person.

---

## Migration Path

### For Operators

1. **No action required** — old bookings.jsonl files remain valid
2. **Parsing**: Build queries that handle both formats:
   ```python
   # Handles both old and new formats
   def get_visitor_name(record):
       return record.get("visitor_name") or record.get("name") or record.get("caller")
   ```

### For New Bookings

- All bookings created after deploying M9 use the new field names
- Recommend a bulk migration if you have a large existing JSONL:
  ```python
  import json
  
  old_records = [json.loads(line) for line in open("bookings.jsonl")]
  migrated = []
  
  for record in old_records:
      record["caller_phone"] = record.pop("caller", "")
      if "name" in record:
          record["visitor_name"] = record.pop("name")
      migrated.append(record)
  
  with open("bookings.jsonl.migrated", "w") as f:
      for record in migrated:
          f.write(json.dumps(record, ensure_ascii=False) + "\n")
  ```

### Database Sync (CRM Integration, P2)

When you integrate with a CRM:
```python
# Parse new M9 format
caller_phone = record["caller_phone"]
visitor_name = record["visitor_name"]
visit_date = parse_date(record["day"])
visit_time = parse_time(record["time"])

# Upsert to CRM
crm.upsert_appointment(
    phone=caller_phone,
    name=visitor_name,
    scheduled_at=datetime(visit_date, visit_time),
    property="Northern Heights",
)
```

---

## FAQ

**Q: What happens to my old bookings.jsonl with `"caller": "unknown"`?**  
A: They remain unchanged. The M9 code doesn't re-process existing records. New bookings use the correct fields.

**Q: Can I still read the old format?**  
A: Yes. The JSONL is still human-readable JSON. Use `json.loads()` and check for either `"caller"` or `"caller_phone"`.

**Q: Should I migrate my historical data?**  
A: Only if you have downstream systems that need clean data (CRM, reporting). Use the migration script above.

**Q: Why change the field name instead of fixing the value?**  
A: Because the old field name was semantically wrong. `"caller"` should be a person's name or identifier, but it contained "unknown". Renaming clarifies intent: `"caller_phone"` is the phone number, `"visitor_name"` is the person.

**Q: What about Google Calendar?**  
A: That's a bonus in M9. If you haven't configured it, bookings still work as before (JSONL only).

---

## Impact on Downstream Systems

### Immediate (No Action Needed)

- ✅ Bookings still save to JSONL
- ✅ JSONL files are human-readable
- ✅ Scripts using `record.get("caller")` will need to handle None or switch to `"caller_phone"`

### Near Term (Recommended)

- Update any parsing/reporting to use new field names:
  ```python
  # Old (works with M8, breaks silently with M9)
  phone = record["caller"]  # Now a phone number, not a name
  
  # New (works with both M8 and M9)
  phone = record.get("caller_phone", record.get("caller", ""))
  name = record.get("visitor_name", record.get("name", ""))
  ```

### Future (Calendar Reads)

- Google Calendar becomes a secondary index of bookings
- Agents can query it to avoid double-bookings
- Calendar invites are sent to callers for confirmation

---

## Validation Script

Use this to verify your data is correct:

```python
import json

def validate_booking(record):
    """Check if a booking record has all required fields."""
    required = {"ts", "call_id"}
    new_style = {"caller_phone", "visitor_name"}
    old_style = {"caller", "name"}
    
    if not required.issubset(record.keys()):
        return False, f"Missing required fields: {required - set(record.keys())}"
    
    has_new = new_style.issubset(record.keys())
    has_old = old_style.issubset(record.keys())
    
    if not (has_new or has_old):
        return False, "Missing both new-style (caller_phone, visitor_name) and old-style (caller, name)"
    
    if has_new and record["caller_phone"] == "unknown":
        return False, "caller_phone is 'unknown' (should be a phone number)"
    
    if has_old and record["caller"] == "unknown":
        return False, "caller is 'unknown' (old format; should migrate)"
    
    return True, "Valid"

# Test
old_record = {"ts": "2026-07-13 14:30:00", "call_id": "abc", "caller": "unknown", "name": "Rohan"}
new_record = {"ts": "2026-07-13T14:30:00", "call_id": "abc", "caller_phone": "+91-98765-43210", "visitor_name": "Rohan"}

print(f"Old: {validate_booking(old_record)}")  # (False, "caller is 'unknown' ...")
print(f"New: {validate_booking(new_record)}")  # (True, "Valid")
```

---

## Summary

| Aspect | M8 | M9 |
|--------|----|----|
| `caller` field | "unknown" (useless) | Removed |
| `caller_phone` field | ❌ Missing | `"+91-98765-43210"` ✅ |
| `visitor_name` field | ❌ Missing | `"Rohan Kumar"` ✅ |
| `name` field | `"Rohan Kumar"` (unclear purpose) | Kept for backwards compat |
| Google Calendar | ❌ No | ✅ Yes (optional, per-agent) |
| `ts` format | `"2026-07-13 14:30:00"` | `"2026-07-13T14:30:00.123456"` (ISO) |

**Outcome**: Booking records now contain all necessary information to create accurate calendar events, send notifications, and populate CRM systems.
