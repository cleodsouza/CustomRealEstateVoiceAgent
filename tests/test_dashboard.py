"""dashboard.py — snapshot builder: call reconstruction from the flat
event stream, booking normalization/dedupe, resilience to torn writes."""
import json

from dashboard import build_snapshot


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_files(tmp_path, transcript_records, booking_records):
    t = tmp_path / "transcripts.jsonl"
    b = tmp_path / "bookings.jsonl"
    _write_jsonl(t, transcript_records)
    _write_jsonl(b, booking_records)
    return t, b


def test_groups_events_into_calls(tmp_path):
    t, b = make_files(tmp_path, [
        {"ts": "2026-07-08 09:00:00", "event": "CallStarted",
         "call_id": "unknown", "caller": "unknown", "agent_id": "priya"},
        {"ts": "2026-07-08 09:00:20", "event": "TurnCompleted",
         "call_id": "unknown", "turn_seq": 1, "user_text": "hi",
         "agent_text": "hello", "thinking_s": 1.2, "first_audio_s": 2.5,
         "interrupted": False},
        {"ts": "2026-07-08 09:01:00", "event": "CallEnded",
         "call_id": "unknown"},
    ], [])
    snap = build_snapshot(t, b)
    assert len(snap["calls"]) == 1
    call = snap["calls"][0]
    assert call["agent_id"] == "priya"
    assert call["duration_s"] == 60.0
    assert call["turns"][0]["first_audio_s"] == 2.5
    assert call["turns"][0]["interrupted"] is False


def test_orphan_events_become_implicit_calls(tmp_path):
    # Early logs contain lone CallEnded lines and turns with no CallStarted.
    t, b = make_files(tmp_path, [
        {"ts": "2026-07-06 08:35:12", "event": "CallEnded", "call_id": "unknown"},
        {"ts": "2026-07-07 08:42:17", "event": "TurnCompleted",
         "call_id": "unknown", "turn_seq": 1, "user_text": "x",
         "agent_text": "", "thinking_s": None, "first_audio_s": 1.0,
         "interrupted": True},
        {"ts": "2026-07-07 08:42:42", "event": "CallEnded", "call_id": "unknown"},
    ], [])
    calls = build_snapshot(t, b)["calls"]
    assert len(calls) == 2
    assert calls[0]["turns"] == []
    assert len(calls[1]["turns"]) == 1


def test_unclosed_call_still_appears(tmp_path):
    t, b = make_files(tmp_path, [
        {"ts": "2026-07-08 09:00:00", "event": "CallStarted",
         "call_id": "unknown", "caller": "unknown", "agent_id": "priya"},
    ], [])
    calls = build_snapshot(t, b)["calls"]
    assert len(calls) == 1
    assert calls[0]["ended"] is None
    assert calls[0]["duration_s"] is None


def test_booking_shapes_normalized_and_deduped(tmp_path):
    t, b = make_files(tmp_path, [], [
        # old shape, double-confirmed 17s apart -> one booking
        {"ts": "2026-07-08 09:10:59", "call_id": "unknown",
         "caller": "unknown", "day": "Saturday", "time": "2 PM", "name": "Kriyaan"},
        {"ts": "2026-07-08 09:11:16", "call_id": "unknown",
         "caller": "unknown", "day": "Saturday", "time": "2 PM", "name": "Kriyaan"},
        # new shape (ISO ts, visitor_name), different person -> kept
        {"ts": "2026-07-14T16:27:37.930031", "call_id": "unknown",
         "caller_phone": "unknown", "visitor_name": "Beon",
         "day": "Sunday", "time": "2:00 PM", "name": "Beon"},
    ])
    bookings = build_snapshot(t, b)["bookings"]
    assert len(bookings) == 2
    assert bookings[0]["name"] == "Kriyaan"
    assert bookings[0]["time"] == "2 PM"  # latest confirmation wins
    assert bookings[1]["name"] == "Beon"


def test_same_name_far_apart_is_two_bookings(tmp_path):
    t, b = make_files(tmp_path, [], [
        {"ts": "2026-07-08 09:00:00", "day": "Saturday", "time": "2 PM", "name": "A"},
        {"ts": "2026-07-08 11:00:00", "day": "Saturday", "time": "2 PM", "name": "A"},
    ])
    assert len(build_snapshot(t, b)["bookings"]) == 2


def test_torn_write_and_missing_files_survive(tmp_path):
    t = tmp_path / "transcripts.jsonl"
    t.write_text('{"ts": "2026-07-08 09:00:00", "event": "CallEnded", '
                 '"call_id": "unknown"}\n{"ts": "2026-07-08 09:', # torn line
                 encoding="utf-8")
    snap = build_snapshot(t, tmp_path / "does-not-exist.jsonl")
    assert len(snap["calls"]) == 1
    assert snap["bookings"] == []
