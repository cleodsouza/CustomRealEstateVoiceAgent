"""dashboard.py — read-model builder for the operations dashboard.

Purpose
    Turn the JSONL files the observability sinks already write
    (transcripts.jsonl via TranscriptWriter, bookings.jsonl via the
    booking tool) into one JSON snapshot the dashboard page renders.

Responsibilities
    - Reconstruct calls from the flat event stream (call_id is often
      "unknown" today, so grouping is by CallStarted/CallEnded boundaries
      in file order — the stream is append-only and single-process).
    - Normalize booking records across their historical shapes and
      collapse duplicate confirmations of the same visit.

Ownership / dependencies
    Pure observer over sink output files. No runtime imports beyond the
    stdlib: deleting this module (and its two routes in server.py) changes
    nothing about a call. When transcripts move to a database (M11), this
    module becomes a query against that store; the snapshot shape and the
    page stay the same.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def _parse_ts(raw: str) -> datetime | None:
    """Both historical formats: '2026-07-08 09:10:59' and full ISO."""
    for fmt in ("%Y-%m-%d %H:%M:%S", None):
        try:
            return (datetime.strptime(raw, fmt) if fmt
                    else datetime.fromisoformat(raw))
        except (ValueError, TypeError):
            continue
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn write must not take the dashboard down
            if isinstance(rec, dict):
                records.append(rec)
    return records


def _reconstruct_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group the flat event stream into calls.

    call_id is unreliable ("unknown") in today's data, so grouping is
    positional: CallStarted opens a call, CallEnded closes it. Turns or
    ends arriving with no open call become an implicit call so early,
    partial logs still show up rather than vanish.
    """
    calls: list[dict[str, Any]] = []
    open_call: dict[str, Any] | None = None

    def _new_call(ts: str | None, agent_id: str | None) -> dict[str, Any]:
        return {"started": ts, "ended": None, "duration_s": None,
                "agent_id": agent_id, "turns": []}

    for ev in events:
        kind = ev.get("event")
        ts = ev.get("ts")
        if kind == "CallStarted":
            if open_call is not None:
                calls.append(open_call)  # previous call never saw its end
            open_call = _new_call(ts, ev.get("agent_id"))
        elif kind == "TurnCompleted":
            if open_call is None:
                open_call = _new_call(ts, None)
            open_call["turns"].append({
                "ts": ts,
                "seq": ev.get("turn_seq"),
                "thinking_s": ev.get("thinking_s"),
                "first_audio_s": ev.get("first_audio_s"),
                "interrupted": bool(ev.get("interrupted")),
            })
        elif kind == "CallEnded":
            if open_call is None:
                open_call = _new_call(ts, None)
            open_call["ended"] = ts
            start = _parse_ts(open_call["started"] or "")
            end = _parse_ts(ts or "")
            if start and end and end >= start:
                open_call["duration_s"] = (end - start).total_seconds()
            calls.append(open_call)
            open_call = None

    if open_call is not None:
        calls.append(open_call)  # in-flight or truncated log
    return calls


def _normalize_bookings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize booking shapes and collapse repeat confirmations: the
    booking tool can fire more than once in a call for the same visit, so
    same visitor + same day within a 30-minute window counts once."""
    out: list[dict[str, Any]] = []
    for rec in records:
        ts = rec.get("ts")
        name = rec.get("visitor_name") or rec.get("name") or "unknown"
        day = rec.get("day") or ""
        entry = {
            "ts": ts,
            "name": name,
            "day": day,
            "time": rec.get("time") or "",
            "flat": rec.get("flat") or "",
            "caller": rec.get("caller_phone") or rec.get("caller") or "unknown",
        }
        prev = out[-1] if out else None
        if prev and prev["name"].lower() == name.lower() \
                and prev["day"].lower() == day.lower():
            t_prev, t_cur = _parse_ts(prev["ts"] or ""), _parse_ts(ts or "")
            if t_prev and t_cur and abs((t_cur - t_prev)) <= timedelta(minutes=30):
                out[-1] = entry  # keep the latest confirmation
                continue
        out.append(entry)
    return out


def build_snapshot(transcripts_path: str | Path,
                   bookings_path: str | Path) -> dict[str, Any]:
    """The one payload GET /dashboard/data serves."""
    calls = _reconstruct_calls(_read_jsonl(Path(transcripts_path)))
    bookings = _normalize_bookings(_read_jsonl(Path(bookings_path)))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "calls": calls,
        "bookings": bookings,
    }
