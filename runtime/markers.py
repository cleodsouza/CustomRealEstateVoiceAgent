"""runtime/markers.py — [[MARKER k=v ...]] token parsing.

Since M7 this is the text-level half of the *marker dispatch strategy*
(runtime/tools.py): tools declare a marker token in their ToolSpec, the
model appends `[[TOKEN key=value ...]]` to replies, and this module strips
recognized tokens out of the spoken text and returns them as tool calls.

Markers a call's agent doesn't own are left in the text untouched — the
parser only speaks for the tools it was given.
"""
from __future__ import annotations

import re
from typing import Mapping

_MARKER_RE = re.compile(r"\[\[(\w+)([^\]]*)\]\]")
_KV_RE = re.compile(r"(\w+)\s*=\s*([^=\]]+?)(?=\s+\w+=|$)")

# A marker longer than this is not a marker; release the held text.
MAX_MARKER_LEN = 200


class MarkerGuard:
    """Incremental marker filter for the token stream (S2).

    Per-clause extraction is safe against markers split across deltas
    because the clause buffer reassembles them. A raw token surface has no
    such buffer, so this guard sits at delta granularity: text that might
    be the prefix of a marker is HELD BACK, then either dropped (a complete,
    recognized marker — no fragment of tool syntax ever reaches a token
    consumer), released verbatim (unrecognized or disproven — exactly the
    per-clause behavior of leaving foreign markers in the text), or freed
    by the length cap / end-of-stream flush.

    Stripping only, deliberately: DISPATCH stays on the clause path
    (extract_tool_calls), so a tool fires exactly once however many views
    of the stream exist.
    """

    def __init__(self, markers: Mapping[str, str],
                 max_hold: int = MAX_MARKER_LEN) -> None:
        self._markers = markers
        self._max_hold = max_hold
        self._buf = ""

    def feed(self, text: str) -> str:
        """Add delta text; return whatever is now provably safe to emit."""
        self._buf += text
        return self._scan(final=False)

    def flush(self) -> str:
        """End of stream: everything still held is released (an unclosed
        `[[` can no longer become a marker)."""
        return self._scan(final=True)

    def _scan(self, *, final: bool) -> str:
        out: list[str] = []
        buf = self._buf
        while buf:
            idx = buf.find("[[")
            if idx == -1:
                # A trailing "[" may become "[[" with the next delta.
                if not final and buf.endswith("["):
                    out.append(buf[:-1])
                    buf = "["
                else:
                    out.append(buf)
                    buf = ""
                break
            out.append(buf[:idx])
            held = buf[idx:]
            end = held.find("]]", 2)
            if end == -1:
                if final or len(held) > self._max_hold:
                    # Disproven by exhaustion: release one char, rescan the
                    # rest (a later "[[" may still open a real marker).
                    out.append(held[0])
                    buf = held[1:]
                    continue
                buf = held  # keep holding; more deltas may close it
                break
            candidate, buf = held[: end + 2], held[end + 2:]
            m = _MARKER_RE.fullmatch(candidate)
            if m is None:
                out.append(candidate[0])          # not a marker shape
                buf = candidate[1:] + buf
            elif m.group(1).upper() in self._markers:
                pass                              # ours: strip silently
            else:
                out.append(candidate)             # foreign: leave in text
        self._buf = buf
        return "".join(out)


def extract_tool_calls(
    text: str, markers: Mapping[str, str]
) -> tuple[str, list[tuple[str, dict]]]:
    """Parse recognized markers out of `text`.

    `markers` maps an UPPERCASE marker token to the tool name it triggers.
    Returns (clean_spoken_text, [(tool_name, args), ...]) — args are the
    marker's k=v pairs, lowercased keys, {} for bare markers.
    """
    calls: list[tuple[str, dict]] = []

    def _consume(m: re.Match[str]) -> str:
        tool = markers.get(m.group(1).upper())
        if tool is None:
            return m.group(0)  # not ours: leave it in the text
        args = {k.lower(): v.strip() for k, v in _KV_RE.findall(m.group(2))}
        calls.append((tool, args))
        return ""

    clean = _MARKER_RE.sub(_consume, text).strip()
    return clean, calls
