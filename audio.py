"""
audio.py — Telephony audio bridge (verified round-trip).

Vobiz speaks G.711 mu-law @ 8 kHz, 8-bit, in 20 ms frames (160 bytes).
TTS returns PCM16 @ 24 kHz; local VAD/energy checks need PCM16 @ 8 kHz.
This module converts between them with vectorised numpy (no `audioop`,
so it runs on Python 3.13+ where audioop was removed).

Pure functions — no I/O, no state. Easy to unit test.
"""
from __future__ import annotations
import numpy as np

# --- G.711 constants ---
_BIAS = 0x84          # 132
_CLIP = 32635

FRAME_BYTES = 160     # 20 ms of mu-law @ 8 kHz  (Vobiz ingress framing)
TELEPHONY_RATE = 8000


# ---------------------------------------------------------------------------
# mu-law encode  (PCM16 -> 8-bit mu-law)
# ---------------------------------------------------------------------------
def pcm16_to_ulaw(pcm: np.ndarray) -> bytes:
    """16-bit signed PCM (int16 ndarray) -> mu-law bytes."""
    x = pcm.astype(np.int32)
    sign = (x >> 8) & 0x80
    x = np.minimum(np.abs(x), _CLIP) + _BIAS
    # exponent = number of bits above the bias, clamped 0..7
    exponent = np.clip(np.floor(np.log2(np.maximum(x, 1))).astype(np.int32) - 7, 0, 7)
    mantissa = (x >> (exponent + 3)) & 0x0F
    ulaw = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# mu-law decode  (8-bit mu-law -> PCM16)  — table-driven, fast
# ---------------------------------------------------------------------------
def _build_decode_table() -> np.ndarray:
    out = np.zeros(256, dtype=np.int16)
    for u in range(256):
        uu = ~u & 0xFF
        sign = uu & 0x80
        exponent = (uu >> 4) & 0x07
        mantissa = uu & 0x0F
        sample = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
        out[u] = -sample if sign else sample
    return out


_DECODE = _build_decode_table()


def ulaw_to_pcm16(ulaw: bytes) -> np.ndarray:
    return _DECODE[np.frombuffer(ulaw, dtype=np.uint8)]


# ---------------------------------------------------------------------------
# Resampling (linear interpolation — cheap, good enough for 8 kHz voice)
# ---------------------------------------------------------------------------
def resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(pcm) == 0:
        return pcm.astype(np.int16)
    n_out = int(round(len(pcm) * dst_rate / src_rate))
    if n_out <= 1:
        return pcm.astype(np.int16)
    xi = np.linspace(0, len(pcm) - 1, n_out)
    out = np.interp(xi, np.arange(len(pcm)), pcm.astype(np.float64))
    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Convenience: TTS PCM16 @ src_rate -> list of 20 ms mu-law frames for Vobiz
# ---------------------------------------------------------------------------
def pcm16_to_vobiz_frames(pcm: np.ndarray, src_rate: int) -> list[bytes]:
    pcm8k = resample(pcm, src_rate, TELEPHONY_RATE)
    ulaw = pcm16_to_ulaw(pcm8k)
    return [ulaw[i:i + FRAME_BYTES] for i in range(0, len(ulaw), FRAME_BYTES)]




def pcm16le_bytes_to_vobiz_frames(raw: bytes, src_rate: int) -> list[bytes]:
    """Convert a raw little-endian PCM16 chunk into Vobiz 20 ms mu-law frames."""
    if not raw:
        return []
    if len(raw) % 2:
        raw = raw[:-1]
    if not raw:
        return []
    pcm = np.frombuffer(raw, dtype="<i2")
    return pcm16_to_vobiz_frames(pcm, src_rate=src_rate)


def rms(pcm: np.ndarray) -> float:
    if len(pcm) == 0:
        return 0.0
    return float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))