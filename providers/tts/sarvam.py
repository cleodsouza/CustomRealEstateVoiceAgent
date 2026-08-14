"""providers/tts/sarvam.py — text-to-speech via Sarvam's REST API (Bulbul v3).

One request per clause; frames are re-encoded to the requested format via
the pure audio.py leaf. Successor to sarvam_tts.py, now with a real RIFF
chunk walk instead of assuming a 44-byte header (defect D9).
"""
from __future__ import annotations

import base64
import logging
import struct
from typing import AsyncIterator

import httpx
import numpy as np

import audio
from runtime.types import MULAW_8K, AudioFormat, AudioFrame

log = logging.getLogger("providers.tts.sarvam")

TTS_URL = "https://api.sarvam.ai/text-to-speech"


def _parse_wav(raw: bytes) -> tuple[int, np.ndarray]:
    """Walk RIFF chunks to find fmt/data instead of assuming fixed offsets:
    a legal WAV may carry LIST/fact chunks before data (D9)."""
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("TTS payload is not RIFF/WAVE")
    sample_rate: int | None = None
    channels = 1
    pcm: np.ndarray | None = None
    off = 12
    while off + 8 <= len(raw):
        chunk_id, size = struct.unpack_from("<4sI", raw, off)
        body = raw[off + 8 : off + 8 + size]
        if chunk_id == b"fmt ":
            channels, sample_rate = struct.unpack_from("<HI", body, 2)
        elif chunk_id == b"data":
            pcm = np.frombuffer(body[: len(body) - (len(body) % 2)], dtype="<i2")
        off += 8 + size + (size % 2)  # RIFF chunks are word-aligned
    if sample_rate is None or pcm is None:
        raise ValueError("WAV missing fmt or data chunk")
    if channels == 2:
        pcm = pcm[::2]  # left channel only
    return sample_rate, pcm


class SarvamTTS:
    """Implements runtime.interfaces.TTS.

    Holds ONE pooled httpx client for the adapter's lifetime: a fresh
    client per clause was costing a TCP+TLS handshake on every synthesis —
    pure latency on the reply path (observed 1.8–5.5 s per clause).

    `preprocessing` asks Sarvam to normalize numerals and mixed-language
    text before synthesis ("600" spoken as a number, English loanwords
    pronounced, not spelled). Self-healing: if the endpoint rejects the
    flag (4xx), it is dropped for the adapter's lifetime and the request
    retried once without it — a capability probe, not a hard dependency."""

    supports_streaming_input = False  # REST: one round-trip per clause

    def __init__(
        self, *, api_key: str, model: str, speaker: str, language: str,
        pace: float, preprocessing: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._speaker = speaker
        self._language = language
        self._pace = pace
        self._preprocessing = preprocessing
        self._client = client if client is not None else httpx.AsyncClient(timeout=15)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def healthy(self) -> bool:
        """SupportsHealth probe. Sarvam exposes no free authenticated GET,
        so this checks reachability only: any HTTP response (even 405 on
        this POST-only route) means the endpoint is up."""
        try:
            await self._client.get(TTS_URL)
            return True
        except Exception:  # noqa: BLE001
            return False

    async def synthesize(self, text: str, fmt: AudioFormat) -> AsyncIterator[AudioFrame]:
        if fmt != MULAW_8K:
            raise ValueError(f"SarvamTTS only produces mu-law 8k today, asked for {fmt}")
        text = text.strip()
        if not text:
            return

        headers = {
            "api-subscription-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": [text],
            "target_language_code": self._language,
            "speaker": self._speaker,
            "model": self._model,
            "pace": self._pace,
        }
        if self._preprocessing:
            # Optional niceties, stripped together if the endpoint objects:
            # provider-side text normalization, and synthesis directly at
            # telephony rate (8 kHz) — less audio to generate and ship than
            # 22.05 kHz that we down-sample anyway.
            payload["enable_preprocessing"] = True
            payload["speech_sample_rate"] = 8000

        # Failures RAISE (since M8): the ResilientTTS wrapper owns the
        # retry/breaker decision, and the session owns the fallback line.
        # Swallowing here would hide every failure from both.
        r = await self._client.post(TTS_URL, json=payload, headers=headers)
        if (r.status_code // 100 == 4) and self._preprocessing:
            # Capability probe failed: this endpoint/model rejects the
            # preprocessing flag. Drop it permanently and retry once.
            log.warning("TTS rejected optional params (%d); disabling them",
                        r.status_code)
            self._preprocessing = False
            payload.pop("enable_preprocessing", None)
            payload.pop("speech_sample_rate", None)
            r = await self._client.post(TTS_URL, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()

        audios = data.get("audios", [])
        if not audios:
            raise RuntimeError(f"TTS returned no audio: {data}")

        raw = base64.b64decode(audios[0])
        src_rate, pcm = _parse_wav(raw)
        log.info("TTS WAV: %dHz, %d samples", src_rate, len(pcm))

        frames = audio.pcm16_to_vobiz_frames(pcm, src_rate=src_rate)
        log.info("TTS producing %d frames", len(frames))
        for frame in frames:
            yield AudioFrame(payload=frame, format=MULAW_8K)
