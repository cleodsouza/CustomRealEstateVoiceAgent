"""Sarvam Bulbul v3 TTS adapters.

The normal ``synthesize()`` method keeps the existing REST contract for
static/cached lines.  ``stream_text()`` is the low-latency live-call path:
- one WebSocket per live reply,
- incremental text input,
- progressive audio output,
- μ-law 8 kHz frames directly suitable for Vobiz.

Sarvam's WebSocket API explicitly supports incremental text, progressive audio,
μ-law output, and a completion event; it also recommends closing the socket on
barge-in and opening a fresh one for the next reply.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from typing import AsyncIterator

import httpx
import websockets

import audio
from runtime.types import MULAW_8K, AudioFormat, AudioFrame

log = logging.getLogger("providers.tts.sarvam")

TTS_URL = "https://api.sarvam.ai/text-to-speech"
TTS_WS_URL = "wss://api.sarvam.ai/text-to-speech/ws"


def _parse_wav(raw: bytes):
    """Compatibility helper retained for the existing WAV unit tests."""
    import struct
    import numpy as np
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
        off += 8 + size + (size % 2)
    if sample_rate is None or pcm is None:
        raise ValueError("WAV missing fmt or data chunk")
    if channels == 2:
        pcm = pcm[::2]
    return sample_rate, pcm

_SENTINEL = object()


class SarvamTTS:
    """Sarvam REST + streaming WebSocket TTS adapter."""

    supports_streaming_input = True

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        speaker: str,
        language: str,
        pace: float,
        preprocessing: bool = True,
        streaming_min_buffer_size: int = 30,
        streaming_max_chunk_length: int = 120,
        streaming_sample_rate: int = 8000,
        streaming_audio_queue: int = 96,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._speaker = speaker
        self._language = language
        self._pace = pace
        self._preprocessing = preprocessing
        self._streaming_min_buffer_size = streaming_min_buffer_size
        self._streaming_max_chunk_length = streaming_max_chunk_length
        self._streaming_sample_rate = streaming_sample_rate
        self._streaming_audio_queue = streaming_audio_queue
        self._client = client if client is not None else httpx.AsyncClient(timeout=15)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def healthy(self) -> bool:
        try:
            await self._client.get(TTS_URL)
            return True
        except Exception:  # noqa: BLE001
            return False

    async def synthesize(self, text: str, fmt: AudioFormat) -> AsyncIterator[AudioFrame]:
        """Legacy full-response REST synthesis used for static/cacheable lines."""
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
            payload["enable_preprocessing"] = True
            payload["speech_sample_rate"] = 8000

        r = await self._client.post(TTS_URL, json=payload, headers=headers)
        if (r.status_code // 100 == 4) and self._preprocessing:
            log.warning("TTS rejected optional params (%d); disabling them", r.status_code)
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
        sample_rate, pcm = _parse_wav(raw)
        for frame in audio.pcm16_to_vobiz_frames(pcm, src_rate=sample_rate):
            yield AudioFrame(payload=frame, format=MULAW_8K)

    async def stream_text(
        self,
        text_stream: AsyncIterator[str],
        fmt: AudioFormat,
    ) -> AsyncIterator[AudioFrame]:
        """Stream incremental text to Sarvam and audio back as Vobiz frames.

        The sender and receiver are separate tasks so TTS can keep receiving
        audio while the LLM is still feeding text. Audio is queued locally and
        split into exact 20 ms / 160-byte μ-law frames for the carrier.
        """
        if fmt != MULAW_8K:
            raise ValueError("Sarvam streaming path only supports MULAW_8K")

        headers = {"Api-Subscription-Key": self._api_key}
        url = (
            f"{TTS_WS_URL}?model={self._model}&send_completion_event=true"
        )
        audio_queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue(
            maxsize=self._streaming_audio_queue
        )

        async def receiver(ws) -> None:
            remainder = b""
            try:
                async for raw_message in ws:
                    if not isinstance(raw_message, str):
                        continue
                    msg = json.loads(raw_message)
                    kind = msg.get("type")
                    if kind == "audio":
                        encoded = (msg.get("data") or {}).get("audio")
                        if not encoded:
                            continue
                        chunk = base64.b64decode(encoded)
                        if not chunk:
                            continue
                        data = remainder + chunk
                        full = (len(data) // audio.FRAME_BYTES) * audio.FRAME_BYTES
                        for start in range(0, full, audio.FRAME_BYTES):
                            await audio_queue.put(data[start : start + audio.FRAME_BYTES])
                        remainder = data[full:]
                    elif kind == "event":
                        event_type = (msg.get("data") or {}).get("event_type")
                        if event_type == "final":
                            break
                    elif kind == "error":
                        detail = (msg.get("data") or {}).get("error") or msg
                        raise RuntimeError(f"Sarvam streaming TTS error: {detail}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                await audio_queue.put(exc)
                return
            if remainder:
                await audio_queue.put(remainder.ljust(audio.FRAME_BYTES, b"\xff"))
            await audio_queue.put(None)

        async def sender(ws) -> None:
            async for text in text_stream:
                text = text.strip()
                if not text:
                    continue
                await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
            await ws.send(json.dumps({"type": "flush"}))

        recv_task = send_task = None
        ws = None
        try:
            try:
                ws = await websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=2,
                )
            except TypeError:  # websockets < 14 renamed this kwarg
                ws = await websockets.connect(
                    url,
                    extra_headers=headers,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=2,
                )
            config_message = {
                "type": "config",
                "data": {
                    "target_language_code": self._language,
                    "speaker": self._speaker,
                    "speech_sample_rate": self._streaming_sample_rate,
                    "min_buffer_size": self._streaming_min_buffer_size,
                    "max_chunk_length": self._streaming_max_chunk_length,
                    "output_audio_codec": "mulaw",
                    "pace": self._pace,
                },
            }
            await ws.send(json.dumps(config_message))
            recv_task = asyncio.create_task(receiver(ws), name="sarvam-tts-recv")
            send_task = asyncio.create_task(sender(ws), name="sarvam-tts-send")

            while True:
                item = await audio_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield AudioFrame(payload=item, format=MULAW_8K)

            await send_task
            await recv_task
        except asyncio.CancelledError:
            raise
        finally:
            for task in (send_task, recv_task):
                if task is not None and not task.done():
                    task.cancel()
            for task in (send_task, recv_task):
                if task is not None:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            if ws is not None:
                with contextlib.suppress(Exception):
                    await ws.close()