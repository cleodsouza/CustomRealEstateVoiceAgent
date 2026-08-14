"""SarvamTTS adapter (S6): pooled client, self-healing preprocessing flag."""
import base64
import struct

import httpx

from providers.tts.sarvam import SarvamTTS
from runtime.types import MULAW_8K


def _wav_b64(n_samples=160, rate=8000):
    pcm = struct.pack(f"<{n_samples}h", *([1000] * n_samples))
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt \
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    wav = b"RIFF" + struct.pack("<I", len(body)) + body
    return base64.b64encode(wav).decode()


def make_tts(handler, preprocessing=True):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SarvamTTS(api_key="k", model="bulbul:v3", speaker="ishita",
                     language="hi-IN", pace=1.05,
                     preprocessing=preprocessing, client=client)


async def test_preprocessing_flag_sent_when_enabled():
    seen = []

    def handler(request):
        import json
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"audios": [_wav_b64()]})

    tts = make_tts(handler)
    frames = [f async for f in tts.synthesize("छह सौ", MULAW_8K)]
    assert frames and seen[0]["enable_preprocessing"] is True


async def test_rejected_flag_self_heals_and_stays_off():
    seen = []

    def handler(request):
        import json
        payload = json.loads(request.content)
        seen.append(payload)
        if "enable_preprocessing" in payload:
            return httpx.Response(400, json={"error": "unknown field"})
        return httpx.Response(200, json={"audios": [_wav_b64()]})

    tts = make_tts(handler)
    frames = [f async for f in tts.synthesize("नमस्ते", MULAW_8K)]
    assert frames                                  # first call healed itself
    assert [("enable_preprocessing" in p) for p in seen] == [True, False]

    [f async for f in tts.synthesize("फिर से", MULAW_8K)]
    assert "enable_preprocessing" not in seen[-1]  # remembered: flag off
    assert len(seen) == 3                          # no extra probe round


async def test_disabled_preprocessing_never_sends_flag():
    seen = []

    def handler(request):
        import json
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"audios": [_wav_b64()]})

    tts = make_tts(handler, preprocessing=False)
    [f async for f in tts.synthesize("ठीक", MULAW_8K)]
    assert "enable_preprocessing" not in seen[0]
