"""
recording.py — Fire-and-forget call audio recording to stereo WAV files.

Caller audio (inbound) is recorded on the left channel, agent audio
(outbound TTS) on the right. The two streams are paced very differently —
the caller leg arrives continuously every ~20 ms for the whole call, the
agent leg only produces frames while Priya is actually speaking — so each
frame is timestamped against call start as it's captured and later laid
onto a silence-filled timeline. That's what keeps the two channels in
sync when played back together instead of the agent's speech drifting
earlier as gaps where she was silent get squeezed out.
"""
import asyncio
import logging
import struct
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("recording")

# WAV format constants for mu-law audio
AUDIO_FORMAT_MULAW = 0x0007  # WAVE_FORMAT_MULAW
CHANNELS = 2                 # left = caller, right = agent
SAMPLE_RATE = 8000
BLOCK_ALIGN = CHANNELS * 1   # 1 byte per sample per channel (8-bit mu-law)
BYTE_RATE = SAMPLE_RATE * BLOCK_ALIGN
MULAW_SILENCE = 0xFF         # standard mu-law encoding of zero amplitude


class CallRecorder:
    """Records both legs of a call to a single stereo WAV file.

    Accumulates timestamped frames asynchronously, never blocking. At call
    end, lays each leg onto its own silence-filled timeline and interleaves
    them into a stereo WAV in the background.
    """

    def __init__(self, call_id: str, output_dir: str = "recordings"):
        self.call_id = call_id
        self.output_dir = Path(output_dir)
        self._t0 = time.monotonic()
        self._caller_frames: list[tuple[float, bytes]] = []
        self._agent_frames: list[tuple[float, bytes]] = []

    async def record_caller_frame(self, frame: bytes) -> None:
        """Record one inbound (caller) audio frame. Non-blocking append."""
        self._caller_frames.append((time.monotonic() - self._t0, frame))

    async def record_agent_frame(self, frame: bytes) -> None:
        """Record one outbound (agent TTS) audio frame. Non-blocking append."""
        self._agent_frames.append((time.monotonic() - self._t0, frame))

    async def finalize(self) -> None:
        """Write all accumulated frames to a WAV file. Async, fire-and-forget.
        Failures are logged but do not raise."""
        try:
            await asyncio.to_thread(self._write_wav)
        except Exception as e:
            log.error("Failed to write recording for call %s: %s", self.call_id, e)

    def _write_wav(self) -> None:
        """Actually write the WAV file. Runs in a thread pool to avoid
        blocking the async event loop."""
        if not self._caller_frames and not self._agent_frames:
            log.warning("No audio frames recorded for call %s", self.call_id)
            return

        self.output_dir.mkdir(exist_ok=True, parents=True)
        output_file = self.output_dir / f"{self.call_id}.wav"

        left = self._lay_channel(self._caller_frames)
        right = self._lay_channel(self._agent_frames)
        n = max(len(left), len(right))
        left = self._pad(left, n)
        right = self._pad(right, n)

        stereo = np.empty(n * 2, dtype=np.uint8)
        stereo[0::2] = left
        stereo[1::2] = right
        audio_data = stereo.tobytes()

        wav_data = self._build_wav(audio_data)

        try:
            output_file.write_bytes(wav_data)
            log.info("Recorded call %s to %s (%d bytes audio data)",
                     self.call_id, output_file, len(audio_data))
        except OSError as e:
            log.error("Error writing WAV file %s: %s", output_file, e)

    @staticmethod
    def _lay_channel(frames: list[tuple[float, bytes]]) -> np.ndarray:
        """Place timestamped frames onto a silence-filled mono timeline."""
        if not frames:
            return np.zeros(0, dtype=np.uint8)
        end_samples = max(
            round(ts * SAMPLE_RATE) + len(payload) for ts, payload in frames)
        buf = np.full(end_samples, MULAW_SILENCE, dtype=np.uint8)
        for ts, payload in frames:
            offset = round(ts * SAMPLE_RATE)
            data = np.frombuffer(payload, dtype=np.uint8)
            buf[offset:offset + len(data)] = data
        return buf

    @staticmethod
    def _pad(channel: np.ndarray, length: int) -> np.ndarray:
        if len(channel) >= length:
            return channel
        return np.concatenate(
            [channel, np.full(length - len(channel), MULAW_SILENCE, dtype=np.uint8)])

    def _build_wav(self, audio_data: bytes) -> bytes:
        """Construct a complete WAV file (RIFF header + fmt + data chunks).
        Returns bytes ready to write to disk."""
        audio_data_size = len(audio_data)

        # fmt sub-chunk: 16 bytes of format data
        fmt_chunk = struct.pack(
            "<4sIHHIIHH",
            b"fmt ",           # Subchunk1ID
            16,                # Subchunk1Size (standard for PCM-like)
            AUDIO_FORMAT_MULAW,  # AudioFormat (0x0007 = mu-law)
            CHANNELS,          # NumChannels
            SAMPLE_RATE,       # SampleRate
            BYTE_RATE,         # ByteRate
            BLOCK_ALIGN,       # BlockAlign
            8,                 # BitsPerSample (8 for mu-law)
        )

        # data sub-chunk header
        data_chunk_header = struct.pack(
            "<4sI",
            b"data",               # Subchunk2ID
            audio_data_size,       # Subchunk2Size
        )

        # RIFF header
        file_size = 36 + audio_data_size  # 36 = fmt(8+16) + data(8)
        riff_header = struct.pack(
            "<4sI4s",
            b"RIFF",        # ChunkID
            file_size,      # ChunkSize (file size - 8)
            b"WAVE",        # Format
        )

        return riff_header + fmt_chunk + data_chunk_header + audio_data
