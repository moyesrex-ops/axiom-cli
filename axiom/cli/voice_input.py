"""Voice input — speech-to-text via local Whisper or OpenAI API.

Provides push-to-talk voice input that converts speech to text for
the CLI input loop.  Supports local Whisper (offline, private) and
cloud Whisper API (faster, requires key).
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Audio recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


class VoiceInput:
    """Speech-to-text bridge for the Axiom CLI.

    Usage:
        voice = VoiceInput(mode="local")  # or "cloud"
        text = voice.listen()  # blocks until speech is captured
    """

    def __init__(
        self,
        mode: str = "local",
        model_size: str = "base",
        openai_api_key: Optional[str] = None,
    ):
        self.mode = mode
        self.model_size = model_size
        self.openai_api_key = openai_api_key
        self._whisper_model: Any = None
        self._recording = False
        self._audio_frames: list[bytes] = []

    def listen(self, duration: float = 5.0) -> str:
        """Record audio and transcribe to text.

        Args:
            duration: Maximum recording duration in seconds.

        Returns:
            Transcribed text string.
        """
        audio_data = self._record(duration)
        if not audio_data:
            return ""

        if self.mode == "cloud":
            return self._transcribe_cloud(audio_data)
        else:
            return self._transcribe_local(audio_data)

    def listen_until_silence(self, silence_threshold: float = 1.5) -> str:
        """Record until silence is detected, then transcribe.

        Args:
            silence_threshold: Seconds of silence before stopping.

        Returns:
            Transcribed text string.
        """
        audio_data = self._record_until_silence(silence_threshold)
        if not audio_data:
            return ""

        if self.mode == "cloud":
            return self._transcribe_cloud(audio_data)
        else:
            return self._transcribe_local(audio_data)

    def _record(self, duration: float) -> Optional[bytes]:
        """Record audio for a fixed duration."""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error(
                "sounddevice not installed. Run: pip install sounddevice numpy"
            )
            return None

        try:
            recording = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
            )
            sd.wait()
            return self._numpy_to_wav(recording)
        except Exception as e:
            logger.error("Recording failed: %s", e)
            return None

    def _record_until_silence(
        self, silence_threshold: float = 1.5
    ) -> Optional[bytes]:
        """Record audio until silence is detected."""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error(
                "sounddevice not installed. Run: pip install sounddevice numpy"
            )
            return None

        frames: list = []
        silence_frames = 0
        silence_limit = int(silence_threshold * SAMPLE_RATE / 1024)
        energy_threshold = 500  # Adjustable

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=1024,
            ) as stream:
                while True:
                    data, _ = stream.read(1024)
                    frames.append(data.copy())

                    energy = np.abs(data).mean()
                    if energy < energy_threshold:
                        silence_frames += 1
                    else:
                        silence_frames = 0

                    if silence_frames >= silence_limit and len(frames) > 10:
                        break

                    # Safety limit: 60 seconds max
                    if len(frames) * 1024 / SAMPLE_RATE > 60:
                        break

            import numpy as np

            audio = np.concatenate(frames)
            return self._numpy_to_wav(audio)

        except Exception as e:
            logger.error("Recording failed: %s", e)
            return None

    def _numpy_to_wav(self, audio_data: Any) -> bytes:
        """Convert numpy array to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        return buf.getvalue()

    def _transcribe_local(self, wav_bytes: bytes) -> str:
        """Transcribe using local Whisper model."""
        try:
            import whisper
        except ImportError:
            logger.error(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )
            return ""

        # Lazy-load model
        if self._whisper_model is None:
            logger.info("Loading Whisper model '%s'...", self.model_size)
            self._whisper_model = whisper.load_model(self.model_size)

        # Write to temp file (Whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            result = self._whisper_model.transcribe(
                tmp_path,
                language="en",
                fp16=False,  # Safe for CPU; GPU auto-uses fp16
            )
            return result.get("text", "").strip()
        except Exception as e:
            logger.error("Whisper transcription failed: %s", e)
            return ""
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _transcribe_cloud(self, wav_bytes: bytes) -> str:
        """Transcribe using OpenAI Whisper API."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai not installed. Run: pip install openai")
            return ""

        if not self.openai_api_key:
            logger.error("OpenAI API key required for cloud transcription")
            return ""

        try:
            client = OpenAI(api_key=self.openai_api_key)
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "recording.wav"

            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en",
            )
            return response.text.strip()
        except Exception as e:
            logger.error("Cloud transcription failed: %s", e)
            return ""

    @property
    def available(self) -> bool:
        """Check if voice input dependencies are available."""
        try:
            import sounddevice  # noqa: F401

            return True
        except ImportError:
            return False
