"""Microphone recording and voice message sending.

Records audio from the system microphone, converts to OGG Opus,
and sends as a Telegram voice message.

Bug fixes from original:
- subprocess.run in executor → asyncio.create_subprocess_exec
- asyncio.get_event_loop() → asyncio.get_running_loop()
- Max duration race condition → proper stop via call_soon_threadsafe
- run_coroutine_threadsafe result discarded → done callback for error logging
- Frame count sum() on every callback → running counter
"""

import asyncio
import logging
import os
import tempfile
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

logger = logging.getLogger(__name__)


class VoiceRecorder:
    """Records audio from the microphone and sends it as a voice message."""

    def __init__(
        self,
        client,
        target_chat_id: int,
        sample_rate: int = 48000,
        channels: int = 1,
        max_duration: int = 120,
    ):
        self.client = client
        self.target_chat_id = target_chat_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_duration = max_duration

        self._recording = False
        self._frames: list[np.ndarray] = []
        self._frame_count: int = 0  # Running counter instead of sum()
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def toggle(self, loop: asyncio.AbstractEventLoop) -> None:
        """Toggle recording on/off. Called from the keyboard listener thread."""
        with self._lock:
            if self._recording:
                self._stop_recording(loop)
            else:
                self._loop = loop
                self._start_recording()

    def _start_recording(self) -> None:
        """Start capturing audio from the microphone."""
        self._frames = []
        self._frame_count = 0
        self._recording = True
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                callback=self._audio_callback,
            )
            stream.start()
            self._stream = stream
            logger.info("Recording started")
        except Exception:
            self._recording = False
            logger.exception("Failed to start recording")

    def _stop_recording(self, loop: asyncio.AbstractEventLoop) -> None:
        """Stop recording and schedule send."""
        self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Error stopping audio stream")
            self._stream = None

        if not self._frames:
            logger.warning("No audio frames captured, skipping send")
            return

        logger.info("Recording stopped, processing audio...")
        audio_data = np.concatenate(self._frames, axis=0)
        self._frames = []
        self._frame_count = 0

        # Schedule async send — capture future for error logging
        future = asyncio.run_coroutine_threadsafe(
            self._process_and_send(audio_data), loop
        )
        future.add_done_callback(_log_future_exception)

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """Sounddevice callback — runs in audio thread."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        if not self._recording:
            return

        self._frames.append(indata.copy())
        self._frame_count += indata.shape[0]

        # Enforce max duration using running counter
        if self._frame_count / self.sample_rate >= self.max_duration:
            logger.info("Max recording duration reached")
            self._recording = False
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._max_duration_stop)

    def _max_duration_stop(self) -> None:
        """Called on the event loop thread when max duration is hit."""
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    logger.exception("Error stopping stream after max duration")
                self._stream = None

            if not self._frames:
                return

            logger.info("Processing audio (max duration)...")
            audio_data = np.concatenate(self._frames, axis=0)
            self._frames = []
            self._frame_count = 0

            if self._loop is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self._process_and_send(audio_data), self._loop
                )
                future.add_done_callback(_log_future_exception)

    async def _process_and_send(self, audio_data: np.ndarray) -> None:
        """Convert recorded audio to OGG Opus and send as voice message."""
        fd_wav, wav_path = tempfile.mkstemp(suffix=".wav")
        fd_ogg, ogg_path = tempfile.mkstemp(suffix=".ogg")
        os.close(fd_wav)
        os.close(fd_ogg)

        try:
            # Write WAV
            wavfile.write(wav_path, self.sample_rate, audio_data)

            # Convert to OGG Opus using async ffmpeg
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", wav_path,
                "-c:a", "libopus",
                "-b:a", "64k",
                "-ar", "48000",
                "-ac", "1",
                "-application", "voip",
                ogg_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await proc.wait()

            if rc != 0:
                logger.error(f"ffmpeg conversion failed (exit {rc})")
                return

            duration = len(audio_data) / self.sample_rate

            await self.client.send_voice(
                chat_id=self.target_chat_id,
                voice=ogg_path,
                duration=int(duration),
            )
            logger.info(f"Voice sent to {self.target_chat_id} ({duration:.1f}s)")

        except Exception:
            logger.exception("Failed to process/send voice message")
        finally:
            for p in (wav_path, ogg_path):
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except OSError:
                    pass

    async def cleanup(self) -> None:
        """Stop any active recording."""
        with self._lock:
            self._recording = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        logger.info("Recorder cleaned up")


def _log_future_exception(future) -> None:
    """Callback to log exceptions from run_coroutine_threadsafe."""
    exc = future.exception()
    if exc is not None:
        logger.error(f"Async send failed: {exc}")
