"""Voice message playback with async queue.

Downloads and plays incoming voice messages sequentially.
Uses asyncio.create_subprocess_exec for non-blocking playback.

Bug fixes from original:
- Blocking process.wait() → async subprocess
- subprocess.Popen + polling → asyncio.create_subprocess_exec
- asyncio.get_event_loop() → asyncio.get_running_loop()
- Arbitrary command execution → player whitelist
- Volume not validated → clamped to 0-150
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ALLOWED_PLAYERS = {"mpv", "ffplay"}


class VoicePlayer:
    """Downloads and plays voice messages with an async queue."""

    def __init__(self, player: str = "mpv", volume: int = 100):
        if player not in _ALLOWED_PLAYERS:
            logger.warning(f"Unknown player '{player}', falling back to mpv")
            player = "mpv"
        self.player = player
        self.volume = max(0, min(150, volume))
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._playing = False
        self._current_proc: Optional[asyncio.subprocess.Process] = None
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the playback worker loop."""
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Voice player started")

    async def stop(self) -> None:
        """Stop the player and cancel pending playback."""
        if self._current_proc and self._current_proc.returncode is None:
            self._current_proc.terminate()
            try:
                await asyncio.wait_for(self._current_proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._current_proc.kill()
                await self._current_proc.wait()

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info("Voice player stopped")

    async def enqueue(self, client, message) -> None:
        """Download a voice message and add it to the playback queue."""
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
            tmp_path = tmp.name
            tmp.close()

            path = await client.download_media(message, file_name=tmp_path)
            if path:
                await self._queue.put(path)
                qsize = self._queue.qsize()
                status = f"queued (position {qsize})" if self._playing else "playing next"
                logger.info(f"Voice downloaded: {path} — {status}")
            else:
                logger.error("Failed to download voice message")
                _safe_unlink(tmp_path)
        except Exception:
            logger.exception("Error downloading voice message")

    async def _worker(self) -> None:
        """Process the playback queue sequentially."""
        while True:
            path = await self._queue.get()
            self._playing = True
            try:
                await self._play(path)
            except asyncio.CancelledError:
                _safe_unlink(path)
                raise
            except Exception:
                logger.exception(f"Error playing {path}")
            finally:
                _safe_unlink(path)
                self._playing = False
                self._queue.task_done()

    async def _play(self, path: str) -> None:
        """Play a single audio file using the configured player."""
        if not Path(path).exists():
            logger.warning(f"File not found, skipping: {path}")
            return

        cmd = self._build_command(path)
        logger.info(f"Playing: {path}")

        self._current_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        rc = await self._current_proc.wait()
        self._current_proc = None

        if rc != 0:
            logger.warning(f"Player exited with code {rc} for {path}")
        else:
            logger.info(f"Finished playing: {path}")

    def _build_command(self, path: str) -> list[str]:
        """Build player command. Only whitelisted players allowed."""
        if self.player == "mpv":
            return [
                "mpv", "--no-video", "--really-quiet",
                f"--volume={self.volume}", path,
            ]
        else:  # ffplay (only other allowed option)
            return [
                "ffplay", "-nodisp", "-autoexit",
                "-loglevel", "quiet",
                "-volume", str(self.volume), path,
            ]


def _safe_unlink(path: str) -> None:
    """Remove a file, ignoring errors."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass
