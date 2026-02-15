"""TTS engine abstraction layer.

Routes text-to-speech requests to the configured provider:
- LiteLLM providers: OpenAI, ElevenLabs, MiniMax, Google Vertex, Azure
- Edge TTS: free, no API key, good Turkish support (custom wrapper)

All providers output OGG Opus files for Telegram voice note compatibility.
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Providers that go through LiteLLM
_LITELLM_PROVIDERS = {"openai", "elevenlabs", "minimax", "vertex_ai", "azure"}


class TTSEngine:
    """Unified TTS interface supporting multiple providers via LiteLLM + Edge TTS."""

    def __init__(
        self,
        provider: str = "edge",
        model: str = "",
        voice: str = "tr-TR-AhmetNeural",
        api_key: str = "",
        api_base: str = "",
        timeout: float = 30.0,
    ):
        self.provider = provider.lower().strip()
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: dict) -> "TTSEngine":
        """Create TTSEngine from config dict."""
        return cls(
            provider=config.get("tts_provider", "edge"),
            model=config.get("tts_model", ""),
            voice=config.get("tts_voice", "tr-TR-AhmetNeural"),
            api_key=config.get("tts_api_key", ""),
            api_base=config.get("tts_api_base", ""),
        )

    async def generate(self, text: str) -> Optional[str]:
        """Generate voice audio from text.

        Returns absolute path to .ogg file, or None on failure.
        Caller is responsible for cleaning up the file after use.
        """
        if not text or not text.strip():
            return None

        try:
            if self.provider == "edge":
                return await asyncio.wait_for(
                    self._generate_edge(text), timeout=self.timeout
                )
            else:
                return await asyncio.wait_for(
                    self._generate_litellm(text), timeout=self.timeout
                )
        except asyncio.TimeoutError:
            logger.error(f"TTS generation timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

    async def _generate_litellm(self, text: str) -> Optional[str]:
        """Generate via LiteLLM (OpenAI, ElevenLabs, MiniMax, Vertex, Azure)."""
        import litellm

        # Determine model string
        model = self.model
        if not model:
            # Default models per provider prefix
            defaults = {
                "openai": "openai/tts-1",
                "elevenlabs": "elevenlabs/eleven_multilingual_v2",
                "minimax": "minimax/speech-02-hd",
                "azure": "azure/tts",
            }
            model = defaults.get(self.provider, f"{self.provider}/tts-1")

        # Build kwargs
        kwargs: dict = {"model": model, "input": text}
        if self.voice:
            kwargs["voice"] = self.voice
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        # LiteLLM async speech call
        response = await litellm.aspeech(**kwargs)

        # Write response to temp MP3 file
        fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
        try:
            # response is an HttpxBinaryResponseContent — stream to file
            response.stream_to_file(mp3_path)
        except AttributeError:
            # Fallback: response might be bytes directly
            with os.fdopen(fd, "wb") as f:
                if hasattr(response, "content"):
                    f.write(response.content)
                elif isinstance(response, bytes):
                    f.write(response)
                else:
                    f.write(response.read())
            fd = -1  # Already closed via fdopen
        finally:
            if fd >= 0:
                os.close(fd)

        # Convert MP3 → OGG Opus
        ogg_path = await self._convert_to_ogg(mp3_path)

        # Clean up MP3
        try:
            os.unlink(mp3_path)
        except OSError:
            pass

        return ogg_path

    async def _generate_edge(self, text: str) -> Optional[str]:
        """Generate via edge-tts package (free, no API key)."""
        import edge_tts

        fd, ogg_path = tempfile.mkstemp(suffix=".ogg")
        os.close(fd)

        # Edge TTS can output MP3 natively; we'll convert to OGG
        fd_mp3, mp3_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd_mp3)

        try:
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(mp3_path)

            # Convert to OGG Opus for Telegram
            result = await self._convert_to_ogg(mp3_path)

            if result is None:
                # ffmpeg failed — try sending MP3 path as fallback
                # (won't show as voice bubble but at least audio arrives)
                logger.warning("OGG conversion failed, falling back to MP3")
                return mp3_path

            # Clean up MP3
            try:
                os.unlink(mp3_path)
            except OSError:
                pass

            return result

        except Exception as e:
            logger.error(f"Edge TTS failed: {e}")
            # Clean up temp files on error
            for p in (mp3_path, ogg_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            return None

    async def _convert_to_ogg(self, input_path: str) -> Optional[str]:
        """Convert audio file to OGG Opus using ffmpeg.

        Returns path to .ogg file, or None if conversion fails.
        """
        fd, ogg_path = tempfile.mkstemp(suffix=".ogg")
        os.close(fd)

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y",
                "-i", input_path,
                "-c:a", "libopus",
                "-b:a", "64k",
                "-ar", "48000",
                "-ac", "1",
                "-application", "voip",
                ogg_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

            if proc.returncode != 0:
                logger.error(f"ffmpeg conversion failed (exit {proc.returncode})")
                try:
                    os.unlink(ogg_path)
                except OSError:
                    pass
                return None

            return ogg_path

        except FileNotFoundError:
            logger.error("ffmpeg not found. Install ffmpeg to convert audio formats.")
            try:
                os.unlink(ogg_path)
            except OSError:
                pass
            return None
