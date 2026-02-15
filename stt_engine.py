"""STT engine abstraction layer.

Routes speech-to-text requests to configured providers:
- proxy: Local tts-proxy server (Groq Whisper via /v1/audio/transcriptions)
- LiteLLM: Groq, OpenAI Whisper, GPT-4o transcribe, ElevenLabs, Deepgram, Vertex
"""

import asyncio
import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class STTEngine:
    """Unified STT interface supporting proxy and LiteLLM providers."""

    def __init__(
        self,
        model: str = "whisper-large-v3-turbo",
        api_key: str = "",
        api_base: str = "",
        language: str = "tr",
        provider: str = "proxy",
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.language = language
        self.provider = provider.lower().strip()
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: dict) -> "STTEngine":
        """Create STTEngine from config dict."""
        return cls(
            model=config.get("stt_model", "whisper-large-v3-turbo"),
            api_key=config.get("stt_api_key", ""),
            api_base=config.get("stt_api_base", ""),
            language=config.get("stt_language", "tr"),
            provider=config.get("stt_provider", "proxy"),
        )

    async def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (OGG, WAV, MP3, etc.)

        Returns:
            Transcribed text, or None on failure.
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None

        try:
            if self.provider == "proxy":
                coro = self._transcribe_proxy(audio_path)
            else:
                coro = self._transcribe_litellm(audio_path)

            return await asyncio.wait_for(coro, timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error(f"STT transcription timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return None

    async def _transcribe_proxy(self, audio_path: str) -> Optional[str]:
        """Transcribe via local tts-proxy (Groq Whisper endpoint)."""
        base = self.api_base or "http://127.0.0.1:5111"
        url = f"{base}/v1/audio/transcriptions"

        filename = os.path.basename(audio_path)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "ogg"
        ct_map = {
            "ogg": "audio/ogg", "opus": "audio/ogg", "mp3": "audio/mpeg",
            "wav": "audio/wav", "m4a": "audio/mp4", "webm": "audio/webm",
        }
        content_type = ct_map.get(ext, "audio/ogg")

        form = aiohttp.FormData()
        form.add_field("file", open(audio_path, "rb"), filename=filename, content_type=content_type)
        form.add_field("model", self.model)
        if self.language:
            form.add_field("language", self.language)
        form.add_field("response_format", "json")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, data=form, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Proxy STT HTTP {resp.status}: {body[:200]}")
                    return None

                data = await resp.json()
                return data.get("text")

    async def _transcribe_litellm(self, audio_path: str) -> Optional[str]:
        """Transcribe via LiteLLM (Groq, OpenAI, Deepgram, etc.)."""
        import litellm

        kwargs: dict = {"model": self.model, "file": open(audio_path, "rb")}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.language:
            kwargs["language"] = self.language

        response = await litellm.atranscription(**kwargs)
        return response.text if response and hasattr(response, "text") else None
