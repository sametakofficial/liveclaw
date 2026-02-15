"""STT engine abstraction layer (future-ready).

Routes speech-to-text requests to configured providers via LiteLLM:
- Groq: "groq/whisper-large-v3", "groq/whisper-large-v3-turbo"
- OpenAI: "whisper-1"
- GPT-4o: "openai/gpt-4o-transcribe"
- ElevenLabs, Deepgram, Google Vertex

Interface is defined now; full integration comes when voice-to-text
pipeline is added to the recorder module.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class STTEngine:
    """Unified STT interface using LiteLLM transcription."""

    def __init__(
        self,
        model: str = "groq/whisper-large-v3-turbo",
        api_key: str = "",
        api_base: str = "",
        language: str = "",
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.language = language
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: dict) -> "STTEngine":
        """Create STTEngine from config dict."""
        return cls(
            model=config.get("stt_model", "groq/whisper-large-v3-turbo"),
            api_key=config.get("stt_api_key", ""),
            api_base=config.get("stt_api_base", ""),
            language=config.get("stt_language", ""),
        )

    async def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (OGG, WAV, MP3, etc.)

        Returns:
            Transcribed text, or None on failure.
        """
        try:
            return await asyncio.wait_for(
                self._transcribe(audio_path), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"STT transcription timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return None

    async def _transcribe(self, audio_path: str) -> Optional[str]:
        """Run transcription via LiteLLM."""
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
