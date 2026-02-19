"""TTS engine with pluggable provider architecture.

Supported providers:
- edge      : Microsoft Edge TTS (free, no API key, good Turkish support)
- local     : Any local TTS server with OpenAI-compatible /v1/audio/speech endpoint
- proxy     : Legacy tts-proxy server (OpenAI-compatible)
- openai    : OpenAI TTS API (via LiteLLM)
- elevenlabs: ElevenLabs (via LiteLLM)

All providers output OGG Opus files for Telegram voice note compatibility.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import tempfile
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TTSProvider(abc.ABC):
    """Base class for all TTS providers."""

    name: str = "base"

    @abc.abstractmethod
    async def synthesize(self, text: str) -> Optional[str]:
        """Synthesize *text* and return path to a raw audio file (mp3/wav/etc).

        Returns None on failure.  Caller handles OGG conversion & cleanup.
        """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ---------------------------------------------------------------------------
# Edge TTS  (free, no key)
# ---------------------------------------------------------------------------

class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS — free, no API key required."""

    name = "edge"

    def __init__(self, voice: str = "tr-TR-AhmetNeural"):
        self.voice = voice

    async def synthesize(self, text: str) -> Optional[str]:
        import edge_tts

        fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        try:
            comm = edge_tts.Communicate(text, self.voice)
            await comm.save(mp3_path)
            return mp3_path
        except Exception as e:
            logger.error(f"Edge TTS failed: {e}")
            _safe_unlink(mp3_path)
            return None


# ---------------------------------------------------------------------------
# Local model  (sherpa-onnx, piper, coqui, etc. behind OpenAI-compat API)
# ---------------------------------------------------------------------------

class LocalTTSProvider(TTSProvider):
    """Local TTS server exposing an OpenAI-compatible /v1/audio/speech endpoint.

    Works with sherpa-onnx-server, openai-edge-tts, Coqui TTS server, etc.
    Just point *api_base* to the server URL (e.g. http://127.0.0.1:8000).
    """

    name = "local"

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8000",
        model: str = "tts-1",
        voice: str = "default",
        response_format: str = "mp3",
        timeout: float = 30.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.timeout = timeout

    async def synthesize(self, text: str) -> Optional[str]:
        url = f"{self.api_base}/v1/audio/speech"
        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": self.response_format,
        }

        suffix = f".{self.response_format}"
        fd, raw_path = tempfile.mkstemp(suffix=suffix)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(f"Local TTS HTTP {resp.status}: {body[:200]}")
                        os.close(fd)
                        _safe_unlink(raw_path)
                        return None

                    with os.fdopen(fd, "wb") as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                    fd = -1  # closed via fdopen
        except Exception as e:
            logger.error(f"Local TTS request failed: {e}")
            if fd >= 0:
                os.close(fd)
            _safe_unlink(raw_path)
            return None

        return raw_path


# ---------------------------------------------------------------------------
# Proxy  (legacy tts-proxy, same as local but kept for backward compat)
# ---------------------------------------------------------------------------

class ProxyTTSProvider(LocalTTSProvider):
    """Legacy tts-proxy server — thin wrapper over LocalTTSProvider."""

    name = "proxy"

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:5111",
        model: str = "tts-1",
        voice: str = "Decent_Boy",
        timeout: float = 30.0,
    ):
        super().__init__(
            api_base=api_base,
            model=model,
            voice=voice,
            response_format="mp3",
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# LiteLLM  (OpenAI, ElevenLabs, MiniMax, Azure, Vertex …)
# ---------------------------------------------------------------------------

class LiteLLMProvider(TTSProvider):
    """Cloud TTS via LiteLLM (OpenAI, ElevenLabs, MiniMax, Azure, etc.)."""

    _MODEL_DEFAULTS = {
        "openai": "openai/tts-1",
        "elevenlabs": "elevenlabs/eleven_multilingual_v2",
        "minimax": "minimax/speech-02-hd",
        "azure": "azure/tts",
    }

    def __init__(
        self,
        provider: str = "openai",
        model: str = "",
        voice: str = "alloy",
        api_key: str = "",
        api_base: str = "",
    ):
        self.name = provider.lower().strip()
        self.model = model or self._MODEL_DEFAULTS.get(self.name, f"{self.name}/tts-1")
        self.voice = voice
        self.api_key = api_key
        self.api_base = api_base

    async def synthesize(self, text: str) -> Optional[str]:
        import litellm

        kwargs: dict = {"model": self.model, "input": text}
        if self.voice:
            kwargs["voice"] = self.voice
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base

        try:
            response = await litellm.aspeech(**kwargs)
        except Exception as e:
            logger.error(f"LiteLLM TTS failed: {e}")
            return None

        fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
        try:
            response.stream_to_file(mp3_path)
        except AttributeError:
            with os.fdopen(fd, "wb") as f:
                if hasattr(response, "content"):
                    f.write(response.content)
                elif isinstance(response, bytes):
                    f.write(response)
                else:
                    f.write(response.read())
            fd = -1
        finally:
            if fd >= 0:
                os.close(fd)

        return mp3_path


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[TTSProvider]] = {
    "edge": EdgeTTSProvider,
    "local": LocalTTSProvider,
    "proxy": ProxyTTSProvider,
    # LiteLLM-backed providers share the same class
    "openai": LiteLLMProvider,
    "elevenlabs": LiteLLMProvider,
    "minimax": LiteLLMProvider,
    "azure": LiteLLMProvider,
}


def _build_provider(config: dict) -> TTSProvider:
    """Instantiate the correct provider from config dict."""
    name = config.get("tts_provider", "edge").lower().strip()

    if name == "edge":
        return EdgeTTSProvider(
            voice=config.get("tts_voice", "tr-TR-AhmetNeural"),
        )

    if name == "local":
        return LocalTTSProvider(
            api_base=config.get("tts_api_base", "http://127.0.0.1:8000"),
            model=config.get("tts_model", "tts-1"),
            voice=config.get("tts_voice", "default"),
        )

    if name == "proxy":
        return ProxyTTSProvider(
            api_base=config.get("tts_api_base", "http://127.0.0.1:5111"),
            model=config.get("tts_model", "tts-1"),
            voice=config.get("tts_voice", "Decent_Boy"),
        )

    # Anything else → LiteLLM
    return LiteLLMProvider(
        provider=name,
        model=config.get("tts_model", ""),
        voice=config.get("tts_voice", "alloy"),
        api_key=config.get("tts_api_key", ""),
        api_base=config.get("tts_api_base", ""),
    )


# ---------------------------------------------------------------------------
# TTSEngine  (public API — drop-in replacement for the old class)
# ---------------------------------------------------------------------------

class TTSEngine:
    """Unified TTS interface.  Delegates to a TTSProvider and handles
    OGG Opus conversion so callers always get a Telegram-ready file.
    """

    def __init__(self, provider: TTSProvider, timeout: float = 30.0):
        self._provider = provider
        self.timeout = timeout

    # Keep the same public attribute the rest of the codebase reads
    @property
    def provider(self) -> str:
        return self._provider.name

    @classmethod
    def from_config(cls, config: dict) -> "TTSEngine":
        """Create TTSEngine from config dict."""
        prov = _build_provider(config)
        logger.info(f"TTS provider: {prov.name} ({prov.__class__.__name__})")
        return cls(provider=prov, timeout=30.0)

    async def generate(self, text: str) -> Optional[str]:
        """Generate voice audio from text.

        Returns absolute path to .ogg file, or None on failure.
        Caller is responsible for cleaning up the file after use.
        """
        if not text or not text.strip():
            return None

        try:
            raw_path = await asyncio.wait_for(
                self._provider.synthesize(text), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"TTS generation timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

        if raw_path is None:
            return None

        # Convert to OGG Opus
        ogg_path = await _convert_to_ogg(raw_path)
        if ogg_path is None:
            # Fallback: return raw file (some players handle mp3)
            logger.warning("OGG conversion failed, returning raw audio")
            return raw_path

        _safe_unlink(raw_path)
        return ogg_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


async def _convert_to_ogg(input_path: str) -> Optional[str]:
    """Convert audio file to OGG Opus using ffmpeg."""
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
            _safe_unlink(ogg_path)
            return None

        return ogg_path

    except FileNotFoundError:
        logger.error("ffmpeg not found. Install ffmpeg to convert audio formats.")
        _safe_unlink(ogg_path)
        return None
