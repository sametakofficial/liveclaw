"""Tests for tts_engine.py — TTS abstraction layer."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tts_engine import TTSEngine, EdgeTTSProvider, LocalTTSProvider, ProxyTTSProvider, _convert_to_ogg


class TestTTSEngineInit:
    """Test engine initialization and config parsing."""

    def test_default_provider_is_edge(self):
        engine = TTSEngine.from_config({})
        assert engine.provider == "edge"

    def test_from_config_minimal(self):
        config = {"tts_provider": "edge", "tts_voice": "en-US-GuyNeural"}
        engine = TTSEngine.from_config(config)
        assert engine.provider == "edge"
        assert engine._provider.voice == "en-US-GuyNeural"

    def test_from_config_openai(self):
        config = {
            "tts_provider": "openai",
            "tts_model": "openai/tts-1-hd",
            "tts_voice": "shimmer",
            "tts_api_key": "sk-test",
        }
        engine = TTSEngine.from_config(config)
        assert engine.provider == "openai"
        assert engine._provider.model == "openai/tts-1-hd"
        assert engine._provider.voice == "shimmer"

    def test_from_config_local(self):
        config = {
            "tts_provider": "local",
            "tts_api_base": "http://127.0.0.1:9000",
            "tts_model": "my-model",
            "tts_voice": "my-voice",
        }
        engine = TTSEngine.from_config(config)
        assert engine.provider == "local"
        assert engine._provider.api_base == "http://127.0.0.1:9000"
        assert engine._provider.model == "my-model"

    def test_provider_normalized(self):
        config = {"tts_provider": "  EDGE  "}
        engine = TTSEngine.from_config(config)
        assert engine.provider == "edge"


class TestTTSEngineGenerate:
    """Test generate method edge cases (no actual TTS calls)."""

    def test_empty_text_returns_none(self):
        engine = TTSEngine.from_config({})
        result = asyncio.run(engine.generate(""))
        assert result is None

    def test_whitespace_text_returns_none(self):
        engine = TTSEngine.from_config({})
        result = asyncio.run(engine.generate("   "))
        assert result is None

    def test_none_text_returns_none(self):
        engine = TTSEngine.from_config({})
        result = asyncio.run(engine.generate(None))
        assert result is None


class TestProviders:
    """Test individual provider instantiation."""

    def test_edge_provider(self):
        p = EdgeTTSProvider(voice="tr-TR-AhmetNeural")
        assert p.name == "edge"
        assert p.voice == "tr-TR-AhmetNeural"

    def test_local_provider(self):
        p = LocalTTSProvider(api_base="http://localhost:8000", model="piper", voice="tr")
        assert p.name == "local"
        assert p.api_base == "http://localhost:8000"

    def test_proxy_provider(self):
        p = ProxyTTSProvider()
        assert p.name == "proxy"
        assert p.api_base == "http://127.0.0.1:5111"


class TestOggConversion:
    """Test ffmpeg OGG conversion (requires ffmpeg installed)."""

    def test_convert_nonexistent_file(self):
        result = asyncio.run(_convert_to_ogg("/nonexistent/file.mp3"))
        assert result is None
