"""Tests for tts_engine.py — TTS abstraction layer."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tts_engine import TTSEngine


class TestTTSEngineInit:
    """Test engine initialization and config parsing."""

    def test_default_provider_is_edge(self):
        engine = TTSEngine()
        assert engine.provider == "edge"

    def test_from_config_minimal(self):
        config = {"tts_provider": "edge", "tts_voice": "en-US-GuyNeural"}
        engine = TTSEngine.from_config(config)
        assert engine.provider == "edge"
        assert engine.voice == "en-US-GuyNeural"

    def test_from_config_openai(self):
        config = {
            "tts_provider": "openai",
            "tts_model": "openai/tts-1-hd",
            "tts_voice": "shimmer",
            "tts_api_key": "sk-test",
        }
        engine = TTSEngine.from_config(config)
        assert engine.provider == "openai"
        assert engine.model == "openai/tts-1-hd"
        assert engine.voice == "shimmer"

    def test_provider_normalized(self):
        engine = TTSEngine(provider="  EDGE  ")
        assert engine.provider == "edge"


class TestTTSEngineGenerate:
    """Test generate method edge cases (no actual TTS calls)."""

    def test_empty_text_returns_none(self):
        engine = TTSEngine()
        result = asyncio.run(engine.generate(""))
        assert result is None

    def test_whitespace_text_returns_none(self):
        engine = TTSEngine()
        result = asyncio.run(engine.generate("   "))
        assert result is None

    def test_none_text_returns_none(self):
        engine = TTSEngine()
        result = asyncio.run(engine.generate(None))
        assert result is None


class TestOggConversion:
    """Test ffmpeg OGG conversion (requires ffmpeg installed)."""

    def test_convert_nonexistent_file(self):
        engine = TTSEngine()
        result = asyncio.run(engine._convert_to_ogg("/nonexistent/file.mp3"))
        assert result is None
