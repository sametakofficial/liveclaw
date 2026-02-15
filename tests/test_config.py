"""Tests for config.py — configuration loading and validation."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import validate_config, apply_defaults, _DEFAULTS


def _make_valid_config() -> dict:
    """Return a minimal valid config dict."""
    return {
        "api_id": 12345678,
        "api_hash": "abc123",
        "session_string": "session_abc",
        "target_chat_id": -1001234567890,
        "bot_token": "123:ABC",
        "bot_user_id": 999999,
    }


class TestValidateConfig:
    """Test config validation logic."""

    def test_valid_config_passes(self):
        config = _make_valid_config()
        errors = validate_config(config)
        assert errors == []

    def test_coerces_string_int(self):
        config = _make_valid_config()
        config["api_id"] = "12345678"
        validate_config(config)
        assert config["api_id"] == 12345678
        assert isinstance(config["api_id"], int)

    def test_coerces_string_chat_id(self):
        config = _make_valid_config()
        config["target_chat_id"] = "-1001234567890"
        validate_config(config)
        assert isinstance(config["target_chat_id"], int)


class TestApplyDefaults:
    """Test default value application."""

    def test_applies_missing_defaults(self):
        config = _make_valid_config()
        apply_defaults(config)

        assert config["tts_provider"] == "proxy"
        assert config["tts_voice"] == "Decent_Boy"
        assert config["tts_api_base"] == "http://127.0.0.1:5111"
        assert config["stt_provider"] == "proxy"
        assert config["stt_model"] == "whisper-large-v3-turbo"
        assert config["log_file"] == "liveclaw.log"
        assert config["classifier_model"] == "anthropic/claude-3-5-haiku-20241022"

    def test_preserves_existing_values(self):
        config = _make_valid_config()
        config["tts_provider"] = "openai"
        config["tts_voice"] = "alloy"
        apply_defaults(config)

        assert config["tts_provider"] == "openai"
        assert config["tts_voice"] == "alloy"

    def test_all_defaults_present(self):
        config = _make_valid_config()
        apply_defaults(config)

        for key in _DEFAULTS:
            assert key in config, f"Default key '{key}' not applied"


class TestTTSEngineFromConfig:
    """Test TTSEngine.from_config integration with config defaults."""

    def test_proxy_default(self):
        from tts_engine import TTSEngine

        config = _make_valid_config()
        apply_defaults(config)
        engine = TTSEngine.from_config(config)

        assert engine.provider == "proxy"
        assert engine.voice == "Decent_Boy"
        assert engine.api_base == "http://127.0.0.1:5111"

    def test_openai_config(self):
        from tts_engine import TTSEngine

        config = _make_valid_config()
        config["tts_provider"] = "openai"
        config["tts_model"] = "openai/tts-1"
        config["tts_voice"] = "nova"
        config["tts_api_key"] = "sk-test"
        apply_defaults(config)
        engine = TTSEngine.from_config(config)

        assert engine.provider == "openai"
        assert engine.model == "openai/tts-1"
        assert engine.voice == "nova"
        assert engine.api_key == "sk-test"
