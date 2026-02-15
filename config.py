"""Configuration loading and validation for LiveClaw."""

import json
import sys
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent / "config.json"

# Required keys: (key_name, expected_type)
_REQUIRED = {
    "api_id": int,
    "api_hash": str,
    "session_string": str,
    "target_chat_id": int,
    "bot_token": str,
    "bot_user_id": int,
}

# Optional keys with defaults
_DEFAULTS: dict[str, Any] = {
    "tts_provider": "proxy",
    "tts_model": "tts-1",
    "tts_voice": "Decent_Boy",
    "tts_api_key": "",
    "tts_api_base": "http://127.0.0.1:5111",
    "stt_provider": "proxy",
    "stt_model": "whisper-large-v3-turbo",
    "stt_api_key": "",
    "stt_api_base": "http://127.0.0.1:5111",
    "stt_language": "tr",
    "classifier_model": "anthropic/claude-3-5-haiku-20241022",
    "classifier_api_key": "",
    "classifier_timeout": 3.0,
    "audio_library_dir": "audio_library",
    "shortcuts": {"record": "<ctrl>+<shift>+r"},
    "recording": {"sample_rate": 48000, "channels": 1, "max_duration_seconds": 120},
    "playback": {"player": "mpv", "volume": 100},
    "log_file": "liveclaw.log",
}


def load_config(path: Path | None = None) -> dict:
    """Load config from JSON file, validate required keys, apply defaults.

    Args:
        path: Override config file path (default: config.json next to main.py)

    Returns:
        Validated config dict with defaults applied.

    Raises:
        SystemExit: If config file missing, invalid JSON, or required keys missing.
    """
    config_path = path or CONFIG_PATH

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Copy config.json.example to config.json and fill in your values.")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in {config_path}: {e}")
            sys.exit(1)

    validate_config(config)
    apply_defaults(config)
    return config


def validate_config(config: dict) -> list[str]:
    """Validate required keys exist and have correct types.

    Returns list of error messages (empty if valid).
    Also coerces types where possible (e.g. string "123" → int 123).
    """
    errors = []

    for key, expected_type in _REQUIRED.items():
        value = config.get(key)

        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(f"'{key}' is required")
            continue

        if not isinstance(value, expected_type):
            try:
                config[key] = expected_type(value)
            except (ValueError, TypeError):
                errors.append(f"'{key}' must be {expected_type.__name__}, got: {value!r}")

    if errors:
        for err in errors:
            print(f"Config error: {err}")
        sys.exit(1)

    return errors


def apply_defaults(config: dict) -> None:
    """Apply default values for missing optional keys."""
    for key, default in _DEFAULTS.items():
        if key not in config:
            config[key] = default
