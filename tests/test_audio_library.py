"""Tests for audio_library.py — pre-recorded audio clip management."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_library import AudioLibrary


def _create_test_library(tmp_dir: str) -> tuple[str, str]:
    """Create a test manifest and audio directory.

    Returns (manifest_path, library_dir).
    """
    lib_dir = os.path.join(tmp_dir, "audio_library")
    os.makedirs(lib_dir, exist_ok=True)

    manifest = {
        "ack_done": {
            "file": "ack_done.ogg",
            "description": "Task completed",
            "tts_text": "Done.",
            "examples": ["Done", "Finished"],
        },
        "ack_error": {
            "file": "ack_error.ogg",
            "description": "Error occurred",
            "tts_text": "Error.",
            "examples": ["Error", "Failed"],
        },
    }

    manifest_path = os.path.join(tmp_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path, lib_dir


class TestAudioLibraryLoad:
    """Test manifest loading and validation."""

    def test_load_valid_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, lib_dir = _create_test_library(tmp)
            lib = AudioLibrary(library_dir=lib_dir, manifest_path=manifest_path)
            assert lib.load() is True

    def test_load_missing_manifest(self):
        lib = AudioLibrary(manifest_path="/nonexistent/manifest.json")
        assert lib.load() is False

    def test_load_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = os.path.join(tmp, "bad.json")
            with open(bad_path, "w") as f:
                f.write("not json {{{")
            lib = AudioLibrary(manifest_path=bad_path)
            assert lib.load() is False


class TestAudioLibraryGet:
    """Test audio file retrieval."""

    def test_get_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, lib_dir = _create_test_library(tmp)

            # Create a fake .ogg file
            ogg_path = os.path.join(lib_dir, "ack_done.ogg")
            with open(ogg_path, "wb") as f:
                f.write(b"fake ogg data")

            lib = AudioLibrary(library_dir=lib_dir, manifest_path=manifest_path)
            lib.load()

            result = lib.get("ack_done")
            assert result is not None
            assert result.endswith("ack_done.ogg")

    def test_get_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, lib_dir = _create_test_library(tmp)
            lib = AudioLibrary(library_dir=lib_dir, manifest_path=manifest_path)
            lib.load()

            # File doesn't exist on disk
            assert lib.get("ack_done") is None

    def test_get_unknown_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, lib_dir = _create_test_library(tmp)
            lib = AudioLibrary(library_dir=lib_dir, manifest_path=manifest_path)
            lib.load()
            assert lib.get("nonexistent_key") is None


class TestAudioLibraryManifestPrompt:
    """Test manifest formatting for classifier prompt."""

    def test_manifest_prompt_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, lib_dir = _create_test_library(tmp)
            lib = AudioLibrary(library_dir=lib_dir, manifest_path=manifest_path)
            lib.load()

            prompt = lib.get_manifest_for_prompt()
            assert "ack_done" in prompt
            assert "ack_error" in prompt
            assert "Task completed" in prompt

    def test_manifest_prompt_empty(self):
        lib = AudioLibrary(manifest_path="/nonexistent")
        assert lib.get_manifest_for_prompt() == ""


class TestAudioLibraryKeys:
    """Test keys property."""

    def test_keys_after_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, lib_dir = _create_test_library(tmp)
            lib = AudioLibrary(library_dir=lib_dir, manifest_path=manifest_path)
            lib.load()
            assert set(lib.keys) == {"ack_done", "ack_error"}

    def test_keys_before_load(self):
        lib = AudioLibrary()
        assert lib.keys == []
