"""Pre-recorded audio library for common bot responses.

Manages a directory of .ogg clips mapped to intent keys via a JSON manifest.
Provides instant voice responses for recognized patterns without TTS latency.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AudioLibrary:
    """Manages pre-recorded audio clips for common bot response categories."""

    def __init__(self, library_dir: str = "audio_library", manifest_path: str = "library_manifest.json"):
        self._library_dir = Path(library_dir)
        self._manifest_path = Path(manifest_path)
        self._manifest: dict[str, dict] = {}
        self._loaded = False

    def load(self) -> bool:
        """Load manifest and validate referenced audio files.

        Returns True if manifest loaded successfully (even if some files are missing).
        Returns False if manifest file doesn't exist or is invalid JSON.
        """
        if not self._manifest_path.exists():
            logger.warning(f"Manifest not found: {self._manifest_path}")
            return False

        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load manifest: {e}")
            return False

        # Validate that referenced files exist
        missing = []
        for key, entry in self._manifest.items():
            audio_path = self._library_dir / entry.get("file", "")
            if not audio_path.exists():
                missing.append(key)

        if missing:
            logger.info(
                f"Audio library: {len(missing)} clips missing ({', '.join(missing)}). "
                f"Run --generate-library to create them."
            )

        self._loaded = True
        available = len(self._manifest) - len(missing)
        logger.info(f"Audio library loaded: {available}/{len(self._manifest)} clips available")
        return True

    def get(self, key: str) -> Optional[str]:
        """Get absolute path to audio file for an intent key.

        Returns None if key not found or file doesn't exist.
        """
        entry = self._manifest.get(key)
        if entry is None:
            return None

        audio_path = self._library_dir / entry["file"]
        if not audio_path.exists():
            return None

        return str(audio_path.resolve())

    def get_manifest_for_prompt(self) -> str:
        """Format manifest as text for the classifier LLM prompt.

        Returns a string describing all available categories with their
        descriptions and example phrases, so the LLM knows what to classify into.
        """
        if not self._manifest:
            return ""

        lines = []
        for key, entry in self._manifest.items():
            desc = entry.get("description", "")
            examples = entry.get("examples", [])
            examples_str = ", ".join(f'"{e}"' for e in examples[:4])
            lines.append(f'- "{key}": {desc}. Examples: {examples_str}')

        return "\n".join(lines)

    @property
    def keys(self) -> list[str]:
        """All intent keys defined in the manifest."""
        return list(self._manifest.keys())

    async def generate_library(self, tts_engine) -> None:
        """Auto-generate all audio clips using a TTS engine.

        Reads tts_text from each manifest entry and generates .ogg files.
        Skips entries that already have existing audio files.
        """
        if not self._manifest:
            logger.error("No manifest loaded. Call load() first.")
            return

        self._library_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        skipped = 0
        failed = 0

        for key, entry in self._manifest.items():
            audio_path = self._library_dir / entry["file"]

            if audio_path.exists():
                logger.info(f"  [{key}] already exists, skipping")
                skipped += 1
                continue

            tts_text = entry.get("tts_text", entry.get("description", key))
            logger.info(f"  [{key}] generating: \"{tts_text}\"")

            try:
                result_path = await tts_engine.generate(tts_text)
                if result_path is None:
                    logger.error(f"  [{key}] TTS returned None")
                    failed += 1
                    continue

                # Move generated file to library directory
                result = Path(result_path)
                result.rename(audio_path)
                generated += 1
                logger.info(f"  [{key}] saved to {audio_path}")

            except Exception as e:
                logger.error(f"  [{key}] generation failed: {e}")
                failed += 1

        logger.info(
            f"Library generation complete: {generated} generated, "
            f"{skipped} skipped, {failed} failed"
        )
