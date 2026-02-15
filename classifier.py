"""Hybrid message classifier: regex-first with LLM fallback.

Pipeline:
  1. Try regex pattern match (< 1ms) → return audio library key
  2. No match → call LLM via LiteLLM (with timeout) → classify or generate TTS
  3. LLM fails/timeout → fallback to TTS action

Returns:
  {"action": "library", "key": "ack_started"}   — use pre-recorded clip
  {"action": "tts", "text": "cleaned text"}      — generate TTS on the fly
"""

import asyncio
import json
import logging
from typing import Optional

import patterns

logger = logging.getLogger(__name__)

# Result type aliases
RESULT_LIBRARY = "library"
RESULT_TTS = "tts"


class MessageClassifier:
    """Classifies bot messages into audio library categories or TTS targets."""

    def __init__(
        self,
        model: str = "anthropic/claude-3-5-haiku-20241022",
        api_key: str = "",
        timeout: float = 3.0,
        library_prompt: str = "",
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._system_prompt = self._build_system_prompt(library_prompt)

    @classmethod
    def from_config(cls, config: dict, audio_library) -> "MessageClassifier":
        """Create classifier from config + audio library manifest."""
        return cls(
            model=config.get("classifier_model", "anthropic/claude-3-5-haiku-20241022"),
            api_key=config.get("classifier_api_key", ""),
            timeout=config.get("classifier_timeout", 3.0),
            library_prompt=audio_library.get_manifest_for_prompt(),
        )

    async def classify(self, text: str) -> dict:
        """Classify a message text.

        Returns dict with:
          {"action": "library", "key": "<intent_key>"}
          {"action": "tts", "text": "<cleaned_text>"}
        """
        if not text or not text.strip():
            return {"action": RESULT_TTS, "text": ""}

        # Step 1: Fast regex match
        key = patterns.match(text)
        if key is not None:
            logger.debug(f"Regex match: '{text[:50]}' → {key}")
            return {"action": RESULT_LIBRARY, "key": key}

        # Step 2: LLM classification (with timeout)
        llm_result = await self._classify_llm(text)
        if llm_result is not None:
            return llm_result

        # Step 3: Fallback — send raw text to TTS
        logger.debug(f"Fallback to TTS for: '{text[:50]}'")
        return {"action": RESULT_TTS, "text": self._clean_for_speech(text)}

    async def _classify_llm(self, text: str) -> Optional[dict]:
        """Attempt LLM classification. Returns None on failure/timeout."""
        if not self.model:
            return None

        try:
            return await asyncio.wait_for(
                self._call_llm(text), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"LLM classification timed out ({self.timeout}s)")
            return None
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None

    async def _call_llm(self, text: str) -> Optional[dict]:
        """Make the actual LiteLLM completion call."""
        import litellm

        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 150,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key

        response = await litellm.acompletion(**kwargs)
        raw = response.choices[0].message.content.strip()

        return self._parse_llm_response(raw, text)

    def _parse_llm_response(self, raw: str, original_text: str) -> Optional[dict]:
        """Parse LLM JSON response into a classification result."""
        try:
            # Try to extract JSON from response (LLM might wrap it in markdown)
            cleaned = raw
            if "```" in cleaned:
                # Extract content between code fences
                parts = cleaned.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        cleaned = part
                        break

            data = json.loads(cleaned)
            action = data.get("action", "")

            if action == "library" and "key" in data:
                return {"action": RESULT_LIBRARY, "key": data["key"]}
            elif action == "tts":
                tts_text = data.get("text", original_text)
                return {"action": RESULT_TTS, "text": self._clean_for_speech(tts_text)}

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug(f"Failed to parse LLM response: {e}, raw: {raw[:100]}")

        return None

    @staticmethod
    def _clean_for_speech(text: str) -> str:
        """Clean text for TTS: remove markdown, code blocks, URLs, etc."""
        import re

        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code
        text = re.sub(r"`[^`]+`", "", text)
        # Remove markdown headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Remove markdown bold/italic
        text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
        # Remove markdown links [text](url) — must run before bare URL removal
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove bare URLs
        text = re.sub(r"https?://\S+", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def _build_system_prompt(library_prompt: str) -> str:
        """Build the system prompt for the classifier LLM."""
        base = (
            "You classify bot messages into categories. "
            "Respond with JSON only, no explanation.\n\n"
            "If the message matches a common acknowledgment or status update, return:\n"
            '{"action": "library", "key": "<category_key>"}\n\n'
            "If the message contains unique/specific content that needs to be spoken, return:\n"
            '{"action": "tts", "text": "<cleaned text suitable for speech>"}\n\n'
            "For the tts action, clean the text: remove markdown, code blocks, URLs. "
            "Keep it natural for spoken delivery.\n"
        )

        if library_prompt:
            base += f"\nAvailable categories:\n{library_prompt}\n"

        return base
