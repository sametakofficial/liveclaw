"""Tests for classifier.py — message classification logic."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classifier import MessageClassifier


class TestCleanForSpeech:
    """Test the text cleaning function for TTS input."""

    def test_removes_code_blocks(self):
        text = "Here's the code:\n```python\nprint('hello')\n```\nThat's it."
        result = MessageClassifier._clean_for_speech(text)
        assert "```" not in result
        assert "print" not in result
        assert "That's it" in result

    def test_removes_inline_code(self):
        text = "Use the `print()` function to output text."
        result = MessageClassifier._clean_for_speech(text)
        assert "`" not in result
        assert "function" in result

    def test_removes_urls(self):
        text = "Check out https://example.com for more info."
        result = MessageClassifier._clean_for_speech(text)
        assert "https://" not in result
        assert "more info" in result

    def test_removes_markdown_links(self):
        text = "See [the docs](https://example.com/docs) for details."
        result = MessageClassifier._clean_for_speech(text)
        assert "the docs" in result
        assert "https://" not in result
        assert "[" not in result

    def test_removes_markdown_headers(self):
        text = "## Section Title\nSome content here."
        result = MessageClassifier._clean_for_speech(text)
        assert "##" not in result
        assert "Section Title" in result

    def test_removes_bold_italic(self):
        text = "This is **bold** and *italic* and ***both***."
        result = MessageClassifier._clean_for_speech(text)
        assert "**" not in result
        assert "*" not in result
        assert "bold" in result
        assert "italic" in result

    def test_collapses_whitespace(self):
        text = "Too   many    spaces   here."
        result = MessageClassifier._clean_for_speech(text)
        assert "  " not in result

    def test_empty_string(self):
        assert MessageClassifier._clean_for_speech("") == ""

    def test_plain_text_unchanged(self):
        text = "This is a normal sentence."
        result = MessageClassifier._clean_for_speech(text)
        assert result == text


class TestBuildSystemPrompt:
    """Test system prompt construction."""

    def test_prompt_without_library(self):
        prompt = MessageClassifier._build_system_prompt("")
        assert "library" in prompt.lower()
        assert "tts" in prompt.lower()
        assert "JSON" in prompt

    def test_prompt_with_library(self):
        lib_prompt = '- "ack_started": Task started. Examples: "On it"'
        prompt = MessageClassifier._build_system_prompt(lib_prompt)
        assert "ack_started" in prompt
        assert "Available categories" in prompt


class TestClassifierRegexPath:
    """Test that classifier uses regex before LLM."""

    def test_regex_hit_returns_library(self):
        classifier = MessageClassifier(model="", api_key="", timeout=1.0)
        result = asyncio.run(classifier.classify("Tamam, başlıyorum"))
        assert result["action"] == "library"
        assert result["key"] == "ack_started"

    def test_no_model_falls_back_to_tts(self):
        classifier = MessageClassifier(model="", api_key="", timeout=1.0)
        result = asyncio.run(
            classifier.classify("The API endpoint accepts a JSON payload with three required fields")
        )
        assert result["action"] == "tts"
        assert len(result["text"]) > 0

    def test_empty_text(self):
        classifier = MessageClassifier(model="", api_key="", timeout=1.0)
        result = asyncio.run(classifier.classify(""))
        assert result["action"] == "tts"


class TestParseResponse:
    """Test LLM response parsing."""

    def test_valid_library_response(self):
        classifier = MessageClassifier(model="")
        result = classifier._parse_llm_response(
            '{"action": "library", "key": "ack_done"}', "original"
        )
        assert result == {"action": "library", "key": "ack_done"}

    def test_valid_tts_response(self):
        classifier = MessageClassifier(model="")
        result = classifier._parse_llm_response(
            '{"action": "tts", "text": "some cleaned text"}', "original"
        )
        assert result["action"] == "tts"
        assert result["text"] == "some cleaned text"

    def test_json_in_code_fence(self):
        classifier = MessageClassifier(model="")
        raw = '```json\n{"action": "library", "key": "ack_started"}\n```'
        result = classifier._parse_llm_response(raw, "original")
        assert result is not None
        assert result["key"] == "ack_started"

    def test_invalid_json(self):
        classifier = MessageClassifier(model="")
        result = classifier._parse_llm_response("not json at all", "original")
        assert result is None

    def test_missing_action(self):
        classifier = MessageClassifier(model="")
        result = classifier._parse_llm_response('{"key": "ack_done"}', "original")
        assert result is None
