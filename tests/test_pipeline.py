"""Integration/simulation tests for the interceptor pipeline.

Tests the full flow: message → classify → voice generation
without needing actual Telegram or external API connections.
Uses mocks for Pyrogram client, Bot API, and TTS.
"""

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from classifier import MessageClassifier
from interceptor import MessageInterceptor
from audio_library import AudioLibrary
from tts_engine import TTSEngine


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _make_fake_ogg(path: str) -> None:
    """Write a tiny fake OGG file for testing."""
    with open(path, "wb") as f:
        f.write(b"OggS" + b"\x00" * 100)


def _make_test_library(tmp_dir: str) -> AudioLibrary:
    """Create a test audio library with real files."""
    lib_dir = os.path.join(tmp_dir, "audio_library")
    os.makedirs(lib_dir, exist_ok=True)

    manifest = {
        "ack_started": {
            "file": "ack_started.ogg",
            "description": "Task started",
            "tts_text": "Basladim.",
            "examples": ["Tamam, basliyorum", "On it"],
        },
        "ack_done": {
            "file": "ack_done.ogg",
            "description": "Task done",
            "tts_text": "Tamamlandi.",
            "examples": ["Done", "Tamamlandi"],
        },
    }

    manifest_path = os.path.join(tmp_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    # Create actual audio files
    for entry in manifest.values():
        _make_fake_ogg(os.path.join(lib_dir, entry["file"]))

    lib = AudioLibrary(library_dir=lib_dir, manifest_path=manifest_path)
    lib.load()
    return lib


def _make_mock_message(text: str, msg_id: int = 1, user_id: int = 999, chat_id: int = -100123):
    """Create a mock Pyrogram Message object."""
    msg = MagicMock()
    msg.text = text
    msg.id = msg_id
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.chat = MagicMock()
    msg.chat.id = chat_id
    return msg


# ─── Classification Pipeline Tests ────────────────────────────────────────────


class TestClassificationPipeline:
    """Test the full classify → route decision pipeline."""

    def test_ack_message_routes_to_library(self):
        """Common ack messages should hit the audio library."""
        classifier = MessageClassifier(model="", api_key="", timeout=1.0)
        result = asyncio.run(classifier.classify("Tamam, başlıyorum"))
        assert result["action"] == "library"
        assert result["key"] == "ack_started"

    def test_done_message_routes_to_library(self):
        classifier = MessageClassifier(model="", api_key="", timeout=1.0)
        result = asyncio.run(classifier.classify("Tamamlandı"))
        assert result["action"] == "library"
        assert result["key"] == "ack_done"

    def test_unique_content_routes_to_tts(self):
        """Specific content should fall through to TTS."""
        classifier = MessageClassifier(model="", api_key="", timeout=1.0)
        result = asyncio.run(classifier.classify(
            "Python 3.14 ile uyumluluk sorunu var, scipy kurulmuyor"
        ))
        assert result["action"] == "tts"
        assert len(result["text"]) > 0

    def test_markdown_cleaned_for_tts(self):
        """Markdown should be stripped before TTS."""
        classifier = MessageClassifier(model="", api_key="", timeout=1.0)
        result = asyncio.run(classifier.classify(
            "**Sonuc:** `pip install scipy` komutu ile kurabilirsin. Detaylar: https://scipy.org"
        ))
        assert result["action"] == "tts"
        assert "**" not in result["text"]
        assert "`" not in result["text"]
        assert "https://" not in result["text"]

    def test_multiple_messages_classified_independently(self):
        """Each message should be classified on its own."""
        classifier = MessageClassifier(model="", api_key="", timeout=1.0)

        messages = [
            ("Bakıyorum", "library"),
            ("API endpoint 3 parametre aliyor", "tts"),
            ("Hallettim", "library"),
            ("Düşünüyorum", "library"),
        ]

        for text, expected_action in messages:
            result = asyncio.run(classifier.classify(text))
            assert result["action"] == expected_action, f"Failed for: {text}"


# ─── Interceptor Pipeline Tests ───────────────────────────────────────────────


class TestInterceptorPipeline:
    """Test the interceptor message handling with mocks."""

    def _make_interceptor(self, tmp_dir: str) -> tuple[MessageInterceptor, MagicMock]:
        """Create an interceptor with mocked dependencies."""
        client = MagicMock()
        client.send_voice = AsyncMock()
        client.on_message = MagicMock(return_value=lambda f: f)

        audio_lib = _make_test_library(tmp_dir)
        classifier = MessageClassifier(model="", api_key="", timeout=1.0)

        tts = MagicMock(spec=TTSEngine)
        # TTS generate returns a fake ogg path
        fake_ogg = os.path.join(tmp_dir, "tts_output.ogg")
        _make_fake_ogg(fake_ogg)
        tts.generate = AsyncMock(return_value=fake_ogg)

        interceptor = MessageInterceptor(
            client=client,
            bot_token="fake:token",
            bot_user_id=999,
            target_chat_id=-100123,
            classifier=classifier,
            tts_engine=tts,
            audio_library=audio_lib,
        )

        return interceptor, client

    def test_library_hit_sends_cached_audio(self):
        """Ack message should send pre-recorded audio, not call TTS."""
        with tempfile.TemporaryDirectory() as tmp:
            interceptor, client = self._make_interceptor(tmp)

            asyncio.run(interceptor._process_normal("Tamam, başlıyorum", -100123))

            # Should have called send_voice
            assert client.send_voice.called
            # TTS should NOT have been called (library hit)
            assert not interceptor._tts.generate.called

    def test_unique_content_calls_tts(self):
        """Non-ack message should trigger TTS generation."""
        with tempfile.TemporaryDirectory() as tmp:
            interceptor, client = self._make_interceptor(tmp)

            asyncio.run(interceptor._process_normal(
                "Veritabani baglantisi basarili, 42 kayit bulundu", -100123
            ))

            # TTS should have been called
            assert interceptor._tts.generate.called
            # Should have sent voice
            assert client.send_voice.called

    def test_fast_mode_skips_llm(self):
        """Fast mode should only use regex, no LLM call."""
        with tempfile.TemporaryDirectory() as tmp:
            interceptor, client = self._make_interceptor(tmp)

            # Ack message in fast mode → library
            asyncio.run(interceptor._process_fast("Hallettim", -100123))
            assert client.send_voice.called
            assert not interceptor._tts.generate.called

    def test_fast_mode_unique_goes_to_tts(self):
        """Fast mode with unique content should go to TTS directly."""
        with tempfile.TemporaryDirectory() as tmp:
            interceptor, client = self._make_interceptor(tmp)

            asyncio.run(interceptor._process_fast(
                "Dosya 3.2MB boyutunda, indiriliyor", -100123
            ))
            assert interceptor._tts.generate.called

    def test_batch_mode_combines_messages(self):
        """Batch mode should combine texts and make single TTS call."""
        with tempfile.TemporaryDirectory() as tmp:
            interceptor, client = self._make_interceptor(tmp)

            # Pre-fill queue with extra messages
            import time
            asyncio.run(interceptor._queue.put(("Ikinci mesaj", -100123, int(time.time()))))
            asyncio.run(interceptor._queue.put(("Ucuncu mesaj", -100123, int(time.time()))))

            asyncio.run(interceptor._process_batch("Birinci mesaj", -100123))

            # TTS should be called once with combined text
            assert interceptor._tts.generate.call_count == 1
            call_text = interceptor._tts.generate.call_args[0][0]
            assert "Birinci" in call_text
            assert "Ikinci" in call_text
            assert "Ucuncu" in call_text

    def test_delete_message_called(self):
        """Message deletion should be attempted via Bot API."""
        with tempfile.TemporaryDirectory() as tmp:
            interceptor, client = self._make_interceptor(tmp)

            # Mock aiohttp session
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"ok": True})

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            ))
            interceptor._http_session = mock_session

            asyncio.run(interceptor._delete_message(-100123, 42))

            assert mock_session.post.called
            call_args = mock_session.post.call_args
            assert "deleteMessage" in call_args[0][0]

    def test_duplicate_message_ignored(self):
        """Same message ID should not be processed twice."""
        with tempfile.TemporaryDirectory() as tmp:
            interceptor, client = self._make_interceptor(tmp)

            msg = _make_mock_message("Tamam, başlıyorum", msg_id=42, user_id=999, chat_id=-100123)

            # Mock delete to avoid HTTP call
            interceptor._delete_message = AsyncMock()

            # First call — should queue
            asyncio.run(interceptor._on_bot_message(client, msg))
            assert interceptor._queue.qsize() == 1

            # Second call with same ID — should be ignored
            asyncio.run(interceptor._on_bot_message(client, msg))
            assert interceptor._queue.qsize() == 1  # Still 1, not 2

    def test_non_text_message_ignored(self):
        """Non-text messages (voice, media) should not be intercepted."""
        with tempfile.TemporaryDirectory() as tmp:
            interceptor, client = self._make_interceptor(tmp)
            interceptor._delete_message = AsyncMock()

            msg = _make_mock_message(None, msg_id=99)  # text=None (voice message)
            asyncio.run(interceptor._on_bot_message(client, msg))
            assert interceptor._queue.qsize() == 0


# ─── TTS Proxy Integration Test ──────────────────────────────────────────────


class TestTTSProxyIntegration:
    """Test TTS via proxy (requires tts-proxy running on :5111)."""

    @pytest.mark.skipif(
        not os.getenv("TEST_PROXY", ""),
        reason="Set TEST_PROXY=1 to run proxy integration tests",
    )
    def test_proxy_tts_generates_ogg(self):
        """TTS proxy should return a valid OGG file."""
        engine = TTSEngine(
            provider="proxy", model="tts-1", voice="Decent_Boy",
            api_base="http://127.0.0.1:5111",
        )
        path = asyncio.run(engine.generate("Test cümlesi"))
        assert path is not None
        assert os.path.exists(path)
        assert os.path.getsize(path) > 100
        os.unlink(path)

    @pytest.mark.skipif(
        not os.getenv("TEST_PROXY", ""),
        reason="Set TEST_PROXY=1 to run proxy integration tests",
    )
    def test_proxy_stt_transcribes(self):
        """STT proxy should transcribe audio to text."""
        from stt_engine import STTEngine

        # First generate audio
        tts = TTSEngine(
            provider="proxy", model="tts-1", voice="Decent_Boy",
            api_base="http://127.0.0.1:5111",
        )
        path = asyncio.run(tts.generate("Merhaba dünya"))
        assert path is not None

        # Then transcribe it
        stt = STTEngine(
            provider="proxy", model="whisper-large-v3-turbo",
            api_base="http://127.0.0.1:5111", language="tr",
        )
        text = asyncio.run(stt.transcribe(path))
        assert text is not None
        assert len(text) > 0
        os.unlink(path)
