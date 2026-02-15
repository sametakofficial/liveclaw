"""Message interceptor — core pipeline for LiveClaw.

Watches for OpenClaw bot text messages in the target chat, deletes them,
classifies the content, and sends voice replacements.

Flow per message:
  1. DELETE via Bot API HTTP call (~50ms)
  2. CLASSIFY via regex-first + LLM fallback
  3. VOICE via audio library hit or TTS generation
  4. SEND as Telegram voice message

Adaptive queue strategy:
  Queue 0-2:  Normal (regex → LLM fallback → TTS/library)
  Queue 3-5:  Fast (regex only, skip LLM → TTS/library)
  Queue 6+:   Batch (concatenate texts → single TTS)
"""

import asyncio
import logging
import os
import time
from collections import deque
from typing import Optional

import aiohttp
from pyrogram import Client, filters
from pyrogram.types import Message

from audio_library import AudioLibrary
from classifier import MessageClassifier, RESULT_LIBRARY, RESULT_TTS
from tts_engine import TTSEngine

logger = logging.getLogger(__name__)

# Adaptive queue thresholds
_QUEUE_FAST_THRESHOLD = 3
_QUEUE_BATCH_THRESHOLD = 6

# Bot API base URL
_BOT_API = "https://api.telegram.org/bot{token}/{method}"


class MessageInterceptor:
    """Intercepts bot text messages and replaces them with voice."""

    def __init__(
        self,
        client: Client,
        bot_token: str,
        bot_user_id: int,
        target_chat_id: int,
        classifier: MessageClassifier,
        tts_engine: TTSEngine,
        audio_library: AudioLibrary,
    ):
        self._client = client
        self._bot_token = bot_token
        self._bot_user_id = bot_user_id
        self._target_chat_id = target_chat_id
        self._classifier = classifier
        self._tts = tts_engine
        self._library = audio_library

        self._queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Duplicate detection: track recent message IDs
        self._seen_ids: deque[int] = deque(maxlen=100)

    async def start(self) -> None:
        """Register Pyrogram handler and start the processing worker."""
        self._http_session = aiohttp.ClientSession()
        self._worker_task = asyncio.create_task(self._worker())

        # Register handler: text messages from bot in target chat
        self._client.on_message(
            filters.chat(self._target_chat_id)
            & filters.user(self._bot_user_id)
            & filters.text
        )(self._on_bot_message)

        logger.info(
            f"Interceptor started — watching bot {self._bot_user_id} "
            f"in chat {self._target_chat_id}"
        )

    async def stop(self) -> None:
        """Stop the worker and clean up resources."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

        logger.info("Interceptor stopped")

    async def _on_bot_message(self, client: Client, message: Message) -> None:
        """Pyrogram handler for incoming bot text messages."""
        # Skip non-text (voice, media, etc.)
        if not message.text:
            return

        # Duplicate check
        if message.id in self._seen_ids:
            return
        self._seen_ids.append(message.id)

        text = message.text.strip()
        if not text:
            return

        # Delete first, then queue for processing
        asyncio.create_task(self._delete_message(message.chat.id, message.id))

        # Enqueue: (text, chat_id, timestamp)
        await self._queue.put((text, message.chat.id, int(time.time())))
        logger.debug(f"Queued message: '{text[:60]}' (queue size: {self._queue.qsize()})")

    async def _worker(self) -> None:
        """Process queued messages sequentially with adaptive strategy."""
        while True:
            try:
                text, chat_id, ts = await self._queue.get()
                qsize = self._queue.qsize()

                try:
                    if qsize >= _QUEUE_BATCH_THRESHOLD:
                        # Batch mode: drain queue, combine texts
                        await self._process_batch(text, chat_id)
                    elif qsize >= _QUEUE_FAST_THRESHOLD:
                        # Fast mode: regex only, skip LLM
                        await self._process_fast(text, chat_id)
                    else:
                        # Normal mode: full pipeline
                        await self._process_normal(text, chat_id)
                except Exception as e:
                    logger.error(f"Worker error processing message: {e}")
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Worker loop error: {e}")

    async def _process_normal(self, text: str, chat_id: int) -> None:
        """Full pipeline: regex → LLM → library/TTS."""
        result = await self._classifier.classify(text)
        await self._handle_result(result, text, chat_id)

    async def _process_fast(self, text: str, chat_id: int) -> None:
        """Fast pipeline: regex only, skip LLM. Fallback to TTS."""
        import patterns as pat

        key = pat.match(text)
        if key is not None:
            result = {"action": RESULT_LIBRARY, "key": key}
        else:
            result = {"action": RESULT_TTS, "text": MessageClassifier._clean_for_speech(text)}

        await self._handle_result(result, text, chat_id)

    async def _process_batch(self, first_text: str, chat_id: int) -> None:
        """Batch mode: drain queue, combine all texts into single TTS."""
        texts = [first_text]

        # Drain remaining items from queue
        while not self._queue.empty():
            try:
                text, _, _ = self._queue.get_nowait()
                texts.append(text)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        combined = " ".join(texts)
        cleaned = MessageClassifier._clean_for_speech(combined)
        logger.info(f"Batch mode: combined {len(texts)} messages ({len(cleaned)} chars)")

        result = {"action": RESULT_TTS, "text": cleaned}
        await self._handle_result(result, combined, chat_id)

    async def _handle_result(self, result: dict, original_text: str, chat_id: int) -> None:
        """Route classification result to library or TTS, then send voice."""
        audio_path: Optional[str] = None
        is_temp = False

        try:
            if result["action"] == RESULT_LIBRARY:
                key = result["key"]
                audio_path = self._library.get(key)
                if audio_path:
                    logger.info(f"Library hit: {key}")
                else:
                    # Library file missing — fall back to TTS
                    logger.debug(f"Library miss for '{key}', falling back to TTS")
                    cleaned = MessageClassifier._clean_for_speech(original_text)
                    audio_path = await self._tts.generate(cleaned)
                    is_temp = True

            elif result["action"] == RESULT_TTS:
                tts_text = result.get("text", original_text)
                if tts_text:
                    audio_path = await self._tts.generate(tts_text)
                    is_temp = True

            if audio_path:
                await self._send_voice(chat_id, audio_path)
            else:
                logger.warning(f"No audio produced for: '{original_text[:60]}'")

        finally:
            # Clean up temp TTS files (not library files)
            if is_temp and audio_path:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    async def _delete_message(self, chat_id: int, message_id: int) -> None:
        """Delete a message via Bot API HTTP call.

        Uses the bot token (not userbot) so the bot deletes its own message.
        This removes it for everyone on all devices.
        """
        if not self._http_session:
            return

        url = _BOT_API.format(token=self._bot_token, method="deleteMessage")
        payload = {"chat_id": chat_id, "message_id": message_id}

        try:
            async with self._http_session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if not data.get("ok"):
                        logger.debug(f"Delete failed: {data.get('description', 'unknown')}")
                else:
                    logger.debug(f"Delete HTTP {resp.status} for message {message_id}")
        except Exception as e:
            logger.debug(f"Delete error for message {message_id}: {e}")

    async def _send_voice(self, chat_id: int, audio_path: str) -> None:
        """Send an audio file as a Telegram voice message via Pyrogram."""
        try:
            await self._client.send_voice(
                chat_id=chat_id,
                voice=audio_path,
            )
            logger.info(f"Voice sent to {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send voice: {e}")
