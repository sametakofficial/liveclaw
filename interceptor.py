"""Message interceptor — core pipeline for LiveClaw.

Watches for OpenClaw bot text messages in the target chat, deletes them,
classifies the content, and sends voice replacements.

Uses TWO Pyrogram clients:
  - Userbot (MTProto, user account): listens for messages
  - Bot client (MTProto, bot token): deletes messages + sends voice

This is necessary because private chat message IDs are per-account.
The userbot sees different IDs than the bot. Each client must use its own IDs.

Flow per message:
  1. Userbot intercepts bot text → extracts text
  2. Bot client deletes the original text (using bot's own message ID)
  3. CLASSIFY via regex-first + LLM fallback
  4. VOICE via audio library hit or TTS generation
  5. Bot client sends voice message with caption
"""

import asyncio
import logging
import os
import time
from collections import deque
from typing import Optional

from pyrogram import Client, filters
from pyrogram.types import Message

from audio_library import AudioLibrary
from classifier import MessageClassifier, RESULT_LIBRARY, RESULT_TTS
from tts_engine import TTSEngine

logger = logging.getLogger(__name__)

# Adaptive queue thresholds
_QUEUE_FAST_THRESHOLD = 3
_QUEUE_BATCH_THRESHOLD = 6


class MessageInterceptor:
    """Intercepts bot text messages and replaces them with voice."""

    def __init__(
        self,
        client: Client,
        bot_client: Client,
        bot_user_id: int,
        target_chat_id: int,
        user_id: int,
        classifier: MessageClassifier,
        tts_engine: TTSEngine,
        audio_library: AudioLibrary,
    ):
        self._client = client          # Userbot — listens
        self._bot = bot_client          # Bot — deletes + sends voice
        self._bot_user_id = bot_user_id
        self._target_chat_id = target_chat_id
        self._user_id = user_id
        self._classifier = classifier
        self._tts = tts_engine
        self._library = audio_library

        self._queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

        # Duplicate detection
        self._seen_ids: deque[int] = deque(maxlen=100)

        # Map userbot text hash → bot message ID for deletion
        # Bot handler stores: (chat_id, msg_id) keyed by (timestamp, text_hash)
        self._bot_msg_ids: dict[str, tuple[int, int]] = {}

    async def start(self) -> None:
        """Register handlers on both clients and start the worker."""
        self._worker_task = asyncio.create_task(self._worker())

        # Bot client handler: capture message IDs from bot's perspective
        @self._bot.on_message(
            filters.chat(self._user_id)
            & filters.outgoing
            & filters.text
        )
        async def on_bot_outgoing(client: Client, message: Message):
            """Bot sees its own outgoing messages with correct IDs for deletion."""
            if not message.text:
                return
            text_key = message.text.strip()[:100]
            self._bot_msg_ids[text_key] = (message.chat.id, message.id)
            logger.debug(f"Bot tracked msg {message.id}: '{text_key[:40]}'")

        # Userbot handler: intercept bot text messages
        @self._client.on_message(
            filters.chat(self._target_chat_id)
            & filters.user(self._bot_user_id)
            & filters.text
        )
        async def on_userbot_intercept(client: Client, message: Message):
            await self._on_bot_message(client, message)

        logger.info(
            f"Interceptor started — watching bot {self._bot_user_id} "
            f"in chat {self._target_chat_id}"
        )

    async def stop(self) -> None:
        """Stop the worker and clean up."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Interceptor stopped")

    async def _on_bot_message(self, client: Client, message: Message) -> None:
        """Userbot handler for incoming bot text messages."""
        if not message.text:
            return

        if message.id in self._seen_ids:
            return
        self._seen_ids.append(message.id)

        text = message.text.strip()
        if not text:
            return

        # Try to delete via bot client (correct message IDs)
        asyncio.create_task(self._delete_message(text))

        # Enqueue for voice processing
        await self._queue.put((text, self._user_id, int(time.time())))
        logger.info(f"Queued: '{text[:60]}' (queue: {self._queue.qsize()})")

    async def _delete_message(self, text: str) -> None:
        """Delete bot's text message using the bot client.

        Looks up the bot-side message ID by matching text content.
        Waits briefly for the bot handler to capture the ID.
        """
        text_key = text.strip()[:100]

        # Wait up to 2 seconds for bot handler to capture the message ID
        for _ in range(20):
            if text_key in self._bot_msg_ids:
                chat_id, msg_id = self._bot_msg_ids.pop(text_key)
                try:
                    await self._bot.delete_messages(chat_id, msg_id)
                    logger.info(f"Deleted message {msg_id} (via bot)")
                    return
                except Exception as e:
                    logger.warning(f"Bot delete failed for msg {msg_id}: {e}")
                    return
            await asyncio.sleep(0.1)

        # Fallback: try deleting via userbot
        logger.warning(f"Bot msg ID not found for: '{text_key[:40]}', trying userbot")
        try:
            # Search recent messages from bot in chat
            async for msg in self._client.get_chat_history(
                self._target_chat_id, limit=5
            ):
                if (msg.from_user and msg.from_user.id == self._bot_user_id
                        and msg.text and msg.text.strip()[:100] == text_key):
                    await self._client.delete_messages(self._target_chat_id, msg.id)
                    logger.info(f"Deleted message {msg.id} (via userbot fallback)")
                    return
        except Exception as e:
            logger.warning(f"Userbot delete fallback failed: {e}")

    async def _worker(self) -> None:
        """Process queued messages with adaptive strategy."""
        while True:
            try:
                text, chat_id, ts = await self._queue.get()
                qsize = self._queue.qsize()

                try:
                    if qsize >= _QUEUE_BATCH_THRESHOLD:
                        await self._process_batch(text, chat_id)
                    elif qsize >= _QUEUE_FAST_THRESHOLD:
                        await self._process_fast(text, chat_id)
                    else:
                        await self._process_normal(text, chat_id)
                except Exception as e:
                    logger.error(f"Worker error: {e}")
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Worker loop error: {e}")

    async def _process_normal(self, text: str, chat_id: int) -> None:
        result = await self._classifier.classify(text)
        await self._handle_result(result, text, chat_id)

    async def _process_fast(self, text: str, chat_id: int) -> None:
        import patterns as pat
        key = pat.match(text)
        if key is not None:
            result = {"action": RESULT_LIBRARY, "key": key}
        else:
            result = {"action": RESULT_TTS, "text": MessageClassifier._clean_for_speech(text)}
        await self._handle_result(result, text, chat_id)

    async def _process_batch(self, first_text: str, chat_id: int) -> None:
        texts = [first_text]
        while not self._queue.empty():
            try:
                text, _, _ = self._queue.get_nowait()
                texts.append(text)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

        combined = " ".join(texts)
        cleaned = MessageClassifier._clean_for_speech(combined)
        logger.info(f"Batch: {len(texts)} messages ({len(cleaned)} chars)")
        result = {"action": RESULT_TTS, "text": cleaned}
        await self._handle_result(result, combined, chat_id)

    async def _handle_result(self, result: dict, original_text: str, chat_id: int) -> None:
        audio_path: Optional[str] = None
        is_temp = False

        try:
            if result["action"] == RESULT_LIBRARY:
                key = result["key"]
                audio_path = self._library.get(key)
                if audio_path:
                    logger.info(f"Library hit: {key}")
                else:
                    cleaned = MessageClassifier._clean_for_speech(original_text)
                    audio_path = await self._tts.generate(cleaned)
                    is_temp = True

            elif result["action"] == RESULT_TTS:
                tts_text = result.get("text", original_text)
                if tts_text:
                    audio_path = await self._tts.generate(tts_text)
                    is_temp = True

            if audio_path:
                await self._send_voice(chat_id, audio_path, caption=original_text)
            else:
                logger.warning(f"No audio for: '{original_text[:60]}'")

        finally:
            if is_temp and audio_path:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    async def _send_voice(self, chat_id: int, audio_path: str, caption: str = "") -> None:
        """Send voice message via bot client."""
        try:
            await self._bot.send_voice(
                chat_id=chat_id,
                voice=audio_path,
                caption=caption[:1024] if caption else None,
            )
            logger.info(f"Voice sent to {chat_id} (via bot)")
        except Exception as e:
            logger.error(f"Bot send_voice failed: {e}")
            # Fallback to userbot
            try:
                await self._client.send_voice(
                    chat_id=chat_id,
                    voice=audio_path,
                    caption=caption[:1024] if caption else None,
                )
                logger.info(f"Voice sent to {chat_id} (via userbot fallback)")
            except Exception as e2:
                logger.error(f"Userbot fallback also failed: {e2}")
