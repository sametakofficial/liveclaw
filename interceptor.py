"""Message interceptor — core pipeline for LiveClaw.

Watches for OpenClaw bot text messages in the target chat, deletes them,
classifies the content, and sends voice replacements.

Architecture:
  - Userbot (Pyrogram MTProto): listens for messages + deletes them
  - Bot API (HTTP): sends voice messages (so they appear from the bot)

Flow per message:
  1. Userbot intercepts bot text message
  2. Userbot deletes it (using its own message IDs)
  3. CLASSIFY via regex-first + LLM fallback
  4. VOICE via audio library hit or TTS generation
  5. Bot API sends voice message with caption
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
        user_id: int,
        classifier: MessageClassifier,
        tts_engine: TTSEngine,
        audio_library: AudioLibrary,
    ):
        self._client = client          # Userbot — listens + deletes
        self._bot_token = bot_token    # For Bot API HTTP calls (send voice)
        self._bot_user_id = bot_user_id
        self._target_chat_id = target_chat_id
        self._user_id = user_id
        self._classifier = classifier
        self._tts = tts_engine
        self._library = audio_library

        self._queue: asyncio.Queue[tuple[str, int, int]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Duplicate detection
        self._seen_ids: deque[int] = deque(maxlen=100)

    async def start(self) -> None:
        """Register handler and start the worker."""
        self._http_session = aiohttp.ClientSession()
        self._worker_task = asyncio.create_task(self._worker())

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

        if self._http_session:
            await self._http_session.close()
            self._http_session = None

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

        # Delete via userbot (uses userbot's own message IDs — always works)
        asyncio.create_task(self._delete_message(message.chat.id, message.id))

        # Enqueue for voice processing (use user_id for Bot API send)
        await self._queue.put((text, self._user_id, int(time.time())))
        logger.info(f"Queued: '{text[:60]}' (queue: {self._queue.qsize()})")

    async def _delete_message(self, chat_id: int, message_id: int) -> None:
        """Delete a message via userbot (Pyrogram).

        Userbot uses its own message counter, so IDs always match.
        No FloodWait risk — this is a regular user action.
        """
        try:
            await self._client.delete_messages(chat_id, message_id)
            logger.info(f"Deleted message {message_id}")
        except Exception as e:
            logger.warning(f"Delete failed for msg {message_id}: {e}")

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
        """Send voice message via Bot API HTTP (appears from bot, not user)."""
        if not self._http_session:
            logger.error("No HTTP session for send_voice")
            return

        url = _BOT_API.format(token=self._bot_token, method="sendVoice")

        try:
            with open(audio_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field("chat_id", str(chat_id))
                form.add_field("voice", f, filename="voice.ogg", content_type="audio/ogg")
                if caption:
                    form.add_field("caption", caption[:1024])

                async with self._http_session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    data = await resp.json()
                    if resp.status == 200 and data.get("ok"):
                        logger.info(f"Voice sent to {chat_id} (via bot)")
                    else:
                        desc = data.get("description", "unknown")
                        logger.error(f"Bot sendVoice failed: {desc}")
                        # Fallback to userbot
                        await self._send_voice_userbot(audio_path, caption)
        except Exception as e:
            logger.error(f"Bot API send_voice failed: {e}")
            await self._send_voice_userbot(audio_path, caption)

    async def _send_voice_userbot(self, audio_path: str, caption: str = "") -> None:
        """Fallback: send voice via userbot."""
        try:
            await self._client.send_voice(
                chat_id=self._target_chat_id,
                voice=audio_path,
                caption=caption[:1024] if caption else None,
            )
            logger.info(f"Voice sent (via userbot fallback)")
        except Exception as e:
            logger.error(f"Userbot send_voice also failed: {e}")
