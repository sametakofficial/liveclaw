#!/usr/bin/env python3
"""LiveClaw — live voice protocol for OpenClaw AI over Telegram.

Intercepts bot text messages, deletes them, and replaces with voice.
Keeps existing voice playback and mic recording functionality.

Usage:
    python main.py                  # Run LiveClaw
    python main.py --generate-library  # Generate audio library clips
"""

import argparse
import asyncio
import logging
import platform
import shutil
import signal
import sys

from pyrogram import Client, filters
from pyrogram.types import Message
from pynput import keyboard

from audio_library import AudioLibrary
from classifier import MessageClassifier
from config import load_config
from interceptor import MessageInterceptor
from player import VoicePlayer
from recorder import VoiceRecorder
from tts_engine import TTSEngine

logger = logging.getLogger("liveclaw")


def setup_logging(log_file: str) -> None:
    """Configure logging to file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


class LiveClaw:
    """Main application: orchestrates all modules."""

    def __init__(self, config: dict):
        self.config = config
        self.target_chat_id = config["target_chat_id"]
        self.loop: asyncio.AbstractEventLoop | None = None

        # Pyrogram userbot client
        self.client = Client(
            name="liveclaw",
            api_id=config["api_id"],
            api_hash=config["api_hash"],
            session_string=config["session_string"],
            no_updates=False,
        )

        # TTS engine
        self.tts = TTSEngine.from_config(config)

        # Audio library
        lib_dir = config.get("audio_library_dir", "audio_library")
        self.audio_library = AudioLibrary(
            library_dir=lib_dir,
            manifest_path="library_manifest.json",
        )

        # Classifier (needs audio library for prompt context)
        self.classifier: MessageClassifier | None = None

        # Interceptor
        self.interceptor: MessageInterceptor | None = None

        # Player
        playback_cfg = config.get("playback", {})
        self.player = VoicePlayer(
            player=playback_cfg.get("player", "mpv"),
            volume=playback_cfg.get("volume", 100),
        )

        # Recorder (initialized after client starts)
        self.recorder: VoiceRecorder | None = None

        # Keyboard listener
        self._kb_listener: keyboard.GlobalHotkeys | None = None

    async def start(self) -> None:
        """Start all modules."""
        self.loop = asyncio.get_running_loop()

        # Check audio player binary
        player_name = self.config.get("playback", {}).get("player", "mpv")
        if shutil.which(player_name) is None:
            logger.error(
                f"Audio player '{player_name}' not found. "
                f"Install it or change playback.player in config.json."
            )
            sys.exit(1)

        # Check ffmpeg
        if shutil.which("ffmpeg") is None:
            logger.error("ffmpeg not found. Install ffmpeg for audio conversion.")
            sys.exit(1)

        # Load audio library
        self.audio_library.load()

        # Init classifier
        self.classifier = MessageClassifier.from_config(
            self.config, self.audio_library
        )

        # Start Pyrogram
        await self.client.start()
        me = await self.client.get_me()
        logger.info(f"Logged in as {me.first_name} (ID: {me.id})")

        # Init recorder
        rec_cfg = self.config.get("recording", {})
        self.recorder = VoiceRecorder(
            client=self.client,
            target_chat_id=self.target_chat_id,
            sample_rate=rec_cfg.get("sample_rate", 48000),
            channels=rec_cfg.get("channels", 1),
            max_duration=rec_cfg.get("max_duration_seconds", 120),
        )

        # Start player
        await self.player.start()

        # Start interceptor
        self.interceptor = MessageInterceptor(
            client=self.client,
            bot_token=self.config["bot_token"],
            bot_user_id=self.config["bot_user_id"],
            target_chat_id=self.target_chat_id,
            classifier=self.classifier,
            tts_engine=self.tts,
            audio_library=self.audio_library,
        )
        await self.interceptor.start()

        # Register voice message handler (bot voice messages → player)
        bot_user_id = self.config["bot_user_id"]

        @self.client.on_message(
            filters.chat(self.target_chat_id)
            & filters.user(bot_user_id)
            & (filters.voice | filters.audio)
        )
        async def on_voice(client: Client, message: Message):
            sender = message.from_user
            name = sender.first_name if sender else "Unknown"
            logger.info(f"Voice message from {name}")
            await self.player.enqueue(client, message)

        # Start keyboard listener
        self._start_keyboard_listener()

        logger.info(
            f"LiveClaw running. Chat: {self.target_chat_id}. "
            f"Bot: {bot_user_id}. TTS: {self.tts.provider}. "
            f"Press {self.config.get('shortcuts', {}).get('record', 'Ctrl+Shift+R')} to record."
        )

    async def stop(self) -> None:
        """Gracefully shut down all modules."""
        logger.info("Shutting down...")

        if self._kb_listener is not None:
            self._kb_listener.stop()

        if self.interceptor is not None:
            await self.interceptor.stop()

        if self.recorder is not None:
            await self.recorder.cleanup()

        await self.player.stop()

        try:
            await self.client.stop()
        except ConnectionError:
            pass

        logger.info("Shutdown complete.")

    def _start_keyboard_listener(self) -> None:
        """Register global hotkey for mic recording toggle."""
        shortcut = self.config.get("shortcuts", {}).get("record", "<ctrl>+<shift>+r")

        def on_record():
            if self.recorder is None or self.loop is None:
                return
            state = "stop" if self.recorder.is_recording else "start"
            logger.info(f"Hotkey — {state} recording")
            self.recorder.toggle(self.loop)

        listener = keyboard.GlobalHotkeys({shortcut: on_record})
        listener.start()
        self._kb_listener = listener
        logger.info(f"Hotkey registered: {shortcut}")


async def run() -> None:
    """Main entry point: load config, start LiveClaw, wait for shutdown."""
    config = load_config()
    setup_logging(config.get("log_file", "liveclaw.log"))

    app = LiveClaw(config)

    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received shutdown signal")
        stop_event.set()

    # Signal handlers only work on Unix
    if platform.system() != "Windows":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    await app.start()
    await stop_event.wait()
    await app.stop()


async def generate_library() -> None:
    """Generate audio library clips using TTS engine, then exit."""
    config = load_config()
    setup_logging(config.get("log_file", "liveclaw.log"))

    tts = TTSEngine.from_config(config)
    lib = AudioLibrary(
        library_dir=config.get("audio_library_dir", "audio_library"),
        manifest_path="library_manifest.json",
    )

    if not lib.load():
        logger.error("Failed to load library manifest")
        sys.exit(1)

    logger.info("Generating audio library...")
    await lib.generate_library(tts)
    logger.info("Done.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="LiveClaw — voice protocol for OpenClaw")
    parser.add_argument(
        "--generate-library",
        action="store_true",
        help="Generate audio library clips using TTS and exit",
    )
    args = parser.parse_args()

    try:
        if args.generate_library:
            asyncio.run(generate_library())
        else:
            asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
