"""One-time script to generate Pyrogram session string and create config.json.

Run this, enter your credentials, and it will create a ready-to-use config.json.
"""

import asyncio
import json
import os

# Pyrogram 2.0.106 calls asyncio.get_event_loop() at import time,
# which fails on Python 3.12+. Create a loop first.
asyncio.set_event_loop(asyncio.new_event_loop())

from pyrogram import Client

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


async def main():
    print("=== LiveClaw Setup ===\n")

    # Step 1: Telegram API credentials
    print("-- Telegram API (my.telegram.org) --")
    api_id = input("api_id: ").strip()
    api_hash = input("api_hash: ").strip()

    # Step 2: Generate session string
    print("\nTelegram'a baglaniliyor...")
    async with Client(
        name="liveclaw_session",
        api_id=int(api_id),
        api_hash=api_hash,
        in_memory=True,
    ) as app:
        session_string = await app.export_session_string()
        me = await app.get_me()
        print(f"Giris yapildi: {me.first_name} (ID: {me.id})\n")

    # Step 3: Bot credentials
    print("-- OpenClaw Bot --")
    bot_token = input("Bot token (OpenClaw config'inden): ").strip()
    bot_user_id = input("Bot user ID (bilmiyorsan bos birak, otomatik bulunur): ").strip()

    # Auto-detect bot_user_id from token if not provided
    if not bot_user_id and bot_token:
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{bot_token}/getMe"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    data = await resp.json()
                    if data.get("ok"):
                        bot_user_id = str(data["result"]["id"])
                        bot_name = data["result"].get("first_name", "")
                        print(f"Bot bulundu: {bot_name} (ID: {bot_user_id})")
        except Exception as e:
            print(f"Bot ID otomatik bulunamadi: {e}")

    if not bot_user_id:
        bot_user_id = input("Bot user ID (zorunlu): ").strip()

    # Step 4: Target chat
    print()
    target_chat_id = input("Target chat ID (OpenClaw ile konustugum chat): ").strip()

    # Step 5: Write config.json
    config = {
        "api_id": int(api_id),
        "api_hash": api_hash,
        "session_string": session_string,
        "target_chat_id": int(target_chat_id),
        "bot_token": bot_token,
        "bot_user_id": int(bot_user_id),
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
        "classifier_model": "",
        "classifier_api_key": "",
        "classifier_timeout": 3.0,
        "audio_library_dir": "audio_library",
        "shortcuts": {"record": "<ctrl>+<shift>+r"},
        "recording": {"sample_rate": 48000, "channels": 1, "max_duration_seconds": 120},
        "playback": {"player": "mpv", "volume": 100},
        "log_file": "liveclaw.log",
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"\nconfig.json yazildi: {CONFIG_PATH}")
    print("Artik 'python main.py' ile calistirabilirsin.")


if __name__ == "__main__":
    asyncio.run(main())
