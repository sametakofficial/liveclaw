"""Pytest configuration — fix Pyrogram 2.0.106 + Python 3.14 compatibility."""

import asyncio

# Pyrogram calls asyncio.get_event_loop() at import time,
# which raises on Python 3.12+. Set a loop before any imports.
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
