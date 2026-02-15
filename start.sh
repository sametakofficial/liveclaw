#!/bin/bash
# LiveClaw launcher — runs in background, survives terminal close
cd /home/samet/Workspace/liveclaw
exec .venv/bin/python3 main.py >> /home/samet/liveclaw.log 2>&1
