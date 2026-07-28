#!/usr/bin/env bash
set -euo pipefail

WEBUI_PORT="${WEBUI_PORT:-7865}"

cd /workspace/wan-training-webui
source /venv/main/bin/activate
exec uvicorn webui.server:app --host 0.0.0.0 --port "${WEBUI_PORT}"
