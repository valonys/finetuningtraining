#!/usr/bin/env bash
# Launch the FastAPI server + Gradio studio side by side.
set -euo pipefail

PORT_API="${VALONY_API_PORT:-8000}"
PORT_UI="${VALONY_UI_PORT:-7860}"

echo "🚀 FastAPI on :${PORT_API}"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT_API" --workers 1 &
API_PID=$!

echo "🎨 Gradio Studio on :${PORT_UI}"
VALONY_API_URL="http://localhost:${PORT_API}" python ui/studio.py --port "$PORT_UI" &
UI_PID=$!

trap "kill $API_PID $UI_PID 2>/dev/null || true" EXIT
wait
