#!/usr/bin/env bash
# Launch the FastAPI backend + React frontend dev server side by side.
set -euo pipefail

PORT_API="${VALONY_API_PORT:-8000}"
PORT_UI="${VALONY_UI_PORT:-5173}"

echo "🚀 FastAPI on :${PORT_API}"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT_API" --workers 1 &
API_PID=$!

echo "⚛️  React dev server on :${PORT_UI}"
cd frontend && npm run dev -- --port "$PORT_UI" &
UI_PID=$!

trap "kill $API_PID $UI_PID 2>/dev/null || true" EXIT
wait
