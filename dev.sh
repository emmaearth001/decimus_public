#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "No venv found at $ROOT/venv — create one with:"
  echo "  python3 -m venv venv && ./venv/bin/pip install -e api"
  exit 1
fi

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  (cd "$ROOT/frontend" && npm install)
fi

cleanup() {
  echo ""
  echo "Stopping..."
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "$WEB_PID" ]] && kill "$WEB_PID" 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

if lsof -ti:8000 >/dev/null 2>&1; then
  echo "Port 8000 in use — killing existing process"
  lsof -ti:8000 | xargs kill -9 2>/dev/null || true
  sleep 1
fi

echo "Starting API on :8000"
(cd "$ROOT/api" && "$PYTHON" run_web.py) &
API_PID=$!

echo "Starting frontend on :3000"
(cd "$ROOT/frontend" && npm run dev) &
WEB_PID=$!

echo ""
echo "  API      → http://localhost:8000"
echo "  Frontend → http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both."

wait
