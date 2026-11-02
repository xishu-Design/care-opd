#!/usr/bin/env bash
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:5173}"

echo "Checking GuardTCM backend: $BACKEND_URL"
curl -fsS "$BACKEND_URL/health"
echo
curl -fsS "$BACKEND_URL/api/version"
echo
echo "Demo account: doctor_demo / password123"
echo "Open frontend: $FRONTEND_URL"
echo "After login, open: $FRONTEND_URL/harness/demo"
