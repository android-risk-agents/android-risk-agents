#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/vllm.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Env file not found: $ENV_FILE"
  echo "Create it from: $SCRIPT_DIR/vllm.env.example"
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${PROJECT_ID:?Missing PROJECT_ID in vllm.env}"
: "${ZONE:?Missing ZONE in vllm.env}"
: "${INSTANCE:?Missing INSTANCE in vllm.env}"
: "${MODEL_ID:?Missing MODEL_ID in vllm.env}"
: "${HOST:?Missing HOST in vllm.env}"
: "${PORT:?Missing PORT in vllm.env}"

echo "=== Smoke test vLLM ==="
echo "Project : $PROJECT_ID"
echo "Zone    : $ZONE"
echo "VM      : $INSTANCE"
echo "URL     : http://$HOST:$PORT (inside VM)"
echo

gcloud config set project "$PROJECT_ID" >/dev/null

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "
set -euo pipefail

echo 'Waiting for vLLM readiness...'
for i in \$(seq 1 120); do
  if curl -sf http://${HOST}:${PORT}/v1/models >/dev/null; then
    echo 'vLLM is ready.'
    break
  fi
  sleep 2
done

echo
echo '--- /v1/models (truncated) ---'
curl -s http://${HOST}:${PORT}/v1/models | head -c 1200
echo
echo
echo '--- chat/completions (truncated) ---'
curl -s http://${HOST}:${PORT}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    \"model\":\"${MODEL_ID}\",
    \"messages\":[
      {\"role\":\"system\",\"content\":\"You are concise.\"},
      {\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}
    ],
    \"temperature\":0
  }' | head -c 2500
echo
echo
echo '--- last logs ---'
tail -n 120 ~/vllm.log || true
"