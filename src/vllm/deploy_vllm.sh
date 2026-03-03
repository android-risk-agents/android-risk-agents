#!/usr/bin/env bash
set -euo pipefail

# Always resolve paths relative to this script (so you can run it from repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/vllm.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Env file not found: $ENV_FILE"
  echo "Create it from: $SCRIPT_DIR/vllm.env.example"
  exit 1
fi

# Load env vars
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Required vars
: "${PROJECT_ID:?Missing PROJECT_ID in vllm.env}"
: "${ZONE:?Missing ZONE in vllm.env}"
: "${INSTANCE:?Missing INSTANCE in vllm.env}"
: "${MODEL_ID:?Missing MODEL_ID in vllm.env}"
: "${HOST:?Missing HOST in vllm.env}"
: "${PORT:?Missing PORT in vllm.env}"
: "${DTYPE:?Missing DTYPE in vllm.env}"
: "${MAX_MODEL_LEN:?Missing MAX_MODEL_LEN in vllm.env}"
: "${GPU_MEMORY_UTILIZATION:?Missing GPU_MEMORY_UTILIZATION in vllm.env}"
: "${HF_TOKEN:?Missing HF_TOKEN in vllm.env}"

echo "=== Deploy vLLM ==="
echo "Project : $PROJECT_ID"
echo "Zone    : $ZONE"
echo "VM      : $INSTANCE"
echo "Model   : $MODEL_ID"
echo "Bind    : $HOST:$PORT"
echo "DTYPE   : $DTYPE"
echo "MaxLen  : $MAX_MODEL_LEN"
echo "GPU mem : $GPU_MEMORY_UTILIZATION"
echo

# Ensure gcloud uses correct project
gcloud config set project "$PROJECT_ID" >/dev/null

# Run everything on the VM
gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "
set -euo pipefail

echo '--- GPU check ---'
nvidia-smi || true

echo '--- Install OS deps ---'
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential

echo '--- Setup venv + install vLLM ---'
if [ ! -d ~/vllm-venv ]; then
  python3 -m venv ~/vllm-venv
fi
source ~/vllm-venv/bin/activate
pip install --upgrade pip
pip install 'vllm>=0.4.0' transformers accelerate

echo '--- Save HF token on VM (persists across stops) ---'
echo '${HF_TOKEN}' > ~/.hf_token
chmod 600 ~/.hf_token
export HF_TOKEN=\$(cat ~/.hf_token)

echo '--- Stop any existing vLLM server ---'
pkill -f 'vllm.entrypoints.openai.api_server' || true

echo '--- Start vLLM server ---'
nohup ~/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model '${MODEL_ID}' \
  --host '${HOST}' \
  --port '${PORT}' \
  --dtype '${DTYPE}' \
  --max-model-len '${MAX_MODEL_LEN}' \
  --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
  > ~/vllm.log 2>&1 &

echo 'Started vLLM. Last 80 log lines:'
sleep 5
tail -n 80 ~/vllm.log || true
echo 'OK'
"