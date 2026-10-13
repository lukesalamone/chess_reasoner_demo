#!/usr/bin/env bash
# Start the demo: vLLM serves the model, then the web server starts.
# Open http://127.0.0.1:8765 when it says ready. Ctrl-C stops both.
#
#   ./serve.sh                    # lukesalamone/chess-reasoner-0.8b from Hugging Face
#   ./serve.sh /path/to/model     # a local copy
#
# GPUMEM (default 0.5) is the share of GPU memory vLLM may take.
set -uo pipefail
MODEL="${1:-lukesalamone/chess-reasoner-0.8b}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="${VLLM_PORT:-8000}"
DEMO_PORT="${DEMO_PORT:-8765}"
LOG="$HERE/vllm.log"

for p in "$PORT" "$DEMO_PORT"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
    echo "port $p is in use; stop what is using it, or set VLLM_PORT / DEMO_PORT"; exit 1
  fi
done
cleanup() { kill "${DPID:-0}" "${VPID:-0}" 2>/dev/null; sleep 2; kill -9 "${VPID:-0}" 2>/dev/null; }
trap cleanup EXIT INT TERM

# --chat-template: the plain ChatML template the model was trained with
# --limit-mm-per-prompt: the base model is a vision-language model; this demo is text only
# gdn_prefill_backend: Triton kernels for the linear-attention layers, so vLLM
#   does not build a CUDA kernel with the system nvcc at startup
# --enable-prefix-caching: each tool call adds a turn, so most of every prompt is cached
# --generation-config vllm: sampling settings come only from the request
# VLLM_USE_FLASHINFER_SAMPLER=0: the demo samples with top_p 1 and no top_k, which vLLM's
#   PyTorch sampler handles; FlashInfer's sampler would be compiled at startup
echo "starting vLLM with $MODEL (log: $LOG)"
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve "$MODEL" --served-model-name chess-reasoner --port "$PORT" \
  --gpu-memory-utilization "${GPUMEM:-0.5}" --max-model-len 8192 \
  --chat-template "$HERE/chat_template.jinja" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --additional-config '{"gdn_prefill_backend":"triton"}' \
  --enable-prefix-caching --generation-config vllm > "$LOG" 2>&1 &
VPID=$!
until curl -sf "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"chess-reasoner"'; do
  kill -0 "$VPID" 2>/dev/null || { echo "vLLM stopped:"; tail -30 "$LOG"; exit 1; }
  sleep 3
done
echo "vLLM ready"

VLLM_URL="http://127.0.0.1:$PORT/v1/chat/completions" DEMO_PORT="$DEMO_PORT" python "$HERE/server.py" &
DPID=$!
until curl -sf "http://127.0.0.1:$DEMO_PORT/api/info" > /dev/null 2>&1; do
  kill -0 "$DPID" 2>/dev/null || { echo "the demo server stopped"; exit 1; }
  sleep 1
done
echo "demo ready: http://127.0.0.1:$DEMO_PORT"
wait "$DPID"
