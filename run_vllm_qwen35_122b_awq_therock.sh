#!/bin/bash

CONTAINER_NAME="vllm-strix-qwen"
MODEL_CACHE_DIR="$HOME/.cache/huggingface"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}/patches"

# 1. Kill and remove any previously running container with the same name
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Stopping and cleaning up previous container: ${CONTAINER_NAME}..."
    docker rm -f ${CONTAINER_NAME}
fi

echo "Launching Qwen3.5 122B on Strix Halo APU using kyuz0/vllm-therock-gfx1151..."
echo "WARNING: Ensure your system has at least 96GB+ of RAM to avoid an OOM crash."

# 2. Run the container with gfx1151 stability optimizations
#    Patches are mounted into /patches and applied via apply_patches.py entrypoint
docker run -it --rm \
  --name "${CONTAINER_NAME}" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "$MODEL_CACHE_DIR":/root/.cache/huggingface \
  -v "${PATCHES_DIR}":/patches:ro \
  -e LD_LIBRARY_PATH=/opt/rocm/lib \
  -e LD_PRELOAD=/opt/rocm/lib/librocm_smi64.so \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_TORCH_COMPILE_LEVEL=0 \
  docker.io/kyuz0/vllm-therock-gfx1151:latest \
  python /patches/apply_patches.py serve QuantTrio/Qwen3.5-122B-A10B-AWQ \
    --served-model-name qwen35 \
    --quantization awq \
    --trust-remote-code \
    --max-model-len 32768 \
    --language-model-only \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000

# --kv-cache-dtype fp16 \
