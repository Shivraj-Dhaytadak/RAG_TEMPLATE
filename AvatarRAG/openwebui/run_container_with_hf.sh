#!/usr/bin/env bash
set -euo pipefail

# run_container_with_hf.sh
# Usage:
#   ./run_container_with_hf.sh [IMAGE] [CONTAINER_NAME]
# Examples:
#   IMAGE=existing-image:latest ./run_container_with_hf.sh
#   ./run_container_with_hf.sh existing-image:latest my_container
#
# Behavior:
# - IMAGE can be passed as first arg or via $IMAGE env var. Defaults to "existing-image:latest".
# - The script looks for a Hugging Face token in this order:
#     1) env var HF_TOKEN
#     2) a .env file in the current folder containing HF_TOKEN=...
#     3) file $HOME/.huggingface/token (plain token)
# - The token is passed into the container as env var HF_TOKEN.
# - Adjust PORT_MAP or other options below as needed.

IMAGE="${1:-${IMAGE:-existing-image:latest}}"
CONTAINER_NAME="${2:-hf_container}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
PORT_MAP="${PORT_MAP:-7860:7860}"

# Load HF token from environment or .env or ~/.huggingface/token
if [ -z "${HF_TOKEN:-}" ]; then
  if [ -f .env ]; then
    HF_TOKEN_LINE=$(grep -E '^HF_TOKEN=' .env || true)
    if [ -n "$HF_TOKEN_LINE" ]; then
      HF_TOKEN="${HF_TOKEN_LINE#HF_TOKEN=}"
      export HF_TOKEN
    fi
  fi
fi

if [ -z "${HF_TOKEN:-}" ]; then
  if [ -f "$HOME/.huggingface/token" ]; then
    HF_TOKEN=$(cat "$HOME/.huggingface/token")
    export HF_TOKEN
  fi
fi

if [ -z "${HF_TOKEN:-}" ]; then
  cat >&2 <<'EOF'
Hugging Face token not found.
Set it using one of these methods, then re-run this script:

1) Export env var (temporary for this shell):
   export HF_TOKEN="hf_xxx"

2) Add a .env file in this folder with a single line:
   HF_TOKEN=hf_xxx

3) Store token at ~/.huggingface/token (plain token contents):
   mkdir -p ~/.huggingface && echo "hf_xxx" > ~/.huggingface/token

To create a token, visit: https://huggingface.co/settings/tokens
EOF
  exit 1
fi

echo "Using image: $IMAGE"
echo "Container name: $CONTAINER_NAME"

echo "Starting container (using $DOCKER_CMD). Press Ctrl-C to stop."

# Basic docker run — adjust mounts, devices, GPU flags as needed for your environment
$DOCKER_CMD run --rm -it \
  --name "$CONTAINER_NAME" \
  -e HF_TOKEN="$HF_TOKEN" \
  -p "$PORT_MAP" \
  -v "$PWD":/workspace \
  "$IMAGE" "$@"

# End of script
