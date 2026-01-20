#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="/workspace"
MUSUBI_DIR="${WORKSPACE_ROOT}/musubi-tuner"
WEBUI_LINK="${WORKSPACE_ROOT}/wan-training-webui"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON_BIN="${WAN_PYTHON_BIN:-python3}"

ensure_workspace_root() {
  if [[ -d "$WORKSPACE_ROOT" ]]; then
    return
  fi

  if mkdir -p "$WORKSPACE_ROOT" 2>/dev/null; then
    return
  fi

  if command -v sudo >/dev/null 2>&1; then
    sudo mkdir -p "$WORKSPACE_ROOT"
    sudo chown "$(id -u)":"$(id -g)" "$WORKSPACE_ROOT"
    return
  fi

  echo "Error: Unable to create ${WORKSPACE_ROOT}. Run this script with sudo or create ${WORKSPACE_ROOT} manually." >&2
  exit 1
}

ensure_workspace_root

if [[ "$WEBUI_LINK" != "$REPO_ROOT" ]]; then
  if [[ -e "$WEBUI_LINK" && ! -L "$WEBUI_LINK" ]]; then
    echo "Error: ${WEBUI_LINK} exists and is not a symlink. Remove it or set up the repo under /workspace." >&2
    exit 1
  fi
  ln -sfn "$REPO_ROOT" "$WEBUI_LINK"
fi

if [[ ! -d "$MUSUBI_DIR" ]]; then
  git clone --recursive https://github.com/kohya-ss/musubi-tuner.git "$MUSUBI_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install -U pip
pip install -U "huggingface_hub>=0.20.0" fastapi "uvicorn[standard]" python-multipart tomli torch torchvision matplotlib protobuf six

(
  cd "$MUSUBI_DIR"
  pip install -e .
)

mkdir -p "$MUSUBI_DIR/models/text_encoders" "$MUSUBI_DIR/models/vae" "$MUSUBI_DIR/models/diffusion_models"

ensure_file() {
  local target="$1"
  if [[ -f "$target" ]]; then
    echo "Found: $target"
    return 0
  fi
  return 1
}

download_if_missing() {
  local repo="$1"
  local file_path="$2"
  local dest_dir="$3"

  local target="${dest_dir}/${file_path##*/}"
  if ensure_file "$target"; then
    return 0
  fi

  echo "Downloading $file_path from $repo..."
  hf download "$repo" "$file_path" --local-dir "$dest_dir"
}

download_if_missing "Wan-AI/Wan2.1-I2V-14B-720P" \
  "models_t5_umt5-xxl-enc-bf16.pth" \
  "$MUSUBI_DIR/models/text_encoders"

download_if_missing "Comfy-Org/Wan_2.1_ComfyUI_repackaged" \
  "split_files/vae/wan_2.1_vae.safetensors" \
  "$MUSUBI_DIR/models/vae"

download_if_missing "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
  "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors" \
  "$MUSUBI_DIR/models/diffusion_models"

download_if_missing "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
  "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors" \
  "$MUSUBI_DIR/models/diffusion_models"

download_if_missing "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
  "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors" \
  "$MUSUBI_DIR/models/diffusion_models"

download_if_missing "Comfy-Org/Wan_2.2_ComfyUI_Repackaged" \
  "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors" \
  "$MUSUBI_DIR/models/diffusion_models"

echo ""
echo "✅ Local installation complete."
echo ""
echo "To start the WebUI:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  cd ${REPO_ROOT}"
echo "  uvicorn webui.server:app --host 0.0.0.0 --port 7865"
