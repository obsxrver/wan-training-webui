#!/bin/bash
# Provisioning script for Vast.ai to setup musubi-tuner and the training webui.
# Verified on  vastai/pytorch:cuda-12.9.1-auto
# For use with vastai/pytorch:latest docker image
set -euo pipefail
source /venv/main/bin/activate
pids=()
wait_all() {
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ $status -ne 0 ]]; then
    echo "One or more parallel tasks failed." >&2
    exit 1
  fi
}

cd /workspace
if [[ ! -d wan-training-webui ]]; then
  git clone https://github.com/obsxrver/wan-training-webui.git
fi
if [[ ! -d musubi-tuner ]]; then
  git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
fi
cd musubi-tuner
git fetch --all --tags --prune

python3 /workspace/wan-training-webui/configure_training_hardware.py \
  /workspace/wan-training-webui/training-configs/wan22_lora.toml

mkdir -p models/text_encoders models/vae models/diffusion_models

pip install -r /workspace/wan-training-webui/requirements.txt --break-system-packages || \
pip install -r /workspace/wan-training-webui/requirements.txt

HF_DOWNLOAD_VENV=/venv/hf-download
python3 -m venv "${HF_DOWNLOAD_VENV}"
"${HF_DOWNLOAD_VENV}/bin/pip" install \
  -r /workspace/wan-training-webui/requirements-hf-download.txt
HF_DOWNLOAD="${HF_DOWNLOAD_VENV}/bin/hf"


if [[ -n "${VASTAI_KEY:-}" ]]; then
  echo "Setting up Vast.ai API key from VASTAI_KEY..."
  vastai set api-key "$VASTAI_KEY" || echo "Warning: Failed to set vastai API key"
fi

(
  set -euo pipefail
  cd /workspace/musubi-tuner

  pip install -e . "huggingface_hub==0.34.3"
) & pids+=($!)


(
  set -euo pipefail
  cd /workspace/musubi-tuner

  mkdir -p models/text_encoders models/vae models/diffusion_models
) & pids+=($!)

DOWNLOAD_STATUS_DIR=/workspace/musubi-tuner/models/download_status
mkdir -p "${DOWNLOAD_STATUS_DIR}"

start_download() {
  local name="$1"
  shift
  local pid_file="${DOWNLOAD_STATUS_DIR}/${name}.pid"
  local log_file="${DOWNLOAD_STATUS_DIR}/${name}.log"
  local exit_file="${DOWNLOAD_STATUS_DIR}/${name}.exit"

  nohup bash -c "cd /workspace/musubi-tuner && $*; rc=\$?; echo \${rc} > '${exit_file}'; rm -f '${pid_file}'; exit \${rc}" >"${log_file}" 2>&1 </dev/null &
  echo $! >"${pid_file}"
}

echo "Starting model downloads in background..."
start_download text_encoder \
  "${HF_DOWNLOAD}" download \
    Wan-AI/Wan2.1-I2V-14B-720P \
    models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir models/text_encoders

start_download vae \
  "${HF_DOWNLOAD}" download \
    Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/vae/wan_2.1_vae.safetensors \
    --local-dir models/vae

start_download diffusion_high_noise \
  "${HF_DOWNLOAD}" download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models

start_download diffusion_low_noise \
  "${HF_DOWNLOAD}" download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models

start_download diffusion_high_noise_i2v \
  "${HF_DOWNLOAD}" download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models

start_download diffusion_low_noise_i2v \
  "${HF_DOWNLOAD}" download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models

echo "Model downloads running in background. PID files stored in ${DOWNLOAD_STATUS_DIR}."

# ---------- wait for critical tasks ----------
wait_all

DEPLOYMENT_DIR=/workspace/wan-training-webui/deployment/vast

sudo install -d -m 0755 /opt/supervisor-scripts /etc/supervisor/conf.d
sudo install -m 0755 "${DEPLOYMENT_DIR}/start_wan_webui.sh" \
  /opt/supervisor-scripts/start_wan_webui.sh
sudo install -m 0644 "${DEPLOYMENT_DIR}/wan-training-webui.conf" \
  /etc/supervisor/conf.d/wan-training-webui.conf

if command -v supervisorctl >/dev/null 2>&1; then
  sudo supervisorctl reread || true
  sudo supervisorctl update || true
fi

echo "✅ Setup complete."
