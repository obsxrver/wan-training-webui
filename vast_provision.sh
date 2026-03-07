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

mkdir -p models/text_encoders models/vae models/diffusion_models
mkdir -p /workspace/wan-training-webui/dataset-configs

curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan-training-webui/main/dataset-configs/dataset.toml" -o /workspace/wan-training-webui/dataset-configs/dataset.toml
curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan-training-webui/main/dataset-configs/turbo.toml" -o /workspace/wan-training-webui/dataset-configs/turbo.toml

pip install -U "huggingface_hub>=0.20.0" --break-system-packages || \
pip install -U huggingface_hub --break-system-packages || \
pip install -U huggingface_hub

#fix bug vastai introduced in latest image
#TODO check if bug is patched and remove
/usr/bin/python3 -m pip install rich

if ! command -v vastai >/dev/null 2>&1; then
  pip install vastai
fi


if [[ -n "${VASTAI_KEY:-}" ]]; then
  echo "Setting up Vast.ai API key from VASTAI_KEY..."
  vastai set api-key "$VASTAI_KEY" || echo "Warning: Failed to set vastai API key"
fi

(
  set -euo pipefail
  cd /workspace/musubi-tuner

  sudo apt-get update -y
  
  pip install -e .
  pip install protobuf six matplotlib fastapi "uvicorn[standard]" python-multipart tomli torch torchvision
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
  hf download \
    Wan-AI/Wan2.1-I2V-14B-720P \
    models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir models/text_encoders

start_download vae \
  hf download \
    Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/vae/wan_2.1_vae.safetensors \
    --local-dir models/vae

start_download diffusion_high_noise \
  hf download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models

start_download diffusion_low_noise \
  hf download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models

start_download diffusion_high_noise_i2v \
  hf download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models

start_download diffusion_low_noise_i2v \
  hf download \
    Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
    split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors \
    --local-dir models/diffusion_models

echo "Model downloads running in background. PID files stored in ${DOWNLOAD_STATUS_DIR}."

# ---------- wait for critical tasks ----------
wait_all

WEBUI_PORT=7865
cat <<'EOF' >/workspace/wan-training-webui/start_wan_webui.sh
#!/bin/bash
set -euo pipefail
WEBUI_PORT="${WEBUI_PORT:-7865}"
cd /workspace/wan-training-webui
source /venv/main/bin/activate
exec uvicorn webui.server:app --host 0.0.0.0 --port "${WEBUI_PORT}"
EOF
chmod +x /workspace/wan-training-webui/start_wan_webui.sh

if command -v supervisorctl >/dev/null 2>&1; then
  sudo tee /etc/supervisor/conf.d/wan-training-webui.conf >/dev/null <<'EOF'
[program:wan-training-webui]
command=/bin/bash /workspace/wan-training-webui/start_wan_webui.sh
directory=/workspace/wan-training-webui
autostart=true
autorestart=true
stdout_logfile=/workspace/wan-training-webui.out.log
stderr_logfile=/workspace/wan-training-webui.err.log
stopasgroup=true
killasgroup=true
environment=PYTHONUNBUFFERED=1
EOF
  sudo supervisorctl reread || true
  sudo supervisorctl update || true
fi

PORTAL_ENTRY="0.0.0.0:${WEBUI_PORT}:${WEBUI_PORT}:/:WAN Training UI"
if [[ -n "${PORTAL_CONFIG:-}" ]]; then
  case "${PORTAL_CONFIG}" in
    *"${PORTAL_ENTRY}"*) ;;
    *) PORTAL_CONFIG="${PORTAL_CONFIG}|${PORTAL_ENTRY}" ;;
  esac
else
  PORTAL_CONFIG="${PORTAL_ENTRY}"
fi
export PORTAL_CONFIG

sudo tee /etc/profile.d/wan_portal.sh >/dev/null <<EOF
export PORTAL_CONFIG="${PORTAL_CONFIG}"
EOF

if command -v supervisorctl >/dev/null 2>&1; then
  sudo supervisorctl restart instance_portal || true
fi

echo "✅ Setup complete."
