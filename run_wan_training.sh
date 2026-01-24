#!/usr/bin/env bash
set -euo pipefail

# Simple WAN2.2 LoRA training runner
# - Prompts for title prefix, author, and dataset path (with sensible defaults)
# - Caches latents and text encoder outputs
# - Trains HIGH noise and LOW noise models
# - If 2+ GPUs are free, runs them concurrently; otherwise waits for a free GPU

MUSUBI_DIR="/workspace/musubi-tuner"
DEFAULT_DATASET="/workspace/wan-training-webui/dataset-configs/dataset.toml"
PYTHON="/venv/main/bin/python"
ACCELERATE="/venv/main/bin/accelerate" #todo install in provisioning if errors

VAE="$MUSUBI_DIR/models/vae/split_files/vae/wan_2.1_vae.safetensors"
T5="$MUSUBI_DIR/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"
T2V_HIGH_DIT="$MUSUBI_DIR/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors"
T2V_LOW_DIT="$MUSUBI_DIR/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors"
I2V_HIGH_DIT="$MUSUBI_DIR/models/diffusion_models/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
I2V_LOW_DIT="$MUSUBI_DIR/models/diffusion_models/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"

# CLI overrides (populated via command line flags or environment variables)
TITLE_PREFIX_INPUT="${WAN_TITLE_PREFIX:-}"
AUTHOR_INPUT="${WAN_AUTHOR:-}"
DATASET_INPUT="${WAN_DATASET_PATH:-}"
SAVE_EVERY_INPUT="${WAN_SAVE_EVERY:-}"
MAX_EPOCHS_INPUT="${WAN_MAX_EPOCHS:-}"
CPU_THREADS_INPUT="${WAN_CPU_THREADS_PER_PROCESS:-}"
MAX_WORKERS_INPUT="${WAN_MAX_DATA_LOADER_WORKERS:-}"
CLI_UPLOAD_CLOUD="${WAN_UPLOAD_CLOUD:-}"
CLI_SHUTDOWN_INSTANCE="${WAN_SHUTDOWN_INSTANCE:-}"
TRAINING_MODE_INPUT="${WAN_TRAINING_MODE:-}"
NOISE_MODE_INPUT="${WAN_NOISE_MODE:-}"
AUTO_CONFIRM=0

CLOUD_PERMISSION_DENIED=0
CLOUD_CONNECTION_MESSAGE=""
CPU_THREAD_SOURCE=""

print_usage() {
  cat <<'EOF'
Usage: run_wan_training.sh [options]

Optional arguments (all fall back to interactive prompts when omitted):
  --title-prefix VALUE             Set the title prefix for output names
  --author VALUE                   Set the metadata author
  --dataset PATH                   Path to dataset configuration toml
  --save-every N                   Save every N epochs
  --max-epochs N                   Maximum epochs to train
  --cpu-threads-per-process N      Number of CPU threads per process
  --max-data-loader-workers N      Data loader workers
  --upload-cloud [Y|N]             Upload outputs to configured cloud storage
  --shutdown-instance [Y|N]        Shut down Vast.ai instance after training
  --mode [t2v|i2v]                 Select the training task (text-to-video or image-to-video)
  --noise-mode [both|high|low]     Choose whether to train high noise, low noise, or both
  --auto-confirm                   Skip the final confirmation prompt
  --help                           Show this message and exit

Environment variable overrides:
  WAN_TITLE_PREFIX, WAN_AUTHOR, WAN_DATASET_PATH, WAN_SAVE_EVERY,
  WAN_MAX_EPOCHS, WAN_CPU_THREADS_PER_PROCESS, WAN_MAX_DATA_LOADER_WORKERS,
  WAN_UPLOAD_CLOUD, WAN_SHUTDOWN_INSTANCE, WAN_TRAINING_MODE,
  WAN_NOISE_MODE
EOF
}

normalize_yes_no() {
  local value="$1"
  value="${value:-}"
  if [[ -z "$value" ]]; then
    echo ""
    return
  fi
  case "$value" in
    [Yy]|[Yy][Ee][Ss]) echo "Y" ;;
    [Nn]|[Nn][Oo]) echo "N" ;;
    *) echo "$value" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --title-prefix)
      TITLE_PREFIX_INPUT="$2"
      shift 2
      ;;
    --author)
      AUTHOR_INPUT="$2"
      shift 2
      ;;
    --dataset)
      DATASET_INPUT="$2"
      shift 2
      ;;
    --save-every)
      SAVE_EVERY_INPUT="$2"
      shift 2
      ;;
    --max-epochs)
      MAX_EPOCHS_INPUT="$2"
      shift 2
      ;;
    --cpu-threads-per-process)
      CPU_THREADS_INPUT="$2"
      shift 2
      ;;
    --max-data-loader-workers)
      MAX_WORKERS_INPUT="$2"
      shift 2
      ;;
    --upload-cloud)
      CLI_UPLOAD_CLOUD="$2"
      shift 2
      ;;
    --shutdown-instance)
      CLI_SHUTDOWN_INSTANCE="$2"
      shift 2
      ;;
    --mode)
      TRAINING_MODE_INPUT="$2"
      shift 2
      ;;
    --noise-mode)
      NOISE_MODE_INPUT="$2"
      shift 2
      ;;
    --auto-confirm)
      AUTO_CONFIRM=1
      shift 1
      ;;
    --help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Use --help to see available arguments." >&2
      exit 1
      ;;
  esac
done

CLI_UPLOAD_CLOUD=$(normalize_yes_no "$CLI_UPLOAD_CLOUD")
CLI_SHUTDOWN_INSTANCE=$(normalize_yes_no "$CLI_SHUTDOWN_INSTANCE")

is_vast_instance() {
  if [[ -n "${CONTAINER_ID:-}" || -n "${VAST_CONTAINER_ID:-}" || -n "${VAST_TCP_PORT_8080:-}" || -n "${PUBLIC_IPADDR:-}" ]]; then
    return 0
  fi
  return 1
}

VAST_INSTANCE=0
if is_vast_instance; then
  VAST_INSTANCE=1
else
  if [[ "${CLI_UPLOAD_CLOUD:-}" =~ ^[Yy]$ ]]; then
    echo "Cloud uploads are only available on Vast.ai instances. Disabling upload." >&2
  fi
  if [[ "${CLI_SHUTDOWN_INSTANCE:-}" =~ ^[Yy]$ ]]; then
    echo "Auto-shutdown is only available on Vast.ai instances. Disabling shutdown." >&2
  fi
  CLI_UPLOAD_CLOUD="N"
  CLI_SHUTDOWN_INSTANCE="N"
fi

require() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

ensure_accelerate_default() {
  local cfg="$HOME/.cache/huggingface/accelerate/default_config.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "No accelerate default config found; creating one..."
    "$ACCELERATE" config default
  fi
}

is_gpu_free() {
  local idx="$1"
  # If no processes are listed for this GPU, consider it free
  local procs
  procs=$(nvidia-smi -i "$idx" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E "[0-9]" || true)
  if [[ -z "$procs" ]]; then
    return 0
  else
    return 1
  fi
}

wait_for_free_gpu() {
  local excluded="${1:-}"
  while true; do
    local all_idxs
    all_idxs=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
    if [[ -z "$all_idxs" ]]; then
      echo "No NVIDIA GPUs detected (nvidia-smi returned nothing)." >&2
      exit 1
    fi
    for idx in $all_idxs; do
      # skip excluded ids (comma- or space-separated)
      if [[ -n "$excluded" ]] && [[ ",$excluded," == *",$idx,"* ]]; then
        continue
      fi
      if is_gpu_free "$idx"; then
        echo "$idx"
        return 0
      fi
    done
    sleep 10
  done
}

get_free_port() {
  python3 - "$@" <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

check_low_vram() {
  # Get VRAM in MB for first GPU (assuming all GPUs are identical)
  local vram_mb
  vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  if [[ -z "$vram_mb" || ! "$vram_mb" =~ ^[0-9]+$ ]]; then
    echo "Warning: Could not detect GPU VRAM, defaulting to xformers" >&2
    return 1
  fi
  
  local vram_gb=$((vram_mb / 1024))
  echo "Detected GPU VRAM: ${vram_gb}GB" >&2
  
  if [[ "$vram_gb" -lt 33 ]]; then
    return 0  # Low VRAM
  else
    return 1  # High VRAM
  fi
}
# block swap on anything 32GB or less VRAM
determine_attention_flags() {
  if check_low_vram; then
    echo "--sdpa --blocks_to_swap 1"
  else
    echo "--sdpa"
  fi
}

get_vast_vcpus() {
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    return 1
  fi

  if ! command -v vastai >/dev/null 2>&1; then
    return 1
  fi

  local result
  result=$(python3 - "$CONTAINER_ID" <<'PY'
import re
import subprocess
import sys

container_id = sys.argv[1].strip()
if not container_id:
    sys.exit(1)

try:
    output = subprocess.check_output(
        ["vastai", "show", "instance", container_id],
        text=True,
        stderr=subprocess.STDOUT,
    )
except Exception:
    sys.exit(1)

lines = [line.strip() for line in output.splitlines() if line.strip()]
if len(lines) < 2:
    sys.exit(1)

header = re.split(r"\s{2,}", lines[0])
column_names = {name.lower(): idx for idx, name in enumerate(header)}
idx = None
for key in ("vcpus", "vcpu", "cpu"):
    if key in column_names:
        idx = column_names[key]
        break
if idx is None:
    sys.exit(1)

for line in lines[1:]:
    parts = re.split(r"\s{2,}", line)
    if not parts:
        continue
    if parts[0] != container_id:
        continue
    try:
        value = float(parts[idx])
    except (IndexError, ValueError):
        continue
    print(int(value))
    sys.exit(0)

sys.exit(1)
PY
)
  if [[ -n "$result" ]]; then
    echo "$result"
    return 0
  fi

  return 1
}

get_cpu_threads() {
  local value

  CPU_THREAD_SOURCE=""
  if value=$(get_vast_vcpus 2>/dev/null); then
    if [[ -n "$value" && "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]]; then
      CPU_THREAD_SOURCE="vastai show instance"
      echo "$value"
      return 0
    fi
  fi

  value=$(nproc 2>/dev/null || true)
  if [[ -n "$value" && "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]]; then
    CPU_THREAD_SOURCE="nproc"
    echo "$value"
    return 0
  fi

  value=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || true)
  if [[ -n "$value" && "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]]; then
    CPU_THREAD_SOURCE="/proc/cpuinfo"
    echo "$value"
    return 0
  fi

  CPU_THREAD_SOURCE=""
  echo ""
  return 1
}

prompt_for_valid_api_key() {
  if (( AUTO_CONFIRM )); then
    echo ""
    echo "Auto-confirm is enabled; skipping Vast.ai API key prompt. Cloud features will remain disabled."
    return 1
  fi

  echo ""
  echo "Your Vast.ai API key is invalid or expired."
  echo "To get a valid API key with full permissions:"
  echo "  1. Go to https://cloud.vast.ai/manage-keys/"
  echo "  2. Create a new API key"
  echo "  3. Enter it below"
  echo ""
  
  while true; do
    read -r -p "Enter your Vast.ai API key (or 'skip' to continue without cloud features): " api_key
    
    if [[ "$api_key" == "skip" ]]; then
      echo "Skipping API key setup. Cloud features will be disabled."
      return 1
    fi
    
    if [[ -z "$api_key" ]]; then
      echo "Please enter a valid API key or 'skip'."
      continue
    fi
    
    # Set the API key
    if vastai set api-key "$api_key"; then
      # Test if it works
      local output
      output=$(vastai show connections 2>&1)
      
      if echo "$output" | grep -qi "failed with error 401"; then
        echo "API key is still invalid or missing required permissions. Please try again."
        continue
      else
        echo "✅ API key set successfully!"
        return 0
      fi
    else
      echo "Failed to set API key. Please try again."
    fi
  done
}

check_cloud_configured() {
  # Check if Vast.ai cloud connections are configured
  CLOUD_PERMISSION_DENIED=0
  CLOUD_CONNECTION_MESSAGE=""
  if (( ! VAST_INSTANCE )); then
    CLOUD_CONNECTION_MESSAGE="Cloud uploads are available only on Vast.ai instances."
    return 1
  fi
  if ! command -v vastai >/dev/null 2>&1; then
    echo "vastai CLI not found. Try: pip install vastai --user --break-system-packages" >&2
    CLOUD_CONNECTION_MESSAGE="vastai CLI not found. Install it with: pip install vastai --user --break-system-packages"
    return 1
  fi

  # Check if API key is valid by testing vastai show connections
  local output
  output=$(vastai show connections 2>&1 || true)

  if echo "$output" | grep -qi "failed with error 401"; then
    CLOUD_PERMISSION_DENIED=1
    CLOUD_CONNECTION_MESSAGE=$'Current Vast.ai API key lacks the permissions required to list cloud connections.\nCreate a new key at https://cloud.vast.ai/manage-keys (enable user_read and cloud permissions) and run: vastai set api-key <your-key>'
    echo "Current API key cannot access cloud integrations (401)." >&2
    return 1
  fi

  # Check if there are any cloud connections (skip header and URL lines)
  local connections
  connections=$(echo "$output" | awk 'NF && $1 ~ /^[0-9]+$/ {print $0; exit}')
  if [[ -n "$connections" ]]; then
    return 0
  fi
  CLOUD_CONNECTION_MESSAGE=$'No cloud connections detected. Visit https://cloud.vast.ai/settings/ and open "cloud connection" to link a storage provider.'
  return 1
}

setup_vast_api_key() {
  # Set up Vast.ai API key for instance management
  if (( ! VAST_INSTANCE )); then
    echo "Warning: Not running on Vast.ai. Instance shutdown is unavailable." >&2
    return 1
  fi
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    echo "Warning: CONTAINER_ID not found. Cannot set up instance shutdown." >&2
    return 1
  fi

  if ! command -v vastai >/dev/null 2>&1; then
    echo "Warning: vastai CLI not found. Cannot set up instance shutdown." >&2
    return 1
  fi

  local config_path="$HOME/.config/vastai/vast_api_key"
  local existing_key=""
  if [[ -f "$config_path" ]]; then
    existing_key=$(tr -d '\r\n\t ' <"$config_path")
  fi

  if [[ -n "$existing_key" ]]; then
    echo "Using existing Vast.ai API key for instance management."
    return 0
  fi

  if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
    if vastai set api-key "$CONTAINER_API_KEY" >/dev/null 2>&1; then
      echo "Configured container API key for instance management."
      return 0
    else
      echo "Warning: Failed to configure container API key for instance management." >&2
    fi
  fi

  echo "No Vast.ai API key configured for instance shutdown. Run 'vastai set api-key <your-key>' to enable this feature." >&2
  return 1
}

upload_to_cloud() {
  local lora_path="$1"
  local lora_name="$2"

  if (( ! VAST_INSTANCE )); then
    echo "Cloud uploads are only available on Vast.ai instances. Skipping upload." >&2
    return 1
  fi
  
  if ! check_cloud_configured; then
    echo "No cloud connections configured in Vast.ai. Skipping upload." >&2
    return 1
  fi
  
  # Get the first available cloud connection
  local connection_id
  connection_id=$(vastai show connections 2>/dev/null | grep -v "^ID" | grep -v "^https://" | head -1 | awk '{print $1}')
  
  if [[ -z "$connection_id" ]]; then
    echo "No cloud connection ID found. Skipping upload." >&2
    return 1
  fi
  
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    echo "Warning: CONTAINER_ID not found. Cannot upload to cloud." >&2
    return 1
  fi
  
  echo "Uploading $lora_name to cloud storage (connection: $connection_id)..."
  
  # Use vastai cloud copy to upload to cloud storage
  # Format: vastai cloud copy --src <src> --dst <dst> --instance <instance_id> --connection <connection_id> --transfer "Instance to Cloud"
  if vastai cloud copy --src "$lora_path" --dst "/loras/WAN/$lora_name" --instance "$CONTAINER_ID" --connection "$connection_id" --transfer "Instance to Cloud"; then
    echo "✅ Successfully uploaded $lora_name to cloud storage"
    return 0
  else
    echo "❌ Failed to upload $lora_name to cloud storage" >&2
    return 1
  fi
}

shutdown_instance() {
  if (( ! VAST_INSTANCE )); then
    echo "Auto-shutdown is only available on Vast.ai instances. Skipping." >&2
    return 1
  fi
  if [[ -z "${CONTAINER_ID:-}" ]]; then
    echo "Warning: CONTAINER_ID not found. Cannot shutdown instance." >&2
    return 1
  fi
  
  if ! command -v vastai >/dev/null 2>&1; then
    echo "Warning: vastai CLI not found. Cannot shutdown instance." >&2
    return 1
  fi
  
  echo "Shutting down Vast.ai instance $CONTAINER_ID..."
  if vastai stop instance "$CONTAINER_ID"; then
    echo "✅ Instance shutdown initiated"
    return 0
  else
    echo "❌ Failed to shutdown instance" >&2
    return 1
  fi
}

calculate_cpu_params() {
  local threads
  threads=$(get_cpu_threads)
  local cpu_threads_per_process
  local max_data_loader_workers

  if [[ -n "$threads" && "$threads" =~ ^[0-9]+$ && "$threads" -gt 0 ]]; then
    cpu_threads_per_process=$((threads / 4))
    max_data_loader_workers=$cpu_threads_per_process

    if [[ "$cpu_threads_per_process" -lt 1 ]]; then
      cpu_threads_per_process=1
    fi
    if [[ "$max_data_loader_workers" -lt 1 ]]; then
      max_data_loader_workers=1
    fi

    if [[ -n "$CPU_THREAD_SOURCE" ]]; then
      echo "Detected $threads CPU threads via $CPU_THREAD_SOURCE." >&2
    else
      echo "Detected $threads CPU threads." >&2
    fi
    echo "Setting --num_cpu_threads_per_process=$cpu_threads_per_process" >&2
    echo "Setting --max_data_loader_n_workers=$max_data_loader_workers" >&2
  else
    cpu_threads_per_process=8
    max_data_loader_workers=8
    echo "Could not determine CPU threads automatically; defaulting to 8 threads for training and data loading." >&2
  fi

  echo "$cpu_threads_per_process $max_data_loader_workers"
}

main() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required but not found in PATH." >&2
    exit 1
  fi

  # Prompt inputs with defaults
  echo "WAN2.2 LoRA simple runner"

  if [[ -z "${TITLE_PREFIX_INPUT:-}" ]]; then
    if (( AUTO_CONFIRM )); then
      TITLE_PREFIX="mylora"
      echo "Title prefix (auto default): $TITLE_PREFIX"
    else
      read -r -p "Title prefix (default: mylora): " TITLE_PREFIX || true
    fi
  else
    TITLE_PREFIX="$TITLE_PREFIX_INPUT"
    echo "Title prefix (auto): $TITLE_PREFIX"
  fi
  TITLE_PREFIX=${TITLE_PREFIX:-mylora}
  # Trim surrounding whitespace before replacing interior whitespace with dashes to avoid trailing hyphens
  TITLE_PREFIX="$(echo "$TITLE_PREFIX" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/[[:space:]]\+/-/g')"

  if [[ -z "${AUTHOR_INPUT:-}" ]]; then
    if (( AUTO_CONFIRM )); then
      AUTHOR="authorName"
      echo "Author (auto default): $AUTHOR"
    else
      read -r -p "Author (default: authorName): " AUTHOR || true
    fi
  else
    AUTHOR="$AUTHOR_INPUT"
    echo "Author (auto): $AUTHOR"
  fi
  AUTHOR=${AUTHOR:-authorName}

  if [[ -z "${DATASET_INPUT:-}" ]]; then
    if (( AUTO_CONFIRM )); then
      DATASET="$DEFAULT_DATASET"
      echo "Dataset path (auto default): $DATASET"
    else
      read -r -p "Dataset path (default: $DEFAULT_DATASET): " DATASET || true
    fi
  else
    DATASET="$DATASET_INPUT"
    echo "Dataset path (auto): $DATASET"
  fi
  DATASET=${DATASET:-$DEFAULT_DATASET}

  local training_mode="$TRAINING_MODE_INPUT"
  if [[ -z "${training_mode:-}" ]]; then
    if (( AUTO_CONFIRM )); then
      training_mode="t2v"
      echo "Training task (auto default): $training_mode"
    else
      read -r -p "Training task (t2v/i2v, default: t2v): " training_mode || true
    fi
  else
    echo "Training task (auto): $training_mode"
  fi
  training_mode=${training_mode:-t2v}
  training_mode=${training_mode,,}

  local TRAIN_TASK
  local HIGH_TITLE
  local LOW_TITLE
  local -a CACHE_LATENTS_ARGS=()
  local noise_mode="$NOISE_MODE_INPUT"
  local RUN_HIGH=1
  local RUN_LOW=1

  if [[ -z "${noise_mode:-}" ]]; then
    if (( AUTO_CONFIRM )); then
      noise_mode="both"
      echo "Noise selection (auto default): $noise_mode"
    else
      read -r -p "Noise selection (high/low/both, default: both): " noise_mode || true
    fi
  else
    echo "Noise selection (auto): $noise_mode"
  fi
  noise_mode=${noise_mode:-both}
  noise_mode=${noise_mode,,}

  case "$noise_mode" in
    both)
      RUN_HIGH=1
      RUN_LOW=1
      ;;
    high)
      RUN_HIGH=1
      RUN_LOW=0
      ;;
    low)
      RUN_HIGH=0
      RUN_LOW=1
      ;;
    *)
      echo "Invalid noise selection: $noise_mode. Use 'high', 'low', or 'both'." >&2
      exit 1
      ;;
  esac

  local TIMESTEP_BOUNDARY

  case "$training_mode" in
    t2v)
      TRAIN_TASK="t2v-A14B"
      HIGH_DIT="$T2V_HIGH_DIT"
      LOW_DIT="$T2V_LOW_DIT"
      HIGH_TITLE="${TITLE_PREFIX}_Wan2.2_high"
      LOW_TITLE="${TITLE_PREFIX}_Wan2.2_low"
      TIMESTEP_BOUNDARY=875
      ;;
    i2v)
      TRAIN_TASK="i2v-A14B"
      HIGH_DIT="$I2V_HIGH_DIT"
      LOW_DIT="$I2V_LOW_DIT"
      HIGH_TITLE="${TITLE_PREFIX}_Wan2.2_high"
      LOW_TITLE="${TITLE_PREFIX}_Wan2.2_low"
      CACHE_LATENTS_ARGS+=(--i2v)
      TIMESTEP_BOUNDARY=900
      ;;
    *)
      echo "Invalid training mode: $training_mode. Use 't2v' or 'i2v'." >&2
      exit 1
      ;;
  esac

  if [[ ! -f "$DATASET" ]]; then
    echo "Dataset config not found at $DATASET; downloading..."
    mkdir -p "$(dirname "$DATASET")"
    curl -fsSL "https://raw.githubusercontent.com/obsxrver/wan-training-webui/main/dataset-configs/dataset.toml" -o "$DATASET" || echo "Failed to download dataset.toml" >&2
  fi

  if [[ -z "${SAVE_EVERY_INPUT:-}" ]]; then
    if (( AUTO_CONFIRM )); then
      SAVE_EVERY=""
      echo "Save every N epochs (auto default): 20"
    else
      read -r -p "Save every N epochs (default: 100): " SAVE_EVERY || true
    fi
  else
    SAVE_EVERY="$SAVE_EVERY_INPUT"
    echo "Save every N epochs (auto): $SAVE_EVERY"
  fi
  SAVE_EVERY=${SAVE_EVERY:-20}

  if [[ -z "${MAX_EPOCHS_INPUT:-}" ]]; then
    if (( AUTO_CONFIRM )); then
      MAX_EPOCHS="100"
      echo "Max epochs (auto default): $MAX_EPOCHS"
    else
      read -r -p "Max epochs (default: 100): " MAX_EPOCHS || true
    fi
  else
    MAX_EPOCHS="$MAX_EPOCHS_INPUT"
    echo "Max epochs (auto): $MAX_EPOCHS"
  fi
  MAX_EPOCHS=${MAX_EPOCHS:-100}

  CPU_PARAMS=($(calculate_cpu_params))
  DEFAULT_CPU_THREADS_PER_PROCESS=${CPU_PARAMS[0]}
  DEFAULT_MAX_DATA_LOADER_WORKERS=${CPU_PARAMS[1]}

  echo ""
  if [[ -z "${CPU_THREADS_INPUT:-}" ]]; then
    if (( AUTO_CONFIRM )); then
      CPU_THREADS_PER_PROCESS="$DEFAULT_CPU_THREADS_PER_PROCESS"
      echo "CPU threads per process (auto default): $CPU_THREADS_PER_PROCESS"
    else
      read -r -p "CPU threads per process (default: $DEFAULT_CPU_THREADS_PER_PROCESS): " CPU_THREADS_PER_PROCESS || true
    fi
  else
    CPU_THREADS_PER_PROCESS="$CPU_THREADS_INPUT"
    echo "CPU threads per process (auto): $CPU_THREADS_PER_PROCESS"
  fi
  CPU_THREADS_PER_PROCESS=${CPU_THREADS_PER_PROCESS:-$DEFAULT_CPU_THREADS_PER_PROCESS}

  if [[ -z "${MAX_WORKERS_INPUT:-}" ]]; then
    if (( AUTO_CONFIRM )); then
      MAX_DATA_LOADER_WORKERS="$DEFAULT_MAX_DATA_LOADER_WORKERS"
      echo "Max data loader workers (auto default): $MAX_DATA_LOADER_WORKERS"
    else
      read -r -p "Max data loader workers (default: $DEFAULT_MAX_DATA_LOADER_WORKERS): " MAX_DATA_LOADER_WORKERS || true
    fi
  else
    MAX_DATA_LOADER_WORKERS="$MAX_WORKERS_INPUT"
    echo "Max data loader workers (auto): $MAX_DATA_LOADER_WORKERS"
  fi
  MAX_DATA_LOADER_WORKERS=${MAX_DATA_LOADER_WORKERS:-$DEFAULT_MAX_DATA_LOADER_WORKERS}

  echo ""
  echo "=== Post-Training Options ==="

  # Check for cloud storage upload option
  UPLOAD_CLOUD="Y"
  local cloud_ready=0
  if check_cloud_configured; then
    cloud_ready=1
  fi

  if [[ -n "${CLI_UPLOAD_CLOUD:-}" ]]; then
    UPLOAD_CLOUD="$CLI_UPLOAD_CLOUD"
    echo "Upload LoRAs to cloud storage after training? [auto: $UPLOAD_CLOUD]"
    if [[ "$UPLOAD_CLOUD" =~ ^[Yy]$ && $cloud_ready -eq 0 ]]; then
      if (( CLOUD_PERMISSION_DENIED )); then
        echo "$CLOUD_CONNECTION_MESSAGE" >&2
        echo "Disabling cloud upload because the current API key lacks required permissions." >&2
        UPLOAD_CLOUD="N"
      else
        echo "$CLOUD_CONNECTION_MESSAGE" >&2
        echo "Disabling cloud upload because no cloud connections are available." >&2
        UPLOAD_CLOUD="N"
      fi
    fi
  else
    if (( cloud_ready )); then
      echo "Cloud storage is configured in Vast.ai."
      if (( AUTO_CONFIRM )); then
        UPLOAD_CLOUD="Y"
        echo "Upload LoRAs to cloud storage after training? [auto default: $UPLOAD_CLOUD]"
      else
        read -r -p "Upload LoRAs to cloud storage after training? [Y/n]: " UPLOAD_CLOUD || true
        UPLOAD_CLOUD=${UPLOAD_CLOUD:-Y}
      fi
    else
      if (( CLOUD_PERMISSION_DENIED )); then
        echo "$CLOUD_CONNECTION_MESSAGE"
        echo "Cloud uploads will be disabled until a full-access API key is configured."
        UPLOAD_CLOUD="N"
      else
        echo "$CLOUD_CONNECTION_MESSAGE"
        if (( ! VAST_INSTANCE )); then
          UPLOAD_CLOUD="N"
        else
          if (( AUTO_CONFIRM )); then
            UPLOAD_CLOUD="Y"
            echo "Upload LoRAs to cloud storage after training? [auto default: $UPLOAD_CLOUD]"
          else
            read -r -p "Upload LoRAs to cloud storage after training? [Y/n]: " UPLOAD_CLOUD || true
            UPLOAD_CLOUD=${UPLOAD_CLOUD:-Y}
          fi
        fi
      fi
    fi
  fi

  # Check for instance shutdown option
  SHUTDOWN_INSTANCE="Y"
  if [[ -n "${CLI_SHUTDOWN_INSTANCE:-}" ]]; then
    SHUTDOWN_INSTANCE="$CLI_SHUTDOWN_INSTANCE"
    echo "Shut down this instance after training? [auto: $SHUTDOWN_INSTANCE]"
  else
    if (( ! VAST_INSTANCE )); then
      echo "Auto-shutdown is only available on Vast.ai instances."
      SHUTDOWN_INSTANCE="N"
    elif [[ -n "${CONTAINER_ID:-}" ]] && command -v vastai >/dev/null 2>&1; then
      echo "Vast.ai instance management available."
      if (( AUTO_CONFIRM )); then
        SHUTDOWN_INSTANCE="Y"
        echo "Shut down this instance after training to save costs? [auto default: $SHUTDOWN_INSTANCE]"
      else
        read -r -p "Shut down this instance after training to save costs? [Y/n]: " SHUTDOWN_INSTANCE || true
        SHUTDOWN_INSTANCE=${SHUTDOWN_INSTANCE:-Y}
      fi
    else
      echo "Vast.ai CLI not available or not running on Vast.ai instance."
      SHUTDOWN_INSTANCE="N"
    fi
  fi

  echo ""
  echo "=== Configuration Summary ==="
  UPLOAD_CLOUD=$(normalize_yes_no "$UPLOAD_CLOUD")
  SHUTDOWN_INSTANCE=$(normalize_yes_no "$SHUTDOWN_INSTANCE")
  echo "  Dataset: $DATASET"
  if (( RUN_HIGH )); then
    echo "  High title: $HIGH_TITLE"
  else
    echo "  High noise: disabled"
  fi
  if (( RUN_LOW )); then
    echo "  Low title:  $LOW_TITLE"
  else
    echo "  Low noise:  disabled"
  fi
  echo "  Author:     $AUTHOR"
  echo "  Max epochs: $MAX_EPOCHS"
  echo "  Save every: $SAVE_EVERY epochs"
  echo "  Task:       $TRAIN_TASK"
  echo "  Mode:       ${training_mode^^}"
  echo "  Noise mode: ${noise_mode^^}"
  echo "  Upload to cloud: $UPLOAD_CLOUD"
  echo "  Auto-shutdown: $SHUTDOWN_INSTANCE"
  echo ""
  if (( AUTO_CONFIRM )); then
    PROCEED="Y"
    echo "Proceed with training? [auto: Y]"
  else
    read -r -p "Proceed with training? [Y/n]: " PROCEED || true
    PROCEED=${PROCEED:-Y}
    if [[ ! "$PROCEED" =~ ^[Yy]?$ ]]; then
      echo "Training cancelled."
      exit 0
    fi
  fi

  # Validate required files
  require "$PYTHON"
  require "$ACCELERATE"
  require "$VAE"
  require "$T5"
  if (( RUN_HIGH )); then
    require "$HIGH_DIT"
  fi
  if (( RUN_LOW )); then
    require "$LOW_DIT"
  fi
  require "$DATASET"

  cd "$MUSUBI_DIR"

  ensure_accelerate_default

  ATTN_FLAGS=$(determine_attention_flags)
  echo "Using attention flags: $ATTN_FLAGS"

  echo "Using CPU parameters:"
  echo "  --num_cpu_threads_per_process: $CPU_THREADS_PER_PROCESS"
  echo "  --max_data_loader_n_workers: $MAX_DATA_LOADER_WORKERS"

  echo "Caching latents..."
  local CACHE_LATENTS_CMD=(
    "$PYTHON"
    src/musubi_tuner/wan_cache_latents.py
    --dataset_config "$DATASET"
    --vae "$VAE"
  )
  if (( ${#CACHE_LATENTS_ARGS[@]} )); then
    CACHE_LATENTS_CMD+=("${CACHE_LATENTS_ARGS[@]}")
  fi
  "${CACHE_LATENTS_CMD[@]}"

  echo "Caching text encoder outputs..."
  "$PYTHON" src/musubi_tuner/wan_cache_text_encoder_outputs.py \
    --dataset_config "$DATASET" \
    --t5 "$T5"

  # Allocate distinct rendezvous ports to prevent EADDRINUSE
  local HIGH_PORT=""
  local LOW_PORT=""
  local HIGH_GPU=""
  local LOW_GPU=""
  local HIGH_PID=""
  local LOW_PID=""
  local -a WAIT_PIDS=()

  if (( RUN_HIGH )); then
    HIGH_PORT=$(get_free_port)
  fi
  if (( RUN_LOW )); then
    LOW_PORT=$(get_free_port)
    if (( RUN_HIGH )) && [[ "$LOW_PORT" == "$HIGH_PORT" ]]; then
      LOW_PORT=$(get_free_port)
    fi
  fi

  if (( RUN_HIGH )); then
    echo "Waiting for a free GPU for HIGH noise training..."
    HIGH_GPU=$(wait_for_free_gpu)
    echo "Starting HIGH on GPU $HIGH_GPU (port $HIGH_PORT) -> run_high.log"
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$HIGH_PORT" CUDA_VISIBLE_DEVICES="$HIGH_GPU" \
    "$ACCELERATE" launch --num_cpu_threads_per_process "$CPU_THREADS_PER_PROCESS" --num_processes 1 --main_process_port "$HIGH_PORT" src/musubi_tuner/wan_train_network.py \
      --task "$TRAIN_TASK" \
      --dit "$HIGH_DIT" \
      --vae "$VAE" \
      --t5 "$T5" \
      --dataset_config "$DATASET" \
      $ATTN_FLAGS \
      --mixed_precision fp16 \
      --fp8_base \
      --optimizer_type adamw \
      --learning_rate 3e-4 \
      --gradient_checkpointing \
      --gradient_accumulation_steps 1 \
      --max_data_loader_n_workers "$MAX_DATA_LOADER_WORKERS" \
      --network_module networks.lora_wan \
      --network_dim 16 \
      --network_alpha 16 \
      --timestep_sampling shift \
      --discrete_flow_shift 1.0 \
      --max_train_epochs "$MAX_EPOCHS" \
      --save_every_n_epochs "$SAVE_EVERY" \
      --seed 5 \
      --optimizer_args weight_decay=0.1 \
      --max_grad_norm 0 \
      --lr_scheduler polynomial \
      --lr_scheduler_power 8 \
      --lr_scheduler_min_lr_ratio=5e-5 \
      --output_dir "$MUSUBI_DIR/output" \
      --output_name "$HIGH_TITLE" \
      --metadata_title "$HIGH_TITLE" \
      --metadata_author "$AUTHOR" \
      --preserve_distribution_shape \
      --min_timestep "$TIMESTEP_BOUNDARY" \
      --max_timestep 1000 \
      > "$PWD/run_high.log" 2>&1 &
    HIGH_PID=$!
    WAIT_PIDS+=("$HIGH_PID")
  else
    echo "Skipping HIGH noise training per noise selection."
  fi

  if (( RUN_LOW )); then
    local GPU_COUNT
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    echo "Waiting for a free GPU for LOW noise training..."
    if (( GPU_COUNT > 1 )) && (( RUN_HIGH )); then
      LOW_GPU=$(wait_for_free_gpu "$HIGH_GPU")
    else
      LOW_GPU=$(wait_for_free_gpu)
    fi
    echo "Starting LOW on GPU $LOW_GPU (port $LOW_PORT) -> run_low.log"
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$LOW_PORT" CUDA_VISIBLE_DEVICES="$LOW_GPU" \
    "$ACCELERATE" launch --num_cpu_threads_per_process "$CPU_THREADS_PER_PROCESS" --num_processes 1 --main_process_port "$LOW_PORT" src/musubi_tuner/wan_train_network.py \
      --task "$TRAIN_TASK" \
      --dit "$LOW_DIT" \
      --vae "$VAE" \
      --t5 "$T5" \
      --dataset_config "$DATASET" \
      $ATTN_FLAGS \
      --mixed_precision fp16 \
      --fp8_base \
      --optimizer_type adamw \
      --learning_rate 3e-4 \
      --gradient_checkpointing \
      --gradient_accumulation_steps 1 \
      --max_data_loader_n_workers "$MAX_DATA_LOADER_WORKERS" \
      --network_module networks.lora_wan \
      --network_dim 16 \
      --network_alpha 16 \
      --timestep_sampling shift \
      --discrete_flow_shift 1.0 \
      --max_train_epochs "$MAX_EPOCHS" \
      --save_every_n_epochs "$SAVE_EVERY" \
      --seed 5 \
      --optimizer_args weight_decay=0.1 \
      --max_grad_norm 0 \
      --lr_scheduler polynomial \
      --lr_scheduler_power 8 \
      --lr_scheduler_min_lr_ratio=5e-5 \
      --output_dir "$MUSUBI_DIR/output" \
      --output_name "$LOW_TITLE" \
      --metadata_title "$LOW_TITLE" \
      --metadata_author "$AUTHOR" \
      --preserve_distribution_shape \
      --min_timestep 0 \
      --max_timestep "$TIMESTEP_BOUNDARY" \
      > "$PWD/run_low.log" 2>&1 &
    LOW_PID=$!
    WAIT_PIDS+=("$LOW_PID")
  else
    echo "Skipping LOW noise training per noise selection."
  fi

  if (( RUN_HIGH )); then
    echo "HIGH PID: $HIGH_PID${HIGH_GPU:+ (GPU $HIGH_GPU)}, log: $PWD/run_high.log"
  fi
  if (( RUN_LOW )); then
    echo "LOW  PID: $LOW_PID${LOW_GPU:+ (GPU $LOW_GPU)}, log: $PWD/run_low.log"
  fi

  if (( RUN_HIGH )) && (( RUN_LOW )); then
    echo "Waiting for both trainings to finish..."
  elif (( RUN_HIGH )); then
    echo "Waiting for high noise training to finish..."
  elif (( RUN_LOW )); then
    echo "Waiting for low noise training to finish..."
  fi

  for pid in "${WAIT_PIDS[@]}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid"
    fi
  done
  echo "✅ Training completed!"

  OUTPUT_DIR="$MUSUBI_DIR/output"
  RENAMED_OUTPUT="$MUSUBI_DIR/output-${TITLE_PREFIX}"
  if [[ -d "$OUTPUT_DIR" ]]; then
    mv "$OUTPUT_DIR" "$RENAMED_OUTPUT"
  fi
  
  # Analyze training logs and generate plots
  echo ""
  echo "=== Analyzing Training Logs ==="
  if [[ -f "$PWD/run_high.log" || -f "$PWD/run_low.log" ]]; then
    "$PYTHON" /workspace/analyze_training_logs.py "$PWD" || echo "Warning: Log analysis failed"
    if [[ -d "$PWD/training_analysis" ]]; then
      mv "$PWD/training_analysis" "$RENAMED_OUTPUT/training_analysis"
    fi

    [[ -f "$PWD/run_high.log" ]] && cp "$PWD/run_high.log" "$RENAMED_OUTPUT/"
    [[ -f "$PWD/run_low.log" ]] && cp "$PWD/run_low.log" "$RENAMED_OUTPUT/"
  else
    echo "Warning: No log files found to analyze"
  fi
  
  # Execute pre-configured post-training actions
  if [[ "$UPLOAD_CLOUD" =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Uploading to Cloud Storage ==="
    upload_to_cloud "$RENAMED_OUTPUT" "${TITLE_PREFIX}" || echo "Failed to upload output directory"
  fi
  
  if [[ "$SHUTDOWN_INSTANCE" =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Shutting Down Instance ==="
    if setup_vast_api_key; then
      echo "Instance will shut down in 10 seconds. Press Ctrl+C to cancel."
      sleep 10
      shutdown_instance
    else
      echo "Could not set up instance shutdown. Skipping auto-shutdown."
    fi
  fi
  
  echo "✅ All done."
}

main "$@" 
