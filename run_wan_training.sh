#!/usr/bin/env bash
set -euo pipefail

# Simple WAN2.2 LoRA training runner
# - Uses CLI inputs (with sensible defaults)
# - Caches latents and text encoder outputs
# - Loads advanced training options from an editable Musubi Tuner TOML file
# - Trains HIGH noise, LOW noise, or COMBINED noise models
# - If 2+ GPUs are free, runs them concurrently; otherwise waits for a free GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MUSUBI_DIR="/workspace/musubi-tuner"
DEFAULT_DATASET="/workspace/wan-training-webui/dataset-configs/dataset.toml"
DEFAULT_TRAINING_CONFIG="$SCRIPT_DIR/training-configs/wan22_lora.toml"
PYTHON="/venv/main/bin/python"
ACCELERATE="/venv/main/bin/accelerate" #todo install in provisioning if errors
ACCELERATE_CPU_THREADS_PER_PROCESS=4
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
TRAINING_CONFIG_INPUT="${WAN_TRAINING_CONFIG_PATH:-}"
SAVE_EVERY_INPUT="${WAN_SAVE_EVERY:-}"
SAVE_OPTIMIZER_STATE_INPUT="${WAN_SAVE_OPTIMIZER_STATE:-}"
RESUME_HIGH_OPTIMIZER_STATE_INPUT="${WAN_RESUME_HIGH_OPTIMIZER_STATE_PATH:-}"
RESUME_LOW_OPTIMIZER_STATE_INPUT="${WAN_RESUME_LOW_OPTIMIZER_STATE_PATH:-}"
MAX_EPOCHS_INPUT="${WAN_MAX_EPOCHS:-}"
CLI_UPLOAD_CLOUD="${WAN_UPLOAD_CLOUD:-}"
CLI_SHUTDOWN_INSTANCE="${WAN_SHUTDOWN_INSTANCE:-}"
TRAINING_MODE_INPUT="${WAN_TRAINING_MODE:-}"
NOISE_MODE_INPUT="${WAN_NOISE_MODE:-}"
CLI_CLOUD_CONNECTION_ID="${WAN_CLOUD_CONNECTION_ID:-}"
AUTO_CONFIRM=0

print_usage() {
  cat <<'EOF'
Usage: run_wan_training.sh [options]

Optional arguments (defaults are used when omitted):
  --title-prefix VALUE             Set the title prefix for output names
  --author VALUE                   Set the metadata author
  --dataset PATH                   Path to dataset configuration toml
  --training-config PATH           Path to Musubi training configuration toml
  --save-every N                   Save every N epochs
  --save-optimizer-state [Y|N]     Save optimizer state with checkpoints and at training end
  --resume-high-optimizer-state PATH
                                    Resume high/combined training from a state directory
  --resume-low-optimizer-state PATH
                                    Resume low training from a state directory
  --max-epochs N                   Maximum epochs to train
  --upload-cloud [Y|N]             Upload outputs to configured cloud storage
  --shutdown-instance [Y|N]        Shut down Vast.ai instance after training
  --mode [t2v|i2v]                 Select the training task (text-to-video or image-to-video)
  --noise-mode [both|high|low|combined]
                                    Choose whether to train high noise, low noise, both, or combined
  --cloud-connection-id VALUE      Upload to a specific Vast.ai cloud connection
  --auto-confirm                   No-op (retained for compatibility)
  --help                           Show this message and exit

Environment variable overrides:
  WAN_TITLE_PREFIX, WAN_AUTHOR, WAN_DATASET_PATH, WAN_TRAINING_CONFIG_PATH,
  WAN_SAVE_EVERY, WAN_SAVE_OPTIMIZER_STATE,
  WAN_RESUME_HIGH_OPTIMIZER_STATE_PATH, WAN_RESUME_LOW_OPTIMIZER_STATE_PATH,
  WAN_MAX_EPOCHS, WAN_UPLOAD_CLOUD, WAN_SHUTDOWN_INSTANCE,
  WAN_TRAINING_MODE, WAN_NOISE_MODE, WAN_CLOUD_CONNECTION_ID
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
    --training-config)
      TRAINING_CONFIG_INPUT="$2"
      shift 2
      ;;
    --save-every)
      SAVE_EVERY_INPUT="$2"
      shift 2
      ;;
    --save-optimizer-state)
      SAVE_OPTIMIZER_STATE_INPUT="$2"
      shift 2
      ;;
    --resume-high-optimizer-state)
      RESUME_HIGH_OPTIMIZER_STATE_INPUT="$2"
      shift 2
      ;;
    --resume-low-optimizer-state)
      RESUME_LOW_OPTIMIZER_STATE_INPUT="$2"
      shift 2
      ;;
    --max-epochs)
      MAX_EPOCHS_INPUT="$2"
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
    --cloud-connection-id)
      CLI_CLOUD_CONNECTION_ID="$2"
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

load_vast_env() {
  local env_file="/etc/environment"
  local line key value

  [[ -f "$env_file" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%$'\r'}"
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      key="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"
      case "$key" in
        CONTAINER_ID|VAST_CONTAINER_ID|CONTAINER_API_KEY|PUBLIC_IPADDR|VAST_TCP_PORT_8080)
          if [[ -z "${!key:-}" ]]; then
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            printf -v "$key" '%s' "$value"
            export "$key"
          fi
          ;;
      esac
    fi
  done < "$env_file"
}

get_container_id() {
  if [[ -n "${CONTAINER_ID:-}" ]]; then
    echo "$CONTAINER_ID"
    return 0
  fi
  if [[ -n "${VAST_CONTAINER_ID:-}" ]]; then
    echo "$VAST_CONTAINER_ID"
    return 0
  fi
  return 1
}

load_vast_env

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

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
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

setup_vast_api_key() {
  # Set up Vast.ai API key for instance management
  if (( ! VAST_INSTANCE )); then
    echo "Warning: Not running on Vast.ai. Instance shutdown is unavailable." >&2
    return 1
  fi
  local container_id
  container_id=$(get_container_id || true)
  if [[ -z "$container_id" ]]; then
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
  local connection_id="${3:-${CLI_CLOUD_CONNECTION_ID:-}}"

  if (( ! VAST_INSTANCE )); then
    echo "Cloud uploads are only available on Vast.ai instances. Skipping upload." >&2
    return 1
  fi

  if [[ -z "$connection_id" ]]; then
    echo "No cloud connection ID provided. Skipping upload." >&2
    return 1
  fi

  local container_id
  container_id=$(get_container_id || true)
  if [[ -z "$container_id" ]]; then
    echo "Warning: CONTAINER_ID not found. Cannot upload to cloud." >&2
    return 1
  fi

  if ! command -v vastai >/dev/null 2>&1; then
    echo "Warning: vastai CLI not found. Cannot upload to cloud." >&2
    return 1
  fi

  echo "Uploading $lora_name to cloud storage (connection: $connection_id)..."
  
  # Use vastai cloud copy to upload to cloud storage
  # Format: vastai cloud copy --src <src> --dst <dst> --instance <instance_id> --connection <connection_id> --transfer "Instance to Cloud"
  if vastai cloud copy --src "$lora_path" --dst "/loras/WAN/$lora_name" --instance "$container_id" --connection "$connection_id" --transfer "Instance to Cloud"; then
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
  local container_id
  container_id=$(get_container_id || true)
  if [[ -z "$container_id" ]]; then
    echo "Warning: CONTAINER_ID not found. Cannot shutdown instance." >&2
    return 1
  fi
  
  if ! command -v vastai >/dev/null 2>&1; then
    echo "Warning: vastai CLI not found. Cannot shutdown instance." >&2
    return 1
  fi
  
  echo "Shutting down Vast.ai instance $container_id..."
  if vastai stop instance "$container_id"; then
    echo "✅ Instance shutdown initiated"
    return 0
  else
    echo "❌ Failed to shutdown instance" >&2
    return 1
  fi
}

main() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required but not found in PATH." >&2
    exit 1
  fi

  # Resolve inputs with defaults
  echo "WAN2.2 LoRA simple runner"

  TITLE_PREFIX="${TITLE_PREFIX_INPUT:-mylora}"
  echo "Title prefix: $TITLE_PREFIX"
  # Trim surrounding whitespace before replacing interior whitespace with dashes to avoid trailing hyphens
  TITLE_PREFIX="$(echo "$TITLE_PREFIX" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/[[:space:]]\+/-/g')"

  AUTHOR="${AUTHOR_INPUT:-authorName}"
  echo "Author: $AUTHOR"

  DATASET="${DATASET_INPUT:-$DEFAULT_DATASET}"
  echo "Dataset path: $DATASET"

  TRAINING_CONFIG="${TRAINING_CONFIG_INPUT:-$DEFAULT_TRAINING_CONFIG}"
  echo "Training config: $TRAINING_CONFIG"

  local training_mode="${TRAINING_MODE_INPUT:-t2v}"
  echo "Training task: $training_mode"
  training_mode=${training_mode,,}

  local TRAIN_TASK
  local TRAIN_MODE_LABEL
  local HIGH_TITLE
  local LOW_TITLE
  local COMBINED_TITLE
  local -a CACHE_LATENTS_ARGS=()
  local noise_mode="${NOISE_MODE_INPUT:-both}"
  local RUN_HIGH=1
  local RUN_LOW=1
  local RUN_COMBINED=0

  echo "Noise selection: $noise_mode"
  noise_mode=${noise_mode,,}

  case "$noise_mode" in
    both)
      RUN_HIGH=1
      RUN_LOW=1
      RUN_COMBINED=0
      ;;
    high)
      RUN_HIGH=1
      RUN_LOW=0
      RUN_COMBINED=0
      ;;
    low)
      RUN_HIGH=0
      RUN_LOW=1
      RUN_COMBINED=0
      ;;
    combined)
      RUN_HIGH=0
      RUN_LOW=0
      RUN_COMBINED=1
      ;;
    *)
      echo "Invalid noise selection: $noise_mode. Use 'high', 'low', 'both', or 'combined'." >&2
      exit 1
      ;;
  esac

  local TIMESTEP_BOUNDARY

  case "$training_mode" in
    t2v)
      TRAIN_TASK="t2v-A14B"
      TRAIN_MODE_LABEL="T2V"
      HIGH_DIT="$T2V_HIGH_DIT"
      LOW_DIT="$T2V_LOW_DIT"
      HIGH_TITLE="WAN2.2-${TRAIN_MODE_LABEL}_HighNoise_${TITLE_PREFIX}"
      LOW_TITLE="WAN2.2-${TRAIN_MODE_LABEL}_LowNoise_${TITLE_PREFIX}"
      COMBINED_TITLE="WAN2.2-${TRAIN_MODE_LABEL}_Combined_${TITLE_PREFIX}"
      TIMESTEP_BOUNDARY=875
      ;;
    i2v)
      TRAIN_TASK="i2v-A14B"
      TRAIN_MODE_LABEL="I2V"
      HIGH_DIT="$I2V_HIGH_DIT"
      LOW_DIT="$I2V_LOW_DIT"
      HIGH_TITLE="WAN2.2-${TRAIN_MODE_LABEL}_HighNoise_${TITLE_PREFIX}"
      LOW_TITLE="WAN2.2-${TRAIN_MODE_LABEL}_LowNoise_${TITLE_PREFIX}"
      COMBINED_TITLE="WAN2.2-${TRAIN_MODE_LABEL}_Combined_${TITLE_PREFIX}"
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

  SAVE_EVERY="${SAVE_EVERY_INPUT:-20}"
  echo "Save every N epochs: $SAVE_EVERY"

  SAVE_OPTIMIZER_STATE="${SAVE_OPTIMIZER_STATE_INPUT:-N}"
  SAVE_OPTIMIZER_STATE=$(normalize_yes_no "$SAVE_OPTIMIZER_STATE")
  case "$SAVE_OPTIMIZER_STATE" in
    Y|N) ;;
    *)
      echo "Invalid save optimizer state option: $SAVE_OPTIMIZER_STATE. Use 'Y' or 'N'." >&2
      exit 1
      ;;
  esac
  RESUME_HIGH_OPTIMIZER_STATE="${RESUME_HIGH_OPTIMIZER_STATE_INPUT:-}"
  RESUME_LOW_OPTIMIZER_STATE="${RESUME_LOW_OPTIMIZER_STATE_INPUT:-}"

  MAX_EPOCHS="${MAX_EPOCHS_INPUT:-100}"
  echo "Max epochs: $MAX_EPOCHS"

  echo ""
  echo "=== Post-Training Options ==="
  UPLOAD_CLOUD="${CLI_UPLOAD_CLOUD:-N}"
  SHUTDOWN_INSTANCE="${CLI_SHUTDOWN_INSTANCE:-N}"

  echo ""
  echo "=== Configuration Summary ==="
  UPLOAD_CLOUD=$(normalize_yes_no "$UPLOAD_CLOUD")
  SHUTDOWN_INSTANCE=$(normalize_yes_no "$SHUTDOWN_INSTANCE")
  echo "  Dataset: $DATASET"
  echo "  Training config: $TRAINING_CONFIG"
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
  if (( RUN_COMBINED )); then
    echo "  Combined title: $COMBINED_TITLE"
  else
    echo "  Combined noise: disabled"
  fi
  echo "  Author:     $AUTHOR"
  echo "  Max epochs: $MAX_EPOCHS"
  echo "  Save every: $SAVE_EVERY epochs"
  echo "  Save optimizer state: $SAVE_OPTIMIZER_STATE"
  if (( RUN_HIGH || RUN_COMBINED )) && [[ -n "$RESUME_HIGH_OPTIMIZER_STATE" ]]; then
    echo "  Resume high/combined state: $RESUME_HIGH_OPTIMIZER_STATE"
  fi
  if (( RUN_LOW )) && [[ -n "$RESUME_LOW_OPTIMIZER_STATE" ]]; then
    echo "  Resume low state: $RESUME_LOW_OPTIMIZER_STATE"
  fi
  echo "  Task:       $TRAIN_TASK"
  echo "  Mode:       ${training_mode^^}"
  echo "  Noise mode: ${noise_mode^^}"
  echo "  Upload to cloud: $UPLOAD_CLOUD"
  if [[ -n "${CLI_CLOUD_CONNECTION_ID:-}" ]]; then
    echo "  Cloud connection: $CLI_CLOUD_CONNECTION_ID"
  fi
  echo "  Auto-shutdown: $SHUTDOWN_INSTANCE"
  echo ""
  PROCEED="Y"
  echo "Proceed with training? [auto: Y]"

  # Validate required files
  require "$PYTHON"
  require "$ACCELERATE"
  require "$VAE"
  require "$T5"
  if (( RUN_COMBINED )); then
    require "$HIGH_DIT"
    require "$LOW_DIT"
  fi
  if (( RUN_HIGH )); then
    require "$HIGH_DIT"
  fi
  if (( RUN_LOW )); then
    require "$LOW_DIT"
  fi
  require "$DATASET"
  require "$TRAINING_CONFIG"
  if (( RUN_HIGH || RUN_COMBINED )) && [[ -n "$RESUME_HIGH_OPTIMIZER_STATE" ]]; then
    require_dir "$RESUME_HIGH_OPTIMIZER_STATE"
    RESUME_HIGH_OPTIMIZER_STATE="$(cd -- "$RESUME_HIGH_OPTIMIZER_STATE" && pwd)"
  fi
  if (( RUN_LOW )) && [[ -n "$RESUME_LOW_OPTIMIZER_STATE" ]]; then
    require_dir "$RESUME_LOW_OPTIMIZER_STATE"
    RESUME_LOW_OPTIMIZER_STATE="$(cd -- "$RESUME_LOW_OPTIMIZER_STATE" && pwd)"
  fi

  cd "$MUSUBI_DIR"

  ensure_accelerate_default
  local EARLY_STOP_MARKER_DIR="$PWD/early_stop_pids"
  rm -rf "$EARLY_STOP_MARKER_DIR"
  mkdir -p "$EARLY_STOP_MARKER_DIR"

  local LOGDIR="$MUSUBI_DIR/logs"
  mkdir -p "$LOGDIR"

  echo "Using --num_cpu_threads_per_process=$ACCELERATE_CPU_THREADS_PER_PROCESS"

  local -a SAVE_STATE_ARGS=()
  local -a HIGH_RESUME_ARGS=()
  local -a LOW_RESUME_ARGS=()
  if [[ "$SAVE_OPTIMIZER_STATE" == "Y" ]]; then
    SAVE_STATE_ARGS+=(--save_state --save_state_on_train_end)
  fi
  if [[ -n "$RESUME_HIGH_OPTIMIZER_STATE" ]]; then
    HIGH_RESUME_ARGS+=(--resume "$RESUME_HIGH_OPTIMIZER_STATE")
  fi
  if [[ -n "$RESUME_LOW_OPTIMIZER_STATE" ]]; then
    LOW_RESUME_ARGS+=(--resume "$RESUME_LOW_OPTIMIZER_STATE")
  fi

  # Musubi loads stable hyperparameters from TOML. These command-line values
  # are intentionally limited to per-run WebUI choices and runtime paths.
  local -a TRAINING_BASE_ARGS=(
    --config_file "$TRAINING_CONFIG"
    --task "$TRAIN_TASK"
    --vae "$VAE"
    --t5 "$T5"
    --dataset_config "$DATASET"
    --max_train_epochs "$MAX_EPOCHS"
    --save_every_n_epochs "$SAVE_EVERY"
    --output_dir "$MUSUBI_DIR/output"
    --metadata_author "$AUTHOR"
    "${SAVE_STATE_ARGS[@]}"
  )

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
  local COMBINED_PORT=""
  local HIGH_GPU=""
  local LOW_GPU=""
  local COMBINED_GPU=""
  local HIGH_PID=""
  local LOW_PID=""
  local COMBINED_PID=""
  local -a WAIT_PIDS=()

  if (( RUN_COMBINED )); then
    COMBINED_PORT=$(get_free_port)
  fi
  if (( RUN_HIGH )); then
    HIGH_PORT=$(get_free_port)
  fi
  if (( RUN_LOW )); then
    LOW_PORT=$(get_free_port)
    if (( RUN_HIGH )) && [[ "$LOW_PORT" == "$HIGH_PORT" ]]; then
      LOW_PORT=$(get_free_port)
    fi
  fi

  if (( RUN_COMBINED )); then
    echo "Waiting for a free GPU for COMBINED noise training..."
    COMBINED_GPU=$(wait_for_free_gpu)
    echo "Starting COMBINED on GPU $COMBINED_GPU (port $COMBINED_PORT) -> run_high.log"
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$COMBINED_PORT" CUDA_VISIBLE_DEVICES="$COMBINED_GPU" \
    "$ACCELERATE" launch --num_cpu_threads_per_process "$ACCELERATE_CPU_THREADS_PER_PROCESS" --num_processes 1 --main_process_port "$COMBINED_PORT" src/musubi_tuner/wan_train_network.py \
      "${TRAINING_BASE_ARGS[@]}" \
      --dit "$LOW_DIT" \
      --dit_high_noise "$HIGH_DIT" \
      --offload_inactive_dit \
      "${HIGH_RESUME_ARGS[@]}" \
      --output_name "$COMBINED_TITLE" \
      --metadata_title "$COMBINED_TITLE" \
      --min_timestep 0 \
      --max_timestep 1000 \
      --timestep_boundary "$TIMESTEP_BOUNDARY" \
      > "$PWD/run_high.log" 2>&1 &
    COMBINED_PID=$!
    WAIT_PIDS+=("$COMBINED_PID")
  fi

  if (( RUN_HIGH )); then
    echo "Waiting for a free GPU for HIGH noise training..."
    HIGH_GPU=$(wait_for_free_gpu)
    echo "Starting HIGH on GPU $HIGH_GPU (port $HIGH_PORT) -> run_high.log"
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$HIGH_PORT" CUDA_VISIBLE_DEVICES="$HIGH_GPU" \
    "$ACCELERATE" launch --num_cpu_threads_per_process "$ACCELERATE_CPU_THREADS_PER_PROCESS" --num_processes 1 --main_process_port "$HIGH_PORT" src/musubi_tuner/wan_train_network.py \
      "${TRAINING_BASE_ARGS[@]}" \
      --dit "$HIGH_DIT" \
      "${HIGH_RESUME_ARGS[@]}" \
      --output_name "$HIGH_TITLE" \
      --metadata_title "$HIGH_TITLE" \
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
    "$ACCELERATE" launch --num_cpu_threads_per_process "$ACCELERATE_CPU_THREADS_PER_PROCESS" --num_processes 1 --main_process_port "$LOW_PORT" src/musubi_tuner/wan_train_network.py \
      "${TRAINING_BASE_ARGS[@]}" \
      --dit "$LOW_DIT" \
      "${LOW_RESUME_ARGS[@]}" \
      --output_name "$LOW_TITLE" \
      --metadata_title "$LOW_TITLE" \
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
  if (( RUN_COMBINED )); then
    echo "COMBINED PID: $COMBINED_PID${COMBINED_GPU:+ (GPU $COMBINED_GPU)}, log: $PWD/run_high.log"
  fi

  if (( RUN_HIGH )) && (( RUN_LOW )); then
    echo "Waiting for both trainings to finish..."
  elif (( RUN_COMBINED )); then
    echo "Waiting for combined noise training to finish..."
  elif (( RUN_HIGH )); then
    echo "Waiting for high noise training to finish..."
  elif (( RUN_LOW )); then
    echo "Waiting for low noise training to finish..."
  fi

  for pid in "${WAIT_PIDS[@]}"; do
    if [[ -n "$pid" ]]; then
      if wait "$pid"; then
        :
      else
        wait_status=$?
        if [[ -f "$EARLY_STOP_MARKER_DIR/$pid" ]]; then
          echo "Training process $pid stopped by configured early stop (exit code $wait_status)."
        elif [[ "$wait_status" -eq 143 || "$wait_status" -eq 137 ]]; then
          echo "Training process $pid stopped before scheduled max epochs."
        else
          echo "Training process $pid failed with exit code $wait_status." >&2
          exit "$wait_status"
        fi
      fi
    fi
  done
  echo "✅ Training completed!"

  OUTPUT_DIR="$MUSUBI_DIR/output"
  RENAMED_OUTPUT_BASE="$MUSUBI_DIR/output-${TITLE_PREFIX}"
  RENAMED_OUTPUT="$RENAMED_OUTPUT_BASE"
  OUTPUT_SUFFIX=2
  while [[ -e "$RENAMED_OUTPUT" ]]; do
    RENAMED_OUTPUT="${RENAMED_OUTPUT_BASE}-${OUTPUT_SUFFIX}"
    OUTPUT_SUFFIX=$((OUTPUT_SUFFIX + 1))
  done
  if [[ -d "$OUTPUT_DIR" ]]; then
    if [[ "$RENAMED_OUTPUT" != "$RENAMED_OUTPUT_BASE" ]]; then
      echo "Output directory already exists; saving this run to $RENAMED_OUTPUT"
    fi
    mv "$OUTPUT_DIR" "$RENAMED_OUTPUT"
  fi
  
  # Analyze training logs and generate plots
  echo ""
  echo "=== Analyzing Training Logs ==="
  if [[ -f "$PWD/run_high.log" || -f "$PWD/run_low.log" ]]; then
    "$PYTHON" /workspace/wan-training-webui/analyze_training_logs.py "$PWD" || echo "Warning: Log analysis failed"
    if [[ -d "$PWD/training_analysis" ]]; then
      mv "$PWD/training_analysis" "$RENAMED_OUTPUT/training_analysis"
    fi

    [[ -f "$PWD/run_high.log" ]] && cp "$PWD/run_high.log" "$RENAMED_OUTPUT/"
    [[ -f "$PWD/run_low.log" ]] && cp "$PWD/run_low.log" "$RENAMED_OUTPUT/"
  else
    echo "Warning: No log files found to analyze"
  fi

  [[ -f "$PWD/webui.log" && -d "$RENAMED_OUTPUT" ]] && cp "$PWD/webui.log" "$RENAMED_OUTPUT/"
  
  # Execute pre-configured post-training actions
  if [[ "$UPLOAD_CLOUD" =~ ^[Yy]$ ]]; then
    echo ""
    echo "=== Uploading to Cloud Storage ==="
    upload_to_cloud "$RENAMED_OUTPUT" "${TITLE_PREFIX}" "$CLI_CLOUD_CONNECTION_ID" || echo "Failed to upload output directory"
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
