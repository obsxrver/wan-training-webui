import re
from pathlib import Path
from typing import FrozenSet, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "run_wan_training.sh"
INDEX_HTML_PATH = Path(__file__).with_name("index.html")

DATASET_ROOT = Path("/workspace/musubi-tuner/dataset")
DATASET_CONFIG_ROOT = Path("/workspace/wan-training-webui/dataset-configs")
LOG_DIR = Path("/workspace/musubi-tuner")
HIGH_LOG = LOG_DIR / "run_high.log"
LOW_LOG = LOG_DIR / "run_low.log"
DOWNLOAD_STATUS_DIR = Path("/workspace/musubi-tuner/models/download_status")

API_KEY_CONFIG_PATH = Path.home() / ".config" / "vastai" / "vast_api_key"
MANAGE_KEYS_URL = "https://cloud.vast.ai/manage-keys"
CLOUD_SETTINGS_URL = "https://cloud.vast.ai/settings/"

DATASET_ROOT.mkdir(parents=True, exist_ok=True)
DATASET_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
DOWNLOAD_STATUS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

CAPTION_PRIORITY = {".txt": 0, ".caption": 1, ".json": 2}
CAPTION_EXTENSIONS = set(CAPTION_PRIORITY)
CAPTION_PREVIEW_LIMIT = 500
PROTECTED_DATASET_DIRS: FrozenSet[str] = frozenset({"cache", "videocache"})
RANGE_HEADER_RE = re.compile(r"bytes=(\d+)-(\d+)?")

TOKEN_ENV_VAR = "JUPYTER_TOKEN"
AUTH_COOKIE_NAME = "token"
AUTH_QUERY_PARAM = "token"
AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
PUBLIC_IP_ENV_VAR = "PUBLIC_IPADDR"
JUPYTER_PORT_ENV_VAR = "VAST_TCP_PORT_8080"
VAST_ENV_VARS = ("CONTAINER_ID", "VAST_CONTAINER_ID", "VAST_TCP_PORT_8080", "PUBLIC_IPADDR")

STEP_PATTERNS = [
    re.compile(r"global_step(?:=|:)\s*(\d+)"),
    re.compile(r"step(?:=|:)\s*(\d+)"),
    re.compile(r"Iteration\s+(\d+)"),
    re.compile(r"steps:.*\|\s*(\d+)\s*/"),
]
EPOCH_PATTERNS = [
    re.compile(r"Epoch\s*\[(\d+)(?:/(\d+))?\]"),
    re.compile(r"Epoch\s+(\d+)(?:\s*/\s*(\d+))?"),
    re.compile(r"Epoch\s+(\d+):"),
    re.compile(r"epoch(?:=|:)\s*(\d+)(?:\s*/\s*(\d+))?"),
    re.compile(r"epoch\s+(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE),
]
LOSS_PATTERNS = [
    re.compile(r"train_loss(?:=|:)\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"loss(?:=|:)\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"Loss\s*=?\s*([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"),
]
# we are parsing lines in /workspace/musubi-tuner/run_high.log and /workspace/musubi-tuner/run_low.log
# that look like this:
# steps:   1%|          | 30/5200 [01:38<4:43:19,  3.29s/it, avr_loss=0.129]
TOTAL_STEP_PATTERNS = [re.compile(r"steps:.*\|\s*\d+\s*/\s*(\d+)")]
TIME_PATTERN = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)<\s*(\d{1,2}:\d{2}(?::\d{2})?)")

# Keep the full run history so refreshes don't drop earlier points; set to an int
# to re-enable trimming if memory ever becomes a concern.
MAX_HISTORY_POINTS: Optional[int] = None
MAX_LOG_LINES = 400
