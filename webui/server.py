import asyncio
import json
import mimetypes
import os
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Awaitable, Callable, Literal

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

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


def _load_auth_token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR)
    if token:
        return token
    generated = secrets.token_hex(32)
    print(
        "[webui] JUPYTER_TOKEN environment variable not set. "
        "Generated temporary token for this process: %s" % generated
    )
    return generated


AUTH_TOKEN = _load_auth_token()


def is_vast_instance() -> bool:
    return any(os.environ.get(var) for var in VAST_ENV_VARS)


def _build_jupyter_base_url() -> Optional[str]:
    public_ip = os.environ.get(PUBLIC_IP_ENV_VAR)
    port = os.environ.get(JUPYTER_PORT_ENV_VAR)
    if not public_ip or not port:
        return None
    return f"https://{public_ip}:{port}"


class TokenAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, token: str) -> None:
        super().__init__(app)
        self._token = token

    def _set_auth_cookie(self, response, scheme: str) -> None:
        response.set_cookie(
            AUTH_COOKIE_NAME,
            self._token,
            httponly=True,
            secure=scheme == "https",
            samesite="lax",
            path="/",
            max_age=AUTH_COOKIE_MAX_AGE,
            expires=AUTH_COOKIE_MAX_AGE,
        )

    async def dispatch(self, request: Request, call_next):
        if not self._token:
            return await call_next(request)

        cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
        query_token = request.query_params.get(AUTH_QUERY_PARAM)
        token_source = None

        if cookie_token == self._token:
            token_source = "cookie"
        elif query_token == self._token:
            token_source = "query"

        if token_source is None:
            if request.method == "OPTIONS":
                return await call_next(request)
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        if token_source == "query" and request.method in {"GET", "HEAD"}:
            redirect_url = request.url.remove_query_params(AUTH_QUERY_PARAM)
            response = RedirectResponse(url=str(redirect_url), status_code=303)
            self._set_auth_cookie(response, redirect_url.scheme)
            return response

        response = await call_next(request)

        if token_source == "query" and cookie_token != self._token:
            self._set_auth_cookie(response, request.url.scheme)
        elif token_source == "cookie":
            self._set_auth_cookie(response, request.url.scheme)

        return response

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
# we are parsing lines in /workspace/musubi-tuner/run_high.log and /workspace/musubi-tuner/run_low.log that look like this:
# steps:   1%|          | 30/5200 [01:38<4:43:19,  3.29s/it, avr_loss=0.129]
TOTAL_STEP_PATTERNS = [
    re.compile(r"steps:.*\|\s*\d+\s*/\s*(\d+)")
]
TIME_PATTERN = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)<\s*(\d{1,2}:\d{2}(?::\d{2})?)")
# Keep the full run history so refreshes don't drop earlier points; set to an int
# to re-enable trimming if memory ever becomes a concern.
MAX_HISTORY_POINTS: Optional[int] = None
MAX_LOG_LINES = 400


def parse_cloud_connections(output: str) -> List[Dict[str, str]]:
    connections: List[Dict[str, str]] = []
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    json_start = next(
        (index for index, line in enumerate(lines) if line.startswith("[") or line.startswith("{")),
        None,
    )
    if json_start is not None:
        json_text = "\n".join(lines[json_start:])
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                connection_id = item.get("id")
                if connection_id is None:
                    continue
                name = str(item.get("name") or "").strip()
                cloud_type = str(item.get("cloud_type") or "").strip()
                connections.append(
                    {"id": str(connection_id), "name": name, "cloud_type": cloud_type}
                )
            if connections:
                return connections

    for line in lines:
        if line.startswith("https://"):
            continue
        if line.lower().startswith("id"):
            continue
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        connection_id = parts[0]
        cloud_type = parts[-1]
        name = " ".join(parts[1:-1]) if len(parts) > 2 else ""
        connections.append({"id": connection_id, "name": name, "cloud_type": cloud_type})
    return connections


def is_api_key_configured() -> bool:
    try:
        return API_KEY_CONFIG_PATH.exists() and API_KEY_CONFIG_PATH.read_text(encoding="utf-8").strip() != ""
    except OSError:
        return False


def maybe_set_container_api_key() -> None:
    container_key = os.environ.get("CONTAINER_API_KEY")
    if not container_key or is_api_key_configured():
        return
    if shutil.which("vastai") is None:
        return
    try:
        subprocess.run(
            ["vastai", "set", "api-key", container_key],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return


def _read_pid(pid_path: Path) -> Optional[int]:
    try:
        text = pid_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def get_download_status() -> Dict[str, Any]:
    active: List[str] = []
    if not DOWNLOAD_STATUS_DIR.exists():
        return {"active": active, "pending": False}

    for pid_file in DOWNLOAD_STATUS_DIR.glob("*.pid"):
        pid_value = _read_pid(pid_file)
        if pid_value is None or not _process_is_running(pid_value):
            try:
                pid_file.unlink()
            except OSError:
                pass
            exit_marker = pid_file.with_suffix(".exit")
            try:
                if exit_marker.exists():
                    exit_marker.unlink()
            except OSError:
                pass
            continue
        active.append(pid_file.stem)

    return {"active": sorted(active), "pending": bool(active)}


def build_snapshot() -> Dict:
    snapshot = training_state.snapshot()
    snapshot["downloads"] = get_download_status()
    return snapshot


async def gather_cloud_status() -> Dict[str, Any]:
    vast_instance = is_vast_instance()
    cli_available = shutil.which("vastai") is not None
    api_key_configured = is_api_key_configured()

    if not vast_instance:
        return {
            "is_vast_instance": False,
            "cli_available": cli_available,
            "api_key_configured": api_key_configured,
            "permission_error": False,
            "has_connections": False,
            "can_upload": False,
            "message": "Cloud uploads are available only on Vast.ai instances.",
            "connections": [],
        }

    if not cli_available:
        return {
            "is_vast_instance": vast_instance,
            "cli_available": False,
            "api_key_configured": api_key_configured,
            "permission_error": False,
            "has_connections": False,
            "can_upload": False,
            "message": "vastai CLI not found. Install it with: pip install vastai --user --break-system-packages",
            "connections": [],
        }

    try:
        process = await asyncio.create_subprocess_exec(
            "vastai",
            "show",
            "connections",
            "--raw",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
    except FileNotFoundError:
        return {
            "is_vast_instance": vast_instance,
            "cli_available": False,
            "api_key_configured": api_key_configured,
            "permission_error": False,
            "has_connections": False,
            "can_upload": False,
            "message": "vastai CLI not found. Install it with: pip install vastai --user --break-system-packages",
            "connections": [],
        }

    output = (stdout_bytes + stderr_bytes).decode("utf-8", errors="ignore")
    lower_output = output.lower()
    permission_error = "failed with error 401" in lower_output
    connections: List[Dict[str, str]] = []

    if not permission_error:
        connections = parse_cloud_connections(output)

    has_connections = bool(connections)
    can_upload = cli_available and not permission_error and has_connections

    if permission_error:
        message = (
            "Current Vast.ai API key lacks the permissions required for cloud uploads. "
            f"Create a new key at {MANAGE_KEYS_URL} and save it below."
        )
    elif not has_connections:
        message = (
            "No cloud connections detected. Configure one at "
            f"{CLOUD_SETTINGS_URL} and open \"cloud connection\" to link storage."
        )
    elif process.returncode != 0:
        message = output.strip() or "Failed to query cloud connections."
    else:
        message = "Cloud uploads are ready to use."

    return {
        "is_vast_instance": vast_instance,
        "cli_available": cli_available,
        "api_key_configured": api_key_configured,
        "permission_error": permission_error,
        "has_connections": has_connections,
        "can_upload": can_upload,
        "message": message,
        "connections": connections,
    }


class VideoConversionError(Exception):
    """Raised when automatic video conversion cannot be completed."""


def _relative_to_any(path: Path, bases: Iterable[Path]) -> str:
    for base in bases:
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name


def _should_skip_path(path: Path) -> bool:
    return any(part in PROTECTED_DATASET_DIRS for part in path.parts)


def _load_video_directories(config_path: Path) -> Set[Path]:
    try:
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise VideoConversionError(f"Dataset config '{config_path}' not found.") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise VideoConversionError(f"Failed to read dataset config '{config_path}': {exc}") from exc

    directories: Set[Path] = set()
    datasets = config.get("datasets")
    if isinstance(datasets, list):
        for entry in datasets:
            if not isinstance(entry, dict):
                continue
            video_dir = entry.get("video_directory")
            if isinstance(video_dir, str) and video_dir.strip():
                directories.add(Path(video_dir).expanduser())

    if not directories:
        parent = config_path.parent
        if parent.exists() and parent.is_dir():
            directories.add(parent)

    resolved: Set[Path] = set()
    missing: List[str] = []
    for directory in directories:
        resolved_dir = directory.resolve()
        if resolved_dir.exists() and resolved_dir.is_dir():
            resolved.add(resolved_dir)
        else:
            missing.append(str(directory))

    if missing and not resolved:
        raise VideoConversionError(
            "No valid video directories were found. Checked: " + ", ".join(missing)
        )

    return resolved


async def _probe_video_fps(path: Path, ffprobe_path: str) -> Optional[float]:
    process = await asyncio.create_subprocess_exec(
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", "ignore").strip() or stdout.decode("utf-8", "ignore").strip()
        raise VideoConversionError(
            f"Failed to inspect '{path.name}' with ffprobe: {message or 'Unknown error.'}"
        )

    rate_text = stdout.decode("utf-8", "ignore").strip()
    if not rate_text or rate_text == "0/0":
        return None
    if "/" in rate_text:
        numerator, denominator = rate_text.split("/", 1)
        try:
            num = float(numerator)
            den = float(denominator)
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den
    try:
        return float(rate_text)
    except ValueError:
        return None


DEFAULT_DATASET_CONFIG = DATASET_CONFIG_ROOT / "dataset.toml"
DATASET_CONFIG_FALLBACK_URL = "https://raw.githubusercontent.com/obsxrver/wan-training-webui/refs/heads/main/dataset-configs/dataset.toml"


def _download_dataset_config(config_path: Path) -> None:
    import urllib.request

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_CONFIG_FALLBACK_URL) as response:
        if getattr(response, "status", 200) != 200:
            raise VideoConversionError(
                f"Failed to download dataset config from {DATASET_CONFIG_FALLBACK_URL}."
            )
        data = response.read()
    with config_path.open("wb") as handle:
        handle.write(data)


async def convert_videos_to_target_fps(
    config_path: Path,
    target_fps: int,
    log_callback: Callable[[str], Awaitable[None]],
) -> Tuple[List[str], int]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if not ffmpeg_path or not ffprobe_path:
        missing = []
        if not ffmpeg_path:
            missing.append("ffmpeg")
        if not ffprobe_path:
            missing.append("ffprobe")
        tools = " and ".join(missing)
        raise VideoConversionError(f"Required tool(s) {tools} not found in PATH.")

    if not config_path.exists():
        await log_callback(
            f"Dataset config '{config_path}' not found locally. Downloading default dataset configuration..."
        )
        try:
            await asyncio.to_thread(_download_dataset_config, config_path)
        except Exception as exc:  # pragma: no cover - network failures surface to user
            raise VideoConversionError(
                f"Unable to download dataset config '{config_path}': {exc}"
            ) from exc
        await log_callback("Default dataset configuration downloaded successfully.")

    directories = _load_video_directories(config_path)
    if not directories:
        await log_callback("No video directories available for conversion. Skipping 16 FPS step.")
        return ([], 0)

    video_files: Set[Path] = set()
    for directory in directories:
        try:
            for file_path in directory.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                if _should_skip_path(file_path):
                    continue
                video_files.add(file_path.resolve())
        except OSError as exc:
            await log_callback(f"Failed to scan directory '{directory}': {exc}")

    if not video_files:
        await log_callback("No video files found for 16 FPS conversion. Skipping step.")
        return ([], 0)

    needs_conversion: List[Tuple[Path, Optional[float]]] = []
    already_matching = 0
    for path in sorted(video_files):
        fps = await _probe_video_fps(path, ffprobe_path)
        if fps is not None and abs(fps - target_fps) <= 0.05:
            already_matching += 1
            continue
        needs_conversion.append((path, fps))

    if not needs_conversion:
        await log_callback("All dataset videos already at 16 FPS. No conversion needed.")
        return ([], already_matching)

    if already_matching:
        await log_callback(f"Skipping {already_matching} video(s) already at {target_fps} FPS.")

    total = len(needs_conversion)
    await log_callback(f"Converting {total} video(s) to {target_fps} FPS before starting training…")

    completed: List[str] = []
    for index, (video_path, fps) in enumerate(needs_conversion, start=1):
        display_name = _relative_to_any(video_path, directories)
        if fps is None:
            await log_callback(f"[{index}/{total}] Converting {display_name} to {target_fps} FPS…")
        else:
            await log_callback(
                f"[{index}/{total}] Converting {display_name} from {fps:.2f} FPS to {target_fps} FPS…"
            )

        temp_file = Path(
            tempfile.NamedTemporaryFile(
                delete=False, suffix=video_path.suffix, dir=str(video_path.parent)
            ).name
        )
        try:
            process = await asyncio.create_subprocess_exec(
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                f"fps={target_fps}",
                str(temp_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                error_text = stderr.decode("utf-8", "ignore").strip() or "Unknown ffmpeg error."
                raise VideoConversionError(
                    f"Failed to convert '{video_path.name}' to {target_fps} FPS: {error_text}"
                )
            os.replace(temp_file, video_path)
            completed.append(display_name)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    await log_callback(f"Finished converting {len(completed)} video(s) to {target_fps} FPS.")
    return (completed, already_matching)


maybe_set_container_api_key()

class TrainRequest(BaseModel):
    title_prefix: str = Field(default="mylora", min_length=1)
    author: str = Field(default="authorName", min_length=1)
    dataset_path: str = Field(default=str(DEFAULT_DATASET_CONFIG))
    save_every: int = Field(default=100, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    cpu_threads_per_process: Optional[int] = Field(default=None, ge=1)
    max_data_loader_workers: Optional[int] = Field(default=None, ge=1)
    upload_cloud: bool = True
    shutdown_instance: bool = True
    auto_confirm: bool = True
    training_mode: Literal["t2v", "i2v"] = "t2v"
    noise_mode: Literal["both", "high", "low"] = "both"
    convert_videos_to_16fps: bool = False
    cloud_connection_id: Optional[str] = None


class ApiKeyRequest(BaseModel):
    api_key: str = Field(min_length=1)




class DeleteDatasetItem(BaseModel):
    media_path: str = Field(min_length=1)


class UpdateCaptionRequest(BaseModel):
    media_path: str = Field(min_length=1)
    caption_text: Optional[str] = None
    caption_path: Optional[str] = None


class BulkCaptionRequest(BaseModel):
    caption_text: str = Field(min_length=1)
    apply_to: Literal["all_images", "uncaptioned_images"] = "all_images"


class EventManager:
    def __init__(self) -> None:
        self._listeners: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def register(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._listeners.append(queue)
        return queue

    async def unregister(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            if queue in self._listeners:
                self._listeners.remove(queue)

    async def publish(self, event: Dict) -> None:
        async with self._lock:
            listeners = list(self._listeners)
        for queue in listeners:
            await queue.put(event)


class TrainingState:
    def __init__(self) -> None:
        self.process: Optional[asyncio.subprocess.Process] = None
        self.status: str = "idle"
        self.running: bool = False
        self.history: Dict[str, List[Dict[str, float]]] = {"high": [], "low": []}
        self.current: Dict[str, Optional[Dict[str, Any]]] = {"high": None, "low": None}
        self.pending: Dict[str, Dict[str, Optional[float]]] = {
            "high": {"step": None, "loss": None},
            "low": {"step": None, "loss": None},
        }
        self.logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.stop_event: asyncio.Event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []
        self.stop_requested: bool = False
        self.active_runs: Set[str] = {"high", "low"}
        self.noise_mode: str = "both"

    def reset_for_start(self, active_runs: Optional[Set[str]] = None) -> None:
        self.active_runs = set(active_runs or {"high", "low"})
        self.history = {"high": [], "low": []}
        self.current = {"high": None, "low": None}
        self.pending = {
            "high": {"step": None, "loss": None},
            "low": {"step": None, "loss": None},
        }
        self.logs.clear()
        self.stop_event = asyncio.Event()
        self.tasks = []
        self.stop_requested = False

    def mark_started(
        self, process: asyncio.subprocess.Process, active_runs: Set[str], noise_mode: str
    ) -> None:
        self.reset_for_start(active_runs)
        self.process = process
        self.status = "running"
        self.running = True
        self.noise_mode = noise_mode

    def mark_finished(self, status: str) -> None:
        self.status = status
        self.running = False
        self.process = None
        self.stop_requested = False
        if not self.stop_event.is_set():
            self.stop_event.set()

    def snapshot(self) -> Dict:
        return {
            "status": self.status,
            "running": self.running,
            "active_runs": sorted(self.active_runs),
            "noise_mode": self.noise_mode,
            "high": {
                "history": list(self.history["high"]),
                "current": dict(self.current["high"]) if self.current["high"] else None,
            },
            "low": {
                "history": list(self.history["low"]),
                "current": dict(self.current["low"]) if self.current["low"] else None,
            },
            "logs": list(self.logs),
        }

    def add_task(self, task: asyncio.Task) -> None:
        self.tasks.append(task)

    async def wait_for_tasks(self) -> None:
        if not self.tasks:
            return
        done, pending = await asyncio.wait(self.tasks, timeout=0)
        for task in pending:
            task.cancel()

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip())

    async def update_metrics(self, run: str, metrics: Dict[str, Optional[Any]]) -> Optional[Dict[str, Optional[Dict[str, Any]]]]:
        entry = self.pending[run]
        changed = False
        current = dict(self.current[run]) if self.current[run] else {}

        step_value = metrics.get("step")
        if step_value is not None:
            step_int = int(step_value)
            if entry["step"] != step_int:
                entry["step"] = step_int
            if current.get("step") != step_int:
                current["step"] = step_int
                changed = True

        loss_value = metrics.get("loss")
        if loss_value is not None:
            loss_float = float(loss_value)
            if entry["loss"] != loss_float:
                entry["loss"] = loss_float

        total_steps = metrics.get("total_steps")
        if total_steps is not None:
            total_int = int(total_steps)
            if current.get("total_steps") != total_int:
                current["total_steps"] = total_int
                changed = True

        epoch_value = metrics.get("epoch")
        if epoch_value is not None:
            epoch_int = int(epoch_value)
            # Only update epoch if new value is greater (prevent resets)
            current_epoch = current.get("epoch", 0)
            epoch_int = max(current_epoch, epoch_int)
            if current.get("epoch") != epoch_int:
                current["epoch"] = epoch_int
                changed = True

        total_epochs = metrics.get("total_epochs")
        if total_epochs is not None:
            total_epochs_int = int(total_epochs)
            if current.get("total_epochs") != total_epochs_int:
                current["total_epochs"] = total_epochs_int
                changed = True

        elapsed = metrics.get("time_elapsed")
        if elapsed is not None and current.get("time_elapsed") != elapsed:
            current["time_elapsed"] = str(elapsed)
            changed = True

        remaining = metrics.get("time_remaining")
        if remaining is not None and current.get("time_remaining") != remaining:
            current["time_remaining"] = str(remaining)
            changed = True

        point: Optional[Dict[str, Any]] = None
        history = self.history[run]
        if entry["step"] is not None and entry["loss"] is not None:
            point = {"step": int(entry["step"]), "loss": float(entry["loss"])}
            if current.get("step") != point["step"] or current.get("loss") != point["loss"]:
                changed = True
            current["step"] = point["step"]
            current["loss"] = point["loss"]
            if history and history[-1]["step"] == point["step"]:
                if history[-1].get("loss") != point["loss"]:
                    history[-1] = point
                    changed = True
            else:
                history.append(point)
                changed = True
                if MAX_HISTORY_POINTS is not None and len(history) > MAX_HISTORY_POINTS:
                    del history[: len(history) - MAX_HISTORY_POINTS]
            self.current[run] = current
            entry["loss"] = None
        else:
            if current:
                if not self.current[run] or current != self.current[run]:
                    changed = True
                self.current[run] = current

        if not changed and point is None:
            return None

        current_snapshot = dict(self.current[run]) if self.current[run] else None
        return {"point": point, "current": current_snapshot}


event_manager = EventManager()
training_state = TrainingState()
download_watchdog_task: Optional[asyncio.Task] = None
app = FastAPI(title="WAN 2.2 Training UI")


app.add_middleware(TokenAuthMiddleware, token=AUTH_TOKEN)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="UI assets missing")
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.get("/jupyter-info")
async def jupyter_info() -> Dict[str, Optional[str]]:
    base_url = _build_jupyter_base_url()
    token = os.environ.get(TOKEN_ENV_VAR) or AUTH_TOKEN
    return {"base_url": base_url, "token": token}


@app.get("/dataset-configs")
async def list_dataset_configs() -> Dict[str, Any]:
    if DATASET_CONFIG_ROOT.exists() and not DATASET_CONFIG_ROOT.is_dir():
        raise HTTPException(status_code=500, detail="Dataset config path is not a directory")

    DATASET_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    configs: List[Dict[str, str]] = []
    for entry in sorted(DATASET_CONFIG_ROOT.glob("*.toml")):
        if entry.is_file():
            configs.append({"name": entry.name, "path": str(entry.resolve())})

    return {"configs": configs, "default_path": str(DEFAULT_DATASET_CONFIG)}




def _clear_dataset_directory() -> None:
    try:
        if DATASET_ROOT.exists() and not DATASET_ROOT.is_dir():
            raise HTTPException(status_code=500, detail="Dataset path is not a directory")
        if not DATASET_ROOT.exists():
            DATASET_ROOT.mkdir(parents=True, exist_ok=True)
            return
        for entry in DATASET_ROOT.iterdir():
            try:
                if entry.is_dir() and entry.name.lower() in PROTECTED_DATASET_DIRS:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except FileNotFoundError:
                continue
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to prepare dataset directory: {exc}") from exc
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)


def _resolve_dataset_file(path: str) -> Path:
    dataset_root = DATASET_ROOT.resolve()
    target_path = (dataset_root / path).resolve()
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if dataset_root not in target_path.parents:
        raise HTTPException(status_code=400, detail="Invalid dataset path")
    return target_path


def _collect_dataset_items() -> List[Dict[str, Any]]:
    if not DATASET_ROOT.exists() or not DATASET_ROOT.is_dir():
        return []

    dataset_root = DATASET_ROOT.resolve()
    captions: Dict[tuple[str, str], Dict[str, Any]] = {}

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in CAPTION_EXTENSIONS:
            continue

        relative = file_path.relative_to(dataset_root)
        parent_key = relative.parent.as_posix()
        stem_key = file_path.stem.lower()
        priority = CAPTION_PRIORITY.get(suffix, 999)
        existing = captions.get((parent_key, stem_key))
        if existing and existing["priority"] <= priority:
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
        if len(text) > CAPTION_PREVIEW_LIMIT:
            text = text[:CAPTION_PREVIEW_LIMIT].rstrip() + "…"

        captions[(parent_key, stem_key)] = {
            "caption_path": relative.as_posix(),
            "caption_text": text,
            "priority": priority,
        }

    items: List[Dict[str, Any]] = []

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            continue

        relative = file_path.relative_to(dataset_root)
        parent_key = relative.parent.as_posix()
        stem_key = file_path.stem.lower()
        caption_info = captions.get((parent_key, stem_key), {})
        media_path = relative.as_posix()
        media_url = f"/dataset/media/{media_path}"
        media_kind = "video" if suffix in VIDEO_EXTENSIONS else "image"

        item: Dict[str, Any] = {
            "media_path": media_path,
            "media_url": media_url,
            "media_kind": media_kind,
            "caption_path": caption_info.get("caption_path"),
            "caption_text": caption_info.get("caption_text"),
        }

        if media_kind == "image":
            item["image_path"] = media_path
            item["image_url"] = media_url
        else:
            item["video_path"] = media_path
            item["video_url"] = media_url

        items.append(item)

    items.sort(key=lambda item: item["media_path"].lower())
    return items


def _is_in_protected_dir(path: Path) -> bool:
    try:
        relative_parts = path.relative_to(DATASET_ROOT).parts
    except ValueError:
        return False
    return any(part.lower() in PROTECTED_DATASET_DIRS for part in relative_parts)


def _find_existing_caption_file(media_file: Path) -> Optional[Path]:
    for suffix, _ in sorted(CAPTION_PRIORITY.items(), key=lambda item: item[1]):
        candidate = media_file.with_suffix(suffix)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_dataset_relative_path(relative_path: str) -> Path:
    dataset_root = DATASET_ROOT.resolve()
    normalized = _normalize_export_path(relative_path)
    target_path = (dataset_root / normalized).resolve()
    if dataset_root not in target_path.parents and target_path != dataset_root:
        raise HTTPException(status_code=400, detail="Invalid dataset path")
    return target_path


def _normalize_export_path(value: str) -> Path:
    normalized = value.replace("\\", "/").strip().lstrip("/")
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid file name")
    candidate = Path(normalized)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise HTTPException(status_code=400, detail="Invalid file path")
    return candidate




async def _write_upload_to_path(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)






def _delete_dataset_item(relative_path: str) -> Dict[str, Any]:
    file_path = _resolve_dataset_file(relative_path)
    deleted: List[str] = []
    captions_removed: List[str] = []
    relative_display = file_path.relative_to(DATASET_ROOT).as_posix()

    try:
        file_path.unlink()
        deleted.append(relative_display)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{relative_display}': {exc}") from exc

    for suffix in CAPTION_EXTENSIONS:
        caption_path = file_path.with_suffix(suffix)
        if not caption_path.exists() or not caption_path.is_file():
            continue
        try:
            caption_path.unlink()
            captions_removed.append(caption_path.relative_to(DATASET_ROOT).as_posix())
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Removed media but failed to delete caption '{caption_path.name}': {exc}",
            ) from exc

    removed_count = 1 + len(captions_removed)
    message = f"Removed '{relative_display}' from the dataset."
    if captions_removed:
        caption_count = len(captions_removed)
        plural = "s" if caption_count != 1 else ""
        message += f" Deleted {caption_count} caption file{plural}."

    return {
        "message": message,
        "deleted": deleted,
        "captions_removed": captions_removed,
        "removed_count": removed_count,
    }




@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)) -> Dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    _clear_dataset_directory()
    saved = []
    for file in files:
        filename = Path(file.filename).name
        if not filename:
            continue
        destination = DATASET_ROOT / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        await file.close()
        saved.append(str(destination))
    return {"saved": saved, "count": len(saved)}


@app.get("/dataset/files")
async def dataset_files() -> Dict[str, Any]:
    items = _collect_dataset_items()
    return {"items": items, "total": len(items)}


@app.get("/dataset/caption")
async def dataset_get_caption(caption_path: Optional[str] = None, media_path: Optional[str] = None) -> Dict[str, Any]:
    if caption_path:
        relative_caption = caption_path.strip()
        if not relative_caption:
            raise HTTPException(status_code=400, detail="caption_path is required")
        caption_file = _resolve_dataset_relative_path(relative_caption)
        if not caption_file.exists() or not caption_file.is_file():
            raise HTTPException(status_code=404, detail="Caption file not found")
    elif media_path:
        normalized_media = media_path.strip()
        if not normalized_media:
            raise HTTPException(status_code=400, detail="media_path is required")
        media_file = _resolve_dataset_file(normalized_media)
        caption_file = None
        for suffix, priority in sorted(CAPTION_PRIORITY.items(), key=lambda item: item[1]):
            candidate = media_file.with_suffix(suffix)
            if candidate.exists() and candidate.is_file():
                caption_file = candidate
                break
        if caption_file is None:
            return {"caption_text": "", "caption_path": None}
        relative_caption = caption_file.relative_to(DATASET_ROOT).as_posix()
    else:
        raise HTTPException(status_code=400, detail="caption_path or media_path is required")

    try:
        caption_text = caption_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read caption: {exc}") from exc

    return {"caption_text": caption_text, "caption_path": relative_caption}


@app.post("/dataset/delete")
async def dataset_delete(payload: DeleteDatasetItem) -> Dict[str, Any]:
    result = _delete_dataset_item(payload.media_path)
    return result


@app.post("/dataset/caption")
async def dataset_update_caption(payload: UpdateCaptionRequest) -> Dict[str, Any]:
    media_path = payload.media_path.strip()
    if not media_path:
        raise HTTPException(status_code=400, detail="media_path is required")

    media_file = _resolve_dataset_file(media_path)
    caption_text_raw = payload.caption_text or ""
    caption_text = caption_text_raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    if payload.caption_path:
        relative_caption = payload.caption_path.strip()
        if not relative_caption:
            raise HTTPException(status_code=400, detail="caption_path is invalid")
        suffix = Path(relative_caption).suffix.lower()
        if suffix not in CAPTION_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Caption file must use a supported extension")
        caption_file = _resolve_dataset_relative_path(relative_caption)
    else:
        caption_file = media_file.with_suffix(".txt")
        relative_caption = caption_file.relative_to(DATASET_ROOT).as_posix()

    caption_file.parent.mkdir(parents=True, exist_ok=True)

    if not caption_text:
        if caption_file.exists():
            try:
                caption_file.unlink()
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Failed to delete caption file: {exc}") from exc
            message = f"Caption removed for '{media_path}'."
        else:
            message = f"No caption text provided for '{media_path}'."
        return {"message": message, "caption_text": "", "caption_path": None}

    try:
        caption_file.write_text(caption_text, encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save caption: {exc}") from exc

    message = f"Caption updated for '{media_path}'."
    return {"message": message, "caption_text": caption_text, "caption_path": relative_caption}


@app.post("/dataset/caption/bulk")
async def dataset_bulk_caption(payload: BulkCaptionRequest) -> Dict[str, Any]:
    caption_text = payload.caption_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not caption_text:
        raise HTTPException(status_code=400, detail="caption_text is required")

    if not DATASET_ROOT.exists() or not DATASET_ROOT.is_dir():
        return {
            "message": "Dataset directory is missing.",
            "updated_count": 0,
            "skipped_count": 0,
            "total_images": 0,
            "caption_text": caption_text,
        }

    total_images = 0
    updated_count = 0
    skipped_count = 0
    dataset_root = DATASET_ROOT.resolve()

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file() or _is_in_protected_dir(file_path):
            continue
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS and file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        total_images += 1
        existing_caption = _find_existing_caption_file(file_path)
        if payload.apply_to == "uncaptioned_images" and existing_caption is not None:
            skipped_count += 1
            continue

        caption_file = file_path.with_suffix(".txt")
        caption_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            caption_file.write_text(caption_text, encoding="utf-8")
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save caption for '{file_path.name}': {exc}",
            ) from exc

        updated_count += 1

    if total_images == 0:
        message = "No images found to caption."
    else:
        plural = "s" if total_images != 1 else ""
        message = f"Applied caption to {updated_count} of {total_images} image{plural}."
        if payload.apply_to == "uncaptioned_images" and skipped_count:
            skipped_plural = "s" if skipped_count != 1 else ""
            message += f" Skipped {skipped_count} image{skipped_plural} with existing captions."

    return {
        "message": message,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "total_images": total_images,
        "caption_text": caption_text,
    }


def _iter_file_chunks(file_path: Path, start: int = 0, end: Optional[int] = None, chunk_size: int = 1024 * 1024):
    with file_path.open("rb") as stream:
        stream.seek(start)
        bytes_remaining = (end - start + 1) if end is not None else None
        while True:
            if bytes_remaining is not None and bytes_remaining <= 0:
                break
            read_size = chunk_size if bytes_remaining is None else min(chunk_size, bytes_remaining)
            data = stream.read(read_size)
            if not data:
                break
            if bytes_remaining is not None:
                bytes_remaining -= len(data)
            yield data


@app.get("/dataset/media/{path:path}")
async def dataset_media(path: str, request: Request) -> StreamingResponse:
    try:
        file_path = _resolve_dataset_file(path)
    except HTTPException:
        raise

    media_type, _ = mimetypes.guess_type(str(file_path))
    media_type = media_type or "application/octet-stream"
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    if range_header:
        range_value = range_header.strip().lower()
        match = RANGE_HEADER_RE.match(range_value)
        if not match:
            raise HTTPException(status_code=416, detail="Invalid Range header")
        start = int(match.group(1))
        end = match.group(2)
        if start >= file_size:
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")
        end = int(end) if end is not None else file_size - 1
        end = min(end, file_size - 1)
        if end < start:
            raise HTTPException(status_code=416, detail="Requested range not satisfiable")
        content_length = end - start + 1
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        }
        return StreamingResponse(
            _iter_file_chunks(file_path, start=start, end=end),
            status_code=206,
            media_type=media_type,
            headers=headers,
        )

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    return StreamingResponse(_iter_file_chunks(file_path), media_type=media_type, headers=headers)


@app.get("/cloud-status")
async def cloud_status() -> Dict[str, Any]:
    return await gather_cloud_status()


@app.post("/vast-api-key")
async def set_vast_api_key(payload: ApiKeyRequest) -> Dict[str, Any]:
    api_key = payload.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required.")
    if not is_vast_instance():
        raise HTTPException(
            status_code=403,
            detail="Vast.ai API keys can only be configured on a Vast.ai instance.",
        )
    if shutil.which("vastai") is None:
        raise HTTPException(status_code=500, detail="vastai CLI is not installed on this instance.")

    env = os.environ.copy()
    try:
        process = await asyncio.create_subprocess_exec(
            "vastai",
            "set",
            "api-key",
            api_key,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="vastai CLI is not installed on this instance.") from exc

    message = (stdout_bytes + stderr_bytes).decode("utf-8", errors="ignore").strip()
    if process.returncode != 0:
        raise HTTPException(status_code=400, detail=message or "Failed to save API key.")

    status = await gather_cloud_status()
    return {"message": message or "API key saved.", "cloud_status": status}


async def stream_process_output(process: asyncio.subprocess.Process) -> None:
    if process.stdout is None:
        return
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="ignore").rstrip()
        if decoded:
            training_state.append_log(decoded)
            await event_manager.publish({"type": "log", "line": decoded})


def parse_metrics(line: str) -> Dict[str, Optional[Any]]:
    step_value: Optional[int] = None
    loss_value: Optional[float] = None
    total_steps: Optional[int] = None
    epoch_value: Optional[int] = None
    total_epochs: Optional[int] = None
    elapsed_time: Optional[str] = None
    remaining_time: Optional[str] = None
    for pattern in STEP_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                step_value = int(match.group(1))
            except ValueError:
                step_value = None
            break
    for pattern in LOSS_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                loss_value = float(match.group(1))
            except ValueError:
                loss_value = None
            break
    for pattern in TOTAL_STEP_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                total_steps = int(match.group(1))
            except ValueError:
                total_steps = None
            break
    for pattern in EPOCH_PATTERNS:
        match = pattern.search(line)
        if match:
            try:
                epoch_value = int(match.group(1))
            except (ValueError, TypeError):
                epoch_value = None
            total_group: Optional[str] = None
            if match.lastindex and match.lastindex >= 2:
                total_group = match.group(2)
            if total_group is not None:
                try:
                    total_epochs = int(total_group)
                except ValueError:
                    total_epochs = None
            break
    time_match = TIME_PATTERN.search(line)
    if time_match:
        elapsed_time = time_match.group(1)
        remaining_time = time_match.group(2)
    return {
        "step": step_value,
        "loss": loss_value,
        "total_steps": total_steps,
        "epoch": epoch_value,
        "total_epochs": total_epochs,
        "time_elapsed": elapsed_time,
        "time_remaining": remaining_time,
    }


async def monitor_log_file(path: Path, run: str) -> None:
    position = 0
    while not training_state.stop_event.is_set():
        if path.exists():
            try:
                size = path.stat().st_size
                if size < position:
                    position = 0
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    handle.seek(position)
                    for line in handle:
                        metrics = parse_metrics(line)
                        result = await training_state.update_metrics(run, metrics)
                        if not result:
                            continue
                        event: Dict[str, Any] = {"type": "metrics", "run": run}
                        point = result.get("point")
                        current = result.get("current")
                        if point:
                            event.update({"step": point["step"], "loss": point["loss"]})
                        if current:
                            event["current"] = current
                        await event_manager.publish(event)
                    position = handle.tell()
            except OSError:
                position = 0
        await asyncio.sleep(1.0)
    # Flush remaining data after stop
    if path.exists():
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(position)
                for line in handle:
                    metrics = parse_metrics(line)
                    result = await training_state.update_metrics(run, metrics)
                    if not result:
                        continue
                    event: Dict[str, Any] = {"type": "metrics", "run": run}
                    point = result.get("point")
                    current = result.get("current")
                    if point:
                        event.update({"step": point["step"], "loss": point["loss"]})
                    if current:
                        event["current"] = current
                    await event_manager.publish(event)
        except OSError:
            pass


async def wait_for_completion(process: asyncio.subprocess.Process) -> None:
    returncode = await process.wait()
    if training_state.stop_requested:
        status = "stopped"
    else:
        status = "completed" if returncode == 0 else "failed"
    training_state.mark_finished(status)
    summary = f"Training {status} (return code {returncode})"
    training_state.append_log(summary)
    await event_manager.publish({"type": "log", "line": summary})
    await event_manager.publish({"type": "status", "status": status, "running": False, "returncode": returncode})


def build_command(payload: TrainRequest) -> List[str]:
    args = ["bash", str(RUN_SCRIPT)]
    args.extend(["--title-prefix", payload.title_prefix])
    args.extend(["--author", payload.author])
    args.extend(["--dataset", payload.dataset_path])
    args.extend(["--save-every", str(payload.save_every)])
    args.extend(["--max-epochs", str(payload.max_epochs)])
    if payload.cpu_threads_per_process is not None:
        args.extend(["--cpu-threads-per-process", str(payload.cpu_threads_per_process)])
    if payload.max_data_loader_workers is not None:
        args.extend(["--max-data-loader-workers", str(payload.max_data_loader_workers)])
    args.extend(["--upload-cloud", "Y" if payload.upload_cloud else "N"])
    args.extend(["--shutdown-instance", "Y" if payload.shutdown_instance else "N"])
    args.extend(["--mode", payload.training_mode])
    args.extend(["--noise-mode", payload.noise_mode])
    if payload.cloud_connection_id:
        args.extend(["--cloud-connection-id", payload.cloud_connection_id])
    if payload.auto_confirm:
        args.append("--auto-confirm")
    return args


@app.post("/train")
async def start_training(payload: TrainRequest) -> Dict:
    if not RUN_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="Training script not found")
    if training_state.running:
        raise HTTPException(status_code=409, detail="Training already in progress")

    download_status = get_download_status()
    if download_status.get("pending"):
        active = download_status.get("active") or []
        detail = "Model downloads are still in progress. Please wait for provisioning to finish."
        if active:
            detail = f"{detail} Pending: {', '.join(active)}."
        raise HTTPException(status_code=409, detail=detail)

    for log_path in (HIGH_LOG, LOW_LOG):
        try:
            log_path.unlink()
        except FileNotFoundError:
            pass

    dataset_config_path = Path(payload.dataset_path).expanduser()
    conversion_logs: List[str] = []

    if payload.convert_videos_to_16fps:
        async def conversion_logger(message: str) -> None:
            conversion_logs.append(message)
            await event_manager.publish({"type": "log", "line": message})

        try:
            await convert_videos_to_target_fps(dataset_config_path, 16, conversion_logger)
        except VideoConversionError as exc:
            error_message = f"Video conversion failed: {exc}"
            await event_manager.publish({"type": "log", "line": error_message})
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not is_vast_instance():
        if payload.upload_cloud:
            payload.upload_cloud = False
            note = "Cloud uploads disabled: running in local mode."
            training_state.append_log(note)
            await event_manager.publish({"type": "log", "line": note})
        if payload.shutdown_instance:
            payload.shutdown_instance = False
            note = "Auto-shutdown disabled: running in local mode."
            training_state.append_log(note)
            await event_manager.publish({"type": "log", "line": note})
    else:
        cloud_status = await gather_cloud_status()
        if payload.upload_cloud and not cloud_status.get("can_upload", False):
            payload.upload_cloud = False
            reason = cloud_status.get("message") or "Cloud uploads are not available."
            note = f"Cloud uploads disabled: {reason}"
            training_state.append_log(note)
            await event_manager.publish({"type": "log", "line": note})
        if payload.cloud_connection_id:
            available_connections = {
                connection.get("id")
                for connection in cloud_status.get("connections") or []
                if connection.get("id")
            }
            if available_connections and payload.cloud_connection_id not in available_connections:
                note = (
                    "Selected cloud connection not found in Vast.ai. "
                    "Falling back to the default connection."
                )
                training_state.append_log(note)
                await event_manager.publish({"type": "log", "line": note})
                payload.cloud_connection_id = None

    noise_mode = payload.noise_mode
    if noise_mode == "high":
        active_runs: Set[str] = {"high"}
    elif noise_mode == "low":
        active_runs = {"low"}
    else:
        active_runs = {"high", "low"}

    command = build_command(payload)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    training_state.mark_started(process, active_runs, noise_mode)

    for line in conversion_logs:
        training_state.append_log(line)

    disabled_messages = []
    if "high" not in active_runs:
        disabled_messages.append("High noise training disabled by configuration.")
    if "low" not in active_runs:
        disabled_messages.append("Low noise training disabled by configuration.")
    for message in disabled_messages:
        training_state.append_log(message)
        await event_manager.publish({"type": "log", "line": message})

    await event_manager.publish({"type": "snapshot", **build_snapshot()})
    await event_manager.publish({"type": "status", "status": "running", "running": True})

    stdout_task = asyncio.create_task(stream_process_output(process))
    training_state.add_task(stdout_task)

    if "high" in active_runs:
        high_task = asyncio.create_task(monitor_log_file(HIGH_LOG, "high"))
        training_state.add_task(high_task)
    if "low" in active_runs:
        low_task = asyncio.create_task(monitor_log_file(LOW_LOG, "low"))
        training_state.add_task(low_task)

    wait_task = asyncio.create_task(wait_for_completion(process))
    training_state.add_task(wait_task)

    return {"status": "started"}


@app.post("/stop")
async def stop_training() -> Dict:
    if not training_state.running or training_state.process is None:
        raise HTTPException(status_code=409, detail="No training process to stop")

    process = training_state.process
    training_state.stop_requested = True
    training_state.status = "stopping"
    training_state.append_log("Stop requested by user. Attempting to terminate training process…")
    await event_manager.publish({"type": "log", "line": "Stop requested by user. Attempting to terminate training process…"})
    await event_manager.publish({"type": "status", "status": "stopping", "running": True})

    try:
        if process.pid is not None:
            os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    else:
        try:
            process.terminate()
        except ProcessLookupError:
            pass

    try:
        await asyncio.wait_for(process.wait(), timeout=15)
    except asyncio.TimeoutError:
        warning = "Training process did not exit after SIGTERM. Sending SIGKILL…"
        training_state.append_log(warning)
        await event_manager.publish({"type": "log", "line": warning})
        try:
            if process.pid is not None:
                os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
        else:
            try:
                process.kill()
            except ProcessLookupError:
                pass

    return {"status": "stopping"}


async def monitor_download_status(poll_interval: float = 5.0) -> None:
    previous: Optional[Dict[str, Any]] = None
    while True:
        status = get_download_status()
        if status != previous:
            await event_manager.publish({"type": "downloads", **status})
            previous = status
        await asyncio.sleep(poll_interval)


@app.get("/status")
async def status() -> Dict:
    return build_snapshot()


@app.get("/events")
async def events() -> StreamingResponse:
    queue = await event_manager.register()
    snapshot = build_snapshot()

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'snapshot', **snapshot})}\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await event_manager.unregister(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.on_event("startup")
async def startup_event() -> None:
    global download_watchdog_task
    download_watchdog_task = asyncio.create_task(monitor_download_status())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await training_state.wait_for_tasks()
    if download_watchdog_task is not None:
        download_watchdog_task.cancel()
        with suppress(asyncio.CancelledError):
            await download_watchdog_task
