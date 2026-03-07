import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Iterable, List, Optional, Set, Tuple

from .config import DATASET_CONFIG_ROOT, PROTECTED_DATASET_DIRS, VIDEO_EXTENSIONS

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


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
        raise VideoConversionError("No valid video directories were found. Checked: " + ", ".join(missing))

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
        raise VideoConversionError(f"Failed to inspect '{path.name}' with ffprobe: {message or 'Unknown error.'}")

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
            raise VideoConversionError(f"Unable to download dataset config '{config_path}': {exc}") from exc
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
            tempfile.NamedTemporaryFile(delete=False, suffix=video_path.suffix, dir=str(video_path.parent)).name
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
