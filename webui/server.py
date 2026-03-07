import asyncio
import json
import mimetypes
import os
import shutil
import signal
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from .auth import AUTH_TOKEN, TokenAuthMiddleware, build_jupyter_base_url, is_vast_instance
from .cloud import gather_cloud_status, maybe_set_container_api_key
from .config import (
    DATASET_CONFIG_ROOT,
    DATASET_ROOT,
    HIGH_LOG,
    INDEX_HTML_PATH,
    LOW_LOG,
    RANGE_HEADER_RE,
    REPO_ROOT,
    RUN_SCRIPT,
    TOKEN_ENV_VAR,
)
from .dataset import (
    bulk_caption,
    clear_dataset_directory,
    collect_dataset_items,
    delete_dataset_item,
    get_caption,
    iter_file_chunks,
    resolve_dataset_file,
    update_caption,
    write_upload_to_path,
)
from .downloads import get_download_status
from .models import ApiKeyRequest, BulkCaptionRequest, DeleteDatasetItem, TrainRequest, UpdateCaptionRequest
from .state import event_manager, training_state
from .training_runtime import build_command, monitor_log_file, stream_process_output, wait_for_completion
from .video_conversion import DEFAULT_DATASET_CONFIG, VideoConversionError, convert_videos_to_target_fps

maybe_set_container_api_key()

download_watchdog_task: Optional[asyncio.Task] = None
app = FastAPI(title="WAN 2.2 Training UI")
app.add_middleware(TokenAuthMiddleware, token=AUTH_TOKEN)


def build_snapshot() -> Dict[str, Any]:
    snapshot = training_state.snapshot()
    snapshot["downloads"] = get_download_status()
    return snapshot


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="UI assets missing")
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


@app.get("/jupyter-info")
async def jupyter_info() -> Dict[str, Optional[str]]:
    base_url = build_jupyter_base_url()
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


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    clear_dataset_directory()
    saved: List[str] = []
    for file in files:
        filename = Path(file.filename or "").name
        if not filename:
            continue
        destination = DATASET_ROOT / filename
        await write_upload_to_path(file, destination)
        await file.close()
        saved.append(str(destination))
    return {"saved": saved, "count": len(saved)}


@app.get("/dataset/files")
async def dataset_files() -> Dict[str, Any]:
    items = collect_dataset_items()
    return {"items": items, "total": len(items)}


@app.get("/dataset/caption")
async def dataset_get_caption(caption_path: Optional[str] = None, media_path: Optional[str] = None) -> Dict[str, Any]:
    return get_caption(caption_path, media_path)


@app.post("/dataset/delete")
async def dataset_delete(payload: DeleteDatasetItem) -> Dict[str, Any]:
    return delete_dataset_item(payload.media_path)


@app.post("/dataset/caption")
async def dataset_update_caption(payload: UpdateCaptionRequest) -> Dict[str, Any]:
    return update_caption(payload.media_path, payload.caption_text, payload.caption_path)


@app.post("/dataset/caption/bulk")
async def dataset_bulk_caption(payload: BulkCaptionRequest) -> Dict[str, Any]:
    return bulk_caption(payload.caption_text, payload.apply_to)


@app.get("/dataset/media/{path:path}")
async def dataset_media(path: str, request: Request) -> StreamingResponse:
    file_path = resolve_dataset_file(path)
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
            iter_file_chunks(file_path, start=start, end=end),
            status_code=206,
            media_type=media_type,
            headers=headers,
        )

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
    }
    return StreamingResponse(iter_file_chunks(file_path), media_type=media_type, headers=headers)


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


@app.post("/train")
async def start_training(payload: TrainRequest) -> Dict[str, str]:
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
    elif noise_mode == "combined":
        # Combined mode writes a single stream to run_high.log.
        active_runs = {"high"}
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
    if noise_mode == "combined":
        disabled_messages.append("Combined noise mode active: live metrics are shown in the High noise panel.")
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
async def stop_training() -> Dict[str, str]:
    if not training_state.running or training_state.process is None:
        raise HTTPException(status_code=409, detail="No training process to stop")

    process = training_state.process
    training_state.stop_requested = True
    training_state.status = "stopping"
    stop_message = "Stop requested by user. Attempting to terminate training process..."
    training_state.append_log(stop_message)
    await event_manager.publish({"type": "log", "line": stop_message})
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
        warning = "Training process did not exit after SIGTERM. Sending SIGKILL..."
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
async def status() -> Dict[str, Any]:
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
