import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import EPOCH_PATTERNS, LOSS_PATTERNS, RUN_SCRIPT, STEP_PATTERNS, TIME_PATTERN, TOTAL_STEP_PATTERNS
from .models import TrainRequest
from .state import event_manager, training_state


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
