import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import signal

from .config import (
    EARLY_STOP_MARKER_DIR,
    EPOCH_PATTERNS,
    LOSS_PATTERNS,
    RUN_SCRIPT,
    STEP_PATTERNS,
    TIME_PATTERN,
    TOTAL_STEP_PATTERNS,
)
from .models import TrainRequest
from .state import event_manager, training_state

RUN_PID_PATTERN = re.compile(r"^(HIGH|LOW|COMBINED)\s+PID:\s*(\d+)", re.IGNORECASE)


def _collect_descendant_pids(pid: int) -> List[int]:
    children_by_parent: Dict[int, List[int]] = {}
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return []

    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        stat_path = entry / "stat"
        try:
            stat = stat_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        try:
            after_name = stat.rsplit(")", 1)[1].strip()
            parent_pid = int(after_name.split()[1])
            child_pid = int(entry.name)
        except (IndexError, ValueError):
            continue
        children_by_parent.setdefault(parent_pid, []).append(child_pid)

    descendants: List[int] = []
    stack = list(children_by_parent.get(pid, []))
    while stack:
        child_pid = stack.pop()
        descendants.append(child_pid)
        stack.extend(children_by_parent.get(child_pid, []))
    return descendants


def _terminate_pid_tree(pid: int) -> None:
    pids = _collect_descendant_pids(pid)
    pids.append(pid)
    for target_pid in pids:
        try:
            os.kill(target_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _write_early_stop_marker(pid: int, run: str, early_stop_epoch: int, reached_epoch: Any) -> None:
    try:
        EARLY_STOP_MARKER_DIR.mkdir(parents=True, exist_ok=True)
        marker = EARLY_STOP_MARKER_DIR / str(pid)
        marker.write_text(
            f"run={run}\nearly_stop_epoch={early_stop_epoch}\nreached_epoch={reached_epoch}\n",
            encoding="utf-8",
        )
    except OSError:
        pass


async def _request_early_stop(run: str, current: Dict[str, Any]) -> None:
    if training_state.has_early_stopped(run):
        return

    early_stop_epoch = training_state.get_early_stop_epoch(run)
    epoch = current.get("epoch")
    if early_stop_epoch is None or epoch is None or int(epoch) < early_stop_epoch + 1:
        return

    pid = training_state.get_run_pid(run)
    if pid is None:
        return

    training_state.mark_early_stopped(run)
    label = "Combined" if training_state.noise_mode == "combined" and run == "high" else run.capitalize()
    message = (
        f"{label} training reached epoch {epoch}; early stop epoch {early_stop_epoch} is complete. "
        "Stopping that training process..."
    )
    training_state.append_log(message)
    await event_manager.publish({"type": "log", "line": message})
    await asyncio.to_thread(_write_early_stop_marker, pid, run, early_stop_epoch, epoch)
    await asyncio.sleep(10)
    await asyncio.to_thread(_terminate_pid_tree, pid)


async def stream_process_output(process: asyncio.subprocess.Process) -> None:
    if process.stdout is None:
        return
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="ignore").rstrip()
        if decoded:
            pid_match = RUN_PID_PATTERN.search(decoded)
            if pid_match:
                pid_label = pid_match.group(1).lower()
                run = "high" if pid_label == "combined" else pid_label
                training_state.set_run_pid(run, int(pid_match.group(2)))
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
                        if current:
                            await _request_early_stop(run, current)
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
                    if current:
                        await _request_early_stop(run, current)
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
