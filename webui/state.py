import asyncio
from collections import deque
from typing import Any, Dict, List, Optional, Set

from .config import MAX_HISTORY_POINTS, MAX_LOG_LINES


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

    def mark_started(self, process: asyncio.subprocess.Process, active_runs: Set[str], noise_mode: str) -> None:
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
        _, pending = await asyncio.wait(self.tasks, timeout=0)
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
