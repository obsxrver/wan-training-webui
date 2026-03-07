import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DOWNLOAD_STATUS_DIR


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
