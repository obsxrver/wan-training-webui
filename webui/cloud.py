import asyncio
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List

from .auth import is_vast_instance
from .config import API_KEY_CONFIG_PATH, CLOUD_SETTINGS_URL, MANAGE_KEYS_URL


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
                connections.append({"id": str(connection_id), "name": name, "cloud_type": cloud_type})
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
