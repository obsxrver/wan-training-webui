from pathlib import Path

import pytest
from fastapi import HTTPException

from webui.models import TrainRequest
from webui.server import normalize_optimizer_state_paths
from webui.training_runtime import build_command


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_build_command_includes_optimizer_state_options() -> None:
    payload = TrainRequest(
        save_optimizer_state=True,
        resume_high_optimizer_state="/states/high",
        resume_low_optimizer_state="/states/low",
    )

    command = build_command(payload)

    assert _option_value(command, "--save-optimizer-state") == "Y"
    assert _option_value(command, "--resume-high-optimizer-state") == "/states/high"
    assert _option_value(command, "--resume-low-optimizer-state") == "/states/low"


def test_build_command_omits_empty_optimizer_state_options() -> None:
    payload = TrainRequest(
        resume_high_optimizer_state=" ",
        resume_low_optimizer_state=None,
    )

    command = build_command(payload)

    assert "--save-optimizer-state" not in command
    assert "--resume-high-optimizer-state" not in command
    assert "--resume-low-optimizer-state" not in command


def test_normalize_optimizer_state_paths_resolves_active_runs(tmp_path: Path) -> None:
    high_state = tmp_path / "high-state"
    low_state = tmp_path / "low-state"
    high_state.mkdir()
    low_state.mkdir()
    payload = TrainRequest(
        noise_mode="both",
        resume_high_optimizer_state=str(high_state),
        resume_low_optimizer_state=str(low_state),
    )

    normalize_optimizer_state_paths(payload)

    assert payload.resume_high_optimizer_state == str(high_state.resolve())
    assert payload.resume_low_optimizer_state == str(low_state.resolve())


def test_normalize_optimizer_state_paths_ignores_inactive_run() -> None:
    payload = TrainRequest(
        noise_mode="high",
        resume_low_optimizer_state="/missing/low-state",
    )

    normalize_optimizer_state_paths(payload)

    assert payload.resume_low_optimizer_state is None


def test_normalize_optimizer_state_paths_rejects_missing_active_state() -> None:
    payload = TrainRequest(
        noise_mode="combined",
        resume_high_optimizer_state="/missing/high-state",
    )

    with pytest.raises(HTTPException) as exc_info:
        normalize_optimizer_state_paths(payload)

    assert exc_info.value.status_code == 400
    assert "directory not found" in str(exc_info.value.detail)
