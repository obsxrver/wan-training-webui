#!/usr/bin/env python3
"""Add a provisioning-time GPU preset to the Wan training config."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PRESET_BEGIN = "# BEGIN AUTO-GENERATED HARDWARE PRESET"
PRESET_END = "# END AUTO-GENERATED HARDWARE PRESET"
BLOCK_SWAP_VRAM_MB = 33 * 1024
FP8_BASE_VRAM_MB = 60 * 1024


def detect_gpu_vram_mb() -> int:
    """Return the smallest total VRAM value reported by nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("could not query GPU VRAM with nvidia-smi") from exc

    values: list[int] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            values.append(int(value))
        except ValueError as exc:
            raise RuntimeError(f"unexpected nvidia-smi VRAM value: {value!r}") from exc

    if not values or any(value <= 0 for value in values):
        raise RuntimeError("nvidia-smi did not report valid GPU VRAM values")
    return min(values)


def build_hardware_preset(vram_mb: int) -> tuple[str, list[str]]:
    settings = ["sdpa = true"]
    enabled = ["sdpa"]

    if vram_mb < BLOCK_SWAP_VRAM_MB:
        settings.append("blocks_to_swap = 1")
        enabled.append("blocks_to_swap=1")
    if vram_mb < FP8_BASE_VRAM_MB:
        settings.append("fp8_base = true")
        enabled.append("fp8_base")

    lines = [
        PRESET_BEGIN,
        "# This section is replaced whenever provisioning detects the GPU hardware.",
        f"# Minimum detected GPU VRAM: {vram_mb} MiB.",
        "[provisioned_hardware]",
        *settings,
        PRESET_END,
    ]
    return "\n".join(lines), enabled


def replace_hardware_preset(config_text: str, preset: str) -> str:
    begin = config_text.find(PRESET_BEGIN)
    end = config_text.find(PRESET_END)

    if (begin == -1) != (end == -1):
        raise RuntimeError("training config contains an incomplete auto-generated hardware preset")
    if begin != -1:
        if end < begin:
            raise RuntimeError("training config hardware preset markers are out of order")
        end += len(PRESET_END)
        before = config_text[:begin].rstrip()
        after = config_text[end:].strip()
        config_text = f"{before}\n\n{after}" if after else before

    return f"{config_text.rstrip()}\n\n{preset}\n"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Wan training TOML to update")
    parser.add_argument(
        "--vram-mb",
        type=positive_int,
        help="Use this VRAM value instead of querying nvidia-smi (primarily for testing)",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        parser.error(f"training config not found: {args.config}")

    try:
        vram_mb = args.vram_mb if args.vram_mb is not None else detect_gpu_vram_mb()
        preset, enabled = build_hardware_preset(vram_mb)
        updated = replace_hardware_preset(
            args.config.read_text(encoding="utf-8"),
            preset,
        )
        args.config.write_text(updated, encoding="utf-8", newline="\n")
    except (OSError, RuntimeError) as exc:
        parser.error(str(exc))

    print(
        f"Configured {args.config} for {vram_mb} MiB minimum GPU VRAM: "
        + ", ".join(enabled)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
