"""Run frozen CARE-DG experiment stages from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def option_name(key: str) -> str:
    """Convert a YAML snake-case key to a command-line option."""

    return "--" + key.replace("_", "-")


def append_options(command: list[str], values: dict[str, object]) -> None:
    """Append scalar and boolean command-line options."""

    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                command.append(option_name(key))
        elif value is not None:
            command.extend([option_name(key), str(value)])


def build_command(
    python_exe: str,
    config: dict[str, object],
    stage: dict[str, object],
    backbone: str,
    variant_name: str,
    seed: int,
    device: str,
) -> tuple[list[str], Path]:
    """Build one strict source-only LODO command from frozen configuration."""

    domains = config["domains"]
    output_dir = (
        Path(str(stage["output_root"]))
        / backbone
        / f"seed{seed}"
        / variant_name
    )
    command = [
        python_exe,
        "scripts/run_graph_temporal_lodo_tacd.py",
        "--domains",
    ]
    for name in stage["domains"]:
        command.append(f"{name}={domains[name]}")
    for target in stage["targets"]:
        command.extend(["--target", str(target)])
    variant_config = dict(config["variants"][variant_name])
    method = str(variant_config.pop("method", "care_dg"))
    command.extend(
        [
            "--method",
            method,
            "--backbone",
            backbone,
            "--output-dir",
            str(output_dir),
            "--seed",
            str(seed),
            "--device",
            device,
        ]
    )
    append_options(command, config["common"])
    append_options(command, config["backbones"][backbone])
    append_options(command, variant_config)
    return command, output_dir


def main() -> None:
    """Run a configured stage and preserve its exact configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["screening", "development", "ablation", "confirmation"],
        required=True,
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage = config["stages"][args.stage]
    output_root = Path(str(stage["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_root / "frozen_config.yaml")

    manifest: list[dict[str, object]] = []
    for backbone in stage["backbones"]:
        for seed in stage["seeds"]:
            for variant in stage["variants"]:
                command, output_dir = build_command(
                    args.python_exe,
                    config,
                    stage,
                    str(backbone),
                    str(variant),
                    int(seed),
                    args.device,
                )
                summary_path = output_dir / "lodo_summary.json"
                manifest.append(
                    {
                        "backbone": backbone,
                        "seed": seed,
                        "variant": variant,
                        "summary": str(summary_path),
                        "command": command,
                    }
                )
                if summary_path.exists():
                    print(f"SKIP backbone={backbone} seed={seed} variant={variant}")
                    continue
                print("RUN " + subprocess.list2cmdline(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    completed = sum(Path(str(row["summary"])).exists() for row in manifest)
    print(f"Completed {completed}/{len(manifest)} runs for stage={args.stage}")


if __name__ == "__main__":
    main()
