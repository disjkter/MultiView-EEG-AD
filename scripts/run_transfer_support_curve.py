"""Run a multi-seed few-shot support-size transfer learning curve."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


def replace_seed(value: object, seed: int) -> object:
    """Replace the canonical seed token recursively in a YAML payload."""

    if isinstance(value, str):
        return value.replace("seed2026", f"seed{seed}")
    if isinstance(value, list):
        return [replace_seed(item, seed) for item in value]
    if isinstance(value, dict):
        return {key: replace_seed(item, seed) for key, item in value.items()}
    return value


def select_named(items: object, name: str, field: str) -> list[dict[str, object]]:
    """Select one named YAML entry and validate its existence."""

    if not isinstance(items, list):
        raise ValueError(f"{field} must be a list")
    selected = [item for item in items if str(item.get("name")) == name]
    if len(selected) != 1:
        raise ValueError(f"Expected one {field} entry named {name}, found {len(selected)}")
    return selected


def run_checked(command: list[str]) -> None:
    """Run a subprocess and fail with a readable command."""

    print("\n" + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    """Generate frozen configs and run audit plus controlled validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config", default="configs/transferability_audit_local.yaml"
    )
    parser.add_argument("--source", default="ADFSU")
    parser.add_argument("--target", default="CAUEEG")
    parser.add_argument("--supports", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(2026, 2031))
    )
    parser.add_argument(
        "--output-root", default="outputs/transfer_support_curve_adfsu_caueeg"
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    base["sources"] = select_named(base.get("sources"), args.source, "sources")
    base["targets"] = select_named(base.get("targets"), args.target, "targets")
    output_root = Path(args.output_root)
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)

    for support in args.supports:
        if support < 1:
            raise ValueError("Support size must be positive")
        for seed in args.seeds:
            config = replace_seed(copy.deepcopy(base), seed)
            config["seed"] = seed
            config["support_subjects_per_class"] = support
            run_name = f"support{support}_seed{seed}"
            config["output_prefix"] = str(output_root / "audits" / run_name)
            config["validation_output_dir"] = str(output_root / run_name)
            config_path = config_root / f"{run_name}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            summary_path = Path(str(config["validation_output_dir"])) / "summary.json"
            if summary_path.exists() and not args.force:
                print(f"SKIP {run_name}: {summary_path} exists")
                continue
            run_checked(
                [
                    args.python_exe,
                    "scripts/audit_graph_transferability.py",
                    "--config",
                    str(config_path),
                ]
            )
            command = [
                args.python_exe,
                "scripts/run_transferability_validation.py",
                "--config",
                str(config_path),
                "--python-exe",
                args.python_exe,
                "--target",
                args.target,
            ]
            if args.force:
                command.append("--force")
            run_checked(command)


if __name__ == "__main__":
    main()
