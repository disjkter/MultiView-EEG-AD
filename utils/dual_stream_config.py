"""Helpers for loading unified dual-stream experiment defaults from YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DUAL_STREAM_CONFIG = PROJECT_ROOT / "configs" / "dual_stream_unified.yaml"


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared config argument to an argument parser."""

    parser.add_argument(
        "--config",
        default=str(DEFAULT_DUAL_STREAM_CONFIG),
        help="Unified YAML config for dual-stream experiments",
    )


def apply_yaml_defaults(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    keys: list[str],
) -> argparse.Namespace:
    """Apply YAML values only for arguments that still equal parser defaults."""

    config_path = Path(args.config)
    if not config_path.exists():
        return args

    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if getattr(action, "dest", None) not in (None, "help")
    }

    for key in keys:
        if key not in config_data or not hasattr(args, key):
            continue
        current_value = getattr(args, key)
        default_value = defaults.get(key)
        if current_value == default_value:
            setattr(args, key, config_data[key])
    return args
