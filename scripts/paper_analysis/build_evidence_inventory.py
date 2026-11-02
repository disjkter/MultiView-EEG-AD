"""Inventory datasets, LODO summaries, checkpoints, and reusable predictions."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper_analysis.common import parse_named_path, write_csv, write_json


def file_digest(path: Path) -> str:
    """Return a lightweight digest based on a file's name, size, and mtime."""

    stat = path.stat()
    payload = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def inventory_result(name: str, root: Path) -> dict[str, object]:
    """Summarize reusable artifacts below one result directory."""

    summaries = sorted(root.rglob("lodo_summary.json")) if root.exists() else []
    checkpoints = sorted(root.rglob("best_model.pth")) if root.exists() else []
    calibration = sorted(root.rglob("*calibration*.csv")) if root.exists() else []
    seeds: set[int] = set()
    variants: set[str] = set()
    targets: set[str] = set()
    import json

    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seeds.add(int(payload["seed"]))
        variants.add(path.parent.name)
        targets.update(str(fold["target"]) for fold in payload.get("folds", []))
    return {
        "name": name,
        "path": str(root),
        "exists": root.exists(),
        "lodo_summaries": len(summaries),
        "checkpoints": len(checkpoints),
        "calibration_csv_files": len(calibration),
        "seeds": ";".join(map(str, sorted(seeds))),
        "variants": ";".join(sorted(variants)),
        "targets": ";".join(sorted(targets)),
    }


def inventory_dataset(name: str, root: Path) -> dict[str, object]:
    """Summarize files proving one processed dataset's identity."""

    files = list(root.glob("*.npy")) if root.exists() else []
    summaries = list(root.glob("*summary*.json")) if root.exists() else []
    return {
        "name": name,
        "path": str(root),
        "exists": root.exists(),
        "npy_files": len(files),
        "summary_files": len(summaries),
        "fingerprint": ";".join(file_digest(path) for path in sorted(files + summaries)),
    }


def main() -> None:
    """Write a machine-readable evidence inventory without loading EEG arrays."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    result_rows = [inventory_result(*parse_named_path(value)) for value in args.result]
    dataset_rows = [inventory_dataset(*parse_named_path(value)) for value in args.dataset]
    prefix = args.output_prefix
    if result_rows:
        write_csv(prefix.with_name(prefix.name + "_results.csv"), result_rows)
    if dataset_rows:
        write_csv(prefix.with_name(prefix.name + "_datasets.csv"), dataset_rows)
    write_json(
        prefix.with_suffix(".json"),
        {"results": result_rows, "datasets": dataset_rows, "content_hash_is_lightweight": True},
    )
    print(f"Saved evidence inventory to {prefix.parent}")


if __name__ == "__main__":
    main()
