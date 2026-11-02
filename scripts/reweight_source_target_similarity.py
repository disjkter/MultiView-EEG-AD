"""Recompute source sampling weights from an existing similarity JSON without rescanning EEG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.analyze_source_target_similarity import exponential_similarity, write_outputs


def main() -> None:
    """Apply scale-normalized exponential distance weighting per target."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    rows = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    targets = list(dict.fromkeys(str(row["target"]) for row in rows))
    for target in targets:
        target_rows = [row for row in rows if str(row["target"]) == target]
        affinities = exponential_similarity(
            [float(row["composite_distance"]) for row in target_rows],
            temperature=args.temperature,
        )
        total = sum(affinities)
        for row, affinity in zip(target_rows, affinities):
            row["similarity_score"] = float(affinity)
            row["sampling_weight"] = float(affinity / total)
            row["weighting_method"] = "scale_normalized_exponential"
            row["similarity_temperature"] = float(args.temperature)

    write_outputs(rows, Path(args.output_prefix))


if __name__ == "__main__":
    main()
