"""Aggregate multi-seed transferability audits and validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
    "delta_balanced_accuracy",
    "delta_auc",
)


def load_seed_tables(
    seeds: list[int],
    validation_pattern: str,
    audit_pattern: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load validation and audit CSV files and attach their random seeds."""

    validation_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    for seed in seeds:
        validation_path = Path(validation_pattern.format(seed=seed))
        audit_path = Path(audit_pattern.format(seed=seed))
        if not validation_path.exists():
            raise FileNotFoundError(f"Missing validation summary: {validation_path}")
        if not audit_path.exists():
            raise FileNotFoundError(f"Missing transferability audit: {audit_path}")
        validation_frames.append(pd.read_csv(validation_path).assign(seed=seed))
        audit_frames.append(pd.read_csv(audit_path).assign(seed=seed))
    return pd.concat(validation_frames, ignore_index=True), pd.concat(
        audit_frames, ignore_index=True
    )


def aggregate_validation(validation: pd.DataFrame) -> pd.DataFrame:
    """Compute per-target and per-initialization multi-seed statistics."""

    columns = [metric for metric in METRICS if metric in validation.columns]
    grouped = validation.groupby(["target", "initialization"], sort=True)[columns]
    result = grouped.agg(["mean", "std", "min", "max"]).reset_index()
    result.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in result.columns
    ]
    wins = (
        validation.assign(positive_transfer=validation["delta_balanced_accuracy"] > 0)
        .groupby(["target", "initialization"], sort=True)["positive_transfer"]
        .mean()
        .rename("positive_transfer_rate")
        .reset_index()
    )
    return result.merge(wins, on=["target", "initialization"], how="left")


def evaluate_ranking(
    validation: pd.DataFrame,
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare audit rankings with observed transfer gains."""

    merged = audit.merge(
        validation,
        left_on=["seed", "target", "source"],
        right_on=["seed", "target", "initialization"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Audit and validation tables have no matching source-target rows")
    merged = merged.rename(columns={"rank_score_x": "rank_score"})
    if "rank_score_y" in merged:
        merged = merged.drop(columns=["rank_score_y"])

    correlations: dict[str, object] = {}
    for target, frame in [("all", merged), *list(merged.groupby("target", sort=True))]:
        correlations[str(target)] = {
            "n": int(len(frame)),
            "spearman_rank_vs_delta_balanced_accuracy": float(
                frame["rank_score"].corr(
                    frame["delta_balanced_accuracy"], method="spearman"
                )
            ),
            "spearman_rank_vs_delta_auc": float(
                frame["rank_score"].corr(frame["delta_auc"], method="spearman")
            ),
        }

    selection_rows: list[dict[str, object]] = []
    for (seed, target), frame in merged.groupby(["seed", "target"], sort=True):
        if len(frame) < 2:
            continue
        rank_max = frame["rank_score"].max()
        gain_max = frame["delta_balanced_accuracy"].max()
        selected = frame[np.isclose(frame["rank_score"], rank_max)]
        best = frame[np.isclose(frame["delta_balanced_accuracy"], gain_max)]
        selected_sources = sorted(selected["source"].astype(str))
        best_sources = sorted(best["source"].astype(str))
        selected_gain = float(selected["delta_balanced_accuracy"].mean())
        selection_rows.append(
            {
                "seed": int(seed),
                "target": str(target),
                "selected_sources": "|".join(selected_sources),
                "best_sources": "|".join(best_sources),
                "selection_hit": bool(set(selected_sources).intersection(best_sources)),
                "selected_gain": selected_gain,
                "best_gain": float(gain_max),
                "selection_regret": float(gain_max - selected_gain),
            }
        )
    selection = pd.DataFrame(selection_rows)
    if not selection.empty:
        correlations["selection"] = {
            "hit_rate": float(selection["selection_hit"].mean()),
            "mean_regret": float(selection["selection_regret"].mean()),
            "by_target": {
                str(target): {
                    "hit_rate": float(frame["selection_hit"].mean()),
                    "mean_regret": float(frame["selection_regret"].mean()),
                }
                for target, frame in selection.groupby("target", sort=True)
            },
        }
    return merged, {"correlations": correlations, "selection": selection_rows}


def write_outputs(
    output_prefix: Path,
    validation: pd.DataFrame,
    aggregate: pd.DataFrame,
    ranking: pd.DataFrame,
    ranking_summary: dict[str, object],
    seeds: list[int],
) -> None:
    """Write machine-readable tables and a concise Markdown report."""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_prefix.with_name(output_prefix.name + "_runs.csv"), index=False)
    aggregate.to_csv(
        output_prefix.with_name(output_prefix.name + "_aggregate.csv"), index=False
    )
    ranking.to_csv(
        output_prefix.with_name(output_prefix.name + "_ranking.csv"), index=False
    )
    payload = {
        "seeds": seeds,
        "aggregate": aggregate.replace({np.nan: None}).to_dict(orient="records"),
        **ranking_summary,
    }
    output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Multi-seed Transferability Summary",
        "",
        f"Seeds: {', '.join(map(str, seeds))}",
        "",
        "| Target | Initialization | Balanced accuracy | AUROC | Delta BA | Positive rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            f"| {row['target']} | {row['initialization']} | "
            f"{row['balanced_accuracy_mean']:.4f} +/- {row['balanced_accuracy_std']:.4f} | "
            f"{row['auc_mean']:.4f} +/- {row['auc_std']:.4f} | "
            f"{row['delta_balanced_accuracy_mean']:.4f} +/- "
            f"{row['delta_balanced_accuracy_std']:.4f} | "
            f"{row['positive_transfer_rate']:.2f} |"
        )
    all_corr = ranking_summary["correlations"]["all"]
    lines.extend(
        [
            "",
            "## Ranking validity",
            "",
            f"- Spearman(rank, delta BA): "
            f"{all_corr['spearman_rank_vs_delta_balanced_accuracy']:.4f}",
            f"- Spearman(rank, delta AUROC): "
            f"{all_corr['spearman_rank_vs_delta_auc']:.4f}",
        ]
    )
    selection = ranking_summary["correlations"].get("selection")
    if selection:
        lines.append(f"- Source-selection hit rate: {selection['hit_rate']:.4f}")
        lines.append(f"- Mean selection regret: {selection['mean_regret']:.4f}")
    output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Aggregate transferability results from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2026, 2031)))
    parser.add_argument(
        "--validation-pattern",
        default=(
            "outputs/transferability_validation_local_lossselect_seed{seed}/summary.csv"
        ),
    )
    parser.add_argument(
        "--audit-pattern",
        default="outputs/transferability_audit_local_lossselect/seed{seed}.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/transferability_multiseed_lossselect/summary",
    )
    args = parser.parse_args()

    validation, audit = load_seed_tables(
        args.seeds, args.validation_pattern, args.audit_pattern
    )
    aggregate = aggregate_validation(validation)
    ranking, ranking_summary = evaluate_ranking(validation, audit)
    output_prefix = Path(args.output_prefix)
    write_outputs(
        output_prefix,
        validation,
        aggregate,
        ranking,
        ranking_summary,
        args.seeds,
    )
    print(f"Saved multi-seed summary to {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
