"""Export phase-2 single-source cross-domain results as markdown/csv tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path


METRIC_NAMES = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]


def load_metrics(output_dir: Path, run_name: str) -> dict[str, tuple[float, float]]:
    """Load one aggregate cross-domain result."""

    payload = json.loads((output_dir / run_name / "aggregate_summary.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    return {name: (float(metrics[name]["mean"]), float(metrics[name]["std"])) for name in METRIC_NAMES}


def fmt(value: tuple[float, float]) -> str:
    """Format one metric as percentage mean +/- std."""

    mean, std = value
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def main() -> None:
    """Write markdown and csv tables for phase 2."""

    output_dir = Path("outputs")
    rows = [
        ("ADSZ -> ADFSU", "SVM", load_metrics(output_dir, "mc_cd_adsz_to_adfsu_svm_ea_thr")),
        ("ADSZ -> ADFSU", "EEGNet", load_metrics(output_dir, "mc_cd_adsz_to_adfsu_eegnet_ea_thr")),
        ("ADSZ -> ADFSU", "ShallowConvNet", load_metrics(output_dir, "mc_cd_adsz_to_adfsu_shallow_ea_thr")),
        ("ADSZ -> ADFSU", "TCN", load_metrics(output_dir, "mc_cd_adsz_to_adfsu_tcn_ea_thr")),
        ("ADSZ -> ADFSU", "TAGE-Net", load_metrics(output_dir, "mc_cd_adsz_to_adfsu_tage_ea_thr")),
        ("ADFSU -> ADSZ", "SVM", load_metrics(output_dir, "mc_cd_adfsu_to_adsz_svm_ea_thr")),
        ("ADFSU -> ADSZ", "EEGNet", load_metrics(output_dir, "mc_cd_adfsu_to_adsz_eegnet_ea_thr")),
        ("ADFSU -> ADSZ", "ShallowConvNet", load_metrics(output_dir, "mc_cd_adfsu_to_adsz_shallow_ea_thr")),
        ("ADFSU -> ADSZ", "TCN", load_metrics(output_dir, "mc_cd_adfsu_to_adsz_tcn_ea_thr")),
        ("ADFSU -> ADSZ", "TAGE-Net", load_metrics(output_dir, "mc_cd_adfsu_to_adsz_tage_ea_thr")),
        ("ADSZ -> ds004504", "SVM", load_metrics(output_dir, "mc_cd_adsz_to_ds004504_svm_ea_thr")),
        ("ADSZ -> ds004504", "EEGNet", load_metrics(output_dir, "mc_cd_adsz_to_ds004504_eegnet_ea_thr")),
        ("ADSZ -> ds004504", "ShallowConvNet", load_metrics(output_dir, "mc_cd_adsz_to_ds004504_shallow_ea_thr")),
        ("ADSZ -> ds004504", "TCN", load_metrics(output_dir, "mc_cd_adsz_to_ds004504_tcn_ea_thr")),
        ("ADSZ -> ds004504", "TAGE-Net", load_metrics(output_dir, "mc_cd_adsz_to_ds004504_tage_ea_thr")),
        ("ADFSU -> ds004504", "SVM", load_metrics(output_dir, "mc_cd_adfsu_to_ds004504_svm_ea_thr")),
        ("ADFSU -> ds004504", "EEGNet", load_metrics(output_dir, "mc_cd_adfsu_to_ds004504_eegnet_ea_thr")),
        ("ADFSU -> ds004504", "ShallowConvNet", load_metrics(output_dir, "mc_cd_adfsu_to_ds004504_shallow_ea_thr")),
        ("ADFSU -> ds004504", "TCN", load_metrics(output_dir, "mc_cd_adfsu_to_ds004504_tcn_ea_thr")),
        ("ADFSU -> ds004504", "TAGE-Net", load_metrics(output_dir, "mc_cd_adfsu_to_ds004504_tage_ea_thr")),
        ("ds004504 -> ADSZ", "SVM", load_metrics(output_dir, "mc_cd_ds004504_to_adsz_svm_ea_thr")),
        ("ds004504 -> ADSZ", "EEGNet", load_metrics(output_dir, "mc_cd_ds004504_to_adsz_eegnet_ea_thr")),
        ("ds004504 -> ADSZ", "ShallowConvNet", load_metrics(output_dir, "mc_cd_ds004504_to_adsz_shallow_ea_thr")),
        ("ds004504 -> ADSZ", "TCN", load_metrics(output_dir, "mc_cd_ds004504_to_adsz_tcn_ea_thr")),
        ("ds004504 -> ADSZ", "TAGE-Net", load_metrics(output_dir, "mc_cd_ds004504_to_adsz_tage_ea_thr")),
        ("ds004504 -> ADFSU", "SVM", load_metrics(output_dir, "mc_cd_ds004504_to_adfsu_svm_ea_thr")),
        ("ds004504 -> ADFSU", "EEGNet", load_metrics(output_dir, "mc_cd_ds004504_to_adfsu_eegnet_ea_thr")),
        ("ds004504 -> ADFSU", "ShallowConvNet", load_metrics(output_dir, "mc_cd_ds004504_to_adfsu_shallow_ea_thr")),
        ("ds004504 -> ADFSU", "TCN", load_metrics(output_dir, "mc_cd_ds004504_to_adfsu_tcn_ea_thr")),
        ("ds004504 -> ADFSU", "TAGE-Net", load_metrics(output_dir, "mc_cd_ds004504_to_adfsu_tage_ea_thr")),
    ]

    md_lines = [
        "| Transfer | Method | Acc | Bal Acc | F1 | AUROC | Sens | Spec |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = [["Transfer", "Method", "Acc", "Bal Acc", "F1", "AUROC", "Sens", "Spec"]]

    for transfer, method, metrics in rows:
        md_lines.append(
            f"| {transfer} | {method} | {fmt(metrics['accuracy'])} | {fmt(metrics['balanced_accuracy'])} | "
            f"{fmt(metrics['f1'])} | {fmt(metrics['auc'])} | {fmt(metrics['sensitivity'])} | {fmt(metrics['specificity'])} |"
        )
        csv_rows.append(
            [
                transfer,
                method,
                fmt(metrics["accuracy"]),
                fmt(metrics["balanced_accuracy"]),
                fmt(metrics["f1"]),
                fmt(metrics["auc"]),
                fmt(metrics["sensitivity"]),
                fmt(metrics["specificity"]),
            ]
        )

    md_path = output_dir / "phase2_cross_domain_table.md"
    csv_path = output_dir / "phase2_cross_domain_table.csv"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        csv.writer(file_obj).writerows(csv_rows)

    print(f"Saved markdown table to: {md_path}")
    print(f"Saved csv table to: {csv_path}")


if __name__ == "__main__":
    main()
