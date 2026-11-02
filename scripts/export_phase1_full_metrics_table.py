"""Export the phase-1 LOSO results as a full metrics markdown/csv table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_metrics(
    output_dir: Path,
    key: str,
    fallback_summary: str | None = None,
    prefer_fallback: bool = False,
) -> dict[str, tuple[float, float]]:
    """Load mean/std metrics from aggregate results or a single-run fallback."""

    aggregate_path = output_dir / key / "aggregate_summary.json"
    metric_names = ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]
    if aggregate_path.exists() and not prefer_fallback:
        payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload)
        return {name: (float(metrics[name]["mean"]), float(metrics[name]["std"])) for name in metric_names}

    if fallback_summary is None:
        raise FileNotFoundError(f"Missing aggregate summary and no fallback provided: {aggregate_path}")

    summary_path = output_dir / fallback_summary / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return {name: (float(metrics[name]), 0.0) for name in metric_names}


def fmt(value: tuple[float, float]) -> str:
    """Format one metric as percentage mean ± std."""

    mean, std = value
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def main() -> None:
    """Write markdown and csv summary tables."""

    output_dir = Path("outputs")
    rows = [
        ("ADSZ", "SVM", load_metrics(output_dir, "mc_loso_adsz_svm")),
        ("ADSZ", "EEGNet", load_metrics(output_dir, "mc_loso_adsz_eegnet")),
        ("ADSZ", "ShallowConvNet", load_metrics(output_dir, "mc_loso_adsz_shallowconvnet")),
        ("ADSZ", "MTDNet", load_metrics(output_dir, "mc_loso_adsz_mtdnet")),
        ("ADSZ", "iTransformer", load_metrics(output_dir, "mc_loso_adsz_itransformer")),
        ("ADSZ", "TCN", load_metrics(output_dir, "mc_loso_adsz_tcn")),
        ("ADSZ", "TAGE-Net", load_metrics(output_dir, "mc_loso_adsz_dualstream", "loso_adsz_lite_auc", prefer_fallback=True)),
        ("ADSZ", "Spectrogram-CNN", load_metrics(output_dir, "mc_loso_adsz_spectrogram_cnn")),
        ("ADFSU", "SVM", load_metrics(output_dir, "mc_loso_external_svm")),
        ("ADFSU", "EEGNet", load_metrics(output_dir, "mc_loso_external_eegnet")),
        ("ADFSU", "ShallowConvNet", load_metrics(output_dir, "mc_loso_adfsu_shallowconvnet")),
        ("ADFSU", "MTDNet", load_metrics(output_dir, "mc_loso_adfsu_mtdnet")),
        ("ADFSU", "iTransformer", load_metrics(output_dir, "mc_loso_adfsu_itransformer")),
        ("ADFSU", "TCN", load_metrics(output_dir, "mc_loso_adfsu_tcn")),
        ("ADFSU", "TAGE-Net", load_metrics(output_dir, "mc_loso_external_dualstream")),
        ("ADFSU", "Spectrogram-CNN", load_metrics(output_dir, "mc_loso_external_spectrogram_cnn")),
        ("ds004504", "SVM", load_metrics(output_dir, "mc_loso_ds005408_svm", "baseline_svm_loso_ds005408")),
        ("ds004504", "EEGNet", load_metrics(output_dir, "mc_loso_ds005408_eegnet", "baseline_eegnet_loso_ds005408")),
        ("ds004504", "ShallowConvNet", load_metrics(output_dir, "mc_loso_ds004504_shallowconvnet")),
        ("ds004504", "MTDNet", load_metrics(output_dir, "mc_loso_ds004504_mtdnet")),
        ("ds004504", "iTransformer", load_metrics(output_dir, "mc_loso_ds004504_itransformer")),
        ("ds004504", "TCN", load_metrics(output_dir, "mc_loso_ds004504_tcn")),
        ("ds004504", "TAGE-Net", load_metrics(output_dir, "mc_loso_ds005408_tage_np_dim16_aug_cw", "mc_loso_ds005408_tage_np_dim16_aug_cw/seed_2026")),
        ("ds004504", "Spectrogram-CNN", load_metrics(output_dir, "mc_loso_ds005408_spectrogram_cnn")),
    ]

    md_lines = [
        "| Dataset | Method | Acc | Bal Acc | F1 | AUROC | Sens | Spec |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    csv_rows = [["Dataset", "Method", "Acc", "Bal Acc", "F1", "AUROC", "Sens", "Spec"]]

    for dataset, method, metrics in rows:
        md_lines.append(
            f"| {dataset} | {method} | {fmt(metrics['accuracy'])} | {fmt(metrics['balanced_accuracy'])} | "
            f"{fmt(metrics['f1'])} | {fmt(metrics['auc'])} | {fmt(metrics['sensitivity'])} | {fmt(metrics['specificity'])} |"
        )
        csv_rows.append(
            [
                dataset,
                method,
                fmt(metrics["accuracy"]),
                fmt(metrics["balanced_accuracy"]),
                fmt(metrics["f1"]),
                fmt(metrics["auc"]),
                fmt(metrics["sensitivity"]),
                fmt(metrics["specificity"]),
            ]
        )

    md_path = output_dir / "phase1_full_metrics_table.md"
    csv_path = output_dir / "phase1_full_metrics_table.csv"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerows(csv_rows)

    print(f"Saved markdown table to: {md_path}")
    print(f"Saved csv table to: {csv_path}")


if __name__ == "__main__":
    main()
