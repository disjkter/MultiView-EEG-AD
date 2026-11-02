"""Find best fine-tuned CAUEEG checkpoints for t-SNE."""
import json
from pathlib import Path

root = Path(r"E:\ProgramFiles\PythonProject\BioFoundation-main\outputs")
targets = [
    "caueeg_tage_timefreq_consistency_bl19_40hz_seed2026",
    "caueeg_tage_timefreq_consistency_adssl_seed2026",
    "caueeg_biofoundation_v1_seed2026",
]

for d in targets:
    p = root / d / "summary.json"
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        m = data.get("metrics", {})
        print(f"{d}")
        for k in ["accuracy", "balanced_accuracy", "f1", "auc", "sensitivity", "specificity"]:
            print(f"  {k}: {m.get(k, 'N/A')}")
    else:
        print(f"{d}: NO summary.json")
