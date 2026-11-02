"""Check existing ablation results."""
import json
from pathlib import Path

root = Path("outputs/ablation")
existing = []
for p in sorted(root.rglob("summary.json")):
    try:
        with open(p) as f:
            d = json.load(f)
        m = d["metrics"]
        num_subs = d.get("num_subjects", d.get("num_subjects_evaluated", "?"))
        n_folds = len(d.get("folds", []))
        # derive variant and dataset from path
        parts = p.relative_to(root).parts
        ds = parts[0]
        tag = parts[1]
        existing.append({
            "path": str(p),
            "ds": ds.split("_")[0],
            "tag": tag,
            "n_sub": num_subs,
            "n_folds": n_folds,
            "balacc": m["balanced_accuracy"],
            "f1": m["f1"],
            "sens": m["sensitivity"],
            "spec": m["specificity"],
        })
    except Exception as e:
        print(f"  CORRUPT: {p} -> {e}")

# Print table
print(f"{'Dataset':>8s} {'Tag':>28s} {'Subs':>5s} {'Folds':>5s} {'BalAcc':>7s} {'F1':>7s} {'Sens':>7s} {'Spec':>7s}")
print("-" * 80)
for e in existing:
    print(f"{e['ds']:>8s} {e['tag']:>28s} {str(e['n_sub']):>5s} {e['n_folds']:>5d} {e['balacc']:7.4f} {e['f1']:7.4f} {e['sens']:7.4f} {e['spec']:7.4f}")

print(f"\nTotal: {len(existing)} completed experiments")
