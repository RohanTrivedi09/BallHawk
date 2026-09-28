"""Val-split metrics for every trained run -> results/<run>/val_metrics.json.

Notebook 07 chooses which interventions to stack on these, never on test (plan: choose on val,
report on test). Same per-class format as metrics.json.
"""

import json
import sys

sys.path.insert(0, "/content/ballhawk")
import ballhawk_common as bc

data_yaml = bc.prepare_data()
for f in sorted(bc.RESULTS_DIR.glob("*/metrics.json")):
    r = json.loads(f.read_text())
    weights = f.parent / "best.pt"
    if r["run_name"].endswith("_smoke") or not weights.exists():
        continue
    overall, per_class = bc.evaluate(weights, data_yaml, r["imgsz"], "val")
    (f.parent / "val_metrics.json").write_text(json.dumps(
        {"run_name": r["run_name"], "split": "val", "imgsz": r["imgsz"], "overall": overall, "per_class": per_class}, indent=2))
    ball = next(c for c in per_class if c["class"] == "ball")
    print(f"{r['run_name']:34s} val ball mAP50-95 {ball['mAP50_95']:.3f}  all {overall['mAP50_95']:.3f}")
