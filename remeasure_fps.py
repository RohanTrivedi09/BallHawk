"""Re-time inference FPS for every finished run in one quiet pass, so all FPS numbers share conditions.

Why: FPS measured at the end of each training run is exposed to whatever else the VM is doing
(02 and its seed twin 02b, same architecture, measured 53 vs 40 FPS). Run this only when nothing
else is using the VM. Writes the median of 3 passes into each metrics.json and says so in notes.
"""

import json
import re
import statistics
import sys

sys.path.insert(0, "/content/ballhawk")
import ballhawk_common as bc

PASSES = 3
data_yaml = bc.prepare_data()
for f in sorted(bc.RESULTS_DIR.glob("*/metrics.json")):
    r = json.loads(f.read_text())
    if r["run_name"].endswith("_smoke"):
        continue
    sahi = re.search(r"_s(\d+)$", r["run_name"]) if r["run_name"].startswith("sahi_") else None
    if sahi:  # inference-only run; `model` is "<source run_name> best.pt"
        dm = bc.sahi_model(bc.RESULTS_DIR / r["model"].split()[0] / "best.pt", 0.25)
        size = int(sahi.group(1))
        timings = [bc.measure_fps(None, data_yaml, 640, predict=lambda p: bc.sahi_predict(dm, p, size))
                   for _ in range(PASSES)]
    else:
        timings = [bc.measure_fps(f.parent / "best.pt", data_yaml, r["imgsz"]) for _ in range(PASSES)]
    old, r["inference_fps"] = r["inference_fps"], statistics.median(timings)
    r["notes"] = re.sub(r" ?FPS re-measured.*?at end of run\)\.", "", r["notes"]).strip()
    r["notes"] += f" FPS re-measured in one quiet pass, median of {PASSES} (was {old:.1f} at end of run)."
    bc.write_metrics(r, f.parent)
    print(f"{r['run_name']:36s} {old:6.1f} -> {r['inference_fps']:6.1f}  passes {[round(t, 1) for t in timings]}")
