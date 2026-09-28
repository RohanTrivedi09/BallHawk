"""Choose the pitch-keypoint model on the pitch VAL split -> results/pitch_kp_choice.json.

Compares every results/pitch_kp_yolo11s*/best.pt (960 from notebook 12, 1280 from 12b) by median
reprojection error of the labelled keypoints through the predicted homography, at the keypoint
confidence already chosen on val (0.5, notebook 14). Test is never looked at here.
"""

import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/content/ballhawk")
import ballhawk_common as bc
import ballhawk_pitch as bp
from ultralytics import YOLO

KP_CONF = 0.5
pitch_yaml = bc.prepare_data(bc.PITCH_DATA_DIR, "splits_pitch.json", bc.PITCH_RAW_DIR)
scores = {}
for d in sorted(bc.RESULTS_DIR.glob("pitch_kp_yolo11s*")):
    if d.name.endswith("_smoke") or not (d / "best.pt").exists():
        continue
    imgsz = json.loads((d / "pitch_metrics.json").read_text()).get("imgsz", 960)
    model, errs, ok = YOLO(d / "best.pt"), [], 0
    for img in bc.split_images(pitch_yaml, "val"):
        h, w = cv2.imread(str(img)).shape[:2]
        k = np.array(open(img.parents[1] / "labels" / f"{img.stem}.txt").read().split(), float)[5:].reshape(-1, 3)
        vis = np.flatnonzero(k[:, 2] > 0)
        fit = bp.frame_homography(model, str(img), imgsz, KP_CONF)
        if fit is not None:
            ok += 1
            errs.append(float(np.median(bp.reprojection_error(fit[0], k[vis, :2] * [w, h], bp.PITCH_XY[vis])[0])))
    n = len(bc.split_images(pitch_yaml, "val"))
    scores[d.name] = {"weights": str(d / "best.pt"), "imgsz": imgsz, "calibrated": ok / n,
                      "median_reproj_m": float(np.median(errs)) if errs else float("inf")}
    print(d.name, scores[d.name])
ok = {k: v for k, v in scores.items() if v["calibrated"] >= 0.9} or scores
best = min(ok, key=lambda k: ok[k]["median_reproj_m"])
(bc.RESULTS_DIR / "pitch_kp_choice.json").write_text(json.dumps({"chosen": best, **scores[best], "kp_conf": KP_CONF,
                                                                  "split": "val", "all": scores}, indent=2))
print("chosen on val:", best)
