"""Pitch-keypoint model at imgsz 1280 on a Lightning AI L4 (24 GB), batch 16: the fair version of 12b.
Same training and scoring as modal_jobs/kp1280.py. Run as a Lightning job from a Studio holding
ballhawk_common.py, ballhawk_pitch.py and splits_pitch.json next to this file. Writes OUT/.
"""
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path("/teamspace/uploads/ballhawk_kp1280") if pathlib.Path("/teamspace/uploads").exists() else HERE / "out"
IMGSZ, KP_CONF = 1280, 0.5
subprocess.run(["pip", "install", "-q", "ultralytics==8.4.163", "huggingface_hub"], check=True)
import sys
sys.path.insert(0, str(HERE))
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


import ballhawk_common as bc
import ballhawk_pitch as bp

data_yaml = bc.prepare_data(HERE / "pitch_data", "splits_pitch.json", HERE / "raw_pitch")
model = YOLO("yolo11s-pose.pt")
t0 = time.time()
model.train(data=data_yaml, imgsz=IMGSZ, epochs=150, batch=16, seed=0, deterministic=True,
            project=str(HERE / "runs"), name="pitch_kp_1280_l4", exist_ok=True)
minutes = (time.time() - t0) / 60
save_dir = pathlib.Path(model.trainer.save_dir)  # Pitfall 2: read it, never build it
best, batch_used = save_dir / "weights" / "best.pt", model.trainer.batch_size

m = YOLO(best).val(data=data_yaml, split="test", imgsz=IMGSZ, plots=False, verbose=False)
kp = YOLO(best)

def score(split):
    errs, px, ok, n = [], [], 0, 0
    for img in bc.split_images(data_yaml, split):
        n += 1
        h, w = cv2.imread(str(img)).shape[:2]
        lab = np.array(open(img.parents[1] / "labels" / f"{img.stem}.txt").read().split(), float)[5:].reshape(-1, 3)
        vis = np.flatnonzero(lab[:, 2] > 0); gt = lab[:, :2] * [w, h]
        k = bp.predict_keypoints(kp, str(img), IMGSZ)
        idx, xy = bp.confident(k, KP_CONF)
        common = np.intersect1d(idx, vis)
        if len(common):
            px.append(float(np.median([np.linalg.norm(xy[list(idx).index(i)] - gt[i]) for i in common])))
        fit = bp.keypoint_homography(k, KP_CONF)
        if fit is not None:
            ok += 1
            errs.append(float(np.median(bp.reprojection_error(fit[0], gt[vis], bp.PITCH_XY[vis])[0])))
    return {"calibrated": ok / n, "median_reproj_m": float(np.median(errs)) if errs else None,
            "p90_reproj_m": float(np.percentile(errs, 90)) if errs else None,
            "median_kp_px_err": float(np.median(px)) if px else None}

record = {"model": "yolo11s-pose.pt", "imgsz": IMGSZ, "epochs": 150, "batch": batch_used, "gpu": "L4 24GB (Lightning AI)",
          "train_minutes": minutes, "kp_conf": KP_CONF,
          "pose_mAP50": float(m.pose.map50), "pose_mAP50_95": float(m.pose.map),
          "val": score("val"), "test": score("test"),
          "notes": "Fair 1280 run: 24 GB GPU, batch as recorded. Same data, split, seed and scoring as notebook 12 / eval_kp_val.py."}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "best.pt").write_bytes(best.read_bytes())
(OUT / "results.csv").write_text((save_dir / "results.csv").read_text())
(OUT / "pitch_metrics.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
print("outputs in", OUT)
