"""Pitch-keypoint model at imgsz 1280 on a 24 GB L4 (Modal), batch 16 — the fair version of 12b.

12b ran on a 16 GB T4, where ultralytics cut the batch to 4 (noisy BatchNorm) and the run was
resumed after an interruption, so its worse result was confounded. Everything else matches
notebook 12: yolo11s-pose.pt, 150 epochs, seed 0, deterministic, the video-grouped pitch split.
Scoring matches notebook 12 (test) and eval_kp_val.py (val), keypoint confidence 0.5.

  modal run modal_jobs/kp1280.py   (from the project root; writes results/pitch_kp_yolo11s_1280_l4/)
"""

import json
import pathlib

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.163", "huggingface_hub")
    .add_local_file(ROOT / "ballhawk_common.py", "/root/code/ballhawk_common.py")
    .add_local_file(ROOT / "ballhawk_pitch.py", "/root/code/ballhawk_pitch.py")
    .add_local_file(ROOT / "splits_pitch.json", "/root/code/splits_pitch.json")
)
app = modal.App("ballhawk-kp1280", image=image)
IMGSZ, KP_CONF = 1280, 0.5


@app.function(gpu="L4", timeout=3 * 3600)
def train_and_score():
    import sys
    import time

    import cv2
    import numpy as np
    from ultralytics import YOLO

    sys.path.insert(0, "/root/code")
    import ballhawk_common as bc
    import ballhawk_pitch as bp

    data_yaml = bc.prepare_data(pathlib.Path("/root/pitch_data"), "splits_pitch.json", pathlib.Path("/root/raw_pitch"))
    model = YOLO("yolo11s-pose.pt")
    t0 = time.time()
    model.train(data=data_yaml, imgsz=IMGSZ, epochs=150, batch=16, seed=0, deterministic=True,
                project="/root/runs", name="pitch_kp_1280_l4", exist_ok=True)
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

    record = {"model": "yolo11s-pose.pt", "imgsz": IMGSZ, "epochs": 150, "batch": batch_used, "gpu": "L4 24GB (Modal)",
              "train_minutes": minutes, "kp_conf": KP_CONF,
              "pose_mAP50": float(m.pose.map50), "pose_mAP50_95": float(m.pose.map),
              "val": score("val"), "test": score("test"),
              "notes": "Fair 1280 run: 24 GB GPU, batch as recorded. Same data, split, seed and scoring as notebook 12 / eval_kp_val.py."}
    return record, best.read_bytes(), (save_dir / "results.csv").read_text()


@app.local_entrypoint()
def main():
    record, weights, results_csv = train_and_score.remote()
    out = ROOT / "results" / "pitch_kp_yolo11s_1280_l4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "best.pt").write_bytes(weights)
    (out / "results.csv").write_text(results_csv)
    (out / "pitch_metrics.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
