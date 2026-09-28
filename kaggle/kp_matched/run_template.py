"""BallHawk pitch keypoints, matched batch on Kaggle 2x T4: imgsz 960 vs 1280, both batch 4.

12b (1280) was confounded: the T4 forced batch 4 while the 960 model used batch 16. Here both sizes
train at batch 4 (the most a T4 fits at 1280), fresh, seed 0, deterministic, 150 epochs, one per GPU
in parallel, so resolution is the only difference. Scoring matches notebook 12 / eval_kp_val.py.
Built by build.py, which embeds ballhawk_common.py, ballhawk_pitch.py and splits_pitch.json.
"""
import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path("/kaggle/working"); CODE, OUT = ROOT / "code", ROOT / "kp_matched"
DATA, RAW = pathlib.Path("/kaggle/tmp/pitch_data"), pathlib.Path("/kaggle/tmp/raw_pitch")
EMBED = {{EMBED}}
SIZES, BATCH, KP_CONF = [960, 1280], 4, 0.5


def worker(imgsz):
    sys.path.insert(0, str(CODE))
    import cv2, numpy as np
    from ultralytics import YOLO
    import ballhawk_common as bc, ballhawk_pitch as bp
    data_yaml = str(DATA / "data_fixed.yaml")
    model = YOLO("yolo11s-pose.pt"); t0 = time.time()
    model.train(data=data_yaml, imgsz=imgsz, epochs=150, batch=BATCH, seed=0, deterministic=True, workers=2,
                project=str(ROOT / "runs"), name=f"kp_{imgsz}_b{BATCH}", exist_ok=True)
    minutes = (time.time() - t0) / 60
    save_dir = pathlib.Path(model.trainer.save_dir); best = save_dir / "weights" / "best.pt"
    m = YOLO(best).val(data=data_yaml, split="test", imgsz=imgsz, plots=False, verbose=False)
    kp = YOLO(best)

    def score(split):
        errs, px, ok, n = [], [], 0, 0
        for img in bc.split_images(data_yaml, split):
            n += 1
            h, w = cv2.imread(str(img)).shape[:2]
            lab = np.array(open(img.parents[1] / "labels" / f"{img.stem}.txt").read().split(), float)[5:].reshape(-1, 3)
            vis = np.flatnonzero(lab[:, 2] > 0); gt = lab[:, :2] * [w, h]
            k = bp.predict_keypoints(kp, str(img), imgsz)
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

    rec = {"imgsz": imgsz, "batch": model.trainer.batch_size, "epochs": 150, "train_minutes": minutes,
           "pose_mAP50": float(m.pose.map50), "pose_mAP50_95": float(m.pose.map), "val": score("val"), "test": score("test")}
    (OUT / f"kp_{imgsz}.json").write_text(json.dumps(rec, indent=2))
    (OUT / f"kp_{imgsz}_best.pt").write_bytes(best.read_bytes())
    print(f"[imgsz {imgsz}] batch {rec['batch']} val reproj {rec['val']['median_reproj_m']} test reproj {rec['test']['median_reproj_m']} ({minutes:.1f} min)", flush=True)


def main():
    CODE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    for name, text in EMBED.items():
        (CODE / name).write_text(text)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.163"], check=True)
    sys.path.insert(0, str(CODE))
    import ballhawk_common as bc
    from ultralytics import YOLO
    YOLO("yolo11s-pose.pt")
    bc.prepare_data(DATA, "splits_pitch.json", RAW)
    import torch
    n = torch.cuda.device_count(); print(n, "GPU(s)", flush=True)
    procs = [subprocess.Popen([sys.executable, __file__, "worker", str(s)], cwd=ROOT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i % max(n, 1))}) for i, s in enumerate(SIZES)]
    codes = [p.wait() for p in procs]
    print("workers exited", codes, flush=True)
    summary = {f.stem: json.loads(f.read_text()) for f in sorted(OUT.glob("kp_*.json"))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))
    sys.exit(1 if any(codes) else 0)


if __name__ == "__main__":
    worker(int(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[1] == "worker" else main()
