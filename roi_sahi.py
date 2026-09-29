"""Ball-region slicing: SAHI's second look, spent only where the ball might be. No training.

Full SAHI (notebook 06) re-runs the 640 model on every 960 px tile of the frame: +0.075 ball
mAP50-95 at about a ninth of the speed. Most of those tiles contain no ball. Here the full-frame
pass at a very low threshold proposes ball candidates; only the top k get a crop (640 or 960 native
px, run at 640), whose ball boxes are merged with the full-frame ones by NMS. Players, referees and
goalkeepers come from the full-frame pass unchanged.

Crop size and k are chosen together on the validation split and reported once on test, like every
other setting in the study. Crop size matters: a 640 px crop shows the ball at native resolution,
three times larger than the 640 model saw in training; a 960 px crop (SAHI's slice size) halves that. All methods run here on the same machine and are scored by the same evaluator
(ballhawk_common.evaluate_predictions), so ball mAP and cost are directly comparable.

  uv run --no-project --python 3.12 --with-requirements demo/requirements.txt --with sahi python roi_sahi.py
Rebuilds the video-grouped splits from the pinned HF mirror into frames/hf_data (splits.json), and
reads results/baseline_yolo11s_640/best.pt.
Writes results/roi_sahi/metrics.json.
"""

import json
import pathlib
import platform
import time

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

import ballhawk_common as bc

ROOT = pathlib.Path(__file__).resolve().parent
WEIGHTS = ROOT / "results/baseline_yolo11s_640/best.pt"
OUT = ROOT / "results/roi_sahi"
IMGSZ, CROPS, SWEEP, SAHI_SLICE = 640, (640, 960), (1, 2, 3, 5), 960

model = YOLO(WEIGHTS)
NAMES = [model.names[i] for i in range(len(model.names))]
BALL = NAMES.index("ball")


def full_frame(img):
    return bc.yolo_arrays(model, str(img), IMGSZ)


def roi(img, full, k, crop):
    """Full-frame boxes plus native-resolution crops around the top-k ball candidates.
    Returns ((xyxy, conf, cls), forward passes used)."""
    xyxy, conf, cls = full
    frame = cv2.imread(str(img))
    h, w = frame.shape[:2]
    windows = []
    for j in np.argsort(-conf):
        if cls[j] != BALL:
            continue
        cx, cy = (xyxy[j][:2] + xyxy[j][2:]) / 2
        if any(x0 <= cx < x0 + crop and y0 <= cy < y0 + crop for x0, y0 in windows):
            continue  # already inside a chosen crop
        windows.append((int(np.clip(cx - crop / 2, 0, w - crop)), int(np.clip(cy - crop / 2, 0, h - crop))))
        if len(windows) == k:
            break
    boxes, scores = [xyxy[cls == BALL]], [conf[cls == BALL]]
    for x0, y0 in windows:
        b = model.predict(frame[y0:y0 + crop, x0:x0 + crop], imgsz=IMGSZ, conf=0.001, iou=0.7, max_det=300,
                          classes=[BALL], verbose=False)[0].boxes
        boxes.append(b.xyxy.cpu().numpy() + [x0, y0, x0, y0])
        scores.append(b.conf.cpu().numpy())
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    keep = torchvision.ops.nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), 0.5).numpy()
    other = cls != BALL
    return ((np.concatenate([xyxy[other], boxes[keep]]), np.concatenate([conf[other], scores[keep]]),
             np.concatenate([cls[other], np.full(len(keep), BALL)])), 1 + len(windows))


def sahi_fn():
    from sahi import AutoDetectionModel

    dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(WEIGHTS), confidence_threshold=0.001,
                                            device="cpu", image_size=IMGSZ)
    return lambda img: bc.sahi_arrays(dm, img, SAHI_SLICE)


def score(preds, data_yaml, split="test"):
    _, per_class = bc.evaluate_predictions(preds, data_yaml, split)
    ball = next(c for c in per_class if c["class"] == "ball")
    return {"ball": ball["mAP50_95"], "ball50": ball["mAP50"], "ball_recall": ball["recall"]}


def run_roi(images, k, crop):
    """ROI predictions for every image, timed end to end including the full-frame pass it builds on."""
    t, preds, passes = time.perf_counter(), {}, []
    for p in images:
        preds[p], n = roi(p, full_frame(p), k, crop)
        passes.append(n)
    return preds, {"passes": float(np.mean(passes)), "ms": 1000 * (time.perf_counter() - t) / len(images)}


def main():
    data_yaml = bc.prepare_data(ROOT / "frames/hf_data", "splits.json", ROOT / "frames/hf_raw")
    val, test = bc.split_images(data_yaml, "val"), bc.split_images(data_yaml, "test")
    for p in test[:3]:  # warm-up
        full_frame(p)
    out = {"device": f"CPU ({platform.machine()})", "images": len(test)}

    # Choose crop size and k on val.
    grid = [(c, k) for c in CROPS for k in SWEEP]
    out["val_baseline"] = score({p: full_frame(p) for p in val}, data_yaml, "val")["ball"]
    out["val_grid"] = {f"{c}/{k}": score(run_roi(val, k, c)[0], data_yaml, "val")["ball"] for c, k in grid}
    crop_best, k_best = max(grid, key=lambda g: (round(out["val_grid"][f"{g[0]}/{g[1]}"], 3), -g[1], -g[0]))  # ties: cheaper
    out["crop_px"], out["k"] = crop_best, k_best
    print("val", out["val_baseline"], out["val_grid"], "->", crop_best, k_best, flush=True)

    t = time.perf_counter()
    full = {p: full_frame(p) for p in test}
    out["baseline"] = score(full, data_yaml) | {"passes": 1, "ms": 1000 * (time.perf_counter() - t) / len(test)}
    print("baseline", out["baseline"], flush=True)
    out["roi_grid"] = {}
    for c, k in grid:  # the whole grid on test, for transparency; only the val choice is the reported result
        preds, cost = run_roi(test, k, c)
        out["roi_grid"][f"{c}/{k}"] = score(preds, data_yaml) | cost
        print("roi", c, k, out["roi_grid"][f"{c}/{k}"], flush=True)
    out["roi"] = out["roi_grid"][f"{crop_best}/{k_best}"] | {"crop_px": crop_best, "k": k_best}

    sahi = sahi_fn()
    sahi(test[0])
    t = time.perf_counter()
    preds = {p: sahi(p) for p in test}
    # 1920x1080 in 960 px slices at 0.2 overlap is a 3 x 2 grid, plus the full-frame pass.
    out["sahi"] = score(preds, data_yaml) | {"passes": 7, "ms": 1000 * (time.perf_counter() - t) / len(test)}
    print("sahi", out["sahi"], flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
