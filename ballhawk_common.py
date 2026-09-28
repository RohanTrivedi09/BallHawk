"""Shared code for every BallHawk experiment notebook.

Owns the fixes for the pitfalls in ballhawk-project-plan.md Section 4 and the
metrics JSON schema in Section 8, so notebooks never re-implement them.
"""

import json
import pathlib
import time

import yaml
from ultralytics import YOLO

CODE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = pathlib.Path("/content/raw")
DATA_DIR = pathlib.Path("/content/data")
# Phase 4 pitch-keypoint data (splits_pitch.json), laid out the same way.
PITCH_RAW_DIR = pathlib.Path("/content/raw_pitch")
PITCH_DATA_DIR = pathlib.Path("/content/pitch_data")
RESULTS_DIR = pathlib.Path("/content/results")
RUNS_DIR = pathlib.Path("/content/runs")  # ultralytics project dir; override off Colab (e.g. Kaggle)

# Fixed across all runs unless an experiment says otherwise (plan Section 5).
FIXED = dict(epochs=100, batch=16, seed=0, deterministic=True)

METRICS_KEYS = {
    "run_name", "notebook", "variable_changed", "model", "imgsz", "epochs", "batch",
    "split", "overall", "per_class", "inference_fps", "train_minutes", "notes",
}


def default_albumentations():
    """The transform list ultralytics 8.4.163 uses when `augmentations` is not set.

    Passing `augmentations=[...]` replaces this list, so an experiment that adds one transform
    must pass these plus the new one to change a single variable.
    """
    import albumentations as A

    return [
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_range=(75, 100), p=0.0),
    ]


def build_dataset(data_dir=DATA_DIR, spec_file="splits.json", raw_dir=RAW_DIR):
    """Download the pinned HF mirror named in `spec_file` and lay it out by its video-grouped split.

    The mirror's own train/valid/test folders are ignored: they share clips across splits.
    Class and keypoint config (names, kpt_shape, flip_idx) is carried over from the mirror.
    """
    from huggingface_hub import snapshot_download

    spec = json.loads((CODE_DIR / spec_file).read_text())
    raw = pathlib.Path(snapshot_download(spec["repo"], repo_type="dataset", revision=spec["revision"],
                                         local_dir=raw_dir)) / "data"
    video_split = {v: s for s, vs in spec["videos"].items() for v in vs}
    folder = {"train": "train", "val": "valid", "test": "test"}
    for img in raw.glob("*/images/*.jpg"):
        split = folder[video_split[img.name.split("_")[0]]]
        for kind, src in (("images", img), ("labels", img.parents[1] / "labels" / f"{img.stem}.txt")):
            dst = data_dir / split / kind / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():  # background-only images may have no label file
                dst.symlink_to(src)
    cfg = yaml.safe_load((raw / "data.yaml").read_text())
    cfg = {k: v for k, v in cfg.items() if k not in {"train", "val", "test", "path", "roboflow"}}
    cfg["nc"] = len(cfg["names"])
    (data_dir / "data.yaml").write_text(yaml.safe_dump(cfg))


def prepare_data(data_dir=DATA_DIR, spec_file="splits.json", raw_dir=RAW_DIR):
    """Write data_fixed.yaml with absolute split paths (Pitfall 1: folder `valid`, key `val`)."""
    data_dir = pathlib.Path(data_dir)
    if not (data_dir / "data.yaml").exists():
        build_dataset(data_dir, spec_file, raw_dir)
    cfg = yaml.safe_load((data_dir / "data.yaml").read_text())
    for key, folder in {"train": "train", "val": "valid", "test": "test"}.items():
        p = data_dir / folder / "images"
        if not p.exists():
            p = data_dir / key / "images"
        if p.exists():
            cfg[key] = str(p)
        else:
            cfg.pop(key, None)
    if "test" not in cfg:
        raise FileNotFoundError(f"No test split under {data_dir}; test metrics are required.")
    cfg.pop("path", None)
    out = data_dir / "data_fixed.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return str(out)


def train(model_cfg, data_yaml, run_name, imgsz=640, smoke=False, model=None, **overrides):
    """Train and return (best_weights_path, save_dir, train_minutes, args_used).

    `model`: an already-built YOLO to train (e.g. with layers edited in memory); `model_cfg` is then
    only its label. Ultralytics builds the trainer model from the in-memory weights.
    """
    args = {**FIXED, "imgsz": imgsz, **overrides}
    if smoke:
        args.update(epochs=1, fraction=0.25)
    model = model or YOLO(model_cfg)
    start = time.time()
    model.train(data=data_yaml, project=str(RUNS_DIR), name=run_name, exist_ok=True, **args)
    minutes = (time.time() - start) / 60
    # ultralytics silently halves the batch on CUDA OOM in epoch 1; record what actually ran.
    args["requested_batch"], args["batch"] = args["batch"], model.trainer.batch_size
    save_dir = pathlib.Path(model.trainer.save_dir)  # Pitfall 2: never build this path by hand
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt missing in {save_dir}")
    return best, save_dir, minutes, args


def evaluate_test(weights, data_yaml, imgsz):
    """Test-split metrics: (overall, per_class)."""
    return evaluate(weights, data_yaml, imgsz, "test")


def evaluate(weights, data_yaml, imgsz, split):
    """Metrics on one split: (overall, per_class). Split is always passed explicitly;
    use "val" to choose settings and "test" only to report them."""
    m = YOLO(weights).val(data=data_yaml, split=split, imgsz=imgsz, batch=16, plots=False, verbose=False)
    overall = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map)}
    per_class = []
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class.append({"class": m.names[int(c)], "precision": float(p), "recall": float(r),
                          "mAP50": float(ap50), "mAP50_95": float(ap)})
    return overall, per_class


def split_images(data_yaml, split):
    img_dir = pathlib.Path(yaml.safe_load(pathlib.Path(data_yaml).read_text())[split])
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def evaluate_predictions(preds, data_yaml, split):
    """Score externally produced boxes (e.g. SAHI) with ultralytics' own matching and AP code,
    so results are comparable with evaluate_test().

    preds: {image_path: (xyxy [N,4] pixels, conf [N], cls [N])} for every image in the split.
    Returns (overall, per_class) in the metrics JSON format.
    """
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou

    names = yaml.safe_load(pathlib.Path(data_yaml).read_text())["names"]
    matcher = DetectionValidator()  # only for match_predictions and its 10 IoU thresholds
    tps, confs, pred_cls, target_cls = [], [], [], []
    for img in split_images(data_yaml, split):
        xyxy, conf, cls = (np.asarray(a, dtype=np.float32) for a in preds[img])
        xyxy = xyxy.reshape(-1, 4)
        w, h = Image.open(img).size
        lbl = img.parents[1] / "labels" / f"{img.stem}.txt"
        rows = np.array([l.split() for l in lbl.read_text().splitlines() if l.strip()] if lbl.exists() else [],
                        dtype=np.float32).reshape(-1, 5)
        gt_cls = rows[:, 0]
        cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
        gt_xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        if len(xyxy) and len(gt_cls):
            iou = box_iou(torch.from_numpy(gt_xyxy), torch.from_numpy(xyxy))
            tp = matcher.match_predictions(torch.from_numpy(cls), torch.from_numpy(gt_cls), iou).numpy()
        else:
            tp = np.zeros((len(xyxy), 10), dtype=bool)
        tps.append(tp); confs.append(conf); pred_cls.append(cls); target_cls.append(gt_cls)
    _, _, p, r, _, ap, classes, *_ = ap_per_class(np.concatenate(tps), np.concatenate(confs),
                                                  np.concatenate(pred_cls), np.concatenate(target_cls))
    overall = {"mAP50": float(ap[:, 0].mean()), "mAP50_95": float(ap.mean())}
    per_class = [{"class": names[int(c)], "precision": float(p[i]), "recall": float(r[i]),
                  "mAP50": float(ap[i, 0]), "mAP50_95": float(ap[i].mean())} for i, c in enumerate(classes)]
    return overall, per_class


def yolo_arrays(model, img, imgsz):
    """Plain full-frame prediction in the (xyxy, conf, cls) form evaluate_predictions() takes,
    with model.val()'s thresholds — the same-evaluator reference for SAHI."""
    b = model.predict(img, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, verbose=False)[0].boxes
    return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()


def sahi_model(weights, conf, imgsz=640):
    from sahi import AutoDetectionModel

    return AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                              confidence_threshold=conf, device="cuda:0", image_size=imgsz)


def sahi_predict(dm, img, size, overlap=0.2):
    """Native-resolution slices of `size` px + a full-frame pass, merged by NMS (IoU 0.5)."""
    from sahi.predict import get_sliced_prediction

    return get_sliced_prediction(str(img), dm, slice_height=size, slice_width=size,
                                 overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                 perform_standard_pred=True, postprocess_type="NMS",
                                 postprocess_match_metric="IOU", postprocess_match_threshold=0.5, verbose=0)


def sahi_arrays(dm, img, size, overlap=0.2):
    """SAHI prediction in the (xyxy, conf, cls) form evaluate_predictions() takes."""
    ops = sahi_predict(dm, img, size, overlap).object_prediction_list
    return ([op.bbox.to_xyxy() for op in ops], [op.score.value for op in ops], [op.category.id for op in ops])


def measure_fps(weights, data_yaml, imgsz, n_images=100, warmup=10, predict=None):
    """End-to-end FPS (pre + inference + post), batch 1, FP32, on test images.

    `predict(image_path)` overrides the default single-image YOLO predict (used for SAHI).
    """
    images = split_images(data_yaml, "test")
    if predict is None:
        model = YOLO(weights)
        predict = lambda p: model.predict(p, imgsz=imgsz, verbose=False)  # noqa: E731
    for p in images[:warmup]:
        predict(p)
    timed = images[:n_images]
    start = time.time()
    for p in timed:
        predict(p)
    return len(timed) / (time.time() - start)


def write_metrics(record, out_dir):
    """Validate against the Section 8 schema and write metrics.json."""
    missing = METRICS_KEYS - record.keys()
    extra = record.keys() - METRICS_KEYS
    if missing or extra:
        raise ValueError(f"metrics schema mismatch: missing={missing} extra={extra}")
    if not record["variable_changed"] or not isinstance(record["notes"], str):
        raise ValueError("variable_changed must be set and notes must be a string")
    if record["split"] != "test":
        raise ValueError("metrics must be reported on the test split")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    return out_dir / "metrics.json"


def run_experiment(run_name, notebook, variable_changed, model_cfg, imgsz=640, notes="",
                   smoke=False, model=None, **overrides):
    """Train → test-split eval → FPS → metrics.json + artifacts in RESULTS_DIR/<run_name>."""
    if smoke:
        run_name += "_smoke"
    data_yaml = prepare_data()
    best, save_dir, minutes, args = train(model_cfg, data_yaml, run_name, imgsz, smoke, model, **overrides)
    overall, per_class = evaluate_test(best, data_yaml, imgsz)
    fps = measure_fps(best, data_yaml, imgsz)
    if args["batch"] != args["requested_batch"]:
        notes += (f" Batch auto-reduced {args['requested_batch']} -> {args['batch']} by ultralytics after CUDA OOM:"
                  f" a second variable, confounds the comparison.")
    out_dir = RESULTS_DIR / run_name
    record = {
        "run_name": run_name, "notebook": notebook, "variable_changed": variable_changed,
        "model": model_cfg, "imgsz": imgsz, "epochs": args["epochs"], "batch": args["batch"],
        "split": "test", "overall": overall, "per_class": per_class,
        "inference_fps": fps, "train_minutes": minutes, "notes": notes,
    }
    path = write_metrics(record, out_dir)
    for name in ["results.csv", "args.yaml", "results.png", "confusion_matrix.png", "weights/best.pt"]:
        src = save_dir / name
        if src.exists():
            (out_dir / src.name).write_bytes(src.read_bytes())
    print(json.dumps(record, indent=2))
    print(f"metrics → {path}")
    return record
