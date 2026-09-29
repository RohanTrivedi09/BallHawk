"""How far ball-track gap filling can be trusted, measured on the unseen 30-second DFL clip.

Coverage: frames with a ball box before and after ballhawk_video.fill_ball_track, with the
renderer's own settings (ballhawk_tactical.BALL_GAP_S, BALL_STEP).
Accuracy: no ground truth exists for the missing frames, so real detections are hidden instead.
For every run of L consecutive detected frames (L = 1 .. the longest bridged gap) with a detection
on both sides, the run is hidden, filled from its neighbours, and compared with the hidden boxes.

  uv run --no-project --python 3.12 --with-requirements demo/requirements.txt python ball_gaps.py
Writes results/ball_gaps/stats.json.
"""

import json
import pathlib

import cv2
import numpy as np
from ultralytics import YOLO

import ballhawk_tactical as tac
import ballhawk_video as bv

ROOT = pathlib.Path(__file__).resolve().parent
DET = ROOT / "results/res_yolo11s_1280/best.pt"
OUT = ROOT / "results/ball_gaps"


def centre(b):
    return (np.asarray(b[:2]) + np.asarray(b[2:])) / 2


def main():
    video = bv.fetch_clip(str(ROOT / "video/dfl_121364_9.mp4"), bv.DFL_CLIP_URL)
    fps, cap = bv.video_fps(video), cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    role_of = {i: {"goalkeeper": "keeper"}.get(n, n) for i, n in YOLO(DET).names.items()}
    dets = bv.track(str(DET), video, tracker="bytetrack.yaml", imgsz=1280)
    best = bv.best_ball_boxes(dets, bv.track_roles(dets, role_of))
    max_gap, max_step = int(round(tac.BALL_GAP_S * fps)), tac.BALL_STEP * width * 25 / fps

    filled, was_filled = bv.fill_ball_track(best, max_gap, max_step)
    detected = sum(b is not None for b in best)
    kept = sum(b is not None for b in filled) - int(was_filled.sum())

    errors = {L: [] for L in range(1, max_gap + 1)}
    for L in errors:
        for s in range(1, len(best) - L):
            window = best[s - 1:s + L + 1]
            if any(b is None for b in window):
                continue
            out, _ = bv.fill_ball_track([window[0]] + [None] * L + [window[-1]], max_gap, max_step)
            if out[1] is None:  # the endpoints were too far apart to bridge
                continue
            errors[L] += [float(np.linalg.norm(centre(out[j]) - centre(window[j]))) for j in range(1, L + 1)]
    allerr = np.array([e for v in errors.values() for e in v])
    stats = {"clip": bv.DFL_CLIP_CREDIT, "frames": len(best), "fps": fps, "frame_width": width,
             "before": detected / len(best), "after": sum(b is not None for b in filled) / len(best),
             "max_gap_s": tac.BALL_GAP_S, "dropped_jumps": detected - kept,
             "holdout_n": int(len(allerr)),
             "holdout_median_px": float(np.median(allerr)) if len(allerr) else None,
             "holdout_p90_px": float(np.percentile(allerr, 90)) if len(allerr) else None,
             "holdout_by_gap": {str(L): (float(np.median(e)) if e else None) for L, e in errors.items()},
             "holdout_note": f"Median px (frame {int(width)} px wide) between a hidden detection and its filled position, "
                             f"over {len(allerr)} hidden boxes in gaps of 1 to {max_gap} frames"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
