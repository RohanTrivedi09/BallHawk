"""Render the broadcast-style tactical video (v2) for the DFL clip, locally.

  uv run --no-project --python 3.12 --with ultralytics==8.4.163 --with opencv-python-headless \
      --with scikit-learn --with imageio-ffmpeg python render_tactical_v2.py
Writes results/tactical_v2/tactical_v2.mp4 (H.264, web-ready) and stats.json.
"""

import json
import pathlib

import ballhawk_tactical as tac
import ballhawk_video as bv

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "results/tactical_v2"
video = bv.fetch_clip(str(ROOT / "video/dfl_121364_9.mp4"), bv.DFL_CLIP_URL)
stats = tac.render_clip(video, OUT / "tactical_v2.mp4", ROOT / "results/res_yolo11s_1280/best.pt",
                        ROOT / "results/pitch_kp_yolo11s/best.pt", progress=lambda f, m="": print(f"{f:.0%} {m}", flush=True))
stats = {"clip": bv.DFL_CLIP_CREDIT, **stats}
(OUT / "stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
