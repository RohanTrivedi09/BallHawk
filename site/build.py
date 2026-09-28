"""Build site/index.html from site/src.html: numbers from results/, images inlined as data URIs.

  python3 site/build.py          (run from the project root; re-run after results change)
The hero crop is the ball labelled in test frame 121364_7_8; its position in the 1920x1080 frame is
computed here so the page can phase the stride-8 grid to the full frame.
"""

import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
CROP_W, CROP_H, FRAME_W, FRAME_H = 192, 128, 1920, 1080


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def runs():
    out = {}
    for f in sorted((ROOT / "results").glob("*/metrics.json")):
        r = json.loads(f.read_text())
        ball = next(c for c in r["per_class"] if c["class"] == "ball")
        v = f.parent / "val_metrics.json"
        val = next(c for c in json.loads(v.read_text())["per_class"] if c["class"] == "ball")["mAP50_95"] if v.exists() else None
        out[r["run_name"]] = {"ball": ball["mAP50_95"], "ball50": ball["mAP50"], "ballR": ball["recall"],
                              "all": r["overall"]["mAP50_95"], "fps": r["inference_fps"], "batch": r["batch"],
                              "min": r["train_minutes"], "val_ball": val}
    return out


def ball_crop_geometry():
    """Same crop rule as the asset step: 192x128 native px centred on the labelled ball."""
    lbl = next((ROOT / "frames/test_labels").glob("121364_7_8_png*.txt"))
    cx, cy, bw, bh = next(list(map(float, l.split()[1:])) for l in lbl.read_text().splitlines() if l.split()[0] == "0")
    cx, cy, bw, bh = cx * FRAME_W, cy * FRAME_H, bw * FRAME_W, bh * FRAME_H
    x0, y0 = int(cx - CROP_W / 2), int(cy - CROP_H / 2)
    return {"crop_x0": x0, "crop_y0": y0, "ball_x": round(cx - x0 - bw / 2, 2), "ball_y": round(cy - y0 - bh / 2, 2),
            "ball_w": round(bw, 2), "ball_h": round(bh, 2)}


fill = {**{k: str(v) for k, v in ball_crop_geometry().items()},
        "runs_json": json.dumps(runs()),
        "ball_crop": data_uri(SITE / "assets/ball_crop.png", "image/png"),
        "gallery": data_uri(SITE / "assets/gallery.jpg", "image/jpeg"),
        "offside": data_uri(SITE / "assets/offside_tile.jpg", "image/jpeg"),
        "tactical_v2_poster": data_uri(SITE / "assets/tactical_v2_poster.jpg", "image/jpeg")}
html = (SITE / "src.html").read_text()
for k, v in fill.items():
    html = html.replace("{{" + k + "}}", v)
assert "{{" not in html, "unfilled placeholder"
(SITE / "index.html").write_text(html)
print(f"site/index.html {len(html) / 1024:.0f} KB", {k: fill[k] for k in ("crop_x0", "crop_y0", "ball_w")})
