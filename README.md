# BallHawk

Why YOLO misses the football on broadcast footage, what fixes it, and how far that carries into tracking,
team assignment and an offside visualiser. Computer Vision course project, Adani University.

On a wide broadcast shot the ball is about **3.7 px** wide at YOLO's 640 px input, smaller than one cell of its
finest (stride-8) grid. Every intervention was tested one variable at a time, chosen on validation, reported on test.

| Result | Value |
|---|---|
| Ball mAP50-95, baseline 640 → trained at 1280 | **0.110 → 0.298** |
| Seed-to-seed noise (sd, 3 seeds) | ±0.009 |
| Sliced inference (SAHI) on the 640 model | +0.075, 9× slower; no gain on the 1280 model |
| Offside-line shift per pixel of calibration click error | ≈ 10 cm |
| Offside-line error from automatic calibration (16 unseen frames) | 1.17 m median |

Full plan, decisions and every result: [`ballhawk-project-plan.md`](ballhawk-project-plan.md).

## Layout
- `ballhawk_common.py` — data build (video-grouped splits), training, test/val evaluation, FPS, metrics JSON
- `ballhawk_video.py`, `ballhawk_pitch.py` — tracking, teams, ball-track gap filling; pitch model, homography, Law 11 offside, team shape
- `ballhawk_hud.py`, `ballhawk_tactical.py`, `render_tactical_v2.py` — broadcast-style tactical video with space control
- `roi_sahi.py`, `ball_gaps.py` — add-on studies without new training: ball-region slicing, gap-filling accuracy
- `notebooks/` — experiments 02–14 (run on Colab via the `colab` CLI; `executed/` holds the run copies)
- `kaggle/`, `modal_jobs/`, `lightning_jobs/` — seed replication and keypoint jobs on other GPU hosts
- `web/` — the website: Overview, Study, Tactical and Live demo pages (plain HTML, CSS and JS)
- `demo/` — FastAPI server: serves `web/` and the demo API (frame, clip jobs, manual calibration)
- `results/` — metrics, report figures and stats for every run

## Website and live demo
One server runs both. From the repository root:
```
python3 web/build.py        # refresh web/data/site-data.js after results change
uv run --python 3.12 --with-requirements demo/requirements.txt python demo/app.py
```
Then open http://127.0.0.1:7860 (API docs at `/api/docs`). Without the server, `web/` is a static site;
only the demo page needs the API.

## Data and credits
Player detection: `martinjolif/football-player-detection` (Roboflow football-players-detection, CC BY 4.0).
Pitch keypoints: `martinjolif/football-pitch-detection` (CC BY 4.0). Broadcast clips: DFL Deutsche Fußball Liga,
Bundesliga Data Shootout. Phone clip: Wikimedia Commons, CC BY-SA 4.0. Built with Ultralytics 8.4.163.
