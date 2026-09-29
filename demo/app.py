"""BallHawk server: the project website and a JSON API for the live demo, in one FastAPI app.

  python demo/app.py            then open http://127.0.0.1:7860

Frame: detections (YOLO11s trained at 1280), team colours (KMeans on shirt colour against this
frame's grass), pitch calibration (automatic from 32 detected landmarks, or manual from four or more
clicked ones), a top-down map and the Law 11 offside line. Clip: the same plus tracking, speeds,
distance and possession, rendered frame by frame in a background job. Runs on CPU.
"""

import base64
import json
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if (ROOT / "ballhawk_pitch.py").exists():  # repository layout: shared modules live one level up
    sys.path.insert(0, str(ROOT))

import anyio  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import ballhawk_hud as hud  # noqa: E402
import ballhawk_pitch as bp  # noqa: E402
import ballhawk_tactical as tac  # noqa: E402
import ballhawk_video as bv  # noqa: E402

DET_PATH, KP_PATH = HERE / "weights" / "detector_1280.pt", HERE / "weights" / "pitch_keypoints_960.pt"
EXAMPLES = HERE / "examples"
WEB = next((d for d in (ROOT / "web", HERE / "web") if (d / "index.html").exists()), None)
MAX_IMAGE_MB, MAX_CLIP_MB, MAX_SECONDS = 20, 200, 30
IMAGE_TYPES, CLIP_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}, {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

DET, KP = YOLO(DET_PATH), YOLO(KP_PATH)
NAMES = [DET.names[i] for i in range(len(DET.names))]
PLAYER, KEEPER, REF, BALL = (NAMES.index(n) for n in ("player", "goalkeeper", "referee", "ball"))
MODEL_LOCK = threading.Lock()  # the shared models serve one frame request at a time


# ---------- frame ----------

def manual_homography(points):
    """points: [[u, v, k], ...] image pixels and keypoint index. Returns (H, rms residual in m or None)."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    k = pts[:, 2].astype(int)
    if len(pts) < 4:
        raise ValueError("Pick at least four landmark pairs.")
    if len(set(k)) != len(k) or k.min() < 0 or k.max() >= len(bp.PITCH_XY):
        raise ValueError("Each pitch landmark can be used once.")
    H, _ = bp.fit_homography(pts[:, :2], bp.PITCH_XY[k])
    if abs(np.linalg.det(H[:2, :2])) < 1e-12:
        raise ValueError("Those points do not define the pitch plane; spread them out and avoid three on one line.")
    metres, _ = bp.reprojection_error(H, pts[:, :2], bp.PITCH_XY[k])
    # Four points fit exactly, so only a fifth one onwards says anything about click accuracy.
    return H, (float(np.sqrt(np.mean(metres ** 2))) if len(pts) > 4 else None)


def analyse_frame(img, points=None):
    """Tactical view of one BGR frame. Returns (composed BGR image, stats dict, notes list)."""
    with MODEL_LOCK:
        b = DET.predict(img, imgsz=1280, conf=0.25, verbose=False)[0].boxes
        fit = None if points else bp.frame_homography(KP, img, 960, 0.5)
    xyxy, cls, conf = b.xyxy.cpu().numpy(), b.cls.int().cpu().numpy(), b.conf.cpu().numpy()
    role_of = {PLAYER: "player", KEEPER: "keeper", REF: "referee", BALL: "ball"}
    people = [j for j in range(len(cls)) if cls[j] != BALL]
    roles = {j: role_of[int(cls[j])] for j in people}
    notes = []
    note = lambda text, warn=False: notes.append({"text": text, "warn": warn})  # noqa: E731

    # Teams: KMeans (k=2) on torso colour, pitch pixels masked against this frame's grass.
    grass = bv.grass_lab(img)
    cols = {j: c for j in people if roles[j] == "player" and (c := bv.jersey_colour(img, xyxy[j], grass)) is not None}
    teams = {}
    if len(cols) >= 4:
        lab = KMeans(2, n_init=10, random_state=0).fit_predict(np.array(list(cols.values())))
        teams = dict(zip(cols, lab.tolist()))

    balls = [j for j in range(len(cls)) if cls[j] == BALL]
    kb = max(balls, key=lambda j: conf[j]) if balls else None
    ball_img = None if kb is None else ((xyxy[kb][0] + xyxy[kb][2]) / 2, (xyxy[kb][1] + xyxy[kb][3]) / 2)
    counts = {r: sum(v == r for v in roles.values()) for r in ("player", "referee", "keeper")} | {"ball": len(balls)}
    stats = {"counts": counts, "calibration": {"method": None, "points": 0, "residual_m": None},
             "offside": {"line_x": None, "beyond": None}, "space_share": None}

    H = None
    if points:
        H, resid = manual_homography(points)
        stats["calibration"] = {"method": "manual", "points": len(points), "residual_m": resid}
        note(f"Calibrated by hand from {len(points)} landmarks"
             + (f"; the fit leaves {100 * resid:.0f} cm RMS on them." if resid is not None else
                ". A fifth landmark would also measure how accurate the clicks were."))
    elif fit is not None:
        H = fit[0]
        stats["calibration"] = {"method": "auto", "points": int(fit[3].sum()), "residual_m": None}
        note(f"Calibrated automatically from {int(fit[3].sum())} detected landmarks (about 2.8 m median error on the test split).")
    else:
        note("Automatic calibration failed: fewer than four confident pitch landmarks. Calibrate by hand to get the map and the offside line.", True)

    state = hud.PitchState(fps=25)
    ball_xy, current, spreads, line_x, control = None, None, {}, None, None
    if H is not None:
        state.update(people, bp.project(H, bp.foot_points(xyxy[people])))
        # 2 m margin: a keeper on the goal line projects just behind it.
        on = lambda j: -2 <= state.pos[j][0] <= bp.LENGTH + 2 and -2 <= state.pos[j][1] <= bp.WIDTH + 2  # noqa: E731
        if kb is not None:
            ball_xy = bp.project(H, [[ball_img[0], xyxy[kb][3]]])[0]
            near = [(np.linalg.norm(state.pos[j] - ball_xy), teams[j]) for j in teams]
            if near and min(near)[0] <= 3.0:
                current = min(near)[1]
                state.possession[current] += 1
        for t in (0, 1):
            pts = np.array([state.pos[j] for j in teams if teams[j] == t])
            if len(pts) >= 3:
                spreads[t] = (float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
        control = hud.control_map({j: state.pos[j] for j in teams if on(j)}, teams, hud.visible_cells(H, img.shape))
        if (share := hud.space_share(control)) is not None:
            stats["space_share"] = {"A": share[0], "B": share[1]}
        keepers = [state.pos[j][0] for j in people if roles[j] == "keeper" and on(j)]
        # The keeper nearest a goal line fixes the defended goal.
        goal = bp.defending_goal_x(min(keepers, key=lambda x: min(x, bp.LENGTH - x))) if keepers else None
        if goal is not None and teams:
            depth = {t: np.mean([abs(state.pos[j][0] - goal) for j in teams if teams[j] == t and on(j)] or [1e9]) for t in (0, 1)}
            defending = min(depth, key=depth.get)
            dx = [state.pos[j][0] for j in people if on(j) and (roles[j] == "keeper" or teams.get(j) == defending)]
            att = [state.pos[j][0] for j in teams if teams[j] != defending and on(j)]
            line_x, flags = bp.offside(dx, att, goal, ball_xy[0] if ball_xy is not None else None)
            if line_x is not None:
                stats["offside"] = {"line_x": float(line_x), "beyond": int(sum(flags))}
                note(f"Offside line {line_x:.1f} m from the left goal line; {int(sum(flags))} attacker(s) beyond it. A visualiser, not a decision.")
        elif goal is None:
            note("No goalkeeper on the pitch, so the defended goal is unknown and no offside line is drawn.", True)
        else:
            note("Too few players to split into teams, so no offside line is drawn.", True)

    frame = hud.frame_overlay(img, xyxy[people], people, roles, teams, [ball_img], line_x, H)
    canvas = hud.compose(frame, hud.pitch_panel(state, roles, teams, ball_xy, line_x, control=control),
                         hud.hud_bar(0, state, current, spreads, line_x))
    return canvas, stats, notes


# ---------- clip jobs ----------

JOBS, JOBS_LOCK, WORKER = {}, threading.Lock(), ThreadPoolExecutor(max_workers=1)
JOB_TTL_S = 2 * 3600


def _set(job_id, **kw):
    with JOBS_LOCK:
        JOBS[job_id].update(kw)


def _run_clip(job_id, source, seconds, stride):
    job_dir = JOBS[job_id]["dir"]
    try:
        _set(job_id, state="running", progress=0.02, message="Reading the clip")
        trimmed, fps = tac.trim(source, job_dir / "input.mp4", seconds, stride)
        out = job_dir / "tactical_view.mp4"
        s = tac.render_clip(trimmed, out, DET_PATH, KP_PATH, progress=lambda f, m="": _set(job_id, progress=float(f), message=m))
        _set(job_id, state="done", progress=1.0, message="Done", stats=s | {"input_fps": fps}, video=f"/api/jobs/{job_id}/video")
    except Exception as e:  # reported to the page; the server keeps running
        _set(job_id, state="error", message=str(e) or type(e).__name__)


def _prune_jobs():
    now = time.time()
    with JOBS_LOCK:
        old = [k for k, j in JOBS.items() if now - j["created"] > JOB_TTL_S and j["state"] in ("done", "error")]
        for k in old:
            shutil.rmtree(JOBS.pop(k)["dir"], ignore_errors=True)


# ---------- HTTP ----------

app = FastAPI(title="BallHawk", docs_url="/api/docs", openapi_url="/api/openapi.json")


def _example(name, kinds):
    """An example file by name; only files that are actually in examples/ are accepted."""
    p = EXAMPLES / pathlib.Path(name).name
    if not p.is_file() or p.suffix.lower() not in kinds:
        raise HTTPException(404, "Unknown example.")
    return p


async def _read_capped(upload, max_mb):
    data = await upload.read(max_mb * 2 ** 20 + 1)
    if len(data) > max_mb * 2 ** 20:
        raise HTTPException(413, f"File is larger than {max_mb} MB.")
    return data


@app.get("/api/health")
def health():
    return {"ok": True, "detector": DET_PATH.name, "keypoints": KP_PATH.name, "max_seconds": MAX_SECONDS}


@app.get("/api/examples")
def examples():
    files = sorted(EXAMPLES.iterdir()) if EXAMPLES.exists() else []
    return {"frames": [{"name": p.name, "url": f"/examples/{p.name}"} for p in files if p.suffix.lower() in IMAGE_TYPES],
            "clips": [{"name": p.name, "url": f"/examples/{p.name}"} for p in files if p.suffix.lower() in CLIP_TYPES]}


@app.get("/api/pitch")
def pitch():
    return {"length": bp.LENGTH, "width": bp.WIDTH,
            "keypoints": [{"i": i, "name": n, "x": float(x), "y": float(y)} for i, (n, x, y) in enumerate(bp.KEYPOINTS)]}


@app.post("/api/frame")
async def frame(file: UploadFile | None = File(None), example: str | None = Form(None), points: str | None = Form(None)):
    if file is not None:
        data = await _read_capped(file, MAX_IMAGE_MB)
    elif example:
        data = _example(example, IMAGE_TYPES).read_bytes()
    else:
        raise HTTPException(400, "Upload a frame or pick an example.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "That file is not an image this server can read.")
    pts = None
    if points:
        try:
            pts = [[float(u), float(v), int(k)] for u, v, k in json.loads(points)]
        except (ValueError, TypeError):
            raise HTTPException(400, "Malformed calibration points.")
    try:
        canvas, stats, notes = await anyio.to_thread.run_sync(analyse_frame, img, pts)
    except ValueError as e:
        raise HTTPException(422, str(e))
    _, jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"image": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(),
            "size": [int(img.shape[1]), int(img.shape[0])], "stats": stats, "notes": notes}


@app.post("/api/clip")
async def clip(file: UploadFile | None = File(None), example: str | None = Form(None),
               seconds: float = Form(10), stride: int = Form(2)):
    if not 1 <= seconds <= MAX_SECONDS or stride not in (1, 2, 3):
        raise HTTPException(400, f"Seconds must be 1 to {MAX_SECONDS} and stride 1, 2 or 3.")
    if file is None and not example:
        raise HTTPException(400, "Upload a clip or pick the example.")
    _prune_jobs()
    job_id = uuid.uuid4().hex[:12]
    job_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"ballhawk_{job_id}_"))
    try:
        if file is not None:
            suffix = pathlib.Path(file.filename or "").suffix.lower()
            if suffix not in CLIP_TYPES:
                raise HTTPException(415, "Upload an mp4, mov, m4v, webm, mkv or avi clip.")
            source, size = job_dir / f"upload{suffix}", 0
            with source.open("wb") as f:
                while chunk := await file.read(2 ** 20):
                    size += len(chunk)
                    if size > MAX_CLIP_MB * 2 ** 20:
                        raise HTTPException(413, f"Clip is larger than {MAX_CLIP_MB} MB.")
                    f.write(chunk)
        else:
            source = _example(example, CLIP_TYPES)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    with JOBS_LOCK:
        busy = any(j["state"] in ("queued", "running") for j in JOBS.values())
        JOBS[job_id] = {"state": "queued", "progress": 0.0, "created": time.time(), "dir": job_dir,
                        "message": "Waiting for the clip ahead to finish" if busy else "Starting",
                        "stats": None, "video": None}
    WORKER.submit(_run_clip, job_id, str(source), seconds, stride)
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "Unknown or expired job.")
        return {k: j[k] for k in ("state", "progress", "message", "stats", "video")}


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
    if j is None or j["state"] != "done":
        raise HTTPException(404, "No video for this job.")
    return FileResponse(j["dir"] / "tactical_view.mp4", media_type="video/mp4", filename="ballhawk_tactical_view.mp4")


if EXAMPLES.exists():
    app.mount("/examples", StaticFiles(directory=EXAMPLES), name="examples")
if WEB is not None:
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")  # last: the website owns every other path

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
