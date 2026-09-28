"""BallHawk live demo: upload a broadcast football frame and get the tactical view back.

Detections (YOLO11s trained at 1280), team colours (KMeans on shirt colour against this frame's
grass), automatic pitch calibration (32-keypoint YOLO11s-pose + RANSAC homography), a top-down
minimap and a per-frame offside line (Law 11, visualiser only). Runs on CPU.
"""

import pathlib
import tempfile

import cv2
import gradio as gr
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

import ballhawk_hud as hud
import ballhawk_pitch as bp
import ballhawk_tactical as tac
import ballhawk_video as bv

HERE = pathlib.Path(__file__).resolve().parent
DET_PATH, KP_PATH = HERE / "weights" / "detector_1280.pt", HERE / "weights" / "pitch_keypoints_960.pt"
DET, KP = YOLO(DET_PATH), YOLO(KP_PATH)
NAMES = [DET.names[i] for i in range(len(DET.names))]
PLAYER, KEEPER, REF, BALL = (NAMES.index(n) for n in ("player", "goalkeeper", "referee", "ball"))


def tactical_view(rgb):
    if rgb is None:
        return None, "Upload a frame or pick an example."
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    b = DET.predict(img, imgsz=1280, conf=0.25, verbose=False)[0].boxes
    xyxy, cls, conf = b.xyxy.cpu().numpy(), b.cls.int().cpu().numpy(), b.conf.cpu().numpy()
    role_of = {PLAYER: "player", KEEPER: "keeper", REF: "referee", BALL: "ball"}
    people = [j for j in range(len(cls)) if cls[j] != BALL]
    roles = {j: role_of[int(cls[j])] for j in people}

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
    notes = [f"{sum(r == 'player' for r in roles.values())} players, {sum(r == 'referee' for r in roles.values())} referee(s), "
             f"{sum(r == 'keeper' for r in roles.values())} goalkeeper(s), {len(balls)} ball detection(s)."]
    state = hud.PitchState(fps=25)
    fit = bp.frame_homography(KP, img, 960, 0.5)
    H, ball_xy, current, spreads, line_x = None, None, None, {}, None
    if fit is None:
        notes.append("Pitch calibration failed: fewer than 4 confident pitch landmarks, so there is no pitch map or offside line.")
    else:
        H = fit[0]
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
                notes.append(f"Offside line at {line_x:.1f} m from the left goal line; {int(sum(flags))} attacker(s) beyond it.")
        elif goal is None:
            notes.append("No goalkeeper on the pitch, so the defended goal is unknown and no offside line is drawn.")
        else:
            notes.append("Too few players to split into teams, so no offside line is drawn.")
        notes.append(f"Calibrated from {int(fit[3].sum())} pitch landmarks (automatic calibration is about 2.8 m off on average). "
                     "Speeds and distance covered need video, so they are not shown for a single frame.")

    frame = hud.frame_overlay(img, xyxy[people], people, roles, teams, [ball_img], line_x, H)
    canvas = hud.compose(frame, hud.pitch_panel(state, roles, teams, ball_xy, line_x),
                         hud.hud_bar(0, state, current, spreads, line_x))
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), "\n\n".join(notes)


def tactical_clip(video_path, seconds, stride, progress=gr.Progress()):
    if not video_path:
        return None, "Upload a clip or pick the example."
    tmp = pathlib.Path(tempfile.mkdtemp())
    trimmed, fps = tac.trim(video_path, tmp / "input.mp4", seconds, int(stride))
    out = tmp / "tactical_view.mp4"
    s = tac.render_clip(trimmed, out, DET_PATH, KP_PATH, progress=lambda f, m="": progress(f, desc=m))
    poss = s["possession_share"]
    lines = [f"Processed {s['frames']} frames at {fps:.1f} fps; pitch calibrated in {s['calibrated_share']:.0%} of them; "
             f"team colours consistent in {s['team_agreement']:.0%} of each player's frames; "
             f"{s['rejected_calibrations']} calibration jump(s) rejected"
             + (" (many rejections mean this camera angle is far from the broadcast view the model was trained on)." if s['rejected_calibrations'] > 0.2 * s['frames'] else "."),
             f"Estimated possession: Team A {100 * poss['A']:.0f}%, Team B {100 * poss['B']:.0f}% ({s['possession_frames']} frames with the ball near a player)."
             if poss else "Possession: the ball was never within 3 m of a player, so no estimate.",
             "Top distance covered: " + (", ".join(f"{k} {v} m" for k, v in s["top_distance_m"].items()) or "n/a") + "."]
    if s["median_line_step_m"] is not None:
        lines.append(f"Offside line moved a median {100 * s['median_line_step_m']:.0f} cm between frames.")
    return str(out), "\n\n".join(lines)


frame_examples = sorted(str(p) for p in (HERE / "examples").glob("*.jpg"))
clip_examples = [[str(p), 10, 2] for p in sorted((HERE / "examples").glob("*.mp4"))]
frame_tab = gr.Interface(
    fn=tactical_view,
    inputs=gr.Image(label="Broadcast frame (wide camera works best)"),
    outputs=[gr.Image(label="Tactical view"), gr.Markdown(label="What was found")],
    examples=frame_examples, flagging_mode="never",
    description="One frame: team shapes, IDs, pitch map, possession estimate, team spreads and the offside line. A few seconds per frame.",
)
clip_tab = gr.Interface(
    fn=tactical_clip,
    inputs=[gr.Video(label="Broadcast clip (wide camera works best)"),
            gr.Slider(3, 30, value=10, step=1, label="Seconds to process"),
            gr.Slider(1, 3, value=2, step=1, label="Use every Nth frame (2 = half the frames, twice as fast)")],
    outputs=[gr.Video(label="Tactical view"), gr.Markdown(label="Summary")],
    examples=clip_examples, flagging_mode="never", cache_examples=False,
    description=("A clip adds tracking: player IDs that persist, speeds, distance covered and running possession. "
                 "Each processed frame takes about 1.5 s on a laptop CPU, so 10 s at every 2nd frame is about 3 to 4 minutes."),
)
demo = gr.TabbedInterface(
    [frame_tab, clip_tab], ["Frame", "Clip"],
    title="BallHawk: tactical view and offside visualiser",
)



def web_app():
    """One server for the whole project: the website at /, its media at /media, the Gradio demo at /demo.
    The site is read from ../site (repository layout) or ./site (a copy next to app.py, e.g. on a Space)."""
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    site = next((d for d in (HERE.parent / "site", HERE / "site") if (d / "index_web.html").exists()), None)
    if site is None:
        raise FileNotFoundError("site/index_web.html not found: run `python3 site/build.py` first")
    app = FastAPI(title="BallHawk")
    app.mount("/media", StaticFiles(directory=site / "media"), name="media")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(site / "index_web.html")

    return gr.mount_gradio_app(app, demo, path="/demo")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(web_app(), host="0.0.0.0", port=7860)
