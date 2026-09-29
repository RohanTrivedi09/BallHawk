"""Clip pipeline for the broadcast-style tactical video: tracks, teams, per-frame calibration,
possession, spreads, speeds, distance and the offside line, rendered with ballhawk_hud.
Used by render_tactical_v2.py (full clips) and the demo's Clip tab (trimmed clips).
"""

import collections
import pathlib

import cv2
import imageio_ffmpeg
import numpy as np
from ultralytics import YOLO

import ballhawk_hud as hud
import ballhawk_pitch as bp
import ballhawk_video as bv

EMA, POSSESSION_M = 0.3, 3.0
MAX_JUMP_M, ACCEPT_AFTER = 3.0, 5  # calibration outlier rejection (see pass 2)
# Ball gaps: bridge up to half a second; the ball may move up to 3 % of the frame width per frame at 25 fps
# (about 60 px at 1920, faster than a hard shot seen by a wide camera).
BALL_GAP_S, BALL_STEP = 0.5, 0.03
# Team shape: each outfield track's mean position over the clip (smooths the ~2.8 m calibration noise),
# from tracks seen for at least SHAPE_MIN_S, longest first, at most 10 per team.
SHAPE_MIN_S, MIN_SHAPE_PLAYERS = 1.0, 6
SHAPE_MIN_CLIP_S = 8.0  # shorter clips give averages too noisy to read lines from


def trim(video, out_mp4, max_seconds=None, stride=1):
    """Copy the first max_seconds of a clip, keeping every stride-th frame. Returns (path, fps).
    Files OpenCV cannot decode (e.g. HEVC or ProRes .mov screen recordings) are converted to H.264 first."""
    cap = cv2.VideoCapture(str(video))
    if not cap.read()[0]:
        cap.release()
        converted = pathlib.Path(out_mp4).with_name("converted.mp4")
        import subprocess
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264",
                        "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(converted)], check=True)
        video = converted
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    limit = int(max_seconds * fps) if max_seconds else None
    writer, k = None, 0
    while True:
        ok, frame = cap.read()
        if not ok or (limit is not None and k >= limit):
            break
        if k % stride == 0:
            if writer is None:
                writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, frame.shape[1::-1])
            writer.write(frame)
        k += 1
    cap.release()
    if writer is None:
        raise ValueError("the clip has no readable frames")
    writer.release()
    return str(out_mp4), fps / stride


def render_clip(video, out_mp4, det, kp_weights, progress=lambda frac, msg="": None):
    """Render the tactical video for `video` into `out_mp4` (H.264, 1920x864). Returns a stats dict."""
    fps = bv.video_fps(video)
    names = [YOLO(det).names[i] for i in range(len(YOLO(det).names))]
    PLAYER, KEEPER, REF, BALL = (names.index(n) for n in ("player", "goalkeeper", "referee", "ball"))
    ROLE = {PLAYER: "player", KEEPER: "keeper", REF: "referee", BALL: "ball"}

    # Pass 1: tracks and teams.
    progress(0.05, "Tracking players and the ball")
    dets = bv.track(str(det), video, tracker="bytetrack.yaml", imgsz=1280)
    teams, agreement = bv.assign_teams(video, dets, PLAYER)
    role = bv.track_roles(dets, ROLE)
    # The ball: best ball box per frame, false jumps dropped and gaps up to BALL_GAP_S bridged.
    best = bv.best_ball_boxes(dets, role)
    cap = cv2.VideoCapture(str(video))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    ball_boxes, ball_filled = bv.fill_ball_track(best, max_gap=int(round(BALL_GAP_S * fps)),
                                                 max_step=BALL_STEP * width * 25 / fps)

    # Pass 2: smoothed homography per frame; defending side decided once per clip (as in notebook 13).
    progress(0.45, "Calibrating the pitch")
    # A fit that moves the pitch more than MAX_JUMP_M from the smoothed one in a single frame is
    # an outlier (a real pan moves it about 1 m per frame); it is skipped unless it persists for
    # ACCEPT_AFTER frames, which means the camera really cut.
    kp, Hs, H_s, rejected, streak = YOLO(kp_weights), [], None, 0, 0
    max_jump = MAX_JUMP_M * max(1.0, 25 / fps)  # a clip with skipped frames pans further per frame
    probe = None
    for frame in bv.frames(video):
        if probe is None:  # 3 x 3 image points over the lower 60 % of the frame, where the pitch is
            h, w = frame.shape[:2]
            probe = [[x * w, y * h] for x in (0.2, 0.5, 0.8) for y in (0.5, 0.7, 0.9)]
        fit = bp.frame_homography(kp, frame, 960, 0.5)
        if fit is not None:
            H = fit[0] / fit[0][2, 2]
            jump = 0.0 if H_s is None else float(np.median(np.linalg.norm(bp.project(H, probe) - bp.project(H_s, probe), axis=1)))
            if jump > max_jump and streak < ACCEPT_AFTER:
                rejected, streak = rejected + 1, streak + 1
            else:
                H_s = H if (H_s is None or jump > max_jump) else EMA * H + (1 - EMA) * H_s
                streak = 0
        Hs.append(H_s)
    keeper_x, depth = [], collections.defaultdict(list)
    for d, H in zip(dets, Hs):
        if H is None or not len(d["id"]):
            continue
        P = bp.project(H, bp.foot_points(d["xyxy"]))
        for p, i in zip(P, d["id"]):
            if role.get(int(i)) == "keeper":
                keeper_x.append(p[0])
    goal = bp.defending_goal_x(min(keeper_x, key=lambda x: min(x, bp.LENGTH - x))) if keeper_x else None
    for d, H in zip(dets, Hs):
        if H is None or goal is None or not len(d["id"]):
            continue
        for p, i in zip(bp.project(H, bp.foot_points(d["xyxy"])), d["id"]):
            if role.get(int(i)) == "player" and int(i) in teams:
                depth[teams[int(i)]].append(abs(p[0] - goal))
    defending = min(depth, key=lambda t: np.mean(depth[t])) if depth else None

    # Pass 3: render.
    pathlib.Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    progress(0.7, "Drawing the tactical view")
    writer = imageio_ffmpeg.write_frames(str(out_mp4), (1920, 864), fps=fps, codec="libx264",
                                         pix_fmt_out="yuv420p", output_params=["-crf", "24", "-preset", "medium", "-movflags", "+faststart"])
    writer.send(None)
    state, ball_hist = hud.PitchState(fps), collections.deque(maxlen=8)
    line_steps, prev_line = [], None
    shares, track_sum, track_n = [], collections.defaultdict(lambda: np.zeros(2)), collections.Counter()
    for f, (frame, d, H) in enumerate(zip(bv.frames(video), dets, Hs)):
        ids = [int(i) for i in d["id"]]
        bb = ball_boxes[f]
        ball_img = None if bb is None else ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
        ball_hist.append(ball_img)
        people = [k for k, i in enumerate(ids) if role.get(i) != "ball"]
        boxes, pids = d["xyxy"][people], [ids[k] for k in people]
        roles_now = {i: role[i] for i in pids}
        ball_xy, current, spreads, line_x, control = None, None, {}, None, None
        if H is not None:
            state.update(pids, bp.project(H, bp.foot_points(boxes)))
            if bb is not None:
                ball_xy = bp.project(H, [[ball_img[0], bb[3]]])[0]
                near = [(np.linalg.norm(state.pos[i] - ball_xy), teams.get(i)) for i in pids if roles_now[i] == "player" and i in teams]
                if near and min(near)[0] <= POSSESSION_M:
                    current = min(near)[1]
                    state.possession[current] += 1
            for t in (0, 1):
                pts = np.array([state.pos[i] for i in pids if roles_now[i] == "player" and teams.get(i) == t])
                if len(pts) >= 3:
                    spreads[t] = (float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1])))
            outfield = {i: state.pos[i] for i in pids if roles_now[i] == "player" and i in teams}
            control = hud.control_map(outfield, teams, hud.visible_cells(H, frame.shape))
            if (share := hud.space_share(control)) is not None:
                shares.append(share)
            for i, p in outfield.items():
                if 0 <= p[0] <= bp.LENGTH and 0 <= p[1] <= bp.WIDTH:
                    track_sum[i] += p
                    track_n[i] += 1
            if goal is not None and defending is not None:
                on = lambda i: -2 <= state.pos[i][0] <= bp.LENGTH + 2 and -2 <= state.pos[i][1] <= bp.WIDTH + 2  # noqa: E731
                dx = [state.pos[i][0] for i in pids if on(i) and (roles_now[i] == "keeper" or teams.get(i) == defending and roles_now[i] == "player")]
                att = [state.pos[i][0] for i in pids if on(i) and roles_now[i] == "player" and teams.get(i) not in (None, defending)]
                line_x, _ = bp.offside(dx, att, goal, ball_xy[0] if ball_xy is not None else None)
        if line_x is not None and prev_line is not None:
            line_steps.append(abs(line_x - prev_line))
        prev_line = line_x
        frame = hud.frame_overlay(frame, boxes, pids, roles_now, teams, ball_hist, line_x, H)
        panel = hud.pitch_panel(state, roles_now, teams, ball_xy, line_x, control=control)
        bar = hud.hud_bar(f, state, current, spreads, line_x)
        canvas = hud.compose(frame, panel, bar)
        writer.send(np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        if f % 25 == 0:
            progress(0.7 + 0.3 * f / max(len(dets), 1), f"Drawing frame {f} of {len(dets)}")
    writer.close()

    formation, shape_depths = {}, {}
    if goal is not None and defending is not None and len(dets) / fps >= SHAPE_MIN_CLIP_S:
        for t in (0, 1):  # depth from the goal each team defends
            own = goal if t == defending else bp.LENGTH - goal
            ids = [i for i, n in track_n.most_common() if teams[i] == t and n >= SHAPE_MIN_S * fps][:10]
            if len(ids) >= MIN_SHAPE_PLAYERS:
                depths = sorted(round(float(abs(track_sum[i][0] / track_n[i] - own)), 1) for i in ids)
                shape_depths[("A", "B")[t]] = depths
                if shape := bp.team_shape(depths):
                    formation[("A", "B")[t]] = shape

    total = sum(state.possession.values())
    stats = {"frames": len(dets), "fps": fps, "team_agreement": agreement, "rejected_calibrations": rejected,
             "calibrated_share": float(np.mean([H is not None for H in Hs])), "defending_goal_x": goal, "defending_team": defending,
             "possession_share": {("A", "B")[t]: state.possession[t] / total for t in (0, 1)} if total else None,
             "possession_frames": total,
             "top_distance_m": {f"#{i}": round(dd, 1) for i, dd in state.dist.most_common(5) if role.get(i) == "player"},
             "median_line_step_m": float(np.median(line_steps)) if line_steps else None,
             "ball_detected_share": float(np.mean([b is not None for b in best])),
             "ball_filled_share": float(np.mean([b is not None for b in ball_boxes])),
             "space_share": {"A": float(np.mean([s[0] for s in shares])), "B": float(np.mean([s[1] for s in shares]))} if shares else None,
             "formation": formation or None, "shape_depths_m": shape_depths or None}
    return stats
