"""Video pipeline shared by notebook 08 (tracking + teams) and Phase 5 (tactical video view).

Frames are never held in memory as a whole clip (522 × 1080p frames ≈ 3 GB); every pass
re-reads the video.
"""

import collections

import cv2
import numpy as np
from ultralytics import YOLO

# Reference palette slots (BGR for OpenCV): team 0 blue, team 1 orange, referee yellow, goalkeeper violet.
TEAM_BGR = {0: (214, 120, 42), 1: (52, 104, 235)}
REFEREE_BGR, KEEPER_BGR, BALL_BGR, UNKNOWN_BGR = (0, 161, 237), (167, 58, 74), (72, 73, 227), (160, 160, 160)


CLIP_URL = ("https://upload.wikimedia.org/wikipedia/commons/b/ba/"
            "Match_de_football_France-Allemagne_-_16_octobre_2018_-_Phase_de_jeu_%281%29.ogv")
CLIP_CREDIT = ('"Match de football France-Allemagne - 16 octobre 2018 - Phase de jeu (1)" by Like tears in rain, '
               "Wikimedia Commons, CC BY-SA 4.0")


DFL_CLIP_URL = ("https://huggingface.co/datasets/dbal0503/Bundesliga/resolve/8f854e3e4f7007134410f2040827bba7bf4c3dd8/"
                "Bundesliga/Clips/121364_9.mp4")
DFL_CLIP_CREDIT = "DFL Deutsche Fussball Liga, Bundesliga Data Shootout clip 121364_9 (via HF dbal0503/Bundesliga)."


def fetch_clip(video="/content/video/france_2018.mp4", url=CLIP_URL):
    """Download a clip once. An .mp4 source is used as is; anything else (the Commons .ogv) gets a
    near-lossless H.264 copy, because ultralytics cannot read .ogv. Every detector and notebook
    then sees the same pixels. Returns the MP4 path."""
    import pathlib
    import subprocess
    import urllib.request

    video = pathlib.Path(video)
    mp4_source = url.split("?")[0].endswith(".mp4")
    source = video if mp4_source else video.with_suffix(".ogv")
    video.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "BallHawk-course-project/1.0"})
        source.write_bytes(urllib.request.urlopen(req).read())
    if not video.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", "-an", str(video)], check=True)
    return str(video)


def frames(video):
    cap = cv2.VideoCapture(str(video))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def video_fps(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def track(weights, video, tracker="bytetrack.yaml", conf=0.1, imgsz=640):
    """Run the tracker (always named explicitly) and return one dict per frame:
    xyxy [N,4], cls [N], conf [N], id [N]. Tracking mode only returns tracked boxes.

    conf=0.1 lets low-score boxes reach ByteTrack's second association stage.
    """
    model = YOLO(weights)
    out = []
    for r in model.track(source=str(video), tracker=tracker, persist=True, stream=True,
                         conf=conf, imgsz=imgsz, verbose=False):
        b = r.boxes
        ids = b.id.int().cpu().numpy() if b.id is not None else np.zeros(0, int)
        n = len(ids)
        out.append({"xyxy": b.xyxy.cpu().numpy()[:n], "cls": b.cls.int().cpu().numpy()[:n],
                    "conf": b.conf.cpu().numpy()[:n], "id": ids})
    return out


def track_stats(dets, fps, player_cls, ball_cls):
    """Fragmentation proxies (no ground-truth IDs exist for the clip, so true ID switches
    cannot be counted) and ball track continuity."""
    minutes = len(dets) / fps / 60
    lengths = collections.Counter()
    for d in dets:
        for i, c in zip(d["id"], d["cls"]):
            if c == player_cls:
                lengths[int(i)] += 1
    ball_frames = sum(bool((d["cls"] == ball_cls).any()) for d in dets)
    return {"frames": len(dets), "player_track_ids": len(lengths),
            "player_ids_per_minute": len(lengths) / minutes,
            "median_player_track_frames": float(np.median(list(lengths.values()))) if lengths else 0.0,
            "ball_track_continuity": ball_frames / len(dets)}


def track_roles(dets, role_of_cls):
    """One role per track: the majority class over its frames. role_of_cls: {class id: role name}."""
    votes = collections.defaultdict(collections.Counter)
    for d in dets:
        for c, i in zip(d["cls"], d["id"]):
            votes[int(i)][role_of_cls[int(c)]] += 1
    return {i: v.most_common(1)[0][0] for i, v in votes.items()}


def best_ball_boxes(dets, role):
    """The most confident box of a ball track in each frame, or None."""
    out = []
    for d in dets:
        k = [j for j, i in enumerate(d["id"]) if role.get(int(i)) == "ball"]
        out.append(d["xyxy"][max(k, key=lambda j: d["conf"][j])] if k else None)
    return out


def fill_ball_track(boxes, max_gap, max_step):
    """Bridge the frames where the detector loses the ball.

    boxes: one ball box (xyxy) or None per frame. max_gap: longest run of missing frames to bridge.
    max_step: px per frame the ball can plausibly move. First, a detection that jumps more than that
    from its nearest detections on both sides is dropped as a false positive (a head, a boot, a
    penalty spot). Then each gap of at most max_gap frames between two kept detections, whose jump
    per frame is also plausible, is filled by linear interpolation. Returns (boxes, filled) where
    filled marks the interpolated frames.
    """
    n = len(boxes)
    kept = [None if b is None else np.asarray(b, np.float64) for b in boxes]
    centre = lambda b: (b[:2] + b[2:]) / 2  # noqa: E731
    det = [f for f in range(n) if kept[f] is not None]
    drop = []
    for j, f in enumerate(det):
        ok = [np.linalg.norm(centre(kept[f]) - centre(kept[g])) <= max_step * abs(f - g)
              for g in (det[j - 1] if j > 0 else None, det[j + 1] if j + 1 < len(det) else None) if g is not None]
        if ok and not any(ok):
            drop.append(f)
    for f in drop:
        kept[f] = None
    filled = np.zeros(n, bool)
    det = [f for f in range(n) if kept[f] is not None]
    for a, b in zip(det, det[1:]):
        if 0 < b - a - 1 <= max_gap and np.linalg.norm(centre(kept[b]) - centre(kept[a])) <= max_step * (b - a):
            for f in range(a + 1, b):
                w = (f - a) / (b - a)
                kept[f] = (1 - w) * kept[a] + w * kept[b]
                filled[f] = True
    return kept, filled


def grass_lab(frame):
    """This frame's pitch colour: median Lab of the green-ish pixels in its lower 60 %, where grass
    dominates. Masking against it, not a fixed green hue range, keeps green kits: on 240 labelled
    players (red vs lime kits) the fixed hue mask deleted most lime shirts; this split them 120/120
    with 100 % agreement to a red/lime pixel rule."""
    low = frame[int(frame.shape[0] * 0.4):]
    hue = cv2.cvtColor(low, cv2.COLOR_BGR2HSV)[..., 0]
    return np.median(cv2.cvtColor(low, cv2.COLOR_BGR2LAB)[(hue >= 30) & (hue <= 90)], axis=0)


def jersey_colour(frame, box, grass, min_distance=20):
    """Median Lab colour of the torso with pixels within `min_distance` (Lab) of this frame's grass
    colour masked out. None if too little is left."""
    x1, y1, x2, y2 = box.astype(int)
    w, h = x2 - x1, y2 - y1
    crop = frame[max(y1 + int(0.15 * h), 0):max(y1 + int(0.5 * h), 0), max(x1 + int(0.2 * w), 0):max(x2 - int(0.2 * w), 0)]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    keep = lab[np.linalg.norm(lab - grass, axis=1) > min_distance]
    return np.median(keep, axis=0) if len(keep) >= 10 else None


def assign_teams(video, dets, player_cls, seed=0):
    """KMeans (k=2) on jersey colours of all player boxes in the clip, then one team per track
    by majority vote across frames (per-frame labels flicker). Referees and goalkeepers are
    excluded by predicted class. Returns ({track_id: team}, agreement) where agreement is the
    mean share of a track's frames that agree with its voted team."""
    from sklearn.cluster import KMeans

    ids, colours = [], []
    for frame, d in zip(frames(video), dets):
        grass = grass_lab(frame)
        for box, c, i in zip(d["xyxy"], d["cls"], d["id"]):
            if c == player_cls and (col := jersey_colour(frame, box, grass)) is not None:
                ids.append(int(i)); colours.append(col)
    labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(np.array(colours))
    votes = collections.defaultdict(collections.Counter)
    for i, lab in zip(ids, labels):
        votes[i][int(lab)] += 1
    teams = {i: v.most_common(1)[0][0] for i, v in votes.items()}
    agreement = float(np.mean([v.most_common(1)[0][1] / sum(v.values()) for v in votes.values()]))
    return teams, agreement


def render(video, dets, teams, out_path, names, trail=30):
    """Annotated MP4: players by team colour, referees/keepers by role, ball with a fading trail."""
    fps = video_fps(video)
    writer, history = None, collections.deque(maxlen=trail)
    ball_cls = names.index("ball")
    for frame, d in zip(frames(video), dets):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        ball = None
        for box, c, i in zip(d["xyxy"], d["cls"], d["id"]):
            x1, y1, x2, y2 = box.astype(int)
            name = names[c]
            if name == "ball":
                continue
            colour = {"player": TEAM_BGR.get(teams.get(int(i)), UNKNOWN_BGR),
                      "referee": REFEREE_BGR, "goalkeeper": KEEPER_BGR}[name]
            cv2.ellipse(frame, ((x1 + x2) // 2, y2), (max((x2 - x1) // 2, 6), max((x2 - x1) // 6, 2)),
                        0, -45, 235, colour, 2, cv2.LINE_AA)
        balls = d["cls"] == ball_cls
        if balls.any():
            bx = d["xyxy"][balls][d["conf"][balls].argmax()]
            ball = (int((bx[0] + bx[2]) / 2), int((bx[1] + bx[3]) / 2))
        history.append(ball)
        pts = list(history)
        for k in range(1, len(pts)):
            if pts[k - 1] is not None and pts[k] is not None:
                alpha = k / len(pts)
                cv2.line(frame, pts[k - 1], pts[k], BALL_BGR, max(1, int(3 * alpha)), cv2.LINE_AA)
        if ball is not None:
            cv2.circle(frame, ball, 7, BALL_BGR, 2, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
