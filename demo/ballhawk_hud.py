"""Broadcast-style tactical video: HUD, team shapes, IDs, a styled 2D pitch map, speeds, distance,
possession and the offside line. Built on the tracks (ballhawk_video.track), teams
(ballhawk_video.assign_teams) and per-frame homographies (ballhawk_pitch.frame_homography).

Pitch positions come from automatic calibration (about 2.8 m median error on the test split), so
each track's position is smoothed before speeds and distances are derived from it, and physically
impossible steps are dropped from distance covered.
"""

import collections

import cv2
import numpy as np

import ballhawk_pitch as bp

# BGR. Team A blue, team B green, goalkeeper magenta, referee yellow, ball white, offside red.
TEAM = {0: (214, 120, 42), 1: (80, 190, 70)}
KEEPER, REFEREE, BALL, OFFSIDE, UNKNOWN = (220, 60, 200), (0, 200, 255), (255, 255, 255), (60, 60, 235), (170, 170, 170)
INK, MUTED, BG, PANEL = (235, 235, 235), (150, 160, 155), (18, 22, 20), (26, 32, 29)
FONT = cv2.FONT_HERSHEY_SIMPLEX
DRAW_W = 1320  # width of the broadcast area in the composed video; smaller frames are upscaled before drawing


class PitchState:
    """Smoothed pitch position, speed and distance covered per track, plus possession counts."""

    def __init__(self, fps, alpha=0.3, speed_window_s=0.5, max_speed_kmh=36.0):
        self.fps, self.alpha = fps, alpha
        self.window = max(2, int(round(speed_window_s * fps)))
        self.max_step = max_speed_kmh / 3.6 / fps  # metres per frame a sprinter cannot exceed
        self.pos, self.hist = {}, collections.defaultdict(lambda: collections.deque(maxlen=self.window + 1))
        self.dist = collections.Counter()
        self.possession = collections.Counter()

    def update(self, ids, pitch_xy):
        for i, p in zip(ids, pitch_xy):
            prev = self.pos.get(i)
            s = p if prev is None else self.alpha * p + (1 - self.alpha) * prev
            if prev is not None:
                step = float(np.linalg.norm(s - prev))
                if step <= self.max_step:
                    self.dist[i] += step
            self.pos[i] = s
            self.hist[i].append(s)

    def speed_kmh(self, i):
        h = self.hist.get(i)
        if not h or len(h) <= self.window:
            return None
        return float(np.linalg.norm(h[-1] - h[0]) / (self.window / self.fps) * 3.6)


def _label(img, text, org, colour, scale=0.45, thick=1, bg=None):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + 3), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def _hull(img, pts, colour, alpha=0.18):
    pts = np.asarray(pts, np.int32)
    if len(pts) < 3:
        return img
    hull = cv2.convexHull(pts)
    over = img.copy()
    cv2.fillPoly(over, [hull], colour, cv2.LINE_AA)
    img = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    cv2.polylines(img, [hull], True, colour, 2, cv2.LINE_AA)
    return img


def pitch_panel(state, roles, teams, ball_xy, line_x, size=(600, 744)):
    """Right-hand panel: top distance, dark striped 2D pitch with hulls, IDs, speeds, ball and offside line, legend."""
    W, H = size
    panel = np.full((H, W, 3), PANEL, np.uint8)
    scale = int((W - 30) / (bp.LENGTH + 8))  # integer px per metre (minimap line widths need ints)
    mm, to_px = bp.minimap(scale=scale, margin=4, grass=(34, 58, 40), line=(225, 230, 225))
    # Mowing stripes on grass pixels only, so lines stay crisp.
    grass = np.all(mm == (34, 58, 40), axis=2)
    for k in range(0, 12, 2):
        x0, x1 = to_px(k * bp.LENGTH / 12, 0)[0], to_px((k + 1) * bp.LENGTH / 12, 0)[0]
        band = np.zeros_like(grass); band[:, x0:x1] = True
        mm[grass & band] = (40, 66, 46)
    for t in (0, 1):
        pts = [to_px(*state.pos[i]) for i, r in roles.items() if r == "player" and teams.get(i) == t and i in state.pos]
        mm = _hull(mm, pts, TEAM[t], 0.22)
    if line_x is not None:
        cv2.line(mm, to_px(line_x, 0), to_px(line_x, bp.WIDTH), OFFSIDE, 2, cv2.LINE_AA)
    for i, r in roles.items():
        if i not in state.pos:
            continue
        colour = {"keeper": KEEPER, "referee": REFEREE}.get(r, TEAM.get(teams.get(i), UNKNOWN))
        c = to_px(*state.pos[i])
        cv2.circle(mm, c, 9, colour, -1, cv2.LINE_AA)
        cv2.circle(mm, c, 9, (245, 245, 245), 1, cv2.LINE_AA)
        _label(mm, str(i % 100), (c[0] - 6, c[1] + 4), (255, 255, 255), 0.32)
        v = state.speed_kmh(i)
        if v is not None and v <= 40 and r == "player":  # faster than any sprinter means a calibration error, not a speed
            _label(mm, f"{v:.0f}km/h", (c[0] + 11, c[1] - 6), INK, 0.32)
    if ball_xy is not None:
        cv2.circle(mm, to_px(*ball_xy), 5, BALL, -1, cv2.LINE_AA)
    mh = mm.shape[0]
    y0 = (H - mh) // 2 + 20
    panel[y0:y0 + mh, 20:20 + mm.shape[1]] = mm
    _label(panel, "TACTICAL 2D PITCH MAP (105 m x 68 m)", (20, y0 - 12), INK, 0.5)
    # Top distance
    top = [(i, d) for i, d in state.dist.most_common() if roles.get(i) == "player" and d > 0][:3]
    if top:  # distance needs video; a single frame has none to show
        cv2.rectangle(panel, (20, 18), (230, 40 + 22 * len(top) + 6), (60, 70, 64), 1)
        _label(panel, "TOP DISTANCE", (28, 34), (0, 200, 255), 0.42)
    for k, (i, d) in enumerate(top):
        cv2.circle(panel, (32, 52 + 22 * k), 4, TEAM.get(teams.get(i), UNKNOWN), -1, cv2.LINE_AA)
        _label(panel, f"#{i}  {d:.1f} m", (44, 57 + 22 * k), INK, 0.42)
    # Legend
    x = 20
    for name, col in (("Team A", TEAM[0]), ("Team B", TEAM[1]), ("Goalkeeper", KEEPER), ("Referee", REFEREE), ("Offside", OFFSIDE)):
        cv2.circle(panel, (x + 6, H - 22), 6, col, -1, cv2.LINE_AA)
        _label(panel, name, (x + 16, H - 17), INK, 0.42)
        x += 22 + cv2.getTextSize(name, FONT, 0.42, 1)[0][0] + 18
    return panel


def hud_bar(frame_idx, state, current, spreads, line_x, size=(1920, 120)):
    """Top bar: frame number, possession share and current holder, team spreads, offside line."""
    W, H = size
    bar = np.full((H, W, 3), BG, np.uint8)
    _label(bar, f"FRAME {frame_idx:04d}", (24, 50), (0, 200, 255), 0.8, 2)
    total = sum(state.possession.values())
    a = state.possession[0] / total if total else 0.5
    x0, x1, y = 620, 1080, 58
    _label(bar, f"TEAM A {100 * a:.1f}%", (x0, 36), TEAM[0], 0.62, 2)
    _label(bar, f"TEAM B {100 * (1 - a):.1f}%", (x1 - 175, 36), TEAM[1], 0.62, 2)
    cv2.rectangle(bar, (x0, y - 10), (x1, y + 8), TEAM[1], -1)
    cv2.rectangle(bar, (x0, y - 10), (x0 + int(a * (x1 - x0)), y + 8), TEAM[0], -1)
    holder = {0: "TEAM A", 1: "TEAM B"}.get(current, "UNAVAILABLE")
    _label(bar, f"Est. possession: {holder}", (x0, 90), INK, 0.52)
    _label(bar, f"Offside line: {line_x:.1f} m" if line_x is not None else "Offside line: n/a", (x0, 112), (60, 90, 235), 0.52)
    for k, t in enumerate((0, 1)):
        s = spreads.get(t)
        txt = f"Team {'AB'[t]} spread: {s[0]:.1f} m x {s[1]:.1f} m" if s else f"Team {'AB'[t]} spread: n/a"
        _label(bar, txt, (1300, 44 + 34 * k), TEAM[t], 0.62, 2)
    _label(bar, "Positions from automatic calibration (~2.8 m median error); speeds smoothed", (1300, 112), MUTED, 0.42)
    return bar


def frame_overlay(frame, boxes, ids, roles, teams, ball_hist, line_x, H):
    """Broadcast frame: team hulls on the feet, role-coloured boxes with IDs, ball arrow, offside band.
    Frames narrower than DRAW_W (e.g. 360p downloads) are upscaled first, so labels and lines are drawn
    at display size instead of being blown up with the frame."""
    if frame.shape[1] < DRAW_W:
        u = DRAW_W / frame.shape[1]
        frame = cv2.resize(frame, (DRAW_W, int(round(frame.shape[0] * u))), interpolation=cv2.INTER_CUBIC)
        boxes = np.asarray(boxes, np.float64) * u
        ball_hist = [None if p is None else (p[0] * u, p[1] * u) for p in ball_hist]
        if H is not None:  # image -> pitch for the upscaled coordinates
            H = H @ np.diag([1 / u, 1 / u, 1.0])
    feet = bp.foot_points(boxes)
    for t in (0, 1):
        pts = [feet[k] for k, i in enumerate(ids) if roles.get(i) == "player" and teams.get(i) == t]
        frame = _hull(frame, pts, TEAM[t], 0.16)
    if line_x is not None and H is not None:
        frame = bp.draw_offside_line(frame, H, line_x, colour=OFFSIDE, band_m=0.4, alpha=0.45)
    for (x1, y1, x2, y2), i in zip(boxes.astype(int), ids):
        r = roles.get(i)
        if r == "ball":
            continue
        colour = {"keeper": KEEPER, "referee": REFEREE}.get(r, TEAM.get(teams.get(i), UNKNOWN))
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.ellipse(frame, ((x1 + x2) // 2, y2), (max((x2 - x1) // 2 + 4, 8), max((x2 - x1) // 6, 3)), 0, 0, 360, colour, 2, cv2.LINE_AA)
        _label(frame, f"#{i}", (x1, y1 - 5), (255, 255, 255), 0.45, 1, bg=colour)
    pts = [p for p in ball_hist if p is not None]
    if pts:
        bx, by = pts[-1]
        if len(pts) >= 3:
            v = np.subtract(pts[-1], pts[0]) / (len(pts) - 1)
            if np.linalg.norm(v) > 1.5:
                tip = (int(bx + 8 * v[0]), int(by + 8 * v[1]))
                cv2.arrowedLine(frame, (int(bx), int(by)), tip, (0, 215, 255), 4, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(frame, (int(bx), int(by)), 10, (0, 215, 255), 2, cv2.LINE_AA)
    return frame


def compose(frame, panel, bar, left_w=1320):
    """Always bar height + panel height by bar width (1920 x 864 with the defaults): the frame is fitted
    inside a left_w x panel-height box and centred, whatever its aspect ratio. A fixed size matters because
    the video writer takes raw frames; a taller frame (e.g. a non-16:9 screen recording) would tear."""
    box_h = panel.shape[0]
    s = min(left_w / frame.shape[1], box_h / frame.shape[0])
    fw, fh = max(1, int(frame.shape[1] * s)), max(1, int(frame.shape[0] * s))
    body = np.full((box_h, left_w + panel.shape[1], 3), BG, np.uint8)
    x0, y0 = (left_w - fw) // 2, (box_h - fh) // 2
    body[y0:y0 + fh, x0:x0 + fw] = cv2.resize(frame, (fw, fh))
    body[:, left_w:] = panel
    return np.vstack([bar, body])
