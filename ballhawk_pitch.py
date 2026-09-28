"""Pitch model and homography tools shared by notebooks 10-13 (Phases 3-5).

Coordinates are metres: x along the length from the left goal line, y across the width from the
top touchline. Keypoint ORDER follows SoccerPitchConfiguration.vertices in roboflow/sports (MIT),
which the Phase 4 keypoint dataset uses; DIMENSIONS are the official Laws of the Game values, not
that repo's approximations. Points on the outer touchlines and halfway line depend on pitch size,
which varies by stadium: 105 x 68 m is assumed (Bundesliga standard). Penalty-area geometry is
fixed by the laws, so prefer those points when anchoring a homography.

Indices 10, 11, 18, 19 are where the penalty arc meets the penalty-box front line. roboflow/sports
puts them at the goal-area y instead, but the dataset's annotators placed them on the arc: a
homography fitted to labels alone leaves 1.6-2.1 m residuals on exactly those four points with the
roboflow definition and ~0.2 m on every other point (checked 2026-09-26, notebook 12).
"""

import cv2
import numpy as np

LENGTH, WIDTH = 105.0, 68.0
PENALTY_W, PENALTY_L = 40.32, 16.5
GOAL_AREA_W, GOAL_AREA_L = 18.32, 5.5
PENALTY_SPOT, CIRCLE_R = 11.0, 9.15

_pb = ((WIDTH - PENALTY_W) / 2, (WIDTH + PENALTY_W) / 2)   # penalty box side-line y values
# Where the penalty arc (radius CIRCLE_R around the spot) meets the penalty-box front line.
_arc = float(np.sqrt(CIRCLE_R ** 2 - (PENALTY_L - PENALTY_SPOT) ** 2))
_gb = ((WIDTH - GOAL_AREA_W) / 2, (WIDTH + GOAL_AREA_W) / 2)  # goal area side-line y values
_L, _W, _cx, _cy = LENGTH, WIDTH, LENGTH / 2, WIDTH / 2

# (name, x, y) in the dataset's keypoint order (index 0 = roboflow vertex "01").
KEYPOINTS = [
    ("L corner top", 0, 0), ("L goal line x penalty box top", 0, _pb[0]),
    ("L goal line x goal area top", 0, _gb[0]), ("L goal line x goal area bottom", 0, _gb[1]),
    ("L goal line x penalty box bottom", 0, _pb[1]), ("L corner bottom", 0, _W),
    ("L goal area front top", GOAL_AREA_L, _gb[0]), ("L goal area front bottom", GOAL_AREA_L, _gb[1]),
    ("L penalty spot", PENALTY_SPOT, _cy),
    ("L penalty box front top", PENALTY_L, _pb[0]), ("L penalty box front x arc top", PENALTY_L, _cy - _arc),
    ("L penalty box front x arc bottom", PENALTY_L, _cy + _arc), ("L penalty box front bottom", PENALTY_L, _pb[1]),
    ("halfway top", _cx, 0), ("halfway x circle top", _cx, _cy - CIRCLE_R),
    ("halfway x circle bottom", _cx, _cy + CIRCLE_R), ("halfway bottom", _cx, _W),
    ("R penalty box front top", _L - PENALTY_L, _pb[0]), ("R penalty box front x arc top", _L - PENALTY_L, _cy - _arc),
    ("R penalty box front x arc bottom", _L - PENALTY_L, _cy + _arc), ("R penalty box front bottom", _L - PENALTY_L, _pb[1]),
    ("R penalty spot", _L - PENALTY_SPOT, _cy),
    ("R goal area front top", _L - GOAL_AREA_L, _gb[0]), ("R goal area front bottom", _L - GOAL_AREA_L, _gb[1]),
    ("R corner top", _L, 0), ("R goal line x penalty box top", _L, _pb[0]),
    ("R goal line x goal area top", _L, _gb[0]), ("R goal line x goal area bottom", _L, _gb[1]),
    ("R goal line x penalty box bottom", _L, _pb[1]), ("R corner bottom", _L, _W),
    ("circle left", _cx - CIRCLE_R, _cy), ("circle right", _cx + CIRCLE_R, _cy),
]
NAMES = [k[0] for k in KEYPOINTS]
PITCH_XY = np.array([(x, y) for _, x, y in KEYPOINTS], dtype=np.float64)


def fit_homography(img_pts, pitch_pts, ransac=False):
    """Image -> pitch homography. Plain least squares for hand-clicked points; RANSAC for
    detected keypoints, where outliers are expected. Returns (H, inlier_mask)."""
    method = cv2.RANSAC if ransac else 0
    H, mask = cv2.findHomography(np.asarray(img_pts, np.float64), np.asarray(pitch_pts, np.float64), method, 1.0)
    if H is None:
        raise ValueError("homography fit failed (degenerate or too few points)")
    return H, mask.ravel().astype(bool)


def project(H, pts):
    """Apply H to (N,2) points. Zero points give an empty (0,2) array (OpenCV rejects empty input)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


def reprojection_error(H, img_pts, pitch_pts):
    """Per-point error on held-out correspondences: (metres on the pitch, pixels in the image)."""
    img_pts, pitch_pts = np.asarray(img_pts, np.float64), np.asarray(pitch_pts, np.float64)
    metres = np.linalg.norm(project(H, img_pts) - pitch_pts, axis=1)
    pixels = np.linalg.norm(project(np.linalg.inv(H), pitch_pts) - img_pts, axis=1)
    return metres, pixels


def line_shift_sensitivity(img_pts, pitch_pts, line_x, far_y, px_levels=(1, 2, 3, 4, 5), trials=500, seed=0):
    """How far an offside line moves when the clicks are wrong.

    Perturbs every clicked image point at once by exactly `px` pixels in a random direction,
    refits, and measures where the line x = line_x lands at the far touchline y = far_y: the
    pitch point is projected to the image with the unperturbed fit and back with the perturbed
    one. Returns {px: (mean_cm, p95_cm)}.
    """
    rng = np.random.default_rng(seed)
    img_pts, pitch_pts = np.asarray(img_pts, np.float64), np.asarray(pitch_pts, np.float64)
    H, _ = fit_homography(img_pts, pitch_pts)
    probe_img = project(np.linalg.inv(H), [[line_x, far_y]])
    out = {}
    for px in px_levels:
        shifts = []
        for _ in range(trials):
            ang = rng.uniform(0, 2 * np.pi, len(img_pts))
            noisy = img_pts + px * np.stack([np.cos(ang), np.sin(ang)], 1)
            Hn, _ = fit_homography(noisy, pitch_pts)
            shifts.append(abs(project(Hn, probe_img)[0, 0] - line_x) * 100)
        out[px] = (float(np.mean(shifts)), float(np.percentile(shifts, 95)))
    return out


def minimap(scale=8, margin=4, grass=(34, 139, 34), line=(255, 255, 255)):
    """Top-down pitch image (BGR) at `scale` px per metre, plus a function metres -> pixel."""
    w, h = int((LENGTH + 2 * margin) * scale), int((WIDTH + 2 * margin) * scale)
    img = np.full((h, w, 3), grass, np.uint8)
    to_px = lambda x, y: (int((x + margin) * scale), int((y + margin) * scale))  # noqa: E731
    t = max(1, scale // 4)
    cv2.rectangle(img, to_px(0, 0), to_px(LENGTH, WIDTH), line, t)
    cv2.line(img, to_px(_cx, 0), to_px(_cx, WIDTH), line, t)
    cv2.circle(img, to_px(_cx, _cy), int(CIRCLE_R * scale), line, t)
    for x0, sgn in ((0, 1), (LENGTH, -1)):
        cv2.rectangle(img, to_px(x0, _pb[0]), to_px(x0 + sgn * PENALTY_L, _pb[1]), line, t)
        cv2.rectangle(img, to_px(x0, _gb[0]), to_px(x0 + sgn * GOAL_AREA_L, _gb[1]), line, t)
        cv2.circle(img, to_px(x0 + sgn * PENALTY_SPOT, _cy), max(2, scale // 3), line, -1)
    return img, to_px


def predict_keypoints(model, img, imgsz):
    """All pitch keypoints of the most confident pitch detection as a (32, 3) array of
    x, y, confidence — or None when no pitch is detected. Predict once, then threshold freely."""
    r = model.predict(img, imgsz=imgsz, verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return None
    return r.keypoints.data[int(r.boxes.conf.argmax())].cpu().numpy()


def detect_keypoints(model, img, imgsz, conf=0.5):
    """Pitch keypoints from a YOLO pose model: (indices into KEYPOINTS, image xy [N,2]) with
    keypoint confidence >= conf, from the most confident pitch detection."""
    return confident(predict_keypoints(model, img, imgsz), conf)


def confident(k, conf):
    """(indices, xy) of the rows of a predict_keypoints() array with confidence >= conf."""
    if k is None:
        return np.zeros(0, int), np.zeros((0, 2))
    keep = k[:, 2] >= conf
    return np.flatnonzero(keep), k[keep, :2]


def frame_homography(model, img, imgsz, conf=0.5, min_points=4):
    """Per-frame image -> pitch homography from detected keypoints, fitted with RANSAC (1 m
    threshold, since the fit maps into metres). Returns (H, idx, xy, inliers), or None when fewer
    than `min_points` confident keypoints or inliers exist: an uncalibrated frame is reported,
    never guessed."""
    return keypoint_homography(predict_keypoints(model, img, imgsz), conf, min_points)


def keypoint_homography(k, conf=0.5, min_points=4):
    """frame_homography() for an already predicted keypoint array."""
    idx, xy = confident(k, conf)
    if len(idx) < min_points:
        return None
    try:
        H, inliers = fit_homography(xy, PITCH_XY[idx], ransac=True)
    except ValueError:
        return None
    return (H, idx, xy, inliers) if inliers.sum() >= min_points else None


def foot_points(xyxy):
    """Bottom-centre of each box: the foot-position approximation (wrong for a running player's
    leading foot; limb-level precision is out of scope)."""
    xyxy = np.asarray(xyxy, np.float64).reshape(-1, 4)
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], 1)


def defending_goal_x(keeper_x):
    """The goal a goalkeeper defends: the goal line on their half of the pitch."""
    return 0.0 if keeper_x < LENGTH / 2 else LENGTH


def offside(defenders_x, attackers_x, goal_x, ball_x=None):
    """Law 11 along the pitch length. A player is in an offside position if in the opponents' half
    and nearer to the opponents' goal line than both the ball and the second-last opponent, so the
    line sits at whichever of those two is nearer the goal line.

    defenders_x includes the goalkeeper. Returns (line_x, flags) with flags[i] True if attacker i is
    in an offside position, or (None, all False) with fewer than two defenders (no second-last).
    """
    attackers_x = np.asarray(attackers_x, np.float64)
    depth = np.sort(np.abs(np.asarray(defenders_x, np.float64) - goal_x))
    if len(depth) < 2:
        return None, np.zeros(len(attackers_x), bool)
    line_depth = depth[1] if ball_x is None else min(depth[1], abs(ball_x - goal_x))
    line_x = goal_x + line_depth if goal_x == 0 else goal_x - line_depth
    att_depth = np.abs(attackers_x - goal_x)
    return line_x, (att_depth < line_depth) & (att_depth < LENGTH / 2)


def draw_offside_line(frame, H, line_x, colour=(52, 104, 235), band_m=0.3, alpha=0.4):
    """Translucent band at x = line_x across the full pitch width, built in pitch coordinates and
    projected into the image with the inverse homography."""
    corners = [[line_x - band_m / 2, 0], [line_x + band_m / 2, 0], [line_x + band_m / 2, WIDTH], [line_x - band_m / 2, WIDTH]]
    poly = project(np.linalg.inv(H), corners).round().astype(np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [poly], colour, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
