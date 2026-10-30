"""Pure geometry used by the labeller and validator (no CARLA import, unit-tested).

Frames: CARLA world frame (left-handed, x/y horizontal, metres, yaw in degrees).
Forward vector = (cos yaw, sin yaw); right vector = (-sin yaw, cos yaw).
"""
import math

import numpy as np

INF = float("inf")


def box_corners(x, y, yaw_deg, ex, ey, off_x=0.0, off_y=0.0):
    """Ground-plane footprint (4x2) of a box with half-extents ex, ey.

    (off_x, off_y) is the bounding-box centre offset in the actor frame."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    fwd, right = np.array([c, s]), np.array([-s, c])
    centre = np.array([x, y]) + fwd * off_x + right * off_y
    return np.array([centre + fwd * ex + right * ey, centre + fwd * ex - right * ey,
                     centre - fwd * ex - right * ey, centre - fwd * ex + right * ey])


def _axes(corners):
    edges = np.roll(corners, -1, axis=0) - corners
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def boxes_overlap(a, b):
    """Separating-axis test for two convex quadrilaterals (4x2 each)."""
    for axis in np.vstack([_axes(a)[:2], _axes(b)[:2]]):
        pa, pb = a @ axis, b @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


def _point_segment(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def footprint_gap(a, b):
    """Shortest distance between two footprints; 0 if they overlap."""
    if boxes_overlap(a, b):
        return 0.0
    best = INF
    for P, Q in ((a, b), (b, a)):
        for p in P:
            for i in range(4):
                best = min(best, _point_segment(p, Q[i], Q[(i + 1) % 4]))
    return best


def predicted_ttc(ego_box, ego_v, other_box, other_v, horizon=5.0, step=0.05):
    """Constant-velocity time to collision: earliest t in (0, horizon] at which the
    footprints, each translated by its own velocity, overlap.

    Returns 0 if they already overlap and INF if no overlap is predicted
    (covers non-closing, diverging and side-by-side passing actors)."""
    if boxes_overlap(ego_box, other_box):
        return 0.0
    rel_v = np.asarray(other_v, float) - np.asarray(ego_v, float)
    speed = float(np.linalg.norm(rel_v))
    if speed < 1e-3:
        return INF
    # quick reject: even moving straight at each other they cannot meet within the horizon
    if footprint_gap(ego_box, other_box) > speed * horizon:
        return INF
    ts = np.arange(1, int(round(horizon / step)) + 1) * step                 # (T,)
    moved = other_box[None, :, :] + ts[:, None, None] * rel_v[None, None, :]  # (T, 4, 2)
    overlap = np.ones(len(ts), dtype=bool)
    for axis in np.vstack([_axes(ego_box)[:2], _axes(other_box)[:2]]):
        pe = ego_box @ axis
        po = moved @ axis                                                    # (T, 4)
        overlap &= ~((pe.max() < po.min(axis=1)) | (po.max(axis=1) < pe.min()))
    hit = np.nonzero(overlap)[0]
    return float(round(ts[hit[0]], 3)) if len(hit) else INF


def to_ego_frame(ego_x, ego_y, ego_yaw_deg, x, y):
    """World (x, y) -> (forward, right) metres in the ego frame. Vectorised."""
    yaw = np.radians(ego_yaw_deg)
    dx, dy = np.asarray(x) - ego_x, np.asarray(y) - ego_y
    return dx * np.cos(yaw) + dy * np.sin(yaw), -dx * np.sin(yaw) + dy * np.cos(yaw)
