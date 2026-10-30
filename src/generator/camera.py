"""Camera projection and instance-mask matching for 2D boxes (unit-tested without CARLA).

In CARLA 0.9.15 instance-segmentation ids are NOT actor ids (probe 2026-10-29), so each
actor's projected 3D box is matched to the instance id that dominates that region."""
import numpy as np

# CARLA 0.9.15 semantic tags (R channel of the instance image)
TAGS = {"car": (14, 15, 16), "van": (14, 15, 16), "truck": (14, 15, 16), "bus": (14, 15, 16),
        "motorcycle": (18,), "bicycle": (19,), "walker": (12,)}


def intrinsics(width, height, fov_deg):
    f = width / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def project(points_world, world_to_cam, K):
    """Nx3 world points -> (Nx2 pixels, N depths along the optical axis).

    world_to_cam: 4x4 inverse of the camera's CARLA transform matrix.
    CARLA sensor frame (x fwd, y right, z up) -> optical (x right, y down, z fwd) = (y, -z, x)."""
    p = np.c_[points_world, np.ones(len(points_world))] @ world_to_cam.T
    opt = np.stack([p[:, 1], -p[:, 2], p[:, 0]], axis=1)
    z = opt[:, 2]
    uv = (opt @ K.T)[:, :2] / np.maximum(z, 1e-6)[:, None]
    return uv, z


def decode_instance(bgra):
    """Instance image -> (semantic tag array, instance id array)."""
    tag = bgra[:, :, 2]
    inst = (bgra[:, :, 1].astype(np.int32) << 8) | bgra[:, :, 0].astype(np.int32)
    return tag, inst


def match_boxes(tag, inst, candidates, width, height, min_pixels=20):
    """Assign 2D boxes to actors.

    candidates: list of dicts {actor_id, type, uv (8x2), z (8,), dist}; nearest first.
    Returns {actor_id: (x1, y1, x2, y2, visible_px, projected_area)}."""
    used, out = set(), {}
    for c in sorted(candidates, key=lambda c: c["dist"]):
        front = c["z"] > 0.1
        if not front.any():
            continue
        uv = c["uv"][front]
        x1, y1 = np.floor(uv.min(axis=0)).astype(int)
        x2, y2 = np.ceil(uv.max(axis=0)).astype(int)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
        if x2 <= x1 or y2 <= y1:
            continue
        area = float((x2 - x1) * (y2 - y1))
        region_tag = tag[y1:y2, x1:x2]
        region_inst = inst[y1:y2, x1:x2]
        sel = np.isin(region_tag, TAGS.get(c["type"], (14, 15, 16)))
        ids, counts = np.unique(region_inst[sel], return_counts=True)
        order = np.argsort(-counts)
        chosen = next((int(ids[i]) for i in order if int(ids[i]) not in used and counts[i] >= min_pixels),
                      None)
        if chosen is None:
            out[c["actor_id"]] = (np.nan, np.nan, np.nan, np.nan, 0, area)
            continue
        used.add(chosen)
        ys, xs = np.nonzero((inst == chosen) & np.isin(tag, TAGS.get(c["type"], (14, 15, 16))))
        out[c["actor_id"]] = (float(xs.min()), float(ys.min()), float(xs.max() + 1),
                              float(ys.max() + 1), int(len(xs)), area)
    return out
