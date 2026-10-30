"""Episode recording: per-tick capture into a ring buffer, 10 s window cutting, and writing
every raw stream of a clip (DATASET_SCHEMA.md). The clip folder must not exist beforehand."""
import collections
import math
import time

import cv2
import numpy as np
import pandas as pd

from camera import decode_instance, intrinsics, match_boxes, project
from geometry import to_ego_frame
from io_utils import make_new_dir, pack_ragged, write_json, write_table
from rig import STREAMS

GT_RADIUS = 50.0
NO_VALUE = 99.0


# ---------------------------------------------------------------- pure helpers

def window_indices(frames, anchor_frame, pre_frames, n_frames=100):
    """Index range [start, start + n) in `frames` so that anchor_frame sits at pre_frames.

    frames: consecutive world frame ids held in the buffer. Returns start or None."""
    if anchor_frame not in frames:
        return None
    start = frames.index(anchor_frame) - pre_frames
    if start < 0 or start + n_frames > len(frames):
        return None
    return start


def actor_type(blueprint_id, base_type=""):
    if blueprint_id.startswith("walker"):
        return "walker"
    return base_type if base_type in ("car", "van", "truck", "bus", "motorcycle", "bicycle") else "car"


def radar_summary(det, radar_z):
    """Nearest return above the road surface; closing speed and its TTC (v1 logic)."""
    det = det[radar_z + det[:, 3] * np.sin(det[:, 2]) > 0.3] if len(det) else det
    if len(det) == 0:
        return 0, NO_VALUE, 0.0, NO_VALUE
    vel, depth = det[:, 0], det[:, 3]
    i = int(np.argmin(depth))
    closing = np.maximum(-vel, 0.0)
    with np.errstate(divide="ignore"):
        ttc = np.where(closing > 0.5, depth / np.maximum(closing, 1e-3), NO_VALUE)
    return len(det), float(depth[i]), float(closing[i]), float(min(ttc.min(), NO_VALUE))


def lidar_lane_gap(pts, half_len, sensor_z):
    pts = pts.astype(np.float32)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    lane = (x > half_len) & (np.abs(y) < 1.2) & (z > -(sensor_z - 0.3)) & (z < 0.5)
    return (min(float(x[lane].min() - half_len), NO_VALUE) if lane.any() else NO_VALUE), int(lane.sum())


def depth_lane_gap(depth_cm, cam_height, cam_to_bumper):
    """Gap to the nearest non-road pixel in the ego lane; ignores the ego bonnet (v1 fix)."""
    depth = depth_cm.astype(np.float32) / 100.0
    h, w = depth.shape
    f = w / 2.0
    v, u = np.mgrid[0:h, 0:w]
    height = cam_height - depth * (v - h / 2.0) / f
    lateral = depth * (u - w / 2.0) / f
    lane = (np.abs(lateral) < 1.2) & (height > 0.3) & (height < cam_height + 0.5) \
        & (depth > cam_to_bumper + 0.05) & (depth < 200.0)
    if not lane.any():
        return NO_VALUE, 0
    return min(max(float(depth[lane].min()) - cam_to_bumper, 0.0), NO_VALUE), int(lane.sum())


# ---------------------------------------------------------------- episode

class Episode:
    """Ticks the world and keeps the last `maxlen` frames of decoded data."""

    def __init__(self, world, rig, ego, registry, maxlen=200):
        self.world, self.rig, self.ego, self.registry = world, rig, ego, registry
        self.buffer = collections.deque(maxlen=maxlen)
        r = rig.cfg["rgb"]
        self.width, self.height = r["width"], r["height"]
        self.K = intrinsics(self.width, self.height, r["fov"])
        self.ticks = 0
        self.spawn_frame = None

    @property
    def frames(self):
        return [b["world_frame"] for b in self.buffer]

    def tick(self):
        frame = self.world.tick()
        if self.spawn_frame is None:
            self.spawn_frame = frame
        meas = self.rig.read(frame)
        dec = self.rig.decode(meas)
        snap = self.world.get_snapshot()
        ego_snap = snap.find(self.ego.id)
        etf, ev = ego_snap.get_transform(), ego_snap.get_velocity()
        ctrl = self.ego.get_control()
        ego_row = {
            "world_frame": frame, "sim_time": snap.timestamp.elapsed_seconds,
            "x": etf.location.x, "y": etf.location.y, "z": etf.location.z,
            "roll": etf.rotation.roll, "pitch": etf.rotation.pitch, "yaw": etf.rotation.yaw,
            "vx": ev.x, "vy": ev.y, "vz": ev.z, "speed": math.sqrt(ev.x ** 2 + ev.y ** 2 + ev.z ** 2),
            **dec["imu"], **dec["gnss"],
            "throttle": ctrl.throttle, "steer": ctrl.steer, "brake": ctrl.brake,
            "hand_brake": int(ctrl.hand_brake), "reverse": int(ctrl.reverse), "gear": ctrl.gear,
        }
        # ground-truth actors within GT_RADIUS (world frame), plus 2D boxes
        w2c = self.rig.camera_world_matrix_inv(snap)
        rows, cands = [], []
        for aid, info in self.registry.items():
            if aid == self.ego.id:
                continue
            s = snap.find(aid)
            if s is None:
                continue
            tf, v = s.get_transform(), s.get_velocity()
            dist = math.hypot(tf.location.x - etf.location.x, tf.location.y - etf.location.y)
            if dist > GT_RADIUS:
                continue
            row = {"actor_id": aid, "role": info["role"], "type": info["type"],
                   "blueprint": info["blueprint"],
                   "x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                   "roll": tf.rotation.roll, "pitch": tf.rotation.pitch, "yaw": tf.rotation.yaw,
                   "vx": v.x, "vy": v.y, "vz": v.z, **info["bb"]}
            rows.append(row)
            verts = np.array([[p.x, p.y, p.z] for p in info["bbox"].get_world_vertices(tf)])
            uv, z = project(verts, w2c, self.K)
            cands.append({"actor_id": aid, "type": info["type"], "uv": uv, "z": z, "dist": dist})
        tag, inst = decode_instance(dec.pop("instance"))
        boxes = match_boxes(tag, inst, cands, self.width, self.height)
        for row in rows:
            b = boxes.get(row["actor_id"])
            row["box_x1"], row["box_y1"], row["box_x2"], row["box_y2"] = b[:4] if b else (np.nan,) * 4
            row["visible_px"] = b[4] if b else 0
            row["proj_area"] = b[5] if b else 0.0
        self.buffer.append({"world_frame": frame, "sim_time": ego_row["sim_time"], "ego": ego_row,
                            "actors": rows, **dec})
        self.ticks += 1
        return self.buffer[-1]


# ---------------------------------------------------------------- writing

def write_clip(clip_dir, window, collisions, rig, meta, fps=10):
    """Write all files of one clip. Returns (ego_df, actors_df, collisions_in_clip, bytes)."""
    t0 = time.time()
    make_new_dir(clip_dir)
    n = len(window)
    first_frame = window[0]["world_frame"]

    # frames.parquet: per-stream frame ids and timestamps
    frows = []
    for i, w in enumerate(window):
        r = {"frame_idx": i, "world_frame": w["world_frame"], "sim_time": w["sim_time"]}
        for s in STREAMS:
            r[f"{s}_frame"], r[f"{s}_time"] = w["stream_frames"][s]
        frows.append(r)
    write_table(pd.DataFrame(frows), clip_dir / "frames")

    # rgb.mp4
    h, w_ = window[0]["rgb"].shape[:2]
    vw = cv2.VideoWriter(str(clip_dir / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_, h))
    for f in window:
        vw.write(f["rgb"])
    vw.release()

    np.savez_compressed(clip_dir / "depth.npz", depth_cm=np.stack([f["depth_cm"] for f in window]))
    pts, off = pack_ragged([f["lidar"] for f in window], np.float16, 4)
    np.savez_compressed(clip_dir / "lidar.npz", points=pts, offsets=off)
    det, doff = pack_ragged([f["radar"] for f in window], np.float32, 4)
    np.savez_compressed(clip_dir / "radar.npz", detections=det, offsets=doff)

    # ego.parquet
    m = rig.mount
    ego_rows = [{"frame_idx": i, **f["ego"], "bb_ex": m["bb_ex"], "bb_ey": m["bb_ey"],
                 "bb_cx": m["bb_cx"], "bb_cy": m["bb_cy"]} for i, f in enumerate(window)]
    ego_df = pd.DataFrame(ego_rows)
    write_table(ego_df, clip_dir / "ego")

    # actors.parquet (+ ego-frame columns)
    arows = []
    for i, f in enumerate(window):
        e = f["ego"]
        for a in f["actors"]:
            fx, ry = to_ego_frame(e["x"], e["y"], e["yaw"], a["x"], a["y"])
            vfx, vry = to_ego_frame(0.0, 0.0, e["yaw"], a["vx"] - e["vx"], a["vy"] - e["vy"])
            arows.append({"frame_idx": i, **a, "rel_x": float(fx), "rel_y": float(ry),
                          "rel_z": a["z"] - e["z"], "rel_vx": float(vfx), "rel_vy": float(vry),
                          "dist": math.hypot(a["x"] - e["x"], a["y"] - e["y"])})
    actors_df = pd.DataFrame(arows, columns=None if arows else [
        "frame_idx", "actor_id", "role", "type", "blueprint", "x", "y", "z", "roll", "pitch", "yaw",
        "vx", "vy", "vz", "bb_cx", "bb_cy", "bb_cz", "bb_ex", "bb_ey", "bb_ez", "box_x1", "box_y1",
        "box_x2", "box_y2", "visible_px", "proj_area", "rel_x", "rel_y", "rel_z", "rel_vx", "rel_vy",
        "dist"])
    if len(actors_df):
        actors_df["occluded"] = (actors_df["visible_px"] / actors_df["proj_area"].clip(lower=1.0)) < 0.1

    # collisions.parquet (events inside the clip only; frame_idx relative to the clip)
    clip_coll = []
    for c in collisions:
        idx = c["world_frame"] - first_frame
        if 0 <= idx < n:
            clip_coll.append({"frame_idx": int(idx), **c, "impulse_x": c["impulse"][0],
                              "impulse_y": c["impulse"][1], "impulse_z": c["impulse"][2]})
    cdf = pd.DataFrame(clip_coll, columns=["frame_idx", "world_frame", "sim_time", "other_id",
                                           "other_blueprint", "impulse_x", "impulse_y", "impulse_z",
                                           "intensity"] if not clip_coll else None)
    if len(cdf):
        cdf = cdf.drop(columns=["impulse"])
    write_table(cdf, clip_dir / "collisions")

    # sensors_summary.parquet (derived; not raw)
    srows = []
    for i, f in enumerate(window):
        rn, rd, rc, rt = radar_summary(f["radar"], rig.radar_z)
        lg, lp = lidar_lane_gap(f["lidar"], m["bb_ex"], m["roof_z"] + rig.cfg["lidar"]["mount"]["z_above_roof"])
        dg, dp = depth_lane_gap(f["depth_cm"], m["roof_z"] + rig.cfg["rgb"]["mount"]["z_above_roof"],
                                m["bb_ex"] - rig.cfg["rgb"]["mount"]["x_from_bb_center"])
        srows.append({"frame_idx": i, "radar_n": rn, "radar_min_depth": rd, "radar_closing_speed": rc,
                      "radar_min_ttc": rt, "lidar_front_gap": lg, "lidar_front_points": lp,
                      "lidar_points": len(f["lidar"]), "depth_front_gap": dg, "depth_front_pixels": dp})
    write_table(pd.DataFrame(srows), clip_dir / "sensors_summary")

    meta = dict(meta, write_s=round(time.time() - t0, 2))
    write_json(clip_dir / "meta.json", meta)
    return ego_df, actors_df, clip_coll


def finish_clip(clip_dir, actors_df, gaps, ttcs, labels):
    """Add measured gap/TTC columns to actors, then write actors + labels."""
    if len(actors_df):
        actors_df = actors_df.copy()
        actors_df["gap"] = np.where(np.isinf(gaps), np.nan, gaps)
        actors_df["ttc"] = np.where(np.isinf(ttcs), np.nan, ttcs)
    write_table(actors_df, clip_dir / "actors")
    write_json(clip_dir / "labels.json", labels)
