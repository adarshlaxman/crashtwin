"""Synthetic clip folders for validator tests (no CARLA). The defaults produce a valid normal clip."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC / "generator"))
sys.path.insert(0, str(SRC / "validation"))

from io_utils import pack_ragged, provenance, write_table  # noqa: E402
from labels import RULE_V1  # noqa: E402

STREAMS = ("rgb", "instance", "depth", "lidar", "radar", "imu", "gnss")
W, H = 128, 72   # small video keeps tests fast; meta declares the same size


def make_clip(d, outcome="normal", collision_frame=None, drop=None, dup_frame=False, time_gap=False,
              intended="normal"):
    d = Path(d)
    d.mkdir(parents=True)
    n = 100
    wf = np.arange(1000, 1000 + n)
    st = 50.0 + np.arange(n) * 0.1
    if dup_frame:
        wf[10] = wf[9]
    if time_gap:
        st[50:] += 0.3
    frames = {"frame_idx": np.arange(n), "world_frame": wf, "sim_time": st}
    for s in STREAMS:
        frames[f"{s}_frame"], frames[f"{s}_time"] = wf, st
    write_table(pd.DataFrame(frames), d / "frames")

    vw = cv2.VideoWriter(str(d / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for i in range(n):
        vw.write(np.full((H, W, 3), i % 255, np.uint8))
    vw.release()
    depth = np.full((n, 18, 32), 2000, np.uint16)
    depth[:, :5] = 65535
    np.savez_compressed(d / "depth.npz", depth_cm=depth)
    pts, off = pack_ragged([np.random.rand(5000, 4).astype(np.float16) for _ in range(n)], np.float16, 4)
    np.savez_compressed(d / "lidar.npz", points=pts, offsets=off)
    det, doff = pack_ragged([np.random.rand(20, 4).astype(np.float32) for _ in range(n)], np.float32, 4)
    np.savez_compressed(d / "radar.npz", detections=det, offsets=doff)

    x = np.arange(n) * 0.8   # 8 m/s along x
    ego = pd.DataFrame({"frame_idx": np.arange(n), "world_frame": wf, "sim_time": st, "x": x, "y": 0.0,
                        "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "vx": 8.0, "vy": 0.0, "vz": 0.0,
                        "speed": 8.0, "acc_x": 0.0, "acc_y": 0.0, "acc_z": 9.81, "gyro_x": 0.0,
                        "gyro_y": 0.0, "gyro_z": 0.0, "compass": 0.0, "lat": 0.0, "lon": 0.0, "alt": 0.0,
                        "throttle": 0.3, "steer": 0.0, "brake": 0.0, "hand_brake": 0, "reverse": 0, "gear": 1,
                        "bb_ex": 2.3, "bb_ey": 1.0, "bb_cx": 0.0, "bb_cy": 0.0})
    write_table(ego, d / "ego")
    actors = pd.DataFrame({"frame_idx": np.arange(n), "actor_id": 7, "role": "scenario_actor", "type": "car",
                           "blueprint": "vehicle.audi.a2", "x": x + 30.0, "y": 0.0, "z": 0.0, "roll": 0.0,
                           "pitch": 0.0, "yaw": 0.0, "vx": 8.0, "vy": 0.0, "vz": 0.0, "bb_cx": 0.0, "bb_cy": 0.0,
                           "bb_cz": 0.7, "bb_ex": 2.0, "bb_ey": 0.9, "bb_ez": 0.7, "gap": 25.7, "ttc": np.nan})
    write_table(actors, d / "actors")
    coll = pd.DataFrame(columns=["frame_idx", "world_frame", "sim_time", "other_id", "other_blueprint",
                                 "impulse_x", "impulse_y", "impulse_z", "intensity"])
    if collision_frame is not None:
        coll = pd.DataFrame([{"frame_idx": collision_frame, "world_frame": int(wf[collision_frame]),
                              "sim_time": float(st[collision_frame]), "other_id": 7,
                              "other_blueprint": "vehicle.audi.a2", "impulse_x": 3000.0, "impulse_y": 0.0,
                              "impulse_z": 0.0, "intensity": 3000.0}])
    write_table(coll, d / "collisions")
    write_table(pd.DataFrame({"frame_idx": np.arange(n), "radar_n": 20}), d / "sensors_summary")

    labels = {"outcome": outcome, "rule_version": "v1", "rule": RULE_V1, "min_gap_m": 25.7, "min_ttc_s": None,
              "event_frame": collision_frame, "event_time_s": None, "partner_id": 7 if collision_frame else None,
              "partner_type": "car" if collision_frame else None, "top_actors": [], "warnings": [],
              "invalid_reason": None}
    (d / "labels.json").write_text(json.dumps(labels))
    meta = {"scenario_id": d.name, "family": "R1_car_following", "slot_id": "x_s00000", "attempt": 1, "seed": 1,
            "intended_outcome": intended, "town": "Town03", "weather": {"condition": "clear_noon", "parameters": {}},
            "matrix_version": "1.0", "rule_version": "v1", "schema_version": "1.0",
            "sensor_config": {"rgb": {"width": W, "height": H, "fov": 90}}, "provenance": provenance(),
            "scenario_params": {}}
    (d / "meta.json").write_text(json.dumps(meta))
    if drop:
        for p in d.glob(drop + "*"):
            p.unlink()
    return d
