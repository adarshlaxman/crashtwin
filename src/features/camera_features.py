"""Phase 2 step 3: per-frame camera features from tracks (identical for real and simulated video).
No simulator ground truth is used here.

Per frame (10 Hz): top K=5 tracks ranked by a risk score (box area x closeness to the image's
lower centre), each described by 16 numbers; plus 3 global numbers -> F = 83.
Sim-only sensor features (E4) are built separately by `sensor_features`."""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[3])   # project folder (contains crashtwin/)
TRACKS = ROOT / "outputs" / "processed" / "tracks"
CLASSES = [0, 1, 2, 3, 5, 7]          # person, bicycle, car, motorcycle, bus, truck
K = 5
PER_TRACK = 16
F_CAM = K * PER_TRACK + 3
F_SENS = 8


def _track_kinematics(t):
    """Add area, expansion rate (1/s), image TTC and age to one track's rows (sorted by frame)."""
    t = t.sort_values("frame_idx").copy()
    w, h = (t.x2 - t.x1).clip(lower=1e-4), (t.y2 - t.y1).clip(lower=1e-4)
    t["w"], t["h"], t["area"] = w, h, w * h
    t["cx"] = (t.x1 + t.x2) / 2
    size = np.sqrt(t["area"].to_numpy())
    tt = t["t"].to_numpy()
    exp = np.zeros(len(t))
    for i in range(len(t)):
        j = max(0, i - 2)                      # 0.2 s baseline, smoother than one frame
        if i > j and tt[i] > tt[j]:
            exp[i] = (math.log(size[i]) - math.log(size[j])) / (tt[i] - tt[j])
    t["expansion"] = exp
    t["ttc_img"] = np.where(exp > 0.05, 1.0 / np.maximum(exp, 1e-6), 10.0).clip(0, 10)
    t["darea"] = np.r_[0.0, np.diff(t["area"].to_numpy())] * 10.0
    t["age"] = np.arange(len(t))
    t["risk"] = t["area"] * np.exp(-3.0 * (t["cx"] - 0.5).abs()) * t["y2"]
    return t


def camera_features(track_df):
    """track parquet -> (times [T], X [T, F_CAM])."""
    frames = track_df[track_df.track_id < 0].sort_values("frame_idx")
    T = int(frames.frame_idx.max()) + 1 if len(frames) else 0
    X = np.zeros((T, F_CAM), np.float32)
    times = np.zeros(T, np.float32)
    times[frames.frame_idx.to_numpy()] = frames.t.to_numpy()
    flow = np.zeros(T, np.float32)
    flow[frames.frame_idx.to_numpy()] = frames.flow.fillna(0).to_numpy()
    tr = track_df[track_df.track_id >= 0]
    if len(tr):
        tr = pd.concat([_track_kinematics(g) for _, g in tr.groupby("track_id")])
        for f, g in tr.groupby("frame_idx"):
            g = g.nlargest(K, "risk")
            for s, r in enumerate(g.itertuples()):
                onehot = [float(r.cls == c) for c in CLASSES]
                X[f, s * PER_TRACK:(s + 1) * PER_TRACK] = [
                    1.0, r.cx, r.y2, r.w, r.h, r.area, r.darea, np.clip(r.expansion, -5, 5),
                    r.ttc_img / 10.0, *onehot, min(r.age, 50) / 50.0]
            X[f, K * PER_TRACK] = min(len(tr[tr.frame_idx == f]), 20) / 20.0
            X[f, K * PER_TRACK + 2] = float(np.clip(g.expansion.max(), -5, 5)) if len(g) else 0.0
    X[:, K * PER_TRACK + 1] = np.clip(flow / 5.0, 0, 5)
    return times, X


def sensor_features(clip_dir):
    """Sim-only per-frame sensor features (E4): radar, LiDAR, depth, ego speed/IMU. [100, F_SENS]"""
    import sys
    sys.path.insert(0, str(ROOT / "crashtwin" / "src" / "generator"))
    from io_utils import local_path, read_table
    clip_dir = local_path(clip_dir)
    s = read_table(Path(clip_dir) / "sensors_summary").sort_values("frame_idx")
    e = read_table(Path(clip_dir) / "ego").sort_values("frame_idx")
    cols = [s.radar_min_depth / 100, s.radar_closing_speed.clip(0, 30) / 30, s.radar_min_ttc.clip(0, 10) / 10,
            s.lidar_front_gap.clip(0, 99) / 99, s.depth_front_gap.clip(0, 99) / 99, e.speed / 30,
            e.acc_x.clip(-20, 20) / 20, e.gyro_z.clip(-2, 2) / 2]
    return np.stack([np.asarray(c, np.float32) for c in cols], axis=1)


def build(index_df, dataset, with_sensors=False):
    """Stack features for every clip that has a track file. Returns a dict of arrays + a meta DataFrame."""
    items, Xs, Ts, Ss = [], [], [], []
    for row in index_df.to_dict("records"):
        p = TRACKS / dataset / f"{row['clip_id']}.parquet"
        if not p.exists():
            continue
        times, X = camera_features(pd.read_parquet(p))
        if len(X) == 0:
            continue
        Xs.append(X)
        Ts.append(times)
        if with_sensors:
            Ss.append(sensor_features(Path(row["clip_dir"]))[:len(X)])
        items.append(row)
    T = max(len(x) for x in Xs)
    N = len(Xs)
    X = np.zeros((N, T, F_CAM), np.float32)
    mask = np.zeros((N, T), bool)
    times = np.zeros((N, T), np.float32)
    S = np.zeros((N, T, F_SENS), np.float32) if with_sensors else None
    for i, (x, t) in enumerate(zip(Xs, Ts)):
        X[i, :len(x)], mask[i, :len(x)], times[i, :len(t)] = x, True, t
        if with_sensors:
            S[i, :len(Ss[i])] = Ss[i]
    meta = pd.DataFrame(items)
    ev = meta["event_time_s"].astype(float).to_numpy() if "event_time_s" in meta else np.full(N, np.nan)
    y = meta["label"].astype(int).to_numpy()
    tte = np.where(y[:, None] == 1, ev[:, None] - times, np.inf).astype(np.float32)
    return {"X": X, "S": S, "mask": mask, "times": times, "y": y, "tte": tte}, meta
