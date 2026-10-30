"""Phase 2 step 2: YOLO (frozen COCO yolo11s) + ByteTrack on every clip, resampled to 10 Hz.
The SAME code and weights run on real and simulated video (camera-only input for E1-E3).

Per video it writes outputs/processed/tracks/<dataset>/<clip_id>.parquet with one row per
(frame, track): frame_idx (10 Hz), t (s, video time), track_id, cls, conf, x1,y1,x2,y2 (0-1),
plus frame rows (track_id = -1) carrying the ego-motion proxy `flow` (median Farneback
magnitude, lower-centre region, 160x90). Existing outputs are skipped, never overwritten."""
import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")          # never reach the internet (weights are local)

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[3])   # project folder (contains crashtwin/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))
from io_utils import local_path  # noqa: E402
WEIGHTS = ROOT / "models" / "yolo" / "yolo11s.pt"
OUT = ROOT / "outputs" / "processed" / "tracks"
KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
HZ = 10.0
NEXAR_TRAIN_WINDOW_S = 12.0
SEED = 20261029


def sample_times(row, n_frames, fps):
    """Timestamps (s) to sample at 10 Hz. Nexar train: 12 s ending 0.5 s after the event
    (positives) or a seeded random 12 s window (negatives); everything else: the whole video."""
    dur = n_frames / fps
    if row["dataset"] == "nexar" and row.get("official_partition") == "train":
        if row["label"] == 1 and not pd.isna(row["event_time_s"]):
            end = min(float(row["event_time_s"]) + 0.5, dur)
        else:
            rng = random.Random(f"{SEED}-{row['clip_id']}")
            end = rng.uniform(min(NEXAR_TRAIN_WINDOW_S, dur), dur)
        start = max(0.0, end - NEXAR_TRAIN_WINDOW_S)
    else:
        start, end = 0.0, dur
    return np.arange(start, end - 1e-6, 1.0 / HZ)


def read_frames(path, times, fps):
    """Decode only the frames nearest to the requested timestamps."""
    want = {int(round(t * fps)): k for k, t in enumerate(times)}
    cap = cv2.VideoCapture(str(path))
    frames, i = [None] * len(times), 0
    last = max(want) if want else -1
    while i <= last:
        ok = cap.grab()
        if not ok:
            break
        if i in want:
            ok, img = cap.retrieve()
            if ok:
                frames[want[i]] = img
        i += 1
    cap.release()
    # fill any frame the decoder could not reach with the previous one
    for k in range(len(frames)):
        if frames[k] is None and k > 0:
            frames[k] = frames[k - 1]
    return [f for f in frames if f is not None]


def flow_magnitude(prev, cur):
    a = cv2.cvtColor(cv2.resize(prev, (160, 90)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(cur, (160, 90)), cv2.COLOR_BGR2GRAY)
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 9, 2, 5, 1.1, 0)
    roi = f[45:90, 40:120]
    return float(np.median(np.linalg.norm(roi, axis=2)))


def tracker_args():
    cfg = YAML.load(check_yaml("bytetrack.yaml"))
    return IterableSimpleNamespace(**cfg)


def process(model, row, out_path, batch=32):
    fps = row["fps"] if row["fps"] and row["fps"] > 0 else 10.0
    times = sample_times(row, int(row["n_frames"]), fps)
    frames = read_frames(str(local_path(row["path"])), times, fps)
    if not frames:
        return 0
    times = times[:len(frames)]
    h, w = frames[0].shape[:2]
    tracker = BYTETracker(tracker_args())   # this ultralytics version takes only args
    rows = []
    for s in range(0, len(frames), batch):
        results = model.predict(frames[s:s + batch], conf=0.1, classes=list(KEEP), verbose=False,
                                device=0, half=True, imgsz=640)
        for j, r in enumerate(results):
            k = s + j
            tracks = tracker.update(r.boxes.cpu().numpy(), frames[k])
            for t in tracks:   # x1, y1, x2, y2, track_id, score, cls, idx
                rows.append((k, times[k], int(t[4]), int(t[6]), float(t[5]),
                             t[0] / w, t[1] / h, t[2] / w, t[3] / h, np.nan))
            flow = flow_magnitude(frames[k - 1], frames[k]) if k > 0 else 0.0
            rows.append((k, times[k], -1, -1, np.nan, np.nan, np.nan, np.nan, np.nan, flow))
    df = pd.DataFrame(rows, columns=["frame_idx", "t", "track_id", "cls", "conf", "x1", "y1", "x2", "y2", "flow"])
    tmp = out_path.with_suffix(".parquet.tmp")       # atomic: a killed run never leaves a "done" file
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, out_path)
    return len(frames)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="outputs/index/<dataset>.csv or a sim index.csv")
    p.add_argument("--dataset", required=True, help="output sub-folder name, e.g. ccd, nexar, sim_full")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", default="0/1", help="k/n: process every n-th video starting at k")
    a = p.parse_args()
    idx = pd.read_csv(a.index)
    if "clip_id" not in idx:            # simulated index.csv
        idx = idx.rename(columns={"scenario_id": "clip_id"})
        idx["dataset"], idx["fps"], idx["n_frames"] = "sim", 10.0, 100
        idx["path"] = idx["path"].map(lambda d: str(Path(d) / "rgb.mp4"))
        idx["label"] = idx["outcome"].isin(["crash", "near_miss"]).astype(int)
        idx["event_time_s"] = idx["event_time_s"]
    k, n = map(int, a.shard.split("/"))
    idx = idx.iloc[k::n]
    if a.limit:
        idx = idx.head(a.limit)
    out_dir = OUT / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    torch.backends.cudnn.benchmark = True
    done = total = 0
    for i, row in enumerate(idx.to_dict("records")):
        out = out_dir / f"{row['clip_id']}.parquet"
        if out.exists():
            continue
        total += process(model, row, out)
        done += 1
        if done % 50 == 0:
            print(f"  {a.dataset}: {i + 1}/{len(idx)} videos, {total} frames", flush=True)
    print(f"{a.dataset}: processed {done} videos ({total} frames) -> {out_dir}")


if __name__ == "__main__":
    main()
