"""Visual audit sheet for a clip (VALIDATION_PLAN P4): 6 frames around the event with 2D boxes
(event partner in red), ground-truth gap/TTC, speed and the sensor lane gaps. Read-only on the
clip; writes <clip>/audit.png only."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))
from io_utils import read_table  # noqa: E402


def sheet(clip_dir, out=None):
    clip_dir = Path(clip_dir)
    labels = json.loads((clip_dir / "labels.json").read_text())
    meta = json.loads((clip_dir / "meta.json").read_text())
    actors = read_table(clip_dir / "actors")
    ego = read_table(clip_dir / "ego")
    summ = read_table(clip_dir / "sensors_summary")
    ev = labels.get("event_frame")
    centre = ev if ev is not None else meta["scenario_params"].get("pre_frames", 75)
    picks = [max(0, min(99, centre + d)) for d in (-40, -20, -10, -3, 0, 5)]
    cap = cv2.VideoCapture(str(clip_dir / "rgb.mp4"))
    frames, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in picks:
            frames[i] = fr
        i += 1
    tiles = []
    partner = labels.get("partner_id")
    for f in picks:
        img = frames.get(f, np.zeros((720, 1280, 3), np.uint8)).copy()
        rows = actors[actors["frame_idx"] == f] if len(actors) else actors
        for _, a in rows.iterrows():
            if np.isnan(a["box_x1"]):
                continue
            col = (0, 0, 255) if a["actor_id"] == partner else (0, 255, 0)
            cv2.rectangle(img, (int(a["box_x1"]), int(a["box_y1"])), (int(a["box_x2"]), int(a["box_y2"])), col, 2)
            g = "" if np.isnan(a.get("gap", np.nan)) else f"g{a['gap']:.1f}"
            t = "" if np.isnan(a.get("ttc", np.nan)) else f" t{a['ttc']:.1f}"
            cv2.putText(img, f"{a['type']} {g}{t}", (int(a["box_x1"]), max(15, int(a["box_y1"]) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        s = summ[summ["frame_idx"] == f].iloc[0]
        e = ego[ego["frame_idx"] == f].iloc[0]
        txt = (f"f{f} ({f - centre:+d})  v={e['speed'] * 3.6:.0f}km/h  lidar={s['lidar_front_gap']:.1f}"
               f"  depth={s['depth_front_gap']:.1f}  radar={s['radar_min_depth']:.1f}")
        cv2.rectangle(img, (0, 0), (1280, 40), (0, 0, 0), -1)
        cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (640, 360)))
    grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
    head = np.zeros((50, grid.shape[1], 3), np.uint8)
    cv2.putText(head, f"{meta['family']} | intended {meta['intended_outcome']} | measured {labels['outcome']} "
                      f"| event {ev} | partner {labels.get('partner_type')} | {meta['town']} "
                      f"{meta['weather']['condition']}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    out = Path(out) if out else clip_dir / "audit.png"
    cv2.imwrite(str(out), np.vstack([head, grid]))
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(sheet(p))
