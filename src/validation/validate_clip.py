"""Per-clip validator (docs/VALIDATION_PLAN.md). Read-only: never edits or deletes clip data;
it only writes validation.json. A WARN never counts as PASS."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))
from io_utils import read_table, table_exists, write_json  # noqa: E402

N_FRAMES = 100
DT = 0.1
REQUIRED_FILES = ["rgb.mp4", "depth.npz", "lidar.npz", "radar.npz", "labels.json", "meta.json"]
REQUIRED_TABLES = ["frames", "ego", "actors", "collisions", "sensors_summary"]
STREAMS = ("rgb", "instance", "depth", "lidar", "radar", "imu", "gnss")
META_KEYS = ["scenario_id", "family", "slot_id", "attempt", "seed", "intended_outcome", "town",
             "weather", "matrix_version", "rule_version", "schema_version", "sensor_config", "provenance"]


class Report:
    def __init__(self):
        self.checks = {}

    def add(self, cid, level, ok, detail=""):
        status = "pass" if ok else level.lower()
        self.checks[cid] = {"status": status, "detail": detail}

    @property
    def status(self):
        st = {c["status"] for c in self.checks.values()}
        return "FAIL" if "fail" in st else "WARN" if "warn" in st else "PASS"


def _near_event(i, event, margin=5):
    return event is not None and abs(i - event) <= margin


def validate(clip_dir, write=True):
    clip_dir = Path(clip_dir)
    r = Report()
    missing = [f for f in REQUIRED_FILES if not (clip_dir / f).exists()] + \
              [t for t in REQUIRED_TABLES if not table_exists(clip_dir / t)]
    data = {}
    try:
        if not missing:
            for t in REQUIRED_TABLES:
                data[t] = read_table(clip_dir / t)
            data["labels"] = json.loads((clip_dir / "labels.json").read_text())
            data["meta"] = json.loads((clip_dir / "meta.json").read_text())
            data["depth"] = np.load(clip_dir / "depth.npz")["depth_cm"]
            lz = np.load(clip_dir / "lidar.npz")
            data["lidar"], data["lidar_off"] = lz["points"], lz["offsets"]
            rz = np.load(clip_dir / "radar.npz")
            data["radar"], data["radar_off"] = rz["detections"], rz["offsets"]
    except Exception as e:  # unreadable file
        missing.append(f"unreadable:{type(e).__name__}:{e}")
    r.add("F1", "FAIL", not missing, ",".join(missing))
    if missing:
        return _finish(clip_dir, r, write, None)

    labels, meta = data["labels"], data["meta"]
    event = labels.get("event_frame")

    # F2 video
    import cv2
    cap = cv2.VideoCapture(str(clip_dir / "rgb.mp4"))
    n, size = 0, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        size = fr.shape[:2]
    cap.release()
    exp_size = (meta["sensor_config"]["rgb"]["height"], meta["sensor_config"]["rgb"]["width"])
    r.add("F2", "FAIL", n == N_FRAMES and size == exp_size, f"frames={n} size={size}")

    # F3/F4 frames table
    fr = data["frames"]
    wf = fr["world_frame"].to_numpy()
    st = fr["sim_time"].to_numpy()
    r.add("F3", "FAIL", len(fr) == N_FRAMES and bool(np.all(np.diff(wf) == 1)),
          f"rows={len(fr)} non_consecutive={int(np.sum(np.diff(wf) != 1))}")
    dts = np.diff(st)
    r.add("F4", "FAIL", len(dts) > 0 and bool(np.all(np.abs(dts - DT) <= 1e-3)),
          f"dt_min={dts.min():.4f} dt_max={dts.max():.4f}" if len(dts) else "no rows")

    # S1 per-stream sync
    bad = []
    for s in STREAMS:
        if f"{s}_frame" not in fr:
            bad.append(f"{s}:missing")
            continue
        if not np.all(fr[f"{s}_frame"].to_numpy() == wf):
            bad.append(f"{s}:frame")
        if not np.all(np.abs(fr[f"{s}_time"].to_numpy() - st) < 1e-3):
            bad.append(f"{s}:time")
    r.add("S1", "FAIL", not bad, ",".join(bad))

    # S2 depth
    d = data["depth"]
    valid = float(np.mean((d > 0) & (d < 65535))) if d.size else 0.0
    r.add("S2", "FAIL", d.shape[0] == N_FRAMES and valid >= 0.5,
          f"shape={tuple(d.shape)} valid_fraction={valid:.3f}")

    # S3 LiDAR: 100 frames, no NaN, per-frame counts consistent with the clip median
    off = data["lidar_off"]
    counts = np.diff(off)
    med = float(np.median(counts)) if len(counts) else 0.0
    outliers = int(np.sum(np.abs(counts - med) > 0.3 * med)) if med else len(counts)
    nan = bool(np.isnan(data["lidar"].astype(np.float32)).any())
    r.add("S3", "WARN", len(counts) == N_FRAMES and med >= 2000 and outliers == 0 and not nan,
          f"frames={len(counts)} median_points={med:.0f} outlier_frames={outliers} nan={nan}")

    # S4 radar
    rc = np.diff(data["radar_off"])
    empty = float(np.mean(rc == 0)) if len(rc) else 1.0
    r.add("S4", "WARN", len(rc) == N_FRAMES and empty < 0.95, f"frames={len(rc)} empty_fraction={empty:.2f}")

    # S5 ego / IMU / GNSS
    ego = data["ego"]
    acc = np.sqrt(ego["acc_x"] ** 2 + ego["acc_y"] ** 2 + ego["acc_z"] ** 2).to_numpy()
    acc_bad = [i for i, a in enumerate(acc) if a >= 200 and not _near_event(i, event)]
    sp = ego["speed"].to_numpy()
    nan_cols = [c for c in ("lat", "lon", "acc_x", "gyro_z", "speed") if ego[c].isna().any()]
    r.add("S5", "FAIL", len(ego) == N_FRAMES and sp.min() >= 0 and sp.max() <= 60 and not acc_bad and not nan_cols,
          f"rows={len(ego)} speed_max={sp.max():.1f} acc_spikes={acc_bad[:5]} nan={nan_cols}")

    # S6 ego continuity
    xy = ego[["x", "y"]].to_numpy()
    jumps = []
    for i in range(1, len(xy)):
        lim = max(sp[i], sp[i - 1]) * DT + 0.5
        if np.linalg.norm(xy[i] - xy[i - 1]) > lim and not _near_event(i, event):
            jumps.append(i)
    r.add("S6", "WARN", not jumps, f"jumps={jumps[:5]}")

    # S7 actor continuity
    act = data["actors"]
    teleports = []
    if len(act):
        for aid, g in act.sort_values("frame_idx").groupby("actor_id"):
            f = g["frame_idx"].to_numpy()
            p = g[["x", "y"]].to_numpy()
            v = np.hypot(g["vx"].to_numpy(), g["vy"].to_numpy())
            for i in range(1, len(f)):
                if f[i] - f[i - 1] != 1 or _near_event(f[i], event):
                    continue
                if np.linalg.norm(p[i] - p[i - 1]) > max(v[i], v[i - 1]) * DT + 0.5:
                    teleports.append(int(aid))
                    break
    r.add("S7", "WARN", not teleports, f"actors={teleports[:5]}")

    # ---------------- labels
    coll = data["collisions"]
    outcome = labels.get("outcome")
    has_coll = len(coll) > 0
    r.add("L1", "FAIL", (outcome == "crash") == has_coll or outcome == "invalid",
          f"outcome={outcome} collisions={len(coll)}")
    lo, hi = 60, 90
    if outcome == "crash":
        first = int(coll["frame_idx"].min()) if has_coll else None
        in_range = event is not None and lo <= event <= hi
        r.add("L2", "FAIL", event == first and in_range, f"event={event} first_collision={first}")
        # the recorder centred the window on the first collision: it must sit at pre_frames
        sp = meta.get("scenario_params", {})
        if sp.get("anchor_source") == "collision":
            r.add("L2b", "WARN", event == sp.get("pre_frames"),
                  f"event={event} recorder_pre_frames={sp.get('pre_frames')}")
        partner = labels.get("partner_id")
        if labels.get("partner_type") != "static" and len(act):
            g = act[(act["actor_id"] == partner) & (act["frame_idx"] == event)]
            gap = float(g["gap"].iloc[0]) if len(g) and not math.isnan(g["gap"].iloc[0]) else None
            r.add("L3", "WARN", gap is not None and gap <= 0.3, f"partner_gap_at_event={gap}")
        r.add("L8", "FAIL", partner in set(coll["other_id"]), f"partner={partner}")
    elif outcome == "near_miss":
        part = next((a for a in labels.get("top_actors", []) if a["actor_id"] == labels.get("partner_id")), None)
        rule = labels["rule"]
        ok = (not has_coll and part is not None and part["min_gap"] is not None and part["min_ttc"] is not None
              and part["min_gap"] <= rule["near_miss_gap_m"] and part["min_ttc"] <= rule["near_miss_ttc_s"]
              and event is not None and lo <= event <= hi)
        r.add("L4", "FAIL", ok, f"event={event} partner={part}")
        r.add("L8", "FAIL", len(act) > 0 and labels.get("partner_id") in set(act["actor_id"]),
              f"partner={labels.get('partner_id')}")
    elif outcome == "normal":
        mt = labels.get("min_ttc_s")
        r.add("L5", "FAIL", not has_coll and (mt is None or mt > labels["rule"]["normal_min_ttc_s"]),
              f"min_ttc={mt}")
    else:
        r.add("L_invalid", "WARN", False, f"outcome=invalid reason={labels.get('invalid_reason')}")

    r.add("L6", "WARN", outcome == meta.get("intended_outcome"),
          f"intended={meta.get('intended_outcome')} measured={outcome}")
    warns = list(labels.get("warnings", []))
    planned = meta.get("scenario_params", {}).get("conflict_actor_id")
    if outcome == "crash" and planned is not None and labels.get("partner_id") != planned \
            and meta.get("family", "").startswith("C") and labels.get("partner_type") != "static":
        warns.append("unplanned_partner")
    r.add("L7", "WARN", not warns, ",".join(warns))

    # ---------------- reproducibility
    miss = [k for k in META_KEYS if k not in meta]
    r.add("R1", "FAIL", not miss and "parameters" in meta.get("weather", {}), f"missing={miss}")
    # Code revision: a clean git commit when one exists; otherwise (no commits approved yet)
    # the per-file SHA-256 of the generator source must be recorded.
    prov = meta.get("provenance", {})
    git = prov.get("git", {})
    hashes = prov.get("source_sha256", {})
    clean_commit = bool(git.get("commit")) and git.get("dirty") is False
    r.add("R2", "WARN", clean_commit or len(hashes) >= 5,
          f"commit={git.get('commit')} dirty={git.get('dirty')} source_files_hashed={len(hashes)}")
    return _finish(clip_dir, r, write, outcome)


def _finish(clip_dir, r, write, outcome):
    result = {"status": r.status, "outcome": outcome, "checks": r.checks,
              "failed": [k for k, v in r.checks.items() if v["status"] == "fail"],
              "warned": [k for k, v in r.checks.items() if v["status"] == "warn"]}
    if write:
        write_json(Path(clip_dir) / "validation.json", result)
    return result


if __name__ == "__main__":
    for p in sys.argv[1:]:
        res = validate(p)
        print(p, res["status"], "fail:", res["failed"], "warn:", res["warned"])
