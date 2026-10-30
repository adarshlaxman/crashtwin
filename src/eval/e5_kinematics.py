"""E5 (light, decision 2026-10-29): crash kinematics, Accid3nD (real, roadside 3D labels) vs CARLA.

  accid3nd : read labels straight from the zips (never unpacked), 5 Hz, both 16 mm cameras per
             sequence; the crash = the first footprint contact between two road users with at
             least one moving > 3 m/s. Writes outputs/results/e5_accid3nd_events.csv.
             (Accid3nD has no crash-frame field in the sampled labels, so contact is detected from
             the 3D boxes; the coordinate frame is undocumented but metric, so only relative
             quantities are used.)
  carla    : the same quantities for CARLA vehicle-vehicle crashes (ego + partner) from a run.
  compare  : distributions + Wasserstein distance / KS test -> outputs/results/e5_comparison.csv

Only like-for-like kinematic quantities are compared; viewpoints differ (roadside vs dashcam)."""
import argparse
import json
import math
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))
from geometry import box_corners, footprint_gap, predicted_ttc  # noqa: E402

ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[3])   # project folder (contains crashtwin/)
ZIPS = ROOT / "datasets" / "real_accid3nd" / "zips"
RES = ROOT / "outputs" / "results"
CAMS = ("s040_camera_basler_north_16mm", "s050_camera_basler_south_16mm")
STEP = 5          # every 5th frame of 25 Hz = 5 Hz
PRE_S = 5.0


def yaw_from_quat(qx, qy, qz, qw):
    return math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def label_files():
    """(sequence, camera) -> [(zip path, member)], merged over all zips (sequences span zips)."""
    files = defaultdict(list)
    for zp in sorted(ZIPS.glob("train*.zip")):
        for n in zipfile.ZipFile(zp).namelist():
            parts = n.split("/")
            if len(parts) == 4 and parts[1] == "labels" and parts[2] in CAMS and n.endswith(".json"):
                files[(parts[0], parts[2])].append((zp, n))
    return files


def read_sequence(members):
    zips, rows = {}, []
    for zp, n in sorted(members, key=lambda m: m[1])[::STEP]:
        z = zips.setdefault(zp, zipfile.ZipFile(zp))
        frames = json.loads(z.read(n))["openlabel"]["frames"]
        for f in frames.values():
            ts = f["frame_properties"]["timestamp"]
            for o in f["objects"].values():
                od = o["object_data"]
                c = od["cuboid"]["val"]
                sp = next((a["val"] for a in od["cuboid"].get("attributes", {}).get("num", [])
                           if a["name"] == "speed"), np.nan)
                rows.append({"t": ts, "name": od["name"], "type": od["type"], "x": c[0], "y": c[1],
                             "yaw": yaw_from_quat(*c[3:7]), "l": c[7], "w": c[8], "speed": sp})
    return pd.DataFrame(rows)


def add_velocity(df):
    df = df.sort_values(["name", "t"]).copy()
    g = df.groupby("name")
    dt = g["t"].diff()
    df["vx"] = (g["x"].diff() / dt).fillna(0.0)
    df["vy"] = (g["y"].diff() / dt).fillna(0.0)
    return df


def box(r):
    return box_corners(r["x"], r["y"], r["yaw"], r["l"] / 2, r["w"] / 2)


def pair_series(df, a, b, t_contact):
    """Kinematics of the pair a, b over the PRE_S seconds before contact."""
    out = []
    A = df[df.name == a].set_index("t")
    B = df[df.name == b].set_index("t")
    for t in sorted(set(A.index) & set(B.index)):
        if t_contact - PRE_S <= t <= t_contact:
            ra, rb = A.loc[t], B.loc[t]
            ba, bb = box(ra), box(rb)
            # label speed (m/s, from the dataset's fused tracking) is less noisy than differencing
            sa = ra.speed if np.isfinite(ra.speed) else math.hypot(ra.vx, ra.vy)
            sb = rb.speed if np.isfinite(rb.speed) else math.hypot(rb.vx, rb.vy)
            out.append({"dt": round(t - t_contact, 2), "gap": footprint_gap(ba, bb),
                        "ttc": predicted_ttc(ba, (ra.vx, ra.vy), bb, (rb.vx, rb.vy)),
                        "speed_a": sa, "speed_b": sb})
    return pd.DataFrame(out)


def summarise(s):
    """Shared kinematic summary (same definitions for real and simulated crashes)."""
    if s.empty:
        return None
    s = s.sort_values("dt")
    fast = "speed_a" if s.speed_a.iloc[0] >= s.speed_b.iloc[0] else "speed_b"
    v = s[fast].to_numpy()
    t = s.dt.to_numpy()
    pre = s[s.dt < 0]
    warn = pre[pre.ttc <= 1.5]
    if len(s) < 3:
        return None
    closing = -np.gradient(s.gap.to_numpy(), t)
    return {"impact_speed_fast_mps": float(v[-2] if len(v) > 1 else v[-1]),
            "impact_speed_slow_mps": float(s[{"speed_a": "speed_b", "speed_b": "speed_a"}[fast]].iloc[-2])
            if len(s) > 1 else np.nan,
            "closing_speed_0.5s_mps": float(np.interp(-0.5, t, closing)),
            "min_ttc_pre_s": float(pre.ttc.replace(np.inf, np.nan).min()) if len(pre) else np.nan,
            "warning_window_s": float(-warn.dt.min()) if len(warn) else 0.0,
            "max_decel_fast_mps2": float(np.max(-np.gradient(v, t))) if len(v) > 2 else np.nan}


def is_crash(df, a, b, t):
    """Contact must be preceded by >= 2 s of tracking of both actors and followed by a crash
    signature: sustained contact (gap < 0.5 m for >= 1 s) or a >= 30 % speed loss within 3 s."""
    A, B = df[df.name == a].set_index("t"), df[df.name == b].set_index("t")
    shared_pre = [u for u in sorted(set(A.index) & set(B.index)) if t - 2.0 <= u < t]
    if len(shared_pre) < 8:                       # tracked together for ~2 s before contact (5 Hz)
        return False
    # duplicate labels of one vehicle (e.g. BUS + TRUCK boxes on the same object): close centres
    # and near-equal velocity throughout the pre-contact window
    cd = [math.hypot(A.loc[u].x - B.loc[u].x, A.loc[u].y - B.loc[u].y) for u in shared_pre]
    dv = [math.hypot(A.loc[u].vx - B.loc[u].vx, A.loc[u].vy - B.loc[u].vy) for u in shared_pre]
    if np.median(cd) < 3.0 and np.median(dv) < 2.0:
        return False
    # the two must have been apart one second before contact
    early = [u for u in shared_pre if u <= t - 1.0]
    if not early or footprint_gap(box(A.loc[early[-1]]), box(B.loc[early[-1]])) <= 0.5:
        return False
    post = [u for u in sorted(set(A.index) & set(B.index)) if t <= u <= t + 3.0]
    stuck = [u for u in post if footprint_gap(box(A.loc[u]), box(B.loc[u])) < 0.5]
    if len(stuck) >= 2 and stuck[-1] - t >= 1.0:
        return True
    for X in (A, B):
        v0 = math.hypot(X.loc[t].vx, X.loc[t].vy)
        after = X[(X.index > t) & (X.index <= t + 3.0)]
        if v0 > 3.0 and len(after) and np.hypot(after.vx, after.vy).min() <= 0.7 * v0:
            return True
    return False


def accid3nd():
    events, series = [], []
    files = label_files()
    for seq in sorted({k[0] for k in files}):
        best = None
        for cam in CAMS:
            if (seq, cam) not in files:
                continue
            df = add_velocity(read_sequence(files[(seq, cam)]))
            found = None
            for t, g in df.groupby("t"):
                g = g.reset_index(drop=True)
                if len(g) < 2:
                    continue
                xy = g[["x", "y"]].to_numpy()
                d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
                for i, j in zip(*np.nonzero(np.triu(d < 8.0, 1))):
                    ri, rj = g.iloc[i], g.iloc[j]
                    if max(math.hypot(ri.vx, ri.vy), math.hypot(rj.vx, rj.vy)) < 3.0:
                        continue
                    if footprint_gap(box(ri), box(rj)) <= 0.2 and is_crash(df, ri["name"], rj["name"], t):
                        found = (t, cam, ri["name"], rj["name"], ri["type"], rj["type"], df)
                        break
                if found:
                    break
            if found and (best is None or found[0] < best[0]):
                best = found
        if best is None:
            print(f"{seq}: no crash signature found", flush=True)
            continue
        t, cam, a, b, ta, tb, df = best
        s = pair_series(df, a, b, t)
        summ = summarise(s) or {}
        events.append({"source": "accid3nd", "sequence": seq, "camera": cam, "t_contact": t,
                       "actor_a": a, "type_a": ta, "actor_b": b, "type_b": tb, "pre_samples": len(s), **summ})
        series.append(s.assign(sequence=seq))
        print(f"{seq}: crash {ta}/{tb} at {t:.1f} ({len(s)} pre samples) {summ}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events).to_csv(RES / "e5_accid3nd_events.csv", index=False)
    if series:
        pd.concat(series).to_csv(RES / "e5_accid3nd_series.csv", index=False)


def carla(run_dir):
    sys.path.insert(0, str(ROOT / "crashtwin" / "src" / "generator"))
    from io_utils import local_path, read_table
    idx = pd.read_csv(Path(run_dir) / "index.csv")
    idx = idx[(idx.outcome == "crash") & (idx.validation_status != "FAIL")
              & idx.other_type.isin(["car", "van", "truck", "bus", "motorcycle"])]
    rows = []
    for r in idx.itertuples():
        d = local_path(r.path)
        lab = json.loads((d / "labels.json").read_text())
        ego = read_table(d / "ego").set_index("frame_idx")
        act = read_table(d / "actors")
        p = act[act.actor_id == lab["partner_id"]].set_index("frame_idx")
        ev = lab["event_frame"]
        s = []
        for f in range(max(0, ev - 50), ev + 1):
            if f not in p.index or f not in ego.index:
                continue
            e, a = ego.loc[f], p.loc[f]
            eb = box_corners(e.x, e.y, e.yaw, e.bb_ex, e.bb_ey, e.bb_cx, e.bb_cy)
            ab = box_corners(a.x, a.y, a.yaw, a.bb_ex, a.bb_ey, a.bb_cx, a.bb_cy)
            s.append({"dt": (f - ev) / 10.0, "gap": footprint_gap(eb, ab),
                      "ttc": predicted_ttc(eb, (e.vx, e.vy), ab, (a.vx, a.vy)),
                      "speed_a": math.hypot(e.vx, e.vy), "speed_b": math.hypot(a.vx, a.vy)})
        summ = summarise(pd.DataFrame(s))
        if summ:
            rows.append({"source": "carla", "clip": r.scenario_id, "family": r.family, "town": r.town,
                         "type_b": r.other_type, **summ})
    pd.DataFrame(rows).to_csv(RES / "e5_carla_events.csv", index=False)
    print(f"CARLA vehicle crashes summarised: {len(rows)}")


def compare():
    from scipy.stats import ks_2samp, wasserstein_distance
    real = pd.read_csv(RES / "e5_accid3nd_events.csv")
    sim = pd.read_csv(RES / "e5_carla_events.csv")
    out = []
    for col in ("impact_speed_fast_mps", "impact_speed_slow_mps", "closing_speed_0.5s_mps",
                "min_ttc_pre_s", "warning_window_s", "max_decel_fast_mps2"):
        a, b = real[col].dropna(), sim[col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        ks = ks_2samp(a, b)
        out.append({"quantity": col, "real_n": len(a), "real_median": a.median(), "carla_n": len(b),
                    "carla_median": b.median(), "wasserstein": wasserstein_distance(a, b),
                    "ks_stat": ks.statistic, "ks_p": ks.pvalue})
    pd.DataFrame(out).to_csv(RES / "e5_comparison.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


# ---------------------------------------------------------------- traffic kinematics (revised E5)
# Automatic crash extraction from Accid3nD boxes failed visual verification (0/3 candidates were
# crashes; outputs/results/e5_frames/), so E5 compares TRAFFIC behaviour instead: speeds and
# car-following (gap, time headway, predicted TTC to the lead vehicle in the same lane).

def following_pairs(g):
    """Rows of one time step -> list of (follower, leader) dicts within the same lane."""
    out = []
    recs = g.to_dict("records")
    for f in recs:
        best = None
        for l in recs:
            if l is f:
                continue
            if abs((l["yaw"] - f["yaw"] + 180) % 360 - 180) > 20:
                continue
            c, s_ = math.cos(math.radians(f["yaw"])), math.sin(math.radians(f["yaw"]))
            dx, dy = l["x"] - f["x"], l["y"] - f["y"]
            ahead, lat = dx * c + dy * s_, -dx * s_ + dy * c
            if 0 < ahead < 80 and abs(lat) < 1.8 and (best is None or ahead < best[0]):
                best = (ahead, l)
        if best is not None:
            out.append((f, best[1]))
    return out


def traffic_rows(f, l, source, scene):
    fb = box_corners(f["x"], f["y"], f["yaw"], f["ex"], f["ey"])
    lb = box_corners(l["x"], l["y"], l["yaw"], l["ex"], l["ey"])
    gap = footprint_gap(fb, lb)
    v = f["speed"]
    return {"source": source, "scene": scene, "speed": v, "gap": gap,
            "time_headway": gap / v if v > 1.0 else np.nan,
            "ttc": predicted_ttc(fb, (f["vx"], f["vy"]), lb, (l["vx"], l["vy"]), horizon=10.0)}


def traffic_accid3nd():
    rows = []
    files = label_files()
    for (seq, cam), members in sorted(files.items()):
        if cam != CAMS[0]:
            continue
        df = add_velocity(read_sequence(members))
        df["speed"] = df["speed"].where(np.isfinite(df["speed"]), np.hypot(df.vx, df.vy))
        df["ex"], df["ey"] = df["l"] / 2, df["w"] / 2
        df = df[df.type.isin(["CAR", "TRUCK", "BUS", "VAN", "MOTORCYCLE"])]
        for t, g in list(df.groupby("t"))[::5]:            # 1 Hz to limit autocorrelation
            for f, l in following_pairs(g):
                rows.append(traffic_rows(f, l, "accid3nd", "highway") | {"sequence": seq})
        print(f"{seq}: {len(rows)} following samples so far", flush=True)
    pd.DataFrame(rows).to_csv(RES / "e5_accid3nd_traffic.csv", index=False)


def traffic_carla(run_dir):
    sys.path.insert(0, str(ROOT / "crashtwin" / "src" / "generator"))
    from io_utils import local_path, read_table
    idx = pd.read_csv(Path(run_dir) / "index.csv")
    idx = idx[(idx.outcome == "normal") & (idx.validation_status != "FAIL")]
    rows = []
    for r in idx.itertuples():
        d = local_path(r.path)
        act = read_table(d / "actors")
        ego = read_table(d / "ego")
        act = act[act.type.isin(["car", "van", "truck", "bus", "motorcycle"])]
        for fi in range(0, 100, 10):                        # 1 Hz
            g = act[act.frame_idx == fi]
            e = ego[ego.frame_idx == fi]
            recs = [{"x": a.x, "y": a.y, "yaw": a.yaw, "ex": a.bb_ex, "ey": a.bb_ey, "vx": a.vx, "vy": a.vy,
                     "speed": math.hypot(a.vx, a.vy)} for a in g.itertuples()]
            if len(e):
                e = e.iloc[0]
                recs.append({"x": e.x, "y": e.y, "yaw": e.yaw, "ex": e.bb_ex, "ey": e.bb_ey, "vx": e.vx,
                             "vy": e.vy, "speed": e.speed})
            scene = "highway" if r.town.startswith("Town04") else "urban"
            for f, l in following_pairs(pd.DataFrame(recs)):
                rows.append(traffic_rows(f, l, "carla", scene) | {"clip": r.scenario_id})
    pd.DataFrame(rows).to_csv(RES / "e5_carla_traffic.csv", index=False)
    print(f"CARLA following samples: {len(rows)}")


def compare_traffic():
    from scipy.stats import ks_2samp, wasserstein_distance
    real = pd.read_csv(RES / "e5_accid3nd_traffic.csv")
    sim = pd.read_csv(RES / "e5_carla_traffic.csv")
    out = []
    for scene in ("highway", "all"):
        b_all = sim if scene == "all" else sim[sim.scene == "highway"]
        for col in ("speed", "gap", "time_headway", "ttc"):
            a = real[col].replace(np.inf, np.nan).dropna()
            b = b_all[col].replace(np.inf, np.nan).dropna()
            if len(a) < 10 or len(b) < 10:
                continue
            ks = ks_2samp(a, b)
            out.append({"carla_subset": scene, "quantity": col, "real_n": len(a), "real_median": a.median(),
                        "carla_n": len(b), "carla_median": b.median(), "wasserstein": wasserstein_distance(a, b),
                        "ks_stat": ks.statistic, "ks_p": ks.pvalue})
    pd.DataFrame(out).to_csv(RES / "e5_traffic_comparison.csv", index=False)
    print(pd.DataFrame(out).round(3).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["accid3nd", "carla", "compare", "traffic_accid3nd", "traffic_carla", "compare_traffic"])
    p.add_argument("--run", default=None)
    a = p.parse_args()
    {"accid3nd": accid3nd, "carla": lambda: carla(a.run), "compare": compare,
     "traffic_accid3nd": traffic_accid3nd, "traffic_carla": lambda: traffic_carla(a.run),
     "compare_traffic": compare_traffic}[a.cmd]()
