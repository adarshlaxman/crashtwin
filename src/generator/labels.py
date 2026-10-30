"""Outcome labelling, rule v1 (docs/LABELS.md). Pure functions, unit-tested.

The outcome is computed ONLY from the collision log and simulator ground truth.
The scenario's intended outcome is never an input."""
import math

import numpy as np

from geometry import INF, box_corners, footprint_gap, predicted_ttc

RULE_V1 = {
    "version": "v1",
    "eval_start_s": 0.5,          # ignore the first 0.5 s of the clip
    "candidate_radius_m": 50.0,   # actors considered at all
    "ttc_radius_m": 40.0,         # TTC is only computed for actors this close (performance)
    "ttc_horizon_s": 5.0,
    "ttc_step_s": 0.05,
    "near_miss_gap_m": 1.5,       # same actor must satisfy BOTH gap and TTC
    "near_miss_ttc_s": 1.5,
    "normal_min_ttc_s": 3.0,
    "minor_contact_intensity": 500.0,
    "event_frame_range": [60, 90],
}


def _box(row, prefix=""):
    return box_corners(row[prefix + "x"], row[prefix + "y"], row[prefix + "yaw"],
                       row[prefix + "bb_ex"], row[prefix + "bb_ey"],
                       row[prefix + "bb_cx"], row[prefix + "bb_cy"])


def conflict_measures(ego, actors, rule=RULE_V1, fps=10):
    """Per-row footprint gap and predicted TTC for every actor row.

    ego: DataFrame indexed by frame_idx (x, y, yaw, vx, vy, bb_ex, bb_ey, bb_cx, bb_cy)
    actors: DataFrame (frame_idx, actor_id, x, y, yaw, vx, vy, bb_*)
    Returns (gap, ttc) numpy arrays aligned with actors' rows."""
    gaps = np.full(len(actors), INF)
    ttcs = np.full(len(actors), INF)
    ego_boxes = {f: _box(r) for f, r in ego.iterrows()}
    for i, row in enumerate(actors.itertuples(index=False)):
        r = row._asdict()
        f = r["frame_idx"]
        if f not in ego_boxes:
            continue
        e = ego.loc[f]
        if math.hypot(r["x"] - e["x"], r["y"] - e["y"]) > rule["candidate_radius_m"]:
            continue
        ob = _box(r)
        gaps[i] = footprint_gap(ego_boxes[f], ob)
        if gaps[i] <= rule["ttc_radius_m"]:
            ttcs[i] = predicted_ttc(ego_boxes[f], (e["vx"], e["vy"]), ob, (r["vx"], r["vy"]),
                                    rule["ttc_horizon_s"], rule["ttc_step_s"])
    return gaps, ttcs


def compute_outcome(ego, actors, collisions, n_frames=100, fps=10, structural_errors=(),
                    rule=RULE_V1):
    """Return (labels dict, gap array, ttc array).

    collisions: list of dicts with frame_idx, other_id, other_blueprint, other_type, intensity
                (only events involving the ego; frame_idx relative to the clip)."""
    gaps, ttcs = conflict_measures(ego, actors, rule, fps)
    first = int(round(rule["eval_start_s"] * fps))
    in_eval = actors["frame_idx"].to_numpy() >= first if len(actors) else np.zeros(0, bool)

    per_actor = {}
    for idx in np.nonzero(in_eval)[0]:
        aid = int(actors["actor_id"].iat[idx])
        a = per_actor.setdefault(aid, {"min_gap": INF, "min_gap_frame": None,
                                       "min_ttc": INF, "min_ttc_frame": None,
                                       "type": actors["type"].iat[idx]})
        if gaps[idx] < a["min_gap"]:
            a["min_gap"], a["min_gap_frame"] = float(gaps[idx]), int(actors["frame_idx"].iat[idx])
        if ttcs[idx] < a["min_ttc"]:
            a["min_ttc"], a["min_ttc_frame"] = float(ttcs[idx]), int(actors["frame_idx"].iat[idx])

    min_gap = min((a["min_gap"] for a in per_actor.values()), default=INF)
    min_ttc = min((a["min_ttc"] for a in per_actor.values()), default=INF)
    if per_actor:
        conflict = min(per_actor, key=lambda k: (per_actor[k]["min_ttc"], per_actor[k]["min_gap"]))
    else:
        conflict = None

    labels = {
        "rule_version": rule["version"], "rule": rule,
        "min_gap_m": _num(min_gap), "min_ttc_s": _num(min_ttc),
        "conflict_actor": conflict,
        "conflict_actor_type": per_actor[conflict]["type"] if conflict is not None else None,
        "top_actors": sorted(
            [{"actor_id": k, **{kk: _num(vv) for kk, vv in v.items()}} for k, v in per_actor.items()],
            key=lambda d: (d["min_ttc"] if d["min_ttc"] is not None else 1e9,
                           d["min_gap"] if d["min_gap"] is not None else 1e9))[:5],
        "event_frame": None, "event_time_s": None, "partner_id": None,
        "partner_type": None, "partner_blueprint": None, "intensity": None,
        "warnings": [], "invalid_reason": None,
    }

    clip_coll = [c for c in collisions if 0 <= c["frame_idx"] < n_frames]
    if structural_errors:
        labels["outcome"] = "invalid"
        labels["invalid_reason"] = "structural:" + ",".join(structural_errors)
    elif clip_coll:
        c = min(clip_coll, key=lambda c: c["frame_idx"])
        labels.update(outcome="crash", event_frame=int(c["frame_idx"]),
                      partner_id=c["other_id"], partner_type=c["other_type"],
                      partner_blueprint=c["other_blueprint"], intensity=round(c["intensity"], 1))
        # walkers and bicycles are light: their impacts are far below vehicle-scale impulses
        if c["intensity"] < rule["minor_contact_intensity"] and c["other_type"] not in ("walker", "bicycle"):
            labels["warnings"].append("minor_contact")
    else:
        nm = [k for k, a in per_actor.items()
              if a["min_gap"] <= rule["near_miss_gap_m"] and a["min_ttc"] <= rule["near_miss_ttc_s"]]
        if nm:
            k = min(nm, key=lambda k: (per_actor[k]["min_ttc"], per_actor[k]["min_gap"]))
            labels.update(outcome="near_miss", event_frame=per_actor[k]["min_gap_frame"],
                          partner_id=k, partner_type=per_actor[k]["type"])
        elif min_ttc > rule["normal_min_ttc_s"]:
            labels["outcome"] = "normal"
        else:
            labels["outcome"] = "invalid"
            labels["invalid_reason"] = "borderline_conflict"
    if labels["event_frame"] is not None:
        labels["event_time_s"] = round(labels["event_frame"] / fps, 3)
    return labels, gaps, ttcs


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, str):
        return v
    return None if (isinstance(v, float) and math.isinf(v)) else round(float(v), 4)
