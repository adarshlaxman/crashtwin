"""Run-level report (VALIDATION_PLAN section 7). Read-only on clips; writes run_report.md and
run_summary.csv into the run folder. Success rates are computed from ALL logged attempts."""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generator"))
from io_utils import local_path, read_table  # noqa: E402

STREAM_FILES = {"rgb": "rgb.mp4", "depth": "depth.npz", "lidar": "lidar.npz", "radar": "radar.npz"}


def md(df, index=True):
    """DataFrame -> Markdown table (no tabulate dependency)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(lines)


def sensor_sanity(clip_dirs):
    """P5: |sensor lane gap - ground-truth lane gap| where the GT gap < 30 m."""
    errs = {"lidar": [], "depth": [], "radar": []}
    for d in clip_dirs:
        try:
            act = read_table(d / "actors")
            summ = read_table(d / "sensors_summary").set_index("frame_idx")
        except Exception:
            continue
        if not len(act):
            continue
        lane = act[(act["rel_x"] > 0) & (act["rel_y"].abs() < 1.2 + act["bb_ey"])]
        gt = lane.groupby("frame_idx")["gap"].min()
        for f, g in gt.items():
            if not np.isfinite(g) or g >= 30 or f not in summ.index:
                continue
            s = summ.loc[f]
            for k, col in (("lidar", "lidar_front_gap"), ("depth", "depth_front_gap"), ("radar", "radar_min_depth")):
                if s[col] < 99:
                    errs[k].append(abs(float(s[col]) - float(g)))
    return {k: (round(float(np.median(v)), 3) if v else None, len(v)) for k, v in errs.items()}


def report(run_dir):
    run_dir = Path(run_dir)
    log = [json.loads(l) for l in (run_dir / "run_log.jsonl").read_text().splitlines()]
    finished = {(r["slot_id"], r["attempt"]) for r in log if "slot_id" in r and r.get("status") != "started"}
    aborted = [dict(r, status="process_abort", reason="native client abort", wall_s=float("nan"))
               for r in log if r.get("status") == "started" and (r["slot_id"], r["attempt"]) not in finished]
    att = pd.DataFrame([r for r in log if "slot_id" in r and r.get("status") != "started"] + aborted)
    stops = [r for r in log if r.get("status") == "storage_stop"]
    idx = pd.read_csv(run_dir / "index.csv")
    cfg = json.loads((run_dir / "run_config.json").read_text())
    clips = [local_path(p) for p in idx["path"]]
    L = []
    w = L.append
    w(f"# Run report: `{run_dir.name}`\n")
    w(f"- Mode: **{cfg['mode']}**; slots: {len(cfg['slots'])}; attempts logged: {len(att)}; clips written: {len(idx)}")
    free_now = shutil.disk_usage(run_dir.resolve().anchor).free / 1e9
    w(f"- Free disk at start **{cfg.get('free_gb_at_start')} GB**, now **{free_now:.1f} GB**")
    total = sum(sum(f.stat().st_size for f in c.iterdir() if f.is_file()) for c in clips if c.exists())
    w(f"- Total clip data: **{total / 1e9:.2f} GB**")
    if stops:
        w(f"- **Storage stop:** {stops[-1]['reason']}")

    # ---------------- reliability by family
    w("\n## 1. Scenario reliability by family\n")
    status_cols = ["accepted", "warn_hold", "spillover", "invalid", "no_event", "insufficient_history",
                   "early_collision", "setup_error", "error", "process_abort"]
    rows = []
    slots_per_family = Counter(s["family"] for s in cfg["slots"])
    for fam, g in att.groupby("family"):
        c = Counter(g["status"])
        acc_slots = g[g["status"] == "accepted"]["slot_id"].nunique()
        rows.append({"family": fam, "slots": slots_per_family[fam], "attempts": len(g),
                     **{s: c.get(s, 0) for s in status_cols},
                     "slot_success_%": round(100 * acc_slots / max(slots_per_family[fam], 1), 1),
                     "attempt_success_%": round(100 * c.get("accepted", 0) / max(len(g), 1), 1),
                     "mean_wall_s": round(g["wall_s"].mean(), 1)})
    fam_df = pd.DataFrame(rows)
    w(md(fam_df, index=False))
    fam_df.to_csv(run_dir / "run_summary.csv", index=False)

    w("\n### Failure / non-acceptance reasons\n")
    reasons = att[att["status"] != "accepted"].copy()
    reasons["why"] = reasons["status"] + ": " + reasons["reason"].fillna("") + \
        reasons.get("validation_failed", pd.Series([None] * len(reasons))).apply(
            lambda v: f" failed={v}" if isinstance(v, list) and v else "") + \
        reasons.get("validation_warned", pd.Series([None] * len(reasons))).apply(
            lambda v: f" warned={v}" if isinstance(v, list) and v else "")
    rc = reasons.groupby(["family", "why"]).size().reset_index(name="n")
    w(md(rc, index=False) if len(rc) else "_none_")

    # ---------------- by town and weather
    for key in ("town", "weather", "density"):
        w(f"\n## 2.{'twd'.index(key[0]) + 1} Attempts by {key}\n")
        t = att.groupby(key)["status"].agg(attempts="size", accepted=lambda s: int((s == "accepted").sum()))
        t["accepted_%"] = (100 * t["accepted"] / t["attempts"]).round(1)
        w(md(t.reset_index(), index=False))

    # ---------------- outcomes
    w("\n## 3. Intended vs measured outcome (written clips)\n")
    if len(idx):
        w(md(pd.crosstab(idx["intended_outcome"], idx["outcome"])))
        w("\n### Validation status of written clips\n")
        w(md(idx["validation_status"].value_counts().to_frame("clips")))
        cnt = Counter()
        for c in clips:
            v = c / "validation.json"
            if v.exists():
                j = json.loads(v.read_text())
                cnt.update(f"{k}:{x['status']}" for k, x in j["checks"].items() if x["status"] != "pass")
        w("\n### Non-pass checks (all written clips)\n")
        w(md(pd.DataFrame(cnt.most_common(), columns=["check:status", "clips"]), index=False)
          if cnt else "_none_")
        w("\n### Crash partner types\n")
        cr = idx[idx["outcome"] == "crash"]
        w(md(cr["other_type"].value_counts().to_frame("crashes")) if len(cr) else "_none_")
        w("\n### Event frame (clips with an event)\n")
        ev = idx["event_frame"].dropna()
        w(f"min {ev.min():.0f}, median {ev.median():.0f}, max {ev.max():.0f} (allowed 60-90)" if len(ev) else "_none_")

    # ---------------- sensors (P5)
    w("\n## 4. Sensor sanity in CARLA (P5): median |sensor gap - GT lane gap| where GT < 30 m\n")
    s = sensor_sanity(clips)
    w(md(pd.DataFrame([{"sensor": k, "median_abs_err_m": v[0], "frames": v[1]} for k, v in s.items()]), index=False))
    w("\n_Agreement with CARLA ground truth is not evidence of real-world sensor realism._")

    # ---------------- storage / runtime
    w("\n## 5. Storage per clip (MB)\n")
    per = []
    for c in clips:
        if not c.exists():
            continue
        r = {k: (c / f).stat().st_size / 1e6 for k, f in STREAM_FILES.items() if (c / f).exists()}
        r["tables+json"] = sum(f.stat().st_size for f in c.iterdir()
                               if f.is_file() and f.name not in STREAM_FILES.values()) / 1e6
        r["total"] = sum(v for k, v in r.items())
        per.append(r)
    sz = pd.DataFrame(per)
    if len(sz):
        w(md(sz.describe().loc[["mean", "50%", "max"]].round(2)))
    w("\n## 6. Runtime\n")
    acc = att[att["status"] == "accepted"]
    w(f"- Mean wall time per attempt: **{att['wall_s'].mean():.1f} s** (all statuses)")
    if len(acc):
        per_acc = att["wall_s"].sum() / len(acc)
        w(f"- Wall time per accepted clip (incl. failed attempts): **{per_acc:.1f} s**")
        w(f"- Attempts per accepted clip: **{len(att) / len(acc):.2f}**")
        if len(sz):
            med = float(sz["total"].median())
            written_per_acc = len(idx) / len(acc)
            w(f"- **Projection for 2,000 accepted clips** (same family mix and success rates): "
              f"~{2000 * per_acc / 3600:.1f} h, ~{2000 * written_per_acc * med / 1000:.1f} GB "
              f"(written clips per accepted = {written_per_acc:.2f}, median clip {med:.1f} MB)")
    out = run_dir / "run_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(report(p))
