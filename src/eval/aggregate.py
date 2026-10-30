"""Aggregate finished experiment runs into tables and paper figures (only runs with metrics.json;
nothing is filled in for unfinished runs). Writes outputs/results/RESULTS.md + figures/*.png."""
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[3])   # project folder (contains crashtwin/)
EXP = ROOT / "outputs" / "experiments"
RES = ROOT / "outputs" / "results"
FIG = RES / "figures"
# reference categorical palette, fixed order: slot 1 blue, 2 orange, 3 aqua
C_REAL, C_MIX, C_SIM = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_runs():
    rows = []
    for m in sorted(EXP.glob("*/metrics.json")):
        name = m.parent.name
        if name.startswith("_"):
            continue
        res = json.loads(m.read_text())
        r = {"run": name, "config": re.sub(r"_s\d+$", "", name), "seed": int(re.search(r"_s(\d+)$", name).group(1)),
             "val_AP": res["val_AP"], "best_epoch": res["best_epoch"]}
        for t, v in res["tests"].items():
            if "nexar_official_mAP" in v:
                r[f"{t}:mAP"] = v["mAP"]
            else:
                for k in ("AP", "mTTA", "TTA_R80", "tta_s", "recall", "fa_per_min", "video_AP"):
                    r[f"{t}:{k}"] = v.get(k)
        rows.append(r)
    return pd.DataFrame(rows)


def summary(df):
    num = [c for c in df.columns if ":" in c or c == "val_AP"]
    g = df.groupby("config")[num]
    mean, std, n = g.mean(), g.std(), g.size()
    return mean, std, n


def fmt(m, s):
    if pd.isna(m):
        return "–"
    return f"{m:.3f}" if pd.isna(s) else f"{m:.3f} ± {s:.3f}"


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def efficiency_figure(df, metric, title, fname):
    """Real-data fraction vs metric: real-only (E2) vs CARLA + real (E3); CARLA-only (E1) as a reference line."""
    pts = {}
    for cfg_prefix, label in (("E2_real", "Real only"), ("E3_sim+real", "CARLA + real")):
        sub = df[df.config.str.startswith(cfg_prefix)].copy()
        if sub.empty or metric not in sub:
            continue
        sub["frac"] = sub.config.str.extract(r"(\d+)$")[0].astype(int)
        pts[label] = sub.groupby("frac")[metric].agg(["mean", "min", "max"])
    e1 = df[df.config == "E1_sim"][metric] if metric in df else pd.Series(dtype=float)
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    for label, color in (("Real only", C_REAL), ("CARLA + real", C_MIX)):
        if label not in pts:
            continue
        p = pts[label]
        ax.plot(p.index, p["mean"], color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        ax.fill_between(p.index, p["min"], p["max"], color=color, alpha=0.15, linewidth=0)
        ax.annotate(label, (p.index[-1], p["mean"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, va="center", fontsize=9)
    if len(e1):
        ax.axhline(e1.mean(), color=C_SIM, linewidth=2, linestyle="--", label="CARLA only (no real data)")
        ax.annotate("CARLA only", (10, e1.mean()), xytext=(0, 5), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100])
    ax.set_xticklabels(["10 %", "25 %", "50 %", "100 %"])
    ax.set_xlabel("Real training data used", color=INK2)
    style(ax, title, metric.split(":")[1])
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="best")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return FIG / fname


def e4_figure(df, test="sim_test_t05"):
    rows = []
    for cfg in ["E1_sim", "E4_cam+radar", "E4_cam+lidar", "E4_cam+depth", "E4_cam+all"]:
        sub = df[df.config == cfg]
        col = f"{test}:AP"
        if len(sub) and col in sub:
            rows.append((cfg.replace("E1_sim", "camera").replace("E4_cam+", "camera + "), sub[col].mean(),
                         sub[f"{test}:tta_s"].mean() if f"{test}:tta_s" in sub else np.nan))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    names, ap, _ = zip(*rows)
    y = np.arange(len(names))
    ax.barh(y, ap, color=C_REAL, height=0.6)
    for i, v in enumerate(ap):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    style(ax, f"Sensor fusion in CARLA (test: {'Town05, held out' if test == 'sim_test_t05' else 'familiar towns'})", "")
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AP (CCD protocol)", color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "e4_sensor_fusion.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return FIG / "e4_sensor_fusion.png"


def main():
    df = load_runs()
    RES.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("no finished runs yet")
        return
    df.to_csv(RES / "all_runs.csv", index=False)
    mean, std, n = summary(df)
    cols = [("nexar_test_public:mAP", "Nexar public mAP"), ("nexar_test_private:mAP", "Nexar private mAP"),
            ("ccd_test_grouped:AP", "CCD AP (grouped)"), ("ccd_test_grouped:mTTA", "CCD mTTA (s)"),
            ("ccd_test_grouped:tta_s", "CCD warning (s) @R80"), ("ccd_test_grouped:fa_per_min", "CCD FA/min"),
            ("ccd_test_official:AP", "CCD AP (official)"), ("sim_test_familiar:AP", "CARLA AP (familiar)"),
            ("sim_test_t05:AP", "CARLA AP (Town05)")]
    L = ["# CrashTwin results (auto-generated; only finished runs)\n",
         "Mean ± std over seeds. AP/mTTA: UString/DSTA CCD protocol. Nexar: official mAP over the 0.5/1.0/1.5 s groups. "
         "'Warning' and FA/min use the validation threshold for 80 % recall.\n",
         "| Config | seeds | " + " | ".join(c[1] for c in cols if c[0] in mean) + " |",
         "|---|---|" + "---|" * sum(c[0] in mean for c in cols)]
    for cfg in mean.index:
        L.append(f"| {cfg} | {n[cfg]} | " + " | ".join(fmt(mean.loc[cfg, c], std.loc[cfg, c]) for c, _ in cols
                                                       if c in mean) + " |")
    figs = [efficiency_figure(df.assign(**{"nexar_mean:mAP": df[["nexar_test_public:mAP", "nexar_test_private:mAP"]].mean(1)})
                              if "nexar_test_public:mAP" in df else df, "nexar_mean:mAP",
                              "Nexar (real dashcam): crash anticipation", "e1e3_nexar.png"),
            efficiency_figure(df, "ccd_test_grouped:AP", "CCD (real dashcam, grouped split)", "e1e3_ccd.png"),
            e4_figure(df)]
    L += ["", "## Figures"] + [f"![]({f.relative_to(RES).as_posix()})" for f in figs if f]
    (RES / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
