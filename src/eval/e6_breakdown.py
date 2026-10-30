"""E6 (light, 4-day plan): robustness breakdown of finished runs by condition, no new training.
Video-level AP (max score before the event) per group, mean over seeds, 95 % bootstrap CI
(pooled over seeds). Hypothesis fixed before looking: AP(night) < AP(day), AP(rain/fog) < AP(clear).
Writes outputs/results/e6_breakdown.csv."""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import bootstrap_ci  # noqa: E402

ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[3])   # project folder (contains crashtwin/)
EXP = ROOT / "outputs" / "experiments"
FEAT = ROOT / "outputs" / "processed" / "features"
RES = ROOT / "outputs" / "results"
CONFIGS = ["E1_sim", "E2_real100", "E3_sim+real100"]
TESTS = {"ccd_test_grouped": ("ccd", ["daynight", "weather"]),
         "nexar_test_public": ("nexar", ["daynight", "weather", "scene"]),
         "nexar_test_private": ("nexar", ["daynight", "weather", "scene"]),
         "sim_test_t05": ("sim_full", ["weather_condition"]),
         "sim_test_familiar": ("sim_full", ["weather_condition"])}


def video_scores(probs, clip_ids, meta, cache):
    z = np.load(FEAT / f"{cache}.npz")
    m = meta.set_index("clip_id")
    pos = {c: i for i, c in enumerate(meta.clip_id)}
    out = []
    for p, cid in zip(probs, clip_ids):
        i = pos[cid]
        n = int(z["mask"][i].sum())
        tte = z["tte"][i][:n]
        y = int(z["y"][i])
        stop = int(np.argmax(tte <= 0)) if (y == 1 and (tte <= 0).any()) else n
        out.append({"clip_id": cid, "y": y, "score": float(np.max(p[:max(stop, 1)])), **m.loc[cid].to_dict()})
    return pd.DataFrame(out)


def main():
    rows = []
    for cfg in CONFIGS:
        runs = [d for d in EXP.glob(f"{cfg}_s*") if (d / "metrics.json").exists() and re.fullmatch(rf"{re.escape(cfg)}_s\d+", d.name)]
        for test, (cache, keys) in TESTS.items():
            if not (FEAT / f"{cache}_meta.csv").exists():
                continue
            meta = pd.read_csv(FEAT / f"{cache}_meta.csv", dtype={"clip_id": str})
            if cache == "sim_full" and "weather_condition" not in meta:
                continue
            frames = []
            for d in runs:
                f = d / f"pred_{test}.npz"
                if f.exists():
                    z = np.load(f)
                    frames.append(video_scores(z["probs"], z["clip_id"], meta, cache).assign(seed=d.name))
            if not frames:
                continue
            df = pd.concat(frames)
            for key in keys:
                for val, g in df.groupby(key):
                    if g.y.nunique() < 2 or g.clip_id.nunique() < 20:
                        continue
                    aps = [average_precision_score(s.y, s.score) for _, s in g.groupby("seed") if s.y.nunique() == 2]
                    y, sc = g.y.to_numpy(), g.score.to_numpy()

                    def ap(i):
                        if y[i].min() == y[i].max():
                            raise ValueError
                        return average_precision_score(y[i], sc[i])
                    lo, hi = bootstrap_ci(ap, len(y), n=300)
                    rows.append({"config": cfg, "test": test, "factor": key, "group": val,
                                 "videos": g.clip_id.nunique(), "positives": int(g.drop_duplicates("clip_id").y.sum()),
                                 "video_AP": float(np.mean(aps)), "CI95_low": lo, "CI95_high": hi})
    RES.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(RES / "e6_breakdown.csv", index=False)
    print(out.round(3).to_string(index=False) if len(out) else "no finished runs with predictions yet")


if __name__ == "__main__":
    main()
