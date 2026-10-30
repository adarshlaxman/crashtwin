"""Real-data indexes and clip-level split manifests (Phase 2, step 1). Read-only on datasets/.

CCD   : official split (vgg16_features/train.txt, test.txt; 3,600 / 900). Validation = 10 % of the
        official train, grouped by YouTube source id for crash clips (no source id for normal clips).
Nexar : official partitions. train -> 85 % train / 15 % val (stratified by label, seeded);
        test-public and test-private stay test sets, reported separately.
Output: outputs/index/ccd.csv, outputs/index/nexar.csv (one row per video)."""
import argparse
import ast
import os
import random
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[3])   # project folder (contains crashtwin/)
CCD = ROOT / "datasets" / "real_ccd"
NEXAR = ROOT / "datasets" / "real_nexar"
OUT = ROOT / "outputs" / "index"
SEED = 20261029


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    info = (cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def build_ccd():
    rows = []
    for line in (CCD / "Crash-1500.txt").read_text().splitlines():
        vid, rest = line.split(",", 1)
        vec_txt, tail = rest.split("]", 1)
        labels = ast.literal_eval(vec_txt + "]")
        start, ytid, timing, weather, ego = tail.strip(",").split(",")
        rows.append({"clip_id": f"ccd_pos_{vid}", "file": f"positive/{vid}", "label": 1,
                     "event_frame": labels.index(1) if 1 in labels else None, "daynight": timing,
                     "weather": weather, "ego_involved": ego == "Yes", "source_group": f"yt{ytid}",
                     "path": str(CCD / "videos" / "Crash-1500" / f"{vid}.mp4")})
    for p in sorted((CCD / "videos" / "Normal").glob("*.mp4")):
        rows.append({"clip_id": f"ccd_neg_{p.stem}", "file": f"negative/{p.stem}", "label": 0, "event_frame": None,
                     "daynight": None, "weather": None, "ego_involved": None, "source_group": f"neg{p.stem}",
                     "path": str(p)})
    df = pd.DataFrame(rows)
    part = {}
    for name in ("train", "test"):
        for line in (CCD / "official_split" / f"{name}.txt").read_text().splitlines():
            f, _ = line.split()
            part[f.replace(".npz", "")] = name
    df["official_partition"] = df["file"].map(part)
    missing = df["official_partition"].isna().sum()
    if missing:
        sys.exit(f"{missing} CCD clips are not in the official split files")
    rng = random.Random(SEED)
    # (1) split_official: official train/test; val = 10 % of official train (whole source groups)
    df["split_official"] = df["official_partition"]
    for label in (0, 1):
        sel = (df.official_partition == "train") & (df.label == label)
        groups = sorted(df[sel]["source_group"].unique())
        rng.shuffle(groups)
        chosen, n = set(), 0
        for g in groups:
            if n >= 0.10 * sel.sum():
                break
            chosen.add(g)
            n += int((sel & (df.source_group == g)).sum())
        df.loc[sel & df.source_group.isin(chosen), "split_official"] = "val"
    # (2) split_grouped (PRIMARY for E2/E3): the official split shares 107 of 133 YouTube
    # sources between train and test, so crash clips are re-split by source, 80/10/10.
    df["split_grouped"] = None
    for label in (0, 1):
        sel = df.label == label
        groups = sorted(df[sel]["source_group"].unique())
        rng.shuffle(groups)
        total, n = sel.sum(), 0
        for g in groups:
            frac = n / total
            name = "train" if frac < 0.80 else "val" if frac < 0.90 else "test"
            df.loc[sel & (df.source_group == g), "split_grouped"] = name
            n += int((sel & (df.source_group == g)).sum())
    shared = df[df.label == 1].groupby("source_group")["official_partition"].nunique()
    print(f"  CCD official split: {int((shared > 1).sum())} of {len(shared)} crash sources appear in both "
          f"train and test (reported; split_grouped avoids it)")
    df["split"] = df["split_grouped"]
    df["dataset"], df["fps"] = "ccd", 10.0
    df["event_time_s"] = df["event_frame"] / 10.0
    return df


def build_nexar():
    rows = []
    for part in ("train", "test-public", "test-private"):
        for cls in ("positive", "negative"):
            meta = pd.read_csv(NEXAR / part / cls / "metadata.csv")
            for _, m in meta.iterrows():
                p = NEXAR / part / cls / m["file_name"]
                rows.append({"clip_id": f"nexar_{p.stem}", "label": int(cls == "positive"),
                             "official_partition": part, "event_time_s": m["time_of_event"],
                             "alert_time_s": m["time_of_alert"], "time_to_accident": m["time_to_accident"],
                             "daynight": m["light_conditions"], "weather": m["weather"], "scene": m["scene"],
                             "source_group": f"nexar{p.stem}", "path": str(p)})
    df = pd.DataFrame(rows)
    rng = random.Random(SEED)
    df["split"] = df["official_partition"].map({"train": "train", "test-public": "test_public",
                                                "test-private": "test_private"})
    for label in (0, 1):
        ids = sorted(df[(df.official_partition == "train") & (df.label == label)]["clip_id"])
        rng.shuffle(ids)
        df.loc[df.clip_id.isin(ids[:round(0.15 * len(ids))]), "split"] = "val"
    df["dataset"] = "nexar"
    return df


def add_video_info(df):
    info = [video_info(p) for p in df["path"]]
    df["fps"] = [round(i[0], 3) for i in info]
    df["n_frames"] = [i[1] for i in info]
    df["width"], df["height"] = [i[2] for i in info], [i[3] for i in info]
    df["exists"] = [Path(p).exists() for p in df["path"]]
    return df


def check_no_leakage(df):
    """U8: no clip and no source group in two splits."""
    col = "split_grouped" if "split_grouped" in df else "split"
    by_group = df.groupby("source_group")[col].nunique()
    assert (by_group <= 1).all(), f"source groups across splits: {list(by_group[by_group > 1].index)[:5]}"
    assert df["clip_id"].is_unique


def main():
    argparse.ArgumentParser().parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("ccd", build_ccd), ("nexar", build_nexar)):
        df = add_video_info(fn())
        check_no_leakage(df)
        out = OUT / f"{name}.csv"
        if out.exists():
            out = OUT / f"{name}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"   # never overwrite
        df.to_csv(out, index=False)
        print(f"{name}: {len(df)} videos -> {out}")
        print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
        print(f"  missing files: {(~df['exists']).sum()}  fps: {df['fps'].min()}-{df['fps'].max()}"
              f"  frames: {df['n_frames'].min()}-{df['n_frames'].max()}")


if __name__ == "__main__":
    main()
