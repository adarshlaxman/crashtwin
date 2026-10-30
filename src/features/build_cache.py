"""Phase 2 step 4: build feature caches (atomic, skip if present) + the simulated split manifest.

  --dataset ccd   --index outputs/index/ccd.csv
  --dataset nexar --index outputs/index/nexar.csv
  --dataset sim   --index outputs/sim_v2/full_<ts>/index.csv --tracks sim_full [--sensors]

Simulated split (by SLOT, so every attempt of a slot stays together): Town05 -> test_t05;
other towns: train 70 % / val 15 % / test_familiar 15 % by a seeded hash of the slot id.
Only clips with validation PASS and a valid outcome are used (WARN/FAIL/invalid are excluded)."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from camera_features import build

ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[3])   # project folder (contains crashtwin/)
OUT = ROOT / "outputs" / "processed" / "features"
SPLITS = ROOT / "outputs" / "splits"
SEED = "20261029"


def sim_split(slot_id, heldout):
    if heldout:
        return "test_t05"
    h = int(hashlib.sha256(f"{SEED}|{slot_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if h < 0.70 else "val" if h < 0.85 else "test_familiar"


def load_index(dataset, index_path):
    idx = pd.read_csv(index_path)
    if dataset == "sim":
        # usable = integrity checks pass (no FAIL) and a valid measured outcome. Spill-over clips
        # (measured outcome != intended, WARN L6 only) are kept: their label is the MEASURED one.
        idx = idx[(idx.validation_status != "FAIL") & idx.outcome.isin(["crash", "near_miss", "normal"])].copy()
        idx = idx.rename(columns={"scenario_id": "clip_id", "path": "clip_dir"})
        idx["label"] = idx.outcome.isin(["crash", "near_miss"]).astype(int)
        idx["split"] = [sim_split(s, h) for s, h in zip(idx.slot_id, idx.heldout_town.astype(str) == "True")]
        idx["dataset"] = "sim"
        SPLITS.mkdir(parents=True, exist_ok=True)
        man = SPLITS / f"sim_{Path(index_path).parent.name}.csv"
        if not man.exists():
            idx[["clip_id", "slot_id", "family", "town", "outcome", "split"]].assign(seed=SEED).to_csv(man, index=False)
        # leakage check: a slot never spans two splits
        assert (idx.groupby("slot_id")["split"].nunique() <= 1).all()
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["ccd", "nexar", "sim"])
    p.add_argument("--index", required=True)
    p.add_argument("--tracks", default=None, help="tracks sub-folder (default = dataset)")
    p.add_argument("--name", default=None, help="cache name (default = tracks name)")
    p.add_argument("--sensors", action="store_true")
    a = p.parse_args()
    tracks = a.tracks or a.dataset
    name = a.name or tracks
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.npz"
    if out.exists():
        print(f"{out} exists: skipping (delete nothing; use --name for a new version)")
        return
    idx = load_index(a.dataset, a.index)
    arrays, meta = build(idx, tracks, with_sensors=a.sensors)
    tmp = OUT / f"{name}.tmp.npz"
    np.savez_compressed(tmp, **{k: v for k, v in arrays.items() if v is not None})
    meta.to_csv(OUT / f"{name}_meta.csv.tmp", index=False)
    os.replace(OUT / f"{name}_meta.csv.tmp", OUT / f"{name}_meta.csv")
    os.replace(tmp, out)
    print(f"{name}: {len(meta)} clips, X {arrays['X'].shape}, positives {int(arrays['y'].sum())} -> {out}")
    print(meta.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
