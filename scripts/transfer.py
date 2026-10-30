"""Move the CrashTwin project to another machine. The CODE comes from GitHub; this copies the DATA and RESULTS.

  python crashtwin/scripts/transfer.py plan
  python crashtwin/scripts/transfer.py copy --tier results --dest E:/crashtwin_transfer
  python crashtwin/scripts/transfer.py pack --tier results --dest E:/                  (one .zip file)
  python crashtwin/scripts/transfer.py verify --dest E:/crashtwin_transfer
  # on the new machine: copy back into the new project folder
  python crashtwin/scripts/transfer.py copy --tier all --src E:/crashtwin_transfer --dest C:/CARLA_Project

Tiers (each includes the previous one):
  results   (~0.4 GB)  continue experiments, figures, paper: outputs/ without raw CARLA clips + models/ (YOLO)
  sim       (+41 GB)   raw CARLA clips (outputs/sim_v2/*/clips): re-tracking, E5, audits, more generation
  datasets  (+86 GB)   datasets/ (CCD, Nexar, Accid3nD): re-tracking real video (or download them again)
  all                  the whole project folder except .venv (also your lessons, recordings and other files)

Safe by design: resumable (a file already at the destination with the same size is skipped; each file is
written as .partial and then renamed), never deletes anything, never overwrites a different file.
Licensed data (Nexar, Accid3nD and features derived from them) must stay private: USB / private drive only."""
import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[2])   # project folder (contains crashtwin/)
TIERS = ["results", "sim", "datasets", "all"]
SKIP_DIRS = {".venv", "__pycache__", ".ipynb_checkpoints"}
MANIFEST = "TRANSFER_MANIFEST.json"


def tier_of(rel):
    """Smallest tier that contains this project-relative path (forward slashes)."""
    if rel.startswith("outputs/sim_v2/") and "/clips/" in rel:
        return "sim"
    if rel.startswith(("outputs/", "models/")):
        return "results"
    if rel.startswith("datasets/"):
        return "datasets"
    return "all"


def walk(src, exclude=None):
    """All files under src except virtual envs, caches, the _transfer staging folder and `exclude`
    (a destination inside the source must never copy itself)."""
    for d, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not (Path(d) == src and x == "_transfer")
                   and (exclude is None or (Path(d) / x).resolve() != exclude)]
        for f in files:
            if f.endswith((".partial", ".pyc")) or f == MANIFEST:
                continue
            full = Path(d) / f
            yield full.relative_to(src).as_posix(), full


def selected(src, tier, exclude=None):
    keep = TIERS[:TIERS.index(tier) + 1]
    return [(rel, full) for rel, full in walk(src, exclude) if tier_of(rel) in keep]


def gb(n):
    return f"{n / 1e9:.2f} GB"


def plan(src):
    sizes, counts = {t: 0 for t in TIERS}, {t: 0 for t in TIERS}
    for rel, full in walk(src):
        t = tier_of(rel)
        sizes[t] += full.stat().st_size
        counts[t] += 1
    total = 0
    print(f"Source: {src}  (.venv is never copied: recreate it with scripts/setup_env)")
    for t in TIERS:
        total += sizes[t]
        print(f"  {t:<9} adds {counts[t]:>7} files, {gb(sizes[t]):>10}   (tier total {gb(total)})")


def copy(src, dest, tier):
    files = selected(src, tier, exclude=dest)
    need = sum(f.stat().st_size for rel, f in files if not (dest / rel).exists())
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    if need > free - 1e9:
        sys.exit(f"Not enough space at {dest}: need {gb(need)}, free {gb(free)}")
    print(f"Copying tier '{tier}': {len(files)} files, {gb(need)} still to copy -> {dest}")
    done = skipped = 0
    conflicts = []
    t0 = time.time()
    for i, (rel, f) in enumerate(files, 1):
        out = dest / rel
        size = f.stat().st_size
        if out.exists():
            if out.stat().st_size == size:
                skipped += 1
            else:
                conflicts.append(rel)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".partial")
        shutil.copy2(f, tmp)
        os.replace(tmp, out)
        done += size
        if i % 500 == 0:
            print(f"  {i}/{len(files)} files, {gb(done)} copied, {time.time() - t0:.0f} s", flush=True)
    man = dest / MANIFEST
    old = json.loads(man.read_text()) if man.exists() else {"files": {}}
    old["files"].update({rel: f.stat().st_size for rel, f in files})
    old.update({"source": str(src), "tiers": sorted(set(old.get("tiers", [])) | {tier}),
                "updated": datetime.now().isoformat(timespec="seconds")})
    tmp = man.with_name(man.name + ".partial")
    tmp.write_text(json.dumps(old, indent=1))
    os.replace(tmp, man)
    print(f"Done: {gb(done)} copied, {skipped} already there.")
    if conflicts:
        print(f"{len(conflicts)} files NOT copied because a DIFFERENT file already exists at the destination "
              f"(nothing was overwritten), e.g. {conflicts[:3]}")


def pack(src, dest, tier):
    files = selected(src, tier, exclude=dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"crashtwin_{tier}_{datetime.now():%Y%m%d}.zip"
    if out.exists():
        sys.exit(f"{out} already exists; not overwriting it")
    tmp = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for rel, f in files:
            z.write(f, rel)
        z.writestr(MANIFEST, json.dumps({"source": str(src), "tiers": [tier],
                                         "files": {rel: f.stat().st_size for rel, f in files}}, indent=1))
    os.replace(tmp, out)
    print(f"Wrote {out} ({gb(out.stat().st_size)}, {len(files)} files). Unzip it inside the new project folder.")


def verify(dest):
    man = json.loads((dest / MANIFEST).read_text())
    bad = [rel for rel, size in man["files"].items()
           if not (dest / rel).exists() or (dest / rel).stat().st_size != size]
    print(f"{len(man['files']) - len(bad)}/{len(man['files'])} files present with the right size")
    if bad:
        print("missing or different:", bad[:20])
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["plan", "copy", "pack", "verify"])
    p.add_argument("--tier", choices=TIERS, default="results")
    p.add_argument("--src", default=str(ROOT), help="folder to copy from (default: this project folder)")
    p.add_argument("--dest", help="destination folder")
    a = p.parse_args()
    src = Path(a.src).resolve()
    if a.cmd == "plan":
        plan(src)
    elif not a.dest:
        sys.exit("--dest is required")
    elif a.cmd == "copy":
        copy(src, Path(a.dest).resolve(), a.tier)
    elif a.cmd == "pack":
        pack(src, Path(a.dest).resolve(), a.tier)
    else:
        verify(Path(a.dest).resolve())
