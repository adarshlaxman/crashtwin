"""Check that the CrashTwin environment is ready: packages, GPU, CARLA, weights, datasets, results, disk.
Works on Windows and Linux. Paths come from configs/paths.yaml (relative to the project folder),
CRASHTWIN_ROOT and CARLA_ROOT."""
import hashlib
import importlib
import os
import shutil
import socket
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
cfg = yaml.safe_load(CONFIG.read_text())
ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[2])   # project folder (contains crashtwin/)
YOLO_SHA256 = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"


def p(rel):
    return ROOT / rel


def ok(flag, msg):
    print(("[ OK ] " if flag else "[MISS] ") + msg)


print(f"Project folder: {ROOT}")
print("\n== Packages ==")
for name in ["carla", "torch", "torchvision", "ultralytics", "cv2", "numpy", "pandas", "pyarrow", "scipy",
             "sklearn", "matplotlib", "open3d", "lap", "yaml", "psutil", "huggingface_hub"]:
    try:
        mod = importlib.import_module(name)
        ok(True, f"{name} {getattr(mod, '__version__', '')}")
    except Exception as e:
        ok(False, f"{name}: {e}")

print("\n== GPU ==")
import torch  # noqa: E402
ok(torch.cuda.is_available(), f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"       {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("\n== CARLA ==")
carla_root = Path(os.environ.get("CARLA_ROOT") or cfg["carla_root"])
exe = carla_root / ("CarlaUE4.exe" if os.name == "nt" else "CarlaUE4.sh")
ok(exe.exists(), f"{exe.name} at {exe}" + ("" if exe.exists() else "  (set CARLA_ROOT to the CARLA 0.9.15 folder)"))
with socket.socket() as s:
    s.settimeout(1)
    running = s.connect_ex((cfg["carla"]["host"], cfg["carla"]["port"])) == 0
print(f"       server {'running' if running else 'not running (the generator starts it automatically)'}")

print("\n== YOLO weights ==")
w = p("models/yolo/yolo11s.pt")
ok(w.exists() and hashlib.sha256(w.read_bytes()).hexdigest() == YOLO_SHA256,
   f"{w} (SHA-256 must match WEIGHTS_PROVENANCE.md)")

print("\n== Datasets (only needed to re-run tracking or E5) ==")
crash_dir = p(cfg["real"]["ccd_crash_videos"])
ok(crash_dir.exists() and len(list(crash_dir.glob("*.mp4"))) == 1500,
   f"CCD crash videos: {len(list(crash_dir.glob('*.mp4'))) if crash_dir.exists() else 0}/1500")
normal_dir = p(cfg["real"]["ccd_normal_videos"])
ok(normal_dir.exists() and any(normal_dir.rglob("*.mp4")),
   f"CCD normal videos: {len(list(normal_dir.rglob('*.mp4'))) if normal_dir.exists() else 0}")
ok(p(cfg["real"]["ccd_labels"]).exists(), "CCD crash labels (Crash-1500.txt)")
zips = sorted(p(cfg["real"]["accid3nd_zips"]).glob("train*.zip"))
gb = sum(z.stat().st_size for z in zips) / 1e9
ok(gb > 46.5, f"Accid3nD / TUMTraf-A zips: {len(zips)}/5 files, {gb:.1f}/46.6 GB")
nexar = p(cfg["real"]["nexar"])
ok(nexar.exists() and any(nexar.iterdir()), "Nexar (needs Hugging Face terms accepted + login)")

print("\n== Results and caches (needed to continue experiments) ==")
for f in ("outputs/processed/features/ccd.npz", "outputs/processed/features/nexar.npz",
          "outputs/processed/features/sim_full.npz", "outputs/index/ccd.csv", "outputs/index/nexar.csv"):
    ok(p(f).exists(), f)
exps = [d for d in p("outputs/experiments").glob("E*") if (d / "metrics.json").exists()]
ok(len(exps) >= 26, f"finished experiment runs: {len(exps)} (26 on the lab PC)")
sims = sorted(p("outputs/sim_v2").glob("full_*/index.csv"))
ok(bool(sims), f"CARLA full run index: {sims[-1].parent.name if sims else 'none'}")

print("\n== Disk ==")
free = shutil.disk_usage(ROOT).free / 1e9
ok(free > 60, f"{free:.0f} GB free on the project drive (keep >= 30 GB free while generating)")
