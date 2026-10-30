"""File formats, storage guard, seeds and provenance (no CARLA import, unit-tested)."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CRASHTWIN_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or CRASHTWIN_ROOT.parent)   # folder that contains crashtwin/
ORIGINAL_ROOT = r"C:\CARLA_Project"   # where the 2026-10 data was made; index files store paths under it
GENERATOR_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = "1.0"

try:
    import pyarrow  # noqa: F401
    HAVE_PARQUET = True
except ImportError:  # documented fallback (decision D3): identical columns as .csv.gz
    HAVE_PARQUET = False


def local_path(p):
    """Index files store absolute paths from the machine that made them. If such a path doesn't exist
    here, re-anchor it on this machine's project folder (same relative layout)."""
    p = str(p)
    if os.path.exists(p):
        return Path(p)
    for old in (os.environ.get("CRASHTWIN_OLD_ROOT"), ORIGINAL_ROOT):
        if old and p.replace("/", "\\").lower().startswith(old.replace("/", "\\").lower()):
            return PROJECT_ROOT / p[len(old):].lstrip("\\/").replace("\\", "/")
    return Path(p)


# ---------------------------------------------------------------- tables

def write_table(df, path_no_ext):
    """Write a DataFrame as Parquet (zstd) or, if pyarrow is missing, as csv.gz."""
    path_no_ext = Path(path_no_ext)
    if HAVE_PARQUET:
        p = path_no_ext.with_suffix(".parquet")
        df.to_parquet(p, index=False, compression="zstd")
    else:
        p = path_no_ext.with_suffix(".csv.gz")
        df.to_csv(p, index=False, compression="gzip")
    return p


def read_table(path_no_ext):
    import pandas as pd
    path_no_ext = Path(path_no_ext)
    p = path_no_ext.with_suffix(".parquet")
    if p.exists():
        return pd.read_parquet(p)
    return pd.read_csv(path_no_ext.with_suffix(".csv.gz"))


def table_exists(path_no_ext):
    path_no_ext = Path(path_no_ext)
    return path_no_ext.with_suffix(".parquet").exists() or path_no_ext.with_suffix(".csv.gz").exists()


# ---------------------------------------------------------------- ragged arrays

def pack_ragged(arrays, dtype, width):
    """List of [n_i, width] arrays -> (concatenated [N, width], offsets [len+1])."""
    offsets = np.zeros(len(arrays) + 1, dtype=np.int32)
    for i, a in enumerate(arrays):
        offsets[i + 1] = offsets[i] + len(a)
    data = (np.concatenate([np.asarray(a, dtype=dtype).reshape(-1, width) for a in arrays])
            if offsets[-1] else np.zeros((0, width), dtype=dtype))
    return data, offsets


def unpack_ragged(data, offsets):
    return [data[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


# ---------------------------------------------------------------- depth

DEPTH_MAX_CM = 65535


def depth_bgra_to_metres(bgra):
    """CARLA depth camera encoding: (R + G*256 + B*65536) / (256^3 - 1) * 1000 m."""
    b = bgra.astype(np.float64)
    return (b[..., 2] + b[..., 1] * 256.0 + b[..., 0] * 65536.0) / 16777215.0 * 1000.0


def metres_to_cm_u16(metres):
    """uint16 centimetres; >= 655.35 m (sky) saturates at 65535; 0 is reserved for invalid."""
    cm = np.rint(np.asarray(metres) * 100.0)
    return np.clip(cm, 1, DEPTH_MAX_CM).astype(np.uint16)


# ---------------------------------------------------------------- seeds

def attempt_seed(matrix_version, slot_id, attempt):
    h = hashlib.sha256(f"{matrix_version}|{slot_id}|{attempt}".encode()).digest()
    return int.from_bytes(h[:4], "little") % (2 ** 31)


# ---------------------------------------------------------------- storage guard

def dir_size_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class StorageGuard:
    """Stops a run before the run cap or the free-space floor would be crossed."""

    def __init__(self, run_dir, cap_gb, min_free_gb, reserve_gb, disk_usage=shutil.disk_usage):
        self.run_dir = Path(run_dir)
        self.cap = cap_gb * 1e9
        self.min_free = min_free_gb * 1e9
        self.reserve = reserve_gb * 1e9
        self.disk_usage = disk_usage
        self.used = dir_size_bytes(self.run_dir) if self.run_dir.exists() else 0

    def add(self, nbytes):
        self.used += nbytes

    def check(self):
        """Return None if another attempt fits, else a reason string."""
        free = self.disk_usage(str(self.run_dir.anchor or self.run_dir)).free
        if self.used + self.reserve > self.cap:
            return f"run_cap: used {self.used / 1e9:.2f} GB + reserve > cap {self.cap / 1e9:.1f} GB"
        if free - self.reserve < self.min_free:
            return f"min_free: {free / 1e9:.1f} GB free - reserve < {self.min_free / 1e9:.0f} GB"
        return None


# ---------------------------------------------------------------- no-overwrite

def make_new_dir(path):
    """Create a directory that must not exist yet (never overwrite a clip or run)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing {path}")
    path.mkdir(parents=True)
    return path


# ---------------------------------------------------------------- provenance

def git_info():
    try:
        commit = subprocess.run(["git", "-C", str(CRASHTWIN_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        status = subprocess.run(["git", "-C", str(CRASHTWIN_ROOT), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=10)
        return {"commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def source_hashes():
    out = {}
    for p in sorted(GENERATOR_DIR.rglob("*.py")):
        out[str(p.relative_to(CRASHTWIN_ROOT)).replace("\\", "/")] = \
            hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def provenance(carla_client=None):
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "schema_version": SCHEMA_VERSION, "git": git_info(), "source_sha256": source_hashes()}
    for mod in ("numpy", "pandas", "cv2", "pyarrow"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            info[mod] = None
    if carla_client is not None:
        info["carla_server"] = carla_client.get_server_version()
        info["carla_client"] = carla_client.get_client_version()
    return info


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
