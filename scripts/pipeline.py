"""CrashTwin pipeline controller: crash-safe, resumable, survives a lost connection.

  python crashtwin/scripts/pipeline.py status
  python crashtwin/scripts/pipeline.py launch <job>     (detached background process + log file)
  python crashtwin/scripts/pipeline.py run <job>        (same, in the foreground)

Every job skips work that is already finished and resumes work that is not:
  generator   -> run.py --resume (attempts logged first, atomic clip folders)
  tracking    -> one parquet per video, written atomically; existing ones are skipped
  caches      -> written atomically; existing caches are skipped
  experiments -> per-epoch atomic checkpoints; runs with metrics.json are skipped
Re-launching a job after an interruption simply continues it.

Works on Windows and Linux. Run it with the project's venv python; jobs use the same interpreter.
More seeds: set CRASHTWIN_SEEDS=0,1,2,3,4 and launch 'experiments' (finished seeds are skipped)."""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("CRASHTWIN_ROOT") or Path(__file__).resolve().parents[2])   # project folder (contains crashtwin/)
PY = sys.executable
SRC = ROOT / "crashtwin" / "src"
LOGS = ROOT / "outputs" / "logs"
SIM = ROOT / "outputs" / "sim_v2"
SEEDS = tuple(int(s) for s in os.environ.get("CRASHTWIN_SEEDS", "0,1").split(","))


def latest_run(prefix):
    runs = sorted(SIM.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def experiment_plan():
    real = lambda f: [f"ccd:train@{f}", f"nexar:train@{f}"]          # noqa: E731
    rval = ["ccd:val", "nexar:val"]
    plan = []
    for s in SEEDS:
        plan.append((f"E1_sim_s{s}", ["sim:train"], ["sim:val"], s, ""))
        for f in (0.1, 0.25, 0.5, 1.0):
            plan.append((f"E2_real{int(f * 100)}_s{s}", real(f), rval, s, ""))
            plan.append((f"E3_sim+real{int(f * 100)}_s{s}", ["sim:train"] + real(f), rval, s, ""))
        for mod in ("radar", "lidar", "depth", "radar,lidar,depth,ego"):
            tag = "all" if "," in mod else mod
            plan.append((f"E4_cam+{tag}_s{s}", ["sim:train"], ["sim:val"], s, mod))
    return plan


def jobs():
    full = latest_run("full")
    j = {
        "track_ccd": [[PY, str(SRC / "features" / "track_videos.py"), "--index", str(ROOT / "outputs/index/ccd.csv"),
                       "--dataset", "ccd"]],
        "track_nexar": [[PY, str(SRC / "features" / "track_videos.py"), "--index",
                         str(ROOT / "outputs/index/nexar.csv"), "--dataset", "nexar"]],
        "caches_real": [[PY, str(SRC / "features" / "build_cache.py"), "--dataset", "ccd", "--index",
                         str(ROOT / "outputs/index/ccd.csv")],
                        [PY, str(SRC / "features" / "build_cache.py"), "--dataset", "nexar", "--index",
                         str(ROOT / "outputs/index/nexar.csv")]],
        "experiments": [[PY, str(SRC / "models" / "train.py"), "--name", n, "--train", *tr, "--val", *va,
                         "--seed", str(s)] + (["--sensors", mod] if mod else [])
                        for n, tr, va, s, mod in experiment_plan()],
    }
    if full is not None:
        j["generator"] = [[PY, str(SRC / "generator" / "run.py"), "--supervise", "--resume", str(full)]]
        j["track_sim"] = [[PY, str(SRC / "features" / "track_videos.py"), "--index", str(full / "index.csv"),
                           "--dataset", "sim_full"]]
        j["cache_sim"] = [[PY, str(SRC / "features" / "build_cache.py"), "--dataset", "sim", "--index",
                           str(full / "index.csv"), "--tracks", "sim_full", "--sensors"]]
    return j


def pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
            return str(pid) in out
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def wait_for(job):
    """Block until a previously launched job's process has exited."""
    import time
    f = LOGS / f"{job}.pid"
    if f.exists():
        pid = int(f.read_text())
        while pid_alive(pid):
            time.sleep(60)


def run(job):
    if ":" in job:                       # experiments:E2  -> only runs whose name starts with E2
        base, prefix = job.split(":", 1)
        cmds = [c for c in jobs()[base] if c[c.index("--name") + 1].startswith(prefix)]
    elif job == "real_chain":            # wait for Nexar tracking, then real caches, then E2
        wait_for("track_nexar")
        for j in ("track_nexar", "track_ccd", "caches_real", "experiments:E2"):
            run(j)
        return
    elif job == "sim_chain":             # wait for the generator, then sim tracking, cache, all experiments
        wait_for("generator")
        for j in ("track_sim", "cache_sim", "experiments"):
            run(j)
        return
    else:
        cmds = jobs()[job]
    failed = []
    for cmd in cmds:
        print(">>", " ".join(cmd[1:]), flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            failed.append(rc)
            print(f"step failed with exit code {rc}", flush=True)
            if not job.startswith("experiments"):   # data steps must succeed before the next step
                print(f"re-launch '{job}' to resume", flush=True)
                sys.exit(rc)
    if failed:   # experiment runs are independent; unfinished ones resume from their checkpoint
        print(f"{len(failed)} experiment step(s) failed; re-launch '{job}' to resume them", flush=True)


def launch(job):
    LOGS.mkdir(parents=True, exist_ok=True)
    safe = job.replace(":", "_")          # ':' would create an NTFS alternate data stream
    log = LOGS / f"{safe}_{datetime.now():%Y%m%d_%H%M%S}.log"
    if os.name == "nt":
        detach = {"creationflags": 0x00000008 | 0x00000200}   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        detach = {"start_new_session": True}                   # survives closing the terminal / SSH session
    with open(log, "w") as f:
        p = subprocess.Popen([PY, "-u", __file__, "run", job], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, close_fds=True, **detach)
    (LOGS / f"{safe}.pid").write_text(str(p.pid))
    print(f"launched {job} (pid {p.pid}), log: {log}")


def status():
    import pandas as pd
    print(f"== {datetime.now():%Y-%m-%d %H:%M:%S}")
    for prefix in ("pilot", "full"):
        r = latest_run(prefix)
        if r is None or not (r / "run_log.jsonl").exists():
            continue
        recs = [json.loads(l) for l in (r / "run_log.jsonl").read_text().splitlines()]
        done = [x for x in recs if x.get("status") not in (None, "started") and "slot_id" in x]
        acc = {x["slot_id"] for x in done if x["status"] == "accepted"}
        slots = json.loads((r / "run_config.json").read_text())["slots"]
        print(f"{r.name}: {len(acc)}/{len(slots)} slots accepted, {len(done)} attempts, "
              f"{sum(1 for x in recs if x.get('status') == 'process_abort')} restarts")
    for ds, idx in (("ccd", "outputs/index/ccd.csv"), ("nexar", "outputs/index/nexar.csv")):
        n = len(pd.read_csv(ROOT / idx)) if (ROOT / idx).exists() else 0
        t = len(list((ROOT / "outputs/processed/tracks" / ds).glob("*.parquet")))
        print(f"tracks {ds}: {t}/{n}")
    full = latest_run("full")
    if full is not None and (full / "index.csv").exists():
        n = len(pd.read_csv(full / "index.csv"))
        t = len(list((ROOT / "outputs/processed/tracks/sim_full").glob("*.parquet")))
        print(f"tracks sim_full: {t}/{n} written clips")
    feats = sorted(p.name for p in (ROOT / "outputs/processed/features").glob("*.npz"))
    print("caches:", feats)
    plan = experiment_plan()
    fin = [n for n, *_ in plan if (ROOT / "outputs/experiments" / n / "metrics.json").exists()]
    part = [n for n, *_ in plan if (ROOT / "outputs/experiments" / n / "ckpt_last.pt").exists() and n not in fin]
    print(f"experiments: {len(fin)}/{len(plan)} finished; in progress: {part}")
    for f in sorted(LOGS.glob("*.pid")):
        pid = int(f.read_text())
        print(f"job {f.stem}: pid {pid} ({'RUNNING' if pid_alive(pid) else 'not running'})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    elif cmd == "launch":
        launch(sys.argv[2])
    elif cmd == "run":
        run(sys.argv[2])
    else:
        sys.exit(__doc__)
