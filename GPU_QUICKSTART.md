# GPU quickstart: resume CrashTwin on the new machine

One `git clone` brings everything needed to continue: the code **and** the results kit
(`results_kit/crashtwin_results_20261030.zip`, 318 MB, Git LFS). You combine the two in one project folder,
check the GPU, and carry on exactly where the lab PC stopped (26 of 26 training runs finished).
Full details and data downloads: [docs/SETUP_NEW_MACHINE.md](docs/SETUP_NEW_MACHINE.md).

## How long it takes

| Step | Time |
|---|---|
| Install Git and Python 3.10 | ~10 min |
| Clone the repo (code + 318 MB results kit) | 1–5 min, depends on internet speed |
| Combine: unzip the results kit | ~1 min |
| `setup_env` (packages, including ~3 GB of PyTorch CUDA) | 10–20 min |
| Checks (GPU, tests, status) | ~2 min |
| **Total until you can work** | **~30–40 min** |
| First job: 3 more seeds (39 training runs) | ~30 min on an RTX 4500 Ada-class GPU |
| Optional: install CARLA 0.9.15 (only to generate new data) | 30–60 min, large download |
| Optional: finish the 260 unrun CARLA slots | ~1 h (needs the raw clips and CARLA) |
| Optional: re-track all real video | several hours (needs the 93 GB datasets) |

## 1. Get the code and the results (one clone)

Windows (PowerShell):
```
mkdir C:\CARLA_Project
cd C:\CARLA_Project
git clone https://github.com/adarshlaxman/crashtwin.git crashtwin
```
Linux: install Git LFS first (`sudo apt install git-lfs && git lfs install`), then the same commands
with `~/CARLA_Project`. The repo is private: sign in as `adarshlaxman` when asked.

Check that the kit really downloaded: `crashtwin\results_kit\crashtwin_results_20261030.zip` must be
**~318 MB**. If it is only ~130 bytes, Git LFS was missing: install it (Git for Windows includes it;
Linux: `sudo apt install git-lfs && git lfs install`), then run `git lfs pull` inside `crashtwin`.

## 2. Combine code and results in one project folder

The project folder is the folder that **contains** `crashtwin\`. Unzip the kit into it (fresh folder only):
```
tar -xf crashtwin\results_kit\crashtwin_results_20261030.zip -C C:\CARLA_Project        # Windows
unzip crashtwin/results_kit/crashtwin_results_20261030.zip -d ~/CARLA_Project            # Linux
```
Result:
```
C:\CARLA_Project\
  crashtwin\      from GitHub: code, configs, docs, paper, slides, results_kit\
  outputs\        from the kit: feature caches, tracks, 26 trained runs, results, indexes, splits, logs
  models\yolo\    from the kit: YOLO11s weights
  .venv\          made in step 3
```
Index files still contain the lab PC's paths (`C:\CARLA_Project\...`); the code maps them onto this
folder automatically, so any folder name and Linux both work.

## 3. Install the environment
```
powershell -ExecutionPolicy Bypass -File C:\CARLA_Project\crashtwin\scripts\setup_env.ps1   # Windows
bash ~/CARLA_Project/crashtwin/scripts/setup_env.sh                                          # Linux
```
It creates `.venv`, installs the pinned packages, checks the YOLO weights' SHA-256 and runs `check_env`.
It also creates `C:\CARLA_Project\CLAUDE.md`, a one-line link to `crashtwin\CLAUDE.md`. Open the project
folder in VS Code and Claude Code then loads the project briefing (rules, current state, work plan)
automatically. To start, tell Claude: **"read CLAUDE.md and continue"**.

## 4. Check the GPU and the state
Windows shown; on Linux use `.venv/bin/python` and `/`.
```
nvidia-smi                                                       # driver and GPU visible
.venv\Scripts\python.exe crashtwin\scripts\check_env.py           # "CUDA available: True" + GPU name
.venv\Scripts\python.exe crashtwin\scripts\pipeline.py status     # experiments: 26/26 finished
```

## 5. Work on the GPU
- **Training** (`src/models/train.py`) uses the GPU automatically (CPU fallback works, just slower).
- **Tracking** (`src/features/track_videos.py`) needs an NVIDIA GPU (YOLO on GPU 0, half precision).
- **Several GPUs:** pick one with `$env:CUDA_VISIBLE_DEVICES="1"` (Windows) or `CUDA_VISIBLE_DEVICES=1` (Linux).
- **Long jobs run detached:** they survive closing the terminal or losing the connection, resume where
  they stopped, and never redo finished runs. Watch with `pipeline.py status`, `nvidia-smi -l 5`, and the
  logs in `outputs\logs\`.

**First job: 3 more seeds (5 in total), ~30 min.** It makes small differences between configurations
trustworthy. Keep the same `CRASHTWIN_SEEDS` setting when you check status (65 runs in total):
```
$env:CRASHTWIN_SEEDS="0,1,2,3,4"                                          # Linux: export CRASHTWIN_SEEDS=0,1,2,3,4
.venv\Scripts\python.exe crashtwin\scripts\pipeline.py launch experiments
.venv\Scripts\python.exe crashtwin\scripts\pipeline.py status             # until 65/65 finished
.venv\Scripts\python.exe crashtwin\src\eval\aggregate.py                  # tables + figures, mean ± std over 5 seeds
.venv\Scripts\python.exe crashtwin\src\eval\e6_breakdown.py               # robustness table
```

## 6. Combine both machines (if you keep using the lab PC too)

**Recommended:** make the GPU machine the only place that runs jobs. Use the lab PC only if you need
CARLA generation there (it already has CARLA and the 44 GB of raw clips).

If you do use both:
- **Code** syncs through GitHub: `git pull` before you start, then `git add`, `git commit`, `git push` when you finish.
- **Results** do not sync through git. Move them with `transfer.py`, which never overwrites or deletes:
  1. On the machine with the new results, from the project folder:
     `.venv\Scripts\python.exe crashtwin\scripts\transfer.py pack --tier results --dest D:\usb`
     (writes `crashtwin_results_<date>.zip`).
  2. On the other machine, unzip it into a **temporary folder** (not the project folder), then merge:
     `.venv\Scripts\python.exe crashtwin\scripts\transfer.py copy --tier all --src D:\tmp_kit --dest C:\CARLA_Project`
     New runs are added; identical files are skipped. Files that differ (regenerated tables, figures,
     logs) are listed and left alone, which is expected.
  3. Run `aggregate.py` (and `e6_breakdown.py`): the tables now include the runs from both machines.
- **Never train the same run on both machines.** Split the work by seed: for example
  `CRASHTWIN_SEEDS=0,1,2` on one machine and `CRASHTWIN_SEEDS=0,1,3,4` on the other (finished runs are skipped).
- **Optional:** to share a new kit through GitHub, copy the packed zip into `crashtwin/results_kit/` under a
  new date, then commit and push. Git LFS uploads it. Free GitHub accounts have a monthly LFS download
  limit, so don't clone on many machines.

## 7. If something goes wrong

| Problem | Fix |
|---|---|
| Clone asks for a login | Private repo: sign in as `adarshlaxman` (or use a fine-grained token) |
| `results_kit` zip is ~130 bytes | Install Git LFS, then `git lfs pull` inside `crashtwin` |
| `check_env` says CUDA False | Update the NVIDIA driver (R560 or newer), then run `setup_env` again |
| Push fails with "You must have push access to verify locks" (PC with several GitHub logins) | `git config --local credential.https://github.com.username adarshlaxman` and `git config --local lfs.https://github.com/adarshlaxman/crashtwin.git/info/lfs.access basic` |
| Training ends with exit code 0xC0000409 on Windows after `metrics.json` is written | Harmless: CUDA teardown on Windows. The run is complete |
| A job stopped (power, crash, logout) | `pipeline.py launch <job>` again; it resumes and skips finished work |

**Keep the repository private:** the results kit contains Nexar-derived features (licence under supervisor
review) and the AGPL-3.0 YOLO11s weights.
