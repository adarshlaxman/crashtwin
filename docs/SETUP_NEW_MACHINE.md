# Moving CrashTwin to a new machine

Status on 2026-10-30: everything that needed compute is finished on the lab PC (Windows 11, RTX 4500 Ada 24 GB).
The new machine continues from exactly this state. Nothing is re-generated or re-trained unless you choose to.

## 0. What lives where

| What | Where (inside the project folder) | Size | How it moves |
|---|---|---|---|
| Code, configs, docs, paper, slides | `crashtwin/` (git) | < 5 MB | GitHub: `github.com/adarshlaxman/crashtwin` (keep it **private**) |
| Results, feature caches, checkpoints, predictions, indexes, splits, logs, YOLO weights | `outputs/` (without raw clips), `models/` | 0.4 GB | **in the GitHub repo** as `results_kit/crashtwin_results_20261030.zip` (Git LFS, arrives with the clone), or `transfer.py --tier results` |
| Raw CARLA clips (pilot + full run) | `outputs/sim_v2/*/clips/` | 44 GB | `transfer.py --tier sim` (external disk) |
| Real datasets (CCD, Nexar, Accid3nD) | `datasets/` | 93 GB | `transfer.py --tier datasets`, or download again (section 4) |
| Python environment | `.venv/` | 5 GB | never copied: `setup_env` rebuilds it |

The **project folder** is the folder that contains `crashtwin/` (on the lab PC: `C:\CARLA_Project`).
It can be anywhere on the new machine: the code finds it by itself (or set `CRASHTWIN_ROOT`).
Index files store the lab PC's absolute paths (`C:\CARLA_Project\...`); the code re-anchors them on the
new project folder automatically (`io_utils.local_path`).

**What you need for which next step**

| Next step | Needs |
|---|---|
| More seeds, new models, figures, E6, paper | results tier only (0.4 GB) |
| E5 again, audits, re-tracking CARLA video, finishing the 260 unrun slots | + sim tier (44 GB) and CARLA |
| Re-tracking real video (e.g. a different detector) | + datasets tier (93 GB) |

## 1. On the lab PC (do once)

### 1a. Push the code to GitHub
Already done on the lab PC: the code is committed on branch `main` (code, configs, docs, paper, slides;
`.gitignore` keeps data, weights and logs out), and the remote is
`https://adarshlaxman@github.com/adarshlaxman/crashtwin.git`. The user name in that URL matters: this
PC's saved GitHub login belongs to someone else (`ishaantandon`), and without it git would push as them.
Commits are signed `Adarsh L <adarshlaxman@users.noreply.github.com>` (set for this repo only).

1. On github.com, signed in as `adarshlaxman`: **New repository** → name `crashtwin` → **Private** →
   do NOT add a README, .gitignore or licence (the repo must be empty).
2. In a terminal on the lab PC:
   ```
   cd C:\CARLA_Project\crashtwin
   git push -u origin main
   ```
3. A "Connect to GitHub" window opens: sign in as **adarshlaxman**. If the browser is already signed in
   as someone else, choose **Token** instead and paste a fine-grained personal access token
   (github.com → Settings → Developer settings → Fine-grained tokens → only the `crashtwin` repository,
   permission *Contents: Read and write*).

### 1b. Copy the data
- **Results only (enough to continue experiments):** nothing to carry: the results kit is in the GitHub repo
  (`results_kit/`, Git LFS). A copy also stays in `C:\CARLA_Project\_transfer\`.
- **Everything:** plug in an external disk (needs ~140 GB; say it is `E:`), then from `C:\CARLA_Project`:
  ```
  .venv\Scripts\python.exe crashtwin\scripts\transfer.py plan
  .venv\Scripts\python.exe crashtwin\scripts\transfer.py copy --tier datasets --dest E:\crashtwin_transfer
  .venv\Scripts\python.exe crashtwin\scripts\transfer.py verify --dest E:\crashtwin_transfer
  ```
  `--tier datasets` = results + sim + datasets. Use `--tier all` to also take your other files
  (lessons, manual-drive recordings, …). If the copy is interrupted, run the same command again:
  finished files are skipped. Nothing is ever deleted or overwritten.

## 2. On the new machine

### 2a. Requirements
- NVIDIA GPU with a recent driver (R560 or newer; PyTorch uses CUDA 12.6). CARLA needs ≥ 8 GB VRAM;
  training alone needs ~2 GB.
- Windows 10/11 or Ubuntu 20.04/22.04.
- Disk: 10 GB (results only) up to ~200 GB (everything + room to generate more CARLA data; the generator
  stops itself at 40 GB per run or when < 30 GB would be left free).
- Software: **Git**, **Python 3.10** (3.10.11 on the lab PC; keep 3.10: every pinned package, including
  the `carla==0.9.15` client, was installed and tested with it),
  **CARLA 0.9.15** (only for generation / E5).

### 2b. Install
1. **Python 3.10.11:** Windows: python.org installer, tick "Add to PATH" and "py launcher".
   Ubuntu: `sudo apt install python3.10 python3.10-venv` (deadsnakes PPA on 20.04).
2. **CARLA 0.9.15** (skip if you will only train/evaluate):
   https://github.com/carla-simulator/carla/releases/tag/0.9.15 → `CARLA_0.9.15.zip` (Windows) or
   `CARLA_0.9.15.tar.gz` (Linux). The base package contains every town we use (Town01–05, Town10HD);
   AdditionalMaps are not needed. Then point the code at it:
   - Windows: `setx CARLA_ROOT "C:\CARLA_0.9.15\WindowsNoEditor"` (the folder with `CarlaUE4.exe`), then open a new terminal.
   - Linux: `echo 'export CARLA_ROOT=$HOME/CARLA_0.9.15' >> ~/.bashrc` (the folder with `CarlaUE4.sh`).
3. **Project folder + code:**
   ```
   mkdir C:\CARLA_Project            (Linux: mkdir ~/CARLA_Project)
   cd C:\CARLA_Project
   git clone https://github.com/adarshlaxman/crashtwin.git crashtwin
   ```
4. **Python environment** (creates `.venv`, installs pinned packages, downloads + checks YOLO weights):
   - Windows: `powershell -ExecutionPolicy Bypass -File C:\CARLA_Project\crashtwin\scripts\setup_env.ps1`
   - Linux: `bash ~/CARLA_Project/crashtwin/scripts/setup_env.sh`
   - Exact copy of the lab environment if ever needed: `pip install -r crashtwin/requirements-lock.txt`.
5. **Data and results** (into the project folder):
   - Results kit (came with the clone): Windows `tar -xf crashtwin\results_kit\crashtwin_results_20261030.zip -C C:\CARLA_Project` ·
     Linux `unzip crashtwin/results_kit/crashtwin_results_20261030.zip -d ~/CARLA_Project`
     (on Linux install Git LFS before cloning: `sudo apt install git-lfs && git lfs install`)
   - External disk: `.venv\Scripts\python.exe crashtwin\scripts\transfer.py copy --tier all --src E:\crashtwin_transfer --dest C:\CARLA_Project`
6. **Check** (from the project folder; on Linux use `.venv/bin/python` and `/`):
   ```
   .venv\Scripts\python.exe crashtwin\scripts\check_env.py        # every line should be [ OK ] for what you copied
   cd crashtwin\tests ; ..\..\.venv\Scripts\python.exe -m unittest test_core ; cd ..\..   # 24 tests, OK
   .venv\Scripts\python.exe crashtwin\scripts\pipeline.py status  # experiments: 26/26 finished
   ```

## 3. Continue the work

All long jobs run detached (they survive closing the terminal or losing the connection) and resume
where they stopped: `pipeline.py launch <job>`; check with `pipeline.py status`; logs in `outputs/logs/`.

| Task | Command (Windows; Linux: `.venv/bin/python`, `/`, `export VAR=...`) | Time on lab GPU |
|---|---|---|
| **More seeds (recommended first)** | `$env:CRASHTWIN_SEEDS="0,1,2,3,4"; .venv\Scripts\python.exe crashtwin\scripts\pipeline.py launch experiments` | ~10 min per seed (13 runs) |
| Refresh tables + figures | `.venv\Scripts\python.exe crashtwin\src\eval\aggregate.py` | < 1 min |
| E6 breakdown | `.venv\Scripts\python.exe crashtwin\src\eval\e6_breakdown.py` | < 1 min |
| E5 again (needs sim tier) | `...\e5_kinematics.py traffic_carla --run outputs\sim_v2\full_20261029_182453` then `... compare_traffic` | ~5 min |
| Finish the 260 unrun slots (needs sim tier + CARLA) | raise `full_run_cap_gb` in `configs/storage.yaml` from 40 to 45 (your budget decision), then `pipeline.py launch generator` | ~1 h |

Finished runs are never re-trained (a run with `metrics.json` is skipped). New CARLA clips need a new
feature cache under a new name (`build_cache.py` never overwrites an existing cache).
On Linux, start CARLA yourself before generating, headless: `$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen &`.

## 4. Downloading the datasets again (instead of copying)

| Dataset | Source | Put it in |
|---|---|---|
| CCD (Car Crash Dataset) | https://github.com/Cogito2012/CarCrashDataset → Google Drive folder https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F | `datasets/real_ccd/` (`Crash-1500.txt`, `videos/Crash-1500/*.mp4`, `videos/Normal/*.mp4`, `official_split/{train,test}.txt` from `vgg16_features/`) |
| Nexar collision prediction | https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction (gated: accept the terms with your account) | `datasets/real_nexar/` — `hf auth login` with a **new** read token, then `hf download nexar-ai/nexar_collision_prediction --repo-type dataset --local-dir datasets/real_nexar` |
| Accid3nD / TUMTraf-A (E5 only) | https://mediatum.ub.tum.de/node?id=1796514 | `datasets/real_accid3nd/zips/` (5 × `train*.zip`, 46.6 GB) |

Keep `outputs/index/*.csv` from the results tier (they hold the grouped CCD split and the Nexar partitions);
`build_indexes.py` refuses to overwrite them anyway.

## 5. Things learned the hard way (details: `docs/INCIDENTS.md`)
- Use `Town03`, never `Town03_Opt` (its Traffic Manager file is broken in 0.9.15).
- The CARLA Python client sometimes aborts natively (0xC0000409): always generate with `--supervise`
  (`pipeline.py` does); every attempt is logged before it runs, so nothing is replayed or lost.
- On Windows, training processes can exit with 0xC0000409 **after** saving `metrics.json`: harmless.
- Keep ports 2000 (CARLA) and 8000 (Traffic Manager) free.
- A full run stops itself at its 40 GB cap; that is intended (your storage budget).

## 6. Security and licences
- Keep the GitHub repo **private** until the supervisor has reviewed the Nexar licence (decision D6)
  and the paper is submitted.
- Never commit tokens. Delete the old `carla-pc` Hugging Face token (it appeared in a screenshot);
  create a new one on the new machine.
- Nexar and Accid3nD data, and everything derived from them (tracks, caches, predictions), stay private
  (USB / private drive only). CCD is MIT. YOLO11 weights are AGPL-3.0 and are not in git.
