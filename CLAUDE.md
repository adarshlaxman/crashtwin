# CrashTwin: briefing for Claude (read fully before doing anything)

You are continuing a research project that was run with Claude Code on a lab PC until 30 Oct 2026.
That session's memory is NOT available here; this file replaces it. Keep it accurate: when a milestone
changes the state below, update "Current state" and "Work plan" (commit only with the user's approval).

## 1. The project in one paragraph
**CrashTwin** (Adarsh L, CAVE Labs, PES University; GitHub `adarshlaxman`, private repo `crashtwin`) is a
CARLA 0.9.15 digital twin for sim-to-real crash anticipation. A generator makes measured-outcome crash /
near-miss / normal clips with every sensor; a frozen YOLO11s + ByteTrack pipeline turns CARLA and real
dashcam video (CCD, Nexar) into the same 83 camera features (+8 sensor features in CARLA); a causal GRU
predicts crash probability per frame. Research questions: RQ1 sim-to-real (E1 CARLA only, E2 real only at
10/25/50/100 %, E3 CARLA + real at the same fractions), RQ2 sensor fusion in CARLA (E4), RQ3 twin fidelity
(E5, traffic kinematics vs the real Accid3nD highway data), RQ4 conditions (E6). Target: IEEE ITSC/IV
workshop or arXiv paper (`paper/draft.md`).

## 2. Rules from the user (always follow; ask when in doubt)
**Files and data**
- Never delete or overwrite existing files without the user's explicit permission: datasets, CARLA outputs,
  logs, previous or failed/incomplete runs, results, and the user's other folders in the project
  (lessons, recordings, "mechblock", Blender/Unity files). Write new results to NEW versioned folders.
  No cleanup of failed runs. `build_cache.py` and `transfer.py` already refuse to overwrite: keep it so.
- Never upload or publish data. The only exception the user approved: `results_kit/*.zip` in this
  PRIVATE repo (Git LFS). Keep the repo private (Nexar licence still under supervisor review).
- Storage: at most 40 GB of CARLA data per run, and always keep >= 30 GB free on the project drive.

**Git**
- Commit and push only when the user asks (ask each time; approval does not carry over).
- Code, configs, docs, paper and slides only (+ the approved results kit). Never commit tokens or data.
- Never use another person's git identity or saved login. (The lab PC had someone else's global identity
  and GitHub login; this repo sets its own: `Adarsh L <adarshlaxman@users.noreply.github.com>`.)

**Science (do not break these)**
- Town05 is held out: never in training or validation.
- Never split by frame: CARLA splits by scenario slot, CCD by source video (grouped split), Nexar uses the
  official partitions.
- E1 is strictly CARLA-only, including model selection (CARLA validation set only).
- No simulator ground truth ever reaches a real-video model: real and CARLA video get identical
  camera-only inputs.
- Labels are measured (collision sensor + predicted-overlap TTC), never taken from the scenario's intent.
- Report only real results read from result files. Never estimate or invent numbers. If an earlier
  statement was wrong, say so plainly and fix it.

**Working style the user expects**
- Short, simple English answers; tables and numbered steps; give time estimates.
- Work autonomously and resiliently: long jobs run detached and resumable (`pipeline.py launch`), so a
  lost connection never loses work. Still ask before destructive, outward-facing or budget decisions.

## 3. Current state (30 Oct 2026, lab PC: Windows 11, RTX 4500 Ada 24 GB)
- **All compute is finished.** Pilot: 243 clips, 0 sync failures. Full run `full_20261029_182453`:
  stopped at its 40 GB cap after 3,040 of 3,300 slots (the 260 unrun slots are Town10HD normal driving);
  2,576 clips written, **1,877 usable** (852 crash, 276 near-miss, 749 normal; train 1,031 / val 220 /
  test-familiar 245 / test-Town05 381).
- **26/26 training runs** (13 configurations x 2 seeds), E5 and E6 done; `outputs/results/RESULTS.md`,
  figures, `paper/draft.md` section 7 and the slides `docs/CrashTwin_progress_20261030.pptx` are filled.
- **Headline results** (mean of 2 seeds):

| Training data | Nexar public / private mAP | CCD AP (grouped split) |
|---|---|---|
| E1 CARLA only | 0.621 / 0.638 | 0.249 (below the 0.31 chance level) |
| E2 real 10 % → 100 % | 0.586 / 0.526 → 0.703 / 0.681 | 0.678 → 0.867 |
| E3 CARLA + real 10 % → 100 % | 0.619 / 0.585 → 0.707 / 0.714 | 0.535 → 0.844 |

  E4 (Town05 AP): camera 0.837, + radar 0.930, + LiDAR 0.926, + depth 0.940, + all 0.971.
  E5: CARLA highway traffic is slower and calmer than real (speed 8.1 vs 18.8 m/s; headway 2.15 vs
  1.58 s; speed/gap/headway KS p < 1e-30, TTC p < 1e-8). E6: night/rain effects unclear (small groups);
  CARLA-only training transfers best to Nexar highway videos (0.86 vs 0.69 real-only).
  CCD's official split shares 107 of 133 source videos between train and test (we use a grouped split).
- **Not on this machine unless copied:** raw CARLA clips (44 GB) and datasets (93 GB). They are only
  needed to generate CARLA data, re-run E5, or re-track video. Everything else is in the results kit.

## 4. Start-of-session checklist (in this order)
1. Know the layout: the **project folder** is the folder that contains `crashtwin/` (code). Next to it:
   `outputs/` (results, from `results_kit/`), `models/yolo/`, `.venv/`, and maybe `datasets/`.
   `CRASHTWIN_ROOT` overrides the project folder; `CARLA_ROOT` points to CARLA 0.9.15.
2. If `outputs/` is missing: the results kit was not unzipped. Follow `GPU_QUICKSTART.md` step 2.
3. Run from the project folder (Linux: `.venv/bin/python`, `/`):
   `.venv\Scripts\python.exe crashtwin\scripts\pipeline.py status` → expect `experiments: 26/26 finished`
   (more if extra seeds ran). Also `git -C crashtwin status` and `git -C crashtwin log --oneline -5`.
4. New machine or errors: `.venv\Scripts\python.exe crashtwin\scripts\check_env.py` (GPU must be OK).
5. Read as needed: `outputs/results/RESULTS.md`, `paper/draft.md` §7, `docs/INCIDENTS.md`, `PLAN.md`.
6. Tell the user the state in a few lines, then continue with the next unfinished item of the work plan,
   or with what the user asks.

## 5. Work plan (continue in this order; tick items off here when done)
1. [ ] **More seeds** (~30 min on the GPU): `CRASHTWIN_SEEDS=0,1,2,3,4` then `pipeline.py launch
   experiments` (finished runs are skipped; status with the same variable shows 65 runs). Then
   `src/eval/aggregate.py` and `src/eval/e6_breakdown.py`, and update `paper/draft.md` §7 tables
   (mean ± std over 5 seeds). Say which differences are now larger than the seed noise.
2. [ ] **Why CARLA fails on CCD**: split the CCD grouped test set by `ego_involved` (column in
   `outputs/processed/features/ccd_meta.csv`) and score the existing predictions
   (`outputs/experiments/*/pred_ccd_test_grouped.npz`). Hypothesis: the twin only shows crashes of the
   ego car. New script under `src/eval/`, new output file, paper §7.1 + limitations.
3. [ ] **Realistic CARLA traffic** (closes the E5 gap): tune Traffic Manager (speed, following distance),
   check with `e5_kinematics.py traffic_carla` on a NEW small run; needs CARLA and the user's approval
   for generation and storage.
4. [ ] **Finish the 260 unrun slots** (optional, ~1 h): needs the raw clips, CARLA, and the user raising
   `full_run_cap_gb` (40 → 45) in `configs/storage.yaml`; then `pipeline.py launch generator`.
   New clips need a NEW cache name (`build_cache.py` never overwrites).
5. [ ] **Paper**: check the citations marked `[verify]`, refresh numbers from RESULTS.md, keep limitations
   honest, then help prepare the submission (after the licence review).
6. Waiting on the user (remind them, don't do it for them): Nexar licence review with the supervisor
   (decision D6); delete the exposed Hugging Face token named `carla-pc`; log their GitHub account out of
   shared PCs (`git credential-manager github logout adarshlaxman`).

## 6. Commands (Windows; Linux: `.venv/bin/python`, `/`, `export VAR=...`)
```
.venv\Scripts\python.exe crashtwin\scripts\pipeline.py status
.venv\Scripts\python.exe crashtwin\scripts\pipeline.py launch <job>   # detached, resumable, log in outputs\logs\
#   jobs: experiments, experiments:<prefix> (e.g. experiments:E3), generator, track_ccd, track_nexar,
#         caches_real, track_sim, cache_sim, real_chain, sim_chain
.venv\Scripts\python.exe crashtwin\src\eval\aggregate.py               # RESULTS.md + figures
.venv\Scripts\python.exe crashtwin\src\eval\e6_breakdown.py
.venv\Scripts\python.exe crashtwin\src\eval\e5_kinematics.py traffic_carla --run outputs\sim_v2\<run>
.venv\Scripts\python.exe crashtwin\src\eval\e5_kinematics.py compare_traffic
.venv\Scripts\python.exe crashtwin\src\generator\run.py --supervise --resume outputs\sim_v2\<run>
.venv\Scripts\python.exe crashtwin\scripts\transfer.py plan            # back up / move results
cd crashtwin\tests ; ..\..\.venv\Scripts\python.exe -m unittest test_core   # 24 tests, no CARLA needed
```

## 7. Where things are
- `src/generator/` CARLA generator v2 (`run.py` entry; scenarios C1–C5 crash, N1–N4 near-miss,
  R1–R4 normal; `labels.py` rule v1; `io_utils.py` storage guard, provenance, `local_path`).
- `src/validation/` per-clip checks (PASS/WARN/FAIL), run reports, audit sheets.
- `src/real/build_indexes.py` CCD (grouped split, seed 20261029, 80/10/10) and Nexar indexes.
- `src/features/` `track_videos.py` (YOLO11s + ByteTrack, 10 Hz; needs an NVIDIA GPU),
  `camera_features.py` (83 + 8 features), `build_cache.py` (npz caches, never overwrites).
- `src/models/train.py` CrashGRU (2-layer GRU, hidden 128), exponential anticipation loss (tau 2 s),
  AdamW 1e-3, batch 64, <= 40 epochs, patience 8, per-epoch checkpoints, skips runs with `metrics.json`.
- `src/eval/` `metrics.py` (CCD protocol AP/mTTA, Nexar official mAP, false alarms/min, bootstrap CI),
  `aggregate.py`, `e5_kinematics.py`, `e6_breakdown.py`.
- `scripts/` `pipeline.py`, `check_env.py`, `setup_env.ps1/.sh`, `transfer.py`.
- `configs/` scenario matrix, sensors, weather, storage caps, paths. Docs in `docs/`
  (ARCHITECTURE, LABELS, DATASET_SCHEMA, EXPERIMENT_DESIGN, VALIDATION_PLAN, PILOT_REPORT, INCIDENTS,
  SETUP_NEW_MACHINE). `results_kit/` = Git LFS zip of `outputs/` (no raw clips) + YOLO weights.

## 8. Known pitfalls (details in docs/INCIDENTS.md)
- Use `Town03`, never `Town03_Opt` (its Traffic Manager file is broken in 0.9.15).
- The CARLA Python client can abort natively (0xC0000409): always generate with `--supervise`.
- On Windows, training may exit with 0xC0000409 AFTER writing `metrics.json`: harmless.
- The full run is at its 40 GB cap: launching `generator` stops at once unless the user raises the cap.
- `pipeline.py` uses the newest `outputs/sim_v2/full_*` folder; don't create a newer one by accident.
- Index CSVs store `C:\CARLA_Project\...` paths; `io_utils.local_path` re-anchors them automatically.
- Windows log names must not contain `:` (NTFS alternate data streams).
- PCs with several GitHub logins: this repo sets `credential.https://github.com.username adarshlaxman`
  and `lfs.https://github.com/adarshlaxman/crashtwin.git/info/lfs.access basic` (repo-local config; set
  them again after a fresh clone if a push fails with "You must have push access to verify locks").
- Environment: Python 3.10.11, torch 2.14.1+cu126, ultralytics 8.4.192, carla 0.9.15
  (`requirements.txt`; exact snapshot `requirements-lock.txt`).
- The slides were built with pptxgenjs from `outputs/results` numbers (script not in the repo): for new
  slides, rebuild from the result files rather than editing numbers by hand.
