# CrashTwin

A real-data-anchored CARLA digital twin for sim-to-real multi-sensor crash prediction.
Adarsh L · CAVE Labs, PES University.

Design documents are in `docs/`. Start with `ARCHITECTURE.md`; the pilot results are in `docs/PILOT_REPORT.md`.

**Status (2026-10-30):** generation, tracking, all 26 training runs, E5 and E6 are finished. Results:
`paper/draft.md` (section 7) and the slides `docs/CrashTwin_progress_20261030.pptx`; problems and fixes:
`docs/INCIDENTS.md`. **New GPU machine:** start with `GPU_QUICKSTART.md` (clone, combine code + results
kit, resume in ~30–40 min); full details in `docs/SETUP_NEW_MACHINE.md`.

## Layout
```
configs/        scenario_matrix.yaml · sensors.yaml · weather.yaml · storage.yaml · paths.yaml
src/generator/  run.py (entry) · world.py · rig.py · recorder.py · camera.py · labels.py
                geometry.py · io_utils.py · scenarios/{base,crash,near_miss,normal}.py
src/validation/ validate_clip.py · run_report.py · audit_clip.py
src/real/       build_indexes.py (CCD grouped split, Nexar partitions)
src/features/   track_videos.py (YOLO11s + ByteTrack) · camera_features.py · build_cache.py
src/models/     train.py (causal GRU, E1–E4)
src/eval/       metrics.py · aggregate.py · e5_kinematics.py · e6_breakdown.py
src/sim/generate_crashes.py   v1 (frozen, history only; Windows paths)
scripts/        pipeline.py (resumable jobs) · check_env.py · setup_env.ps1/.sh · transfer.py
tests/          unit tests (no CARLA) + fixtures
```
The project folder is the folder that contains `crashtwin/` (lab PC: `C:\CARLA_Project`; elsewhere set
nothing, or `CRASHTWIN_ROOT`). CARLA's folder: `CARLA_ROOT`. Data never goes into git: datasets in
`<project>/datasets`, everything produced in `<project>/outputs` (simulated runs in `outputs/sim_v2/<mode>_<timestamp>/`).
One exception: the results kit `results_kit/*.zip` (Git LFS) so a clone can resume; keep this repo private.

## Commands (from the project folder; Linux: `.venv/bin/python` and `/`)
```
# unit tests (no CARLA needed)
cd crashtwin\tests && ..\..\.venv\Scripts\python.exe -m unittest test_core -v

# smoke test: one slot per family (clear noon, low traffic) in one town
.venv\Scripts\python.exe crashtwin\src\generator\run.py --supervise --smoke --families all --smoke-town Town03

# pilot: 13 families x 10 slots, up to 5 attempts each
.venv\Scripts\python.exe crashtwin\src\generator\run.py --supervise --pilot --slots-per-family 10 --max-attempts 5

# resume an interrupted run
.venv\Scripts\python.exe crashtwin\src\generator\run.py --supervise --resume outputs\sim_v2\<run>

# full dataset: refused unless --approved-full is given (requires pilot approval)
.venv\Scripts\python.exe crashtwin\src\generator\run.py --supervise --full --approved-full

# validate clips / write a run report / render an audit sheet (read-only on clip data)
.venv\Scripts\python.exe crashtwin\src\validation\validate_clip.py <clip_dir> [...]
.venv\Scripts\python.exe crashtwin\src\validation\run_report.py outputs\sim_v2\<run>
.venv\Scripts\python.exe crashtwin\src\validation\audit_clip.py <clip_dir> [...]

# pipeline: detached, resumable jobs (survive a closed terminal or lost connection)
.venv\Scripts\python.exe crashtwin\scripts\pipeline.py status
.venv\Scripts\python.exe crashtwin\scripts\pipeline.py launch experiments      # skips finished runs
$env:CRASHTWIN_SEEDS="0,1,2,3,4"; .venv\Scripts\python.exe crashtwin\scripts\pipeline.py launch experiments
.venv\Scripts\python.exe crashtwin\src\eval\aggregate.py                       # RESULTS.md + figures

# environment check, and moving data to another machine
.venv\Scripts\python.exe crashtwin\scripts\check_env.py
.venv\Scripts\python.exe crashtwin\scripts\transfer.py plan
```

## Safety behaviour
- **Never overwrites:** a new run folder for every run; `make_new_dir` refuses existing clip folders (unit-tested; v1 outputs checked unchanged by SHA-256).
- **Never deletes:** failed, invalid and spill-over clips are kept and labelled in `index.csv` / `validation.json`.
- **Storage guard:** stops before a run's cap (pilot 8 GB, full 40 GB) or when the project drive would drop below 30 GB free.
- **Supervisor:** `--supervise` resumes after native CARLA client aborts. Every attempt is logged as `started` first, so a crashed attempt is never replayed with the same seed.
