# CrashTwin architecture (Phase B design, v1.0, 2026-10-29)

Status: **design only.** Nothing described here as "planned" exists yet. The table in section 6 separates what exists from what is planned.

## 1. Research questions served
- **Main contribution:** E1–E3, sim-to-real crash anticipation on held-out real dashcam video, camera-derived inputs only.
- **Separate questions:** E4, CARLA sensor analysis against simulator ground truth; E5, CARLA vs Accid3nD camera and 3D-track comparison; E6, optional robustness.

## 2. End-to-end data flow
```
 configs/scenario_matrix.yaml ─┐
 configs/sensors.yaml ─────────┤
 configs/weather.yaml ─────────┤
                               ▼
 ┌────────────────────── SIMULATION (CARLA 0.9.15, synchronous, 0.1 s/tick) ───────────────────────┐
 │ generator/run.py ─► world.py ─► scenarios/<family>.py ─► rig.py ─► recorder.py ─► labels.py      │
 │  (slot queue,        (town,       (deterministic          (sensors,   (ring buffer,  (outcome     │
 │   seeds, resume,      traffic,     controller,             frame       10 s window,   from GT,    │
 │   budget guard)       weather,     triggers)               sync)       raw files)     rule v1)    │
 │                       lights)                                                                    │
 └────────────────────────────────────────────┬─────────────────────────────────────────────────────┘
                                              ▼
              validation/validate_clip.py  → validation.json (PASS / WARN / FAIL)
              validation/run_report.py     → run_report.md + run_summary.csv
                                              ▼
      outputs/sim_v2/<run>/clips/<scenario_id>/   IMMUTABLE RAW DATA (never edited after write)
                                              ▼
 ┌──────────────────────────── REAL DATA (read-only) ─────────────────────────────┐
 │ real/ccd.py · real/nexar.py · real/accid3nd.py (reads zips in place)           │
 └───────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
          splits/make_splits.py → outputs/splits/<dataset>_<version>.csv  (clip-level, seeded)
                                     ▼
 ┌──────────────── PREPROCESSING (derived, rebuildable) ──────────────────────────┐
 │ features/resample.py  (all video → 10 Hz by timestamp)                         │
 │ features/track_videos.py (YOLO, frozen COCO weights + ByteTrack)               │
 │ features/camera_features.py (box kinematics, image TTC, flow ego-motion)       │
 │ features/sensor_features.py (sim only: radar/LiDAR/depth/IMU summaries)        │
 │ → outputs/processed/{tracks,features}/<dataset>/...                            │
 └───────────────────────────────────┬────────────────────────────────────────────┘
                                     ▼
          models/  (baselines, causal GRU)  ← experiments/E*.yaml
                                     ▼
          eval/    (CCD protocol, Nexar official mAP, CARLA protocol, bootstrap CIs)
                                     ▼
          outputs/experiments/<E#>/<run>/  → outputs/reports/
```

## 3. Components, responsibilities and interfaces

| Component | File(s) | Responsibility | Input → output | Must NOT |
|---|---|---|---|---|
| Run controller | `src/generator/run.py` | Expand the matrix into **slots**; give each attempt a seed; resume; enforce the storage budget; write `run_log.jsonl` | matrix + CLI → slot queue, run folder | Split data or train |
| World | `src/generator/world.py` | Load the town, set synchronous mode, background traffic (cars, vans, trucks, buses, motorcycles, bicycles, walkers), weather from full parameters, vehicle lights at night | slot → ready world | Pick outcomes |
| Scenario controllers | `src/generator/scenarios/{base,crash,near_miss,normal}.py` | Deterministic set-up and triggers per family; report the **intended** outcome only | world, ego, seed → actors + per-tick control | Label the clip |
| Sensor rig | `src/generator/rig.py` | Spawn sensors from `sensors.yaml`; per tick, wait for every stream's frame == world frame | ego → per-tick bundle {frame, sim_time, stream data} | Drop or fabricate frames |
| Recorder | `src/generator/recorder.py` | Ring buffer; on the event (or a random anchor for normal clips) cut the 10 s window; write raw files (mp4, npz, parquet, json) | bundles → clip folder | Overwrite an existing clip ID |
| Labeller | `src/generator/labels.py` | Compute the **measured** outcome from collision events + ground truth (LABELS.md rule v1) | buffer + collision log → `labels.json` | Use scenario intent as the label |
| Validator | `src/validation/validate_clip.py` | File, sensor, event, physics and reproducibility checks (VALIDATION_PLAN.md) | clip folder → `validation.json` | Delete or "repair" clips |
| Run report | `src/validation/run_report.py` | Success rates by family / town / weather; MB per clip; seconds per clip | run folder → `run_report.md` | |
| Real-data readers | `src/real/{ccd,nexar,accid3nd}.py` | Uniform clip index; labels from official files; Accid3nD read straight from the zips | dataset files → `outputs/index/<dataset>.csv` | Modify `datasets/` |
| Split manager | `src/splits/make_splits.py` | Clip-level splits; Town05 isolation; official Nexar partitions; grouping by CCD source video | indexes → split manifests | Split by frame |
| Preprocessing | `src/features/*.py` | Same camera pipeline for real and sim; sensor features for sim only | video + split → features `[N, T, F]` + mask | Fit normalisers on validation or test data |
| Models | `src/models/{baselines,gru,train}.py` | Train from an experiment config; save checkpoints and config hash | features + manifest → checkpoint | See test labels |
| Evaluation | `src/eval/{ccd_protocol,nexar_eval,sim_protocol,bootstrap}.py` | Dataset-specific official metrics, plus CIs | predictions + labels → metrics json | Tune on the test set |

**Interface contract between simulation and ML:** the only coupling is the files under `outputs/sim_v2/<run>/` (DATASET_SCHEMA.md) and `index.csv`. The ML side never imports generator code.

## 4. Failure handling
| Failure | Handling |
|---|---|
| CARLA client assertion / Traffic Manager crash | Attempt marked `error` in `run_log.jsonl`; the run controller reconnects on a new TM port, reloads the town, and resumes from the next slot |
| CARLA server crash | Detected by connection timeout; restart via `connect_or_start_carla()` (reused from v1); resume |
| Sensor frame missing / timeout (10 s) | Attempt aborted → `error: sensor_timeout:<stream>`; no clip written |
| Wrong outcome (e.g. a near-miss attempt crashed) | Clip **kept** with its true label in the spill-over pool (`slot_status = spillover`); the slot retries with the next seed |
| No event within 60 s | `no_event`; logged; the slot retries (at most 5 attempts per slot) |
| Validation FAIL | Clip kept and marked invalid; never deleted |
| Storage budget reached (sim_v2 > 40 GB or C: free < 30 GB) | Run stops cleanly after the current clip and reports |
| Interrupted run | `run.py --resume <run_dir>` skips slots already `accepted` in `run_log.jsonl` |

## 5. Reproducibility
- Every attempt seed = `sha256(matrix_version, slot_id, attempt)` truncated to 31 bits.
- Python `random`, NumPy and the Traffic Manager are all seeded from it.
- `meta.json` records:
  - CARLA server and client versions, map name;
  - full weather parameters, sensor config, matrix version;
  - the git commit of `crashtwin/` (plus `dirty` flag) and the SHA-256 of each generator source file;
  - Python and package versions.
- **Caveat:** the same seed is not guaranteed to reproduce an identical clip across machines, GPUs or CARLA builds. Physics and Traffic Manager timing can differ. Reproducibility means identical configuration, not bit-identical output.

## 6. Actual vs planned capability (2026-10-29)
| Capability | Actual (v1, `src/sim/generate_crashes.py`) | Planned (v2) |
|---|---|---|
| Synchronous 10 Hz stepping | ✅ verified | kept |
| Crash families | rear_end, cut_in, junction ✅; pedestrian ❌ unverified | 5 crash families (matrix) |
| Near-miss families | ❌ none | 4 families |
| Normal families | 1 ("safe autopilot") | 4 families |
| Clip | 5 s, post-crash frames cut | 10 s raw, event at 6–9 s, post-event kept |
| Raw depth / LiDAR / radar | ❌ summaries only (`--save-lidar` never run) | raw for every clip (within budget) |
| Actor ground truth | ego-relative position/velocity | world pose, velocity, 3D box, 2D box, visibility |
| Validator | ❌ | PASS/WARN/FAIL |
| Weather | random preset | controlled matrix, full parameters stored, custom fog |
| Version recording | ❌ | git commit + file hashes |
| Real-data indexes, features, models, evaluation | ❌ none | Phase 2/3 modules above |

## 7. Folder layout
See `DATASET_SCHEMA.md` §1 for the data tree. Code tree:
```
crashtwin/            (git: code only; .gitignore excludes mp4/npz/parquet/pyc)
  configs/  paths.yaml · scenario_matrix.yaml · sensors.yaml · weather.yaml · storage.yaml
            experiments/E1.yaml … E6.yaml
  src/sim/generate_crashes.py          v1, frozen (history)
  src/generator/ src/validation/ src/real/ src/splits/ src/features/ src/models/ src/eval/
  tests/    unit + smoke tests
  docs/     PHASE_A_INSPECTION · ARCHITECTURE · DATASET_SCHEMA · LABELS · EXPERIMENT_DESIGN
            · VALIDATION_PLAN · IMPLEMENTATION_PLAN · LIMITATIONS (later) · LICENCES (later)
```
