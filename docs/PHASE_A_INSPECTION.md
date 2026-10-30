# Phase A: inspection report (2026-10-29)

Read-only inspection. Nothing was edited, deleted or generated during Phase A.

## 1. Where the generator is
- **Entry point:** `crashtwin/src/sim/generate_crashes.py` (v1, about 620 lines, one file).
- **Command-line options:** `--towns`, `--crashes-per-town`, `--normals-per-town`, `--scenarios`, `--seed`, `--save-lidar`, `--tm-port`.
- **Structure:**
  - Class `CrashGenerator`: `load_town`, `spawn_traffic`, `spawn_ego`, `attach_sensors`, `step`, `run_crash`, `run_normal`, `save`, and a background writer thread.
  - Sensor summaries: `radar_summary`, `lidar_summary`, `depth_summary`.
- **Other files:**
  - `crashtwin/configs/paths.yaml` (data paths)
  - `crashtwin/scripts/check_env.py` (environment check)
  - `crashtwin/PLAN.md` (approved plan)
- **Older related code:** `lesson8_auto_labeled_dataset.py` (instance-segmentation → YOLO boxes; `boxes_from_instance_image` can be reused for 2D boxes).
- **No unit tests. No source control:** `C:\CARLA_Project` is not a git repository.

## 2. What works today (verified on test clips)
| Feature | Status | Evidence |
|---|---|---|
| Synchronous stepping, 0.1 s per tick (10 fps in **simulation time**; saved frame rate = simulation rate; wall-clock speed varies) | ✅ | `FIXED_DT = 0.1`, `world.tick()` + per-sensor frame matching |
| Rear-end crash | ✅ | 3 clips; ego speed 9.3 → 1.9 m/s at the crash frame; LiDAR and radar gaps fall to about 0.1 m |
| Cut-in crash | ✅ | 4 clips |
| Junction (crossing) crash | ✅ | 2 clips |
| Pedestrian crash | ❌ unverified | 1 clip from run 1 hit a **vehicle**, not the walker (bug, since fixed); then 0 of 4 attempts succeeded; the timing fix is written but **not retested** |
| RGB video 1280×720 mp4 | ✅ | All clips |
| Radar summary (road returns filtered) | ✅ | Run 2 values look plausible |
| LiDAR summary | ✅ | Matches ground truth within about 1 m |
| Depth summary | ❌ | Run 2 reads 0.0 on every frame (the camera sees its own bonnet); the fix is written but **not retested** |
| IMU / GNSS / speed / controls | ✅ | `sensors.csv` |
| Ground-truth actors (ego-relative position and velocity) | ✅ in run 2 only | Run 1 missed the lead car (`is_alive` bug, since fixed) |
| Collision event and frame | ✅ | `collision` column and `crash_index` |
| Full LiDAR point clouds (`--save-lidar`) | ❌ never run | |

## 3. What the existing test clips contain
Only 10 crash and 4 normal clips have been generated so far, all by **earlier code versions**. They are smoke tests, not a dataset.

| Folder | Contents | Code version | Notes |
|---|---|---|---|
| `outputs/crashtwin_sim/run_20261029_141807` | `run_args.json`, an `index.csv` with only the header, an empty `clips/` | v1, Town03_Opt | **Crashed run:** client assertion `px != 0`. Cause: 0.9.15 ships a wrong Traffic Manager file for Town03_Opt (`Town03_Opt.bin` is 587,422 bytes, the same as Town04, not Town03's 335,274). **Kept, not deleted.** |
| `outputs/crashtwin_sim/run_20261029_141955` | 4 crash (3 rear_end, 1 "pedestrian" that hit a vehicle) + 2 normal; 8 MB | v1a | Old depth columns; ground truth missing the lead car |
| `outputs/crashtwin_sim/run_20261029_142153` | 6 crash (4 cut_in, 2 junction) + 2 normal; 18 MB | v1b | Depth gap always 0 (bonnet bug) |

Each clip: 50 frames (5 s), crash frame between 30 and 45; `video.mp4`, `sensors.csv`, `actors.csv`, `meta.json`; about 2.3–3.4 MB per clip.

## 4. Facts that affect the design
- **CARLA 0.9.15** (server = client). Installed maps: Town01–05 and Town10HD (+ `_Opt` versions).
  - **Town06 and Town07 are NOT installed**; they come in CARLA's separate AdditionalMaps package.
- **Road users available:**
  - cars (23 types, plus 2 with no type label: bmw.grandtourer, mini.cooper_s)
  - vans 4 · trucks 4 · bus 1 · **motorcycles 4** · **bicycles 3** · pedestrians 51
- **Weather:** 22 presets (noon, sunset and night × clear, cloudy, wet and rain levels) plus DustStorm. **There is no fog preset**, but fog can be set directly (`fog_density`, `fog_distance`, `fog_falloff`).
- **Accid3nD / TUMTraf-A as downloaded:**
  - 5 zips, 44 GB, but **125.8 GB unpacked**: 38 GB of jpg and **87.6 GB of OpenLABEL json**.
  - About 12 sequences (`r09_s01` to `r09_s12`) from 4 roadside cameras (`s040`/`s050`, north/south, 16 mm/50 mm).
  - **No point-cloud files (.pcd/.bin) in any zip.** E5 cannot compare LiDAR with this download.
- **Disk:** 112 GB free on C:.
  - Real data: Nexar 30 GB, CCD 14 GB (incl. 7.2 GB of zips already unpacked), Accid3nD zips 44 GB.
- **Earlier bug fixes already in the code:** Town03_Opt Traffic Manager crash; actors spawned this tick report `is_alive=False`; Traffic Manager `force_lane_change(True)` means move right.

## 5. Gap analysis: requested vs current
| Requested | Status |
|---|---|
| 10 s clips, event at 6–9 s, keep the raw recording after the event | Missing: now 5 s with the post-crash part cut off |
| Near-miss scenarios + measured near-miss rule | Missing |
| Normal sub-types (following, junction, lane change, pedestrians nearby) | Missing: only one "safe autopilot" type |
| Motorcycle crash, single-vehicle crash | Missing |
| Outcome labels separate from scenario intent (`crash`/`near_miss`/`normal`/`invalid`) | Missing |
| Raw depth, raw LiDAR, raw radar | Missing (summaries only) |
| World-space positions, orientation, 3D boxes, 2D boxes, visibility | Missing |
| Per-frame sensor timestamps + frame-ID checks | Partial: frame matching exists, but timestamps aren't stored per stream |
| Scenario matrix (controlled weather/town/type) | Missing: weather is random |
| Validator (`PASS`/`WARN`/`FAIL`) + run report | Missing |
| Recorded software versions, code revision, full weather parameters | Missing |
| Resumable runs, unique IDs across runs | Missing |
| Retry logging | Missing: failed attempts are dropped silently |
| Town05 held out, Town07 rural | Town05 easy; **Town07 not installed** |

## 6. Storage budget (the key constraint)
**Free: 112 GB.** Keep at least 30 GB free at all times, which leaves **about 80 GB usable**.

| Item | Size |
|---|---|
| Accid3nD read **directly from the zips** (never fully unpacked) | 0 GB extra |
| Features and tracks for real + sim data | ~6 GB |
| Model checkpoints and experiment logs | ~2 GB |
| Pilot (about 140 clips) | ~3 GB |
| **Full simulated dataset** | **≤ 40 GB** |
| Headroom | ≥ 30 GB |

Estimated size per 10 s clip (to be measured in the pilot):

| Stream | Format | Est. size |
|---|---|---|
| RGB 1280×720, 100 frames | mp4 (H.264/mp4v) | ~6 MB |
| Depth 640×360 | uint16 centimetres, compressed npz | ~6–10 MB (to measure) |
| LiDAR 32 channels, 100k points/s (~10k per frame) | float16 x,y,z,intensity, compressed npz | ~6 MB |
| Radar raw | float32 npz | < 0.2 MB |
| Sensors + actors + boxes | parquet | < 1 MB |
| **Total** | | **~20 MB → 2,000 clips ≈ 40 GB** |

If the pilot measures more than 20 MB per clip, I'll reduce **depth to 320×180** for training clips and keep full resolution only for the Town05 test set and a 10 % audit sample.

## 7. Proposed folder architecture
Existing files stay where they are. v1 and its test runs are kept read-only as history.

```
crashtwin/
  configs/
    paths.yaml              (exists)
    scenario_matrix.yaml    families × outcomes × towns × weather × counts
    sensors.yaml            sensor rig: positions, resolution, rates
    weather.yaml            presets + custom fog, stored as full parameters
    storage.yaml            per-stream formats, size budget, headroom
  src/
    sim/generate_crashes.py v1, frozen
    generator/              v2 (new)
      run.py                entry point: resumable, versioned output, run log
      world.py              town loading, traffic (incl. motorcycles/bicycles), weather
      rig.py                sensor rig + synchronous capture (frame ID + timestamp per stream)
      recorder.py           raw streams → files (mp4, npz, parquet)
      scenarios/            base.py, crash.py, near_miss.py, normal.py (one class per family)
      labels.py             outcome rules: crash / near_miss / normal / invalid
    validation/
      validate_clip.py      PASS/WARN/FAIL → validation.json
      run_report.py         per-run summary by family, town, weather
    real/                   ccd.py, nexar.py, accid3nd.py (reads the zips in place)
    splits/                 make_splits.py (scenario-level, seeded manifests)
    features/  models/  eval/   (Phase 2 and 3)
  tests/                    unit tests: labels, near-miss rule, validator, depth decoding
  docs/
    PHASE_A_INSPECTION.md   this file
    ARCHITECTURE.md  SENSORS.md  LABELS.md  DATASET_SCHEMA.md  LIMITATIONS.md  LICENCES.md

outputs/
  crashtwin_sim/            v1 test runs (unchanged, documented above)
  sim_v2/
    pilot_<timestamp>/      index.csv, run_report.md, run_log.jsonl, clips/<scenario_id>/
    full_<timestamp>/       created only after pilot approval
  processed/                tracks/, features/   (derived, rebuildable)
  splits/                   split manifests (clip_id, split, seed)
  experiments/              E1 … E6 configs, checkpoints, metrics
  reports/                  result tables and figures
```

Per clip (`clips/<scenario_id>/`):
```
rgb.mp4            100 frames, 10 fps
frames.parquet     per frame: frame_id, sim_time, one timestamp per stream
depth.npz          uint16 cm, [100, H, W]
lidar.npz          one array per frame: x, y, z, intensity (sensor frame)
radar.npz          one array per frame: velocity, azimuth, altitude, depth
ego.parquet        world pose, velocity, IMU, GNSS, controls, speed
actors.parquet     id, type, blueprint, world pose/velocity, 3D box, ego-relative,
                   2D box, visible flag
labels.json        outcome, event frame/time, partner, min separation, min TTC,
                   rule version
meta.json          scenario family + parameters, seed, town, full weather
                   parameters, versions, code SHA-256
validation.json    PASS/WARN/FAIL + each check
```

## 8. Implementation plan (Phases B–E)
1. **B, design:** write `scenario_matrix.yaml`, `sensors.yaml`, `LABELS.md` (near-miss rule v1) and `DATASET_SCHEMA.md`.
2. **C, implement v2** in small steps, testing each:
   1. rig + recorder (raw streams, timestamps) + validator;
   2. port the 4 existing crash families and retest pedestrian + depth;
   3. add the motorcycle and single-vehicle crash families;
   4. add 4 near-miss families (the ego car's emergency braking is turned on so it avoids impact; the outcome is measured, never assumed);
   5. add 4 normal families;
   6. matrix-driven weather/town assignment, retry logging, resume.
3. **D, test:** unit tests + one smoke clip per family.
4. **E, pilot:** 10 clips per family, 14 families, **≈140 clips, ≈3 GB, ≈1–2 h**. Report success rate by family/town/weather, measured MB per clip and seconds per clip, plus visual checks.
5. **F, review gate:** stop and report. The full run starts only after approval.

Code revision: there's no git, so every `meta.json` stores the **SHA-256 of each generator source file**.

## 9. Decisions needed before Phase C
1. **Rural roads (Town07):**
   - (a) Download CARLA AdditionalMaps 0.9.15 (several GB, added to the CARLA install; an environment change that needs your approval), or
   - (b) **Recommended:** skip it, use Town01 (small town) + Town04's non-highway roads, and list the missing rural map as a limitation.
2. **E5 (real sensor comparison):** this Accid3nD download has **no LiDAR**. Options:
   - (a) compare **camera + 3D track kinematics** (speeds, decelerations, gaps before the crash) between Accid3nD and CARLA crashes, or
   - (b) look for a separate Accid3nD/TUMTraf LiDAR download.
   - **Recommended:** (a) now; check (b) later.
3. **Storage:** accept the **40 GB cap** for the full simulated set (lower depth resolution if the pilot is over budget)?
4. **Space you could free (only with your OK):** CCD's zips are already unpacked (7.2 GB). Leave them unless space gets tight.
5. **Source control:** set up git in `C:\CARLA_Project\crashtwin` only, so commits can be referenced in the paper? (Recommended. Data folders stay out of git.)

## 10. Decisions taken (2026-10-29, by Adarsh)
1. Town07: **skipped.** Use Town01 + Town04's non-highway roads; "no true rural map" is listed as a limitation.
2. E5: **camera + 3D track kinematics** from Accid3nD (no LiDAR in this download).
3. Storage: **40 GB cap** for the full simulated set, always keeping 30 GB free.
4. Git: **yes, for `crashtwin/` code only.** Data stays outside; commits only when asked.

## 11. Superseded by Phase B (2026-10-29)
The Phase A estimates in §6 and §8 were preliminary. Phase B replaces them:
- storage target **≤ 17 MB per clip**, not ~20 MB (DATASET_SCHEMA.md §8);
- pilot = **13 families × 10 = 130 slots**, not ≈ 140 (scenario_matrix.yaml);
- pilot runtime **≈ 2–4 h** and full run **≈ 28–55 h**, not 1–2 h (IMPLEMENTATION_PLAN.md §5);
- LiDAR 100k pts/s is unchanged.
