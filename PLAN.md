# CrashTwin: full project plan
**Real-data-anchored CARLA digital twin for sim-to-real multi-sensor crash prediction**
Adarsh L · CAVE Labs, PES University

---

## 1. Context: why this project
- Real crash datasets (CCD, Nexar) contain only dashcam video. They have no radar, LiDAR or depth.
- Real multi-sensor datasets (nuScenes and similar) contain almost no crashes.
- Real crashes recorded with LiDAR are very few: Accid3nD / TUMTraf-A, roadside only.
- Synthetic crash datasets (DeepAccident) were never checked against real crash videos.

**Our idea:** use CARLA (Unreal Engine 4) to build a digital twin that creates thousands of realistic crashes with every sensor. We anchor the twin to real crash data, train an AI that warns before a crash, and check whether it works on real crashes.

## 2. Objectives
| # | Objective | Measured by |
|---|---|---|
| O1 | Build a CARLA crash generator: 4 crash types + normal driving, with camera, radar, LiDAR, depth, GNSS/IMU, collision sensor and ground truth | About 2,000 clips in CCD format |
| O2 | Put all real data (CCD, Nexar, Accid3nD) and simulated data in one common format | One index and one feature pipeline |
| O3 | Train a crash predictor that warns before the crash | AP, mTTA (seconds of warning) |
| O4 | RQ1 sim-to-real: can a model trained on CARLA predict real crashes? | E1–E3 on Nexar and CCD |
| O5 | RQ2 sensor fusion: how much earlier do radar and LiDAR warn than a camera alone? | E4 |
| O6 | RQ3 twin fidelity: do CARLA's sensors behave like real ones in the same crash? | E5 (Accid3nD replay) |
| O7 | RQ4 conditions: do night and rain affect real and simulated crashes alike? | E6 |
| O8 | Publishable output | Results tables, figures, paper draft (IEEE ITSC/IV workshop or arXiv) |

## 3. Architecture
```
            REAL WORLD                               DIGITAL TWIN (CARLA 0.9.15, UE4)
 ┌──────────────────────────────┐        ┌────────────────────────────────────────────┐
 │ CCD  1.5k crash + 3k normal  │        │ generate_crashes.py                         │
 │ Nexar 1.5k (crash/near-miss) │        │  scenarios: rear_end, cut_in, junction,     │
 │ Accid3nD  roadside cam+LiDAR │──┐     │  pedestrian, normal  × towns × weather      │
 └──────────────┬───────────────┘  │     │  sensors: RGB, depth, LiDAR, radar,          │
                │                  │     │  GNSS/IMU, collision, ground truth           │
                │                  │     │ replay_accid3nd.py (real tracks → CARLA)     │
                │                  └────►└──────────────────────┬─────────────────────┘
                ▼                                               ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ LAYER 1 · Common clip format (10 fps, 50 frames, per-frame label,  │
        │ day/night, weather)  → outputs/index/{ccd,nexar,sim}.csv           │
        └───────────────────────────────┬───────────────────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ LAYER 2 · Perception: YOLO detection + ByteTrack tracking (GPU)     │
        │ same model for real and sim video → outputs/tracks/*.parquet        │
        └───────────────────────────────┬───────────────────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ LAYER 3 · Risk features per frame                                   │
        │ camera (all data): box growth, closing rate, image-TTC, nearest     │
        │   object, object count, ego motion (optical flow)                   │
        │ sensors (sim only): radar range/closing/TTC, LiDAR gap, depth gap,  │
        │   IMU accel/gyro, speed  → outputs/features/*.npz  [N, 50, F]       │
        └───────────────────────────────┬───────────────────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ LAYER 4 · Models                                                    │
        │ baseline: TTC threshold rule  |  GRU / Transformer predictor        │
        │ sensor dropout → one model runs camera-only or fused               │
        │ output: crash probability per frame → warning above a threshold     │
        └───────────────────────────────┬───────────────────────────────────┘
                                        ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ LAYER 5 · Evaluation: AP, mTTA, TTA@R80, false alarms/min,          │
        │ sensor fidelity → E1–E6 → tables, figures, paper                    │
        └───────────────────────────────────────────────────────────────────┘
```

**Folder layout** (all new; existing lessons, recordings, mechblock, Blender and Unity files are untouched)
```
crashtwin/
  configs/   paths.yaml, train_*.yaml, experiments.yaml
  src/sim/       generate_crashes.py, replay_accid3nd.py
  src/real/      index_ccd.py, index_nexar.py, accid3nd.py
  src/features/  track_videos.py, risk_features.py
  src/models/    baseline_ttc.py, predictor.py, train.py
  src/eval/      metrics.py, run_experiments.py, make_figures.py
  scripts/   check_env.py
  paper/     draft
outputs/  crashtwin_sim/, index/, tracks/, features/, checkpoints/, results/, logs/
datasets/ real_ccd/, real_nexar/, real_accid3nd/   (downloaded data, read-only)
```

**Tech stack:** CARLA 0.9.15 + Python API · PyTorch 2.14 + CUDA (RTX 4500 Ada, 24 GB) · Ultralytics YOLO + ByteTrack · OpenCV · Open3D · pandas/NumPy · scikit-learn · TensorBoard.

## 4. Phases

### Phase 1 · Data foundation (about day 1)
**Objective:** all data in one format (O1, O2).
1. Finish the generator (`crashtwin/src/sim/generate_crashes.py`).
   - Already done and tested: rear_end, cut_in and junction scenarios; radar and LiDAR road filtering; ground-truth tracking; the Town03_Opt fix.
   - Written but not yet retested: pedestrian timing (the trigger fires when the walker's arrival time matches the car's) and the depth filter for the car's own bonnet.
   - Then run the full generation in the background: 5 towns × (200 crash + 200 normal), about 2,000 clips, about 3–4 hours, about 3 GB. Add `--save-lidar` on Town03 only.
2. Real-data indexes, all with the same columns (clip_id, path, label, crash_frame, fps, daynight, weather, split):
   - `index_ccd.py`: parse `Crash-1500.txt` (per-frame labels, Day/Night, weather) and list the 3,000 normal clips.
   - `index_nexar.py`: train (time_of_event, time_of_alert) and the test-public and test-private sets; resample 30 fps → 10 fps.
   - `accid3nd.py`: unpack the zips into `datasets/real_accid3nd/extracted` and parse the OpenLABEL 3D tracks and LiDAR file lists.
3. Fixed-seed train/val/test splits → `outputs/index/*.csv`.

**Done when:** `check_env.py` is all OK, the index counts are right, and spot-checked simulated clips show the gaps going to about 0 at the crash.

### Phase 2 · Perception, features, models (about day 2)
**Objective:** a working predictor on real and simulated data (O3).
1. `track_videos.py`: YOLO + ByteTrack on every clip (GPU, resumable).
2. `risk_features.py`: the same camera features for real and simulated clips, plus sensor features for simulated clips.
3. `baseline_ttc.py`: the threshold rule, as a reference.
4. `predictor.py` + `train.py`:
   - GRU (and a small Transformer) with the standard exponential anticipation loss.
   - Sensor dropout.
   - TensorBoard logging; checkpoints in `outputs/checkpoints/`.

**Done when:** tracks overlaid on sample videos look right, the features have no NaN values, the model can overfit a small batch, and the first AP/mTTA numbers are on the CCD test set.

### Phase 3 · Experiments and paper (day 3 onward)
**Objective:** answer RQ1–RQ4 and produce publishable results (O4–O8).

| Exp | Train | Test | RQ |
|---|---|---|---|
| E1 | CARLA only | Nexar, CCD | 1 |
| E2 | Real only | Nexar, CCD | 1 |
| E3 | CARLA + 10/25/50 % real | Nexar, CCD | 1 |
| E4 | Camera / + radar / + LiDAR / + depth / all | CARLA test | 2 |
| E5 | `replay_accid3nd.py`: real crash tracks replayed in CARLA with matching roadside sensors | Real vs simulated sensor comparison | 3 |
| E6 | E1–E4 models | Split by day/night, weather | 4 |

Outputs: `outputs/results/*.csv`, figures, and a paper draft in `crashtwin/paper/`.

**Done when:**
- E2 on CCD falls in the published AP range (a sanity check).
- All 6 experiments have result tables.
- The figures and the paper draft are written.

## 5. Risks and mitigations
- **CARLA client crashes** (Traffic Manager assertion): use `Town03`, not `Town03_Opt`, and `--tm-port` if a port is stuck.
- **Sim-to-real gap too large:** E3's mixed training and sensor dropout give a fallback result that can still be published.
- **Accid3nD format surprises:** E5 runs last and doesn't block E1–E4.
- **Disk:** about 200 GB free. The downloads take 78 GB; simulated data plus features about 10 GB.

## 6. How I'll work in auto mode
- Build the phases in order and test each step before moving on.
- Run long jobs (generation, tracking, training) in the background.
- Never delete or overwrite your existing files; new output goes in new folders only.
- Report at the end of each phase with results and anything that needs your decision.
