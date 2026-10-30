# CrashTwin: Can a Validated CARLA Crash Twin Teach Crash Anticipation on Real Dashcam Video?

*Adarsh L · CAVE Labs, PES University* (draft generated 2026-10-29, results filled 2026-10-30 from `outputs/results/RESULTS.md`; all numbers are mean ± std over 2 seeds)

## Abstract
Real crash videos are scarce, camera-only and hard to label consistently, while simulators can produce unlimited crashes with every sensor. We build **CrashTwin**, a CARLA 0.9.15 generator of 13 deterministic scenario families (rear-end, cut-in, junction, vulnerable-road-user and single-vehicle crashes; their near-miss counterparts; and four kinds of normal driving). It records synchronised RGB, depth, LiDAR, radar, IMU/GNSS, collision events and actor ground truth. Every clip's outcome (*crash*, *near-miss*, *normal* or *invalid*) is **measured** from the collision sensor and a predicted-overlap time-to-collision rule, never taken from the scenario's intent. Each clip also passes 20+ automatic integrity checks. We train one causal GRU crash-anticipation model on camera-derived track features and compare (E1) CARLA-only, (E2) real-only and (E3) CARLA + real training at matched real-data fractions, on held-out Nexar and CCD dashcam videos. A model trained **only on CARLA** reaches Nexar mAP 0.62/0.64 (public/private; chance ≈ 0.5), matching a model trained on 10–25 % of the real data, but it does **not** transfer to CCD (AP 0.25, below the 0.31 base rate). Adding CARLA to the full real training set raises Nexar private mAP from 0.681 to 0.714 but lowers CCD AP slightly (0.867 → 0.844). Inside the twin, adding radar, LiDAR or depth to the camera raises held-out-town AP from 0.84 to 0.93–0.97. CARLA's background traffic is slower and follows less closely than real highway traffic (Accid3nD). We also report that CCD's official split shares 107 of its 133 YouTube source videos between train and test, and we provide a source-grouped split.

## 1. Introduction
Crash anticipation, warning *before* impact, is the safety-relevant version of accident detection. Public benchmarks (CCD [1], DAD [2], Nexar [3]) contain a few thousand dashcam clips, mostly camera-only, collected from the internet. Simulation offers scale and full sensing, but only helps if (i) its crash clips are correct, (ii) what a model learns from them transfers to real video, and (iii) the comparison against real data is fair. Our contributions:
1. **A measured-outcome crash generator** for CARLA: 13 scenario families, controlled across 6 towns, 8 weather/light conditions and 3 traffic densities, with a held-out town (Town05). Outcome labels come from collisions and a predicted-overlap TTC, and every clip is validated automatically (sync, files, labels, physics, provenance).
2. **A matched-fraction sim-to-real study** (E1–E3): identical model, inputs and evaluation; only the training data changes. Inputs are camera-only track features computed identically for real and simulated video, so no simulator ground truth ever reaches the real-video model.
3. **Sensor and twin analyses:** sensor fusion inside the twin (E4), and traffic kinematics of the twin vs real roadside recordings, Accid3nD/TUMTraf-A [4] (E5).
4. **A data-hygiene finding:** CCD's official split leaks source videos between train and test. We release a grouped split.

## 2. Related work
- **Crash anticipation:** DSA [2], UString [1], DSTA [5]; the standard CCD/DAD protocol reports AP and mTTA.
- **Synthetic crash data:** DeepAccident [6] (CARLA, multi-view camera + LiDAR, V2X; not evaluated on real dashcam crashes); Dashcam-to-CARLA reconstruction [7] [verify citation details].
- **Real multi-sensor crash data:** TUMTraf-A / Accid3nD [4] (roadside cameras and 3D labels).
- **Nexar** [3]: a dashcam collision/near-miss benchmark with a public/private test protocol.

## 3. CrashTwin generator
- **Simulation:** CARLA 0.9.15, synchronous at 0.1 s/tick (10 Hz in simulation time) with 0.01 s physics substeps.
- **Towns:** Town01, 02, 03, 04, 10HD for training/validation; **Town05 held out**. Town06/07 aren't installed (no rural map).
- **Sensor rig** (`configs/sensors.yaml`): RGB 1280×720, FOV 90° (roof-front); instance segmentation (for 2D boxes only); depth 640×360; 32-channel LiDAR (100k pts/s, 70 m); front radar (30°×10°, 100 m); IMU; GNSS; collision sensor.
- **Synchronisation:** each tick waits until every stream delivers the same world frame. All seven streams' frame IDs and timestamps are stored and checked.
- **Scenario families** (`configs/scenario_matrix.yaml`): every scenario follows a fixed route, with actors placed in advance to avoid "pop-in". Triggers are dynamic:
  - lead braking at a seeded time (rear-end);
  - a geometric gap trigger (cut-in);
  - closed-loop arrival-time matching of a scripted crossing vehicle (junction);
  - arrival-time matching of a walker or cyclist (VRU);
  - pursuit steering into a roadside pole (single vehicle).
  - Near-miss families add ego emergency braking, armed at a seeded predicted-TTC threshold after a 0.2–0.5 s reaction delay.
- **Clip window:** 10 s (100 frames). The event (first collision, or closest approach) is placed at a random frame in 60–90, after a ≥ 3 s warm-up. Frames after the event stay in the raw data.
- **Labels (rule v1, `docs/LABELS.md`):**
  - *crash* = any ego collision event;
  - *near-miss* = no collision, and the **same** actor reaches footprint gap ≤ 1.5 m and predicted-overlap TTC ≤ 1.5 s;
  - *normal* = no collision and min TTC > 3 s;
  - everything else = *invalid* (borderline), kept on disk but excluded.
  - The predicted-overlap TTC (earliest overlap of constant-velocity footprints, 5 s horizon) avoids the false conflicts that a gap/range-rate TTC gives for oncoming and passing traffic.
- **Validation:** 20+ checks per clip (VALIDATION_PLAN.md), giving PASS/WARN/FAIL. A clip is *usable* when no check FAILs and its measured outcome is valid; clips whose measured outcome differs from the intent are kept with the measured label.

## 4. Pilot (feasibility; `docs/PILOT_REPORT.md`)
- 130 slots, 382 attempts, 243 clips (3.6 GB).
- **0 synchronisation failures.**
- Median |sensor − GT| lane gap: LiDAR 0.08 m, depth 0.04 m, radar 0.11 m (agreement with CARLA ground truth, not proof of real-world realism).
- Slot success rates: 30–100 % by family. Near-miss is the scarcest outcome (≈ 7 % of attempts).
- 14.6 MB per clip; 11.9 s per attempt.

## 5. Real data and splits
- **CCD** [1]: 1,500 crash + 3,000 normal clips, 10 fps, 5 s. Its official split (vgg16_features/train.txt, test.txt: 3,600/900) puts clips from **107 of the 133** YouTube source videos in both train and test; only 1 of 300 official test crashes comes from a source unseen in training. We report the official split for comparability and use a **source-grouped split** (80/10/10, seed 20261029) as the primary CCD test.
- **Nexar** [3]: 1,500 training videos (≈ 40 s, ≈ 30 fps; positives include near-misses), with official test-public (667) and test-private (677) partitions. Test videos end 0.5/1.0/1.5 s before the event. We keep the official partitions; validation = 15 % of the training videos.
- **CARLA:** usable clips split by **slot** (all attempts of a slot together). Town05 → sim-test-T05; other towns → 70/15/15 train/val/test-familiar. Pilot clips are never used.

## 6. Model and protocol
- **Perception:** frozen COCO YOLO11s + ByteTrack, the same weights and code for every dataset. All video is resampled to 10 Hz by timestamp. Nexar training videos are cut to 12 s ending 0.5 s after the event (a seeded random 12 s window for negatives).
- **Features (83 per frame):** the top-5 tracks ranked by size × closeness to the lower image centre, each with normalised box geometry, area growth, scale-expansion rate, image TTC, class and track age; plus the track count, an optical-flow ego-motion proxy, and the maximum expansion.
- **Model:** 2-layer causal GRU (hidden 128) → per-frame crash probability.
  - Loss: exponentially weighted BCE on frames before the event (τ = 2 s) for positives; BCE on all frames for negatives.
  - AdamW, lr 1e-3, batch 64, ≤ 40 epochs, early stopping on validation AP.
  - **E1 selects checkpoints on CARLA validation only.**
- **Metrics:**
  - CCD protocol AP, mTTA and TTA@R80 (UString/DSTA code);
  - warning time in seconds and false alarms per minute at the validation threshold for 80 % recall;
  - Nexar official mAP over the three time-to-accident groups (clip score = last-frame probability);
  - 95 % bootstrap CIs; 2 seeds per configuration.
- **Experiments:**
  - E1: CARLA only.
  - E2: real (CCD + Nexar) at 10/25/50/100 %.
  - E3: CARLA + real at the same fractions (nested subsets).
  - E4: camera vs + radar / + LiDAR / + depth / all, in CARLA.
  - E5: traffic kinematics, CARLA vs Accid3nD (revised from crash kinematics; Section 7.3).
  - E6: AP by condition.

## 7. Results
**Data actually generated.** The full run stopped at its 40 GB storage cap after 3,040 of 3,300 slots; the 260 unrun slots are all in Town10HD (246 normal-driving, 14 N4 near-miss), so the crash families and the held-out Town05 are complete. 2,576 clips were written; **1,877 are usable** (train 1,031 / val 220 / test-familiar 245 / test-Town05 381; label 1 = crash or near-miss).

### 7.1 Sim-to-real (E1–E3)
| Training data | Nexar public mAP | Nexar private mAP | CCD AP (grouped) | CCD AP (official) | CCD FA/min @R80 |
|---|---|---|---|---|---|
| E1 CARLA only | 0.621 ± 0.006 | 0.638 ± 0.004 | 0.249 ± 0.008 | 0.264 ± 0.011 | 11.7 |
| E2 real 10 % | 0.586 ± 0.008 | 0.526 ± 0.025 | 0.678 ± 0.036 | 0.709 ± 0.006 | 7.7 |
| E3 CARLA + real 10 % | 0.619 ± 0.009 | 0.585 ± 0.005 | 0.535 ± 0.034 | 0.587 ± 0.030 | 10.4 |
| E2 real 25 % | 0.640 ± 0.028 | 0.576 ± 0.023 | 0.745 ± 0.001 | 0.779 ± 0.006 | 4.0 |
| E3 CARLA + real 25 % | 0.652 ± 0.038 | 0.606 ± 0.046 | 0.633 ± 0.096 | 0.706 ± 0.038 | 5.7 |
| E2 real 50 % | 0.664 ± 0.029 | 0.666 ± 0.006 | 0.846 ± 0.017 | 0.883 ± 0.018 | 2.1 |
| E3 CARLA + real 50 % | 0.640 ± 0.050 | 0.678 ± 0.007 | 0.797 ± 0.008 | 0.870 ± 0.027 | 3.1 |
| E2 real 100 % | 0.703 ± 0.009 | 0.681 ± 0.009 | 0.867 ± 0.001 | 0.924 ± 0.006 | 1.3 |
| E3 CARLA + real 100 % | 0.707 ± 0.003 | 0.714 ± 0.009 | 0.844 ± 0.030 | 0.917 ± 0.029 | 1.7 |

Figures: `figures/e1e3_nexar.png`, `figures/e1e3_ccd.png`.
- **Nexar:** CARLA-only training beats chance and matches 10–25 % of the real data. Adding CARLA helps most when real data is scarce (private mAP +0.059 at 10 %) and still adds +0.033 private mAP at 100 %; on the public set the gains are within seed variation above 10 %.
- **CCD:** CARLA-only training fails (AP below the 31 % positive rate), and adding CARLA *hurts* at every fraction, most at 10 % (−0.14). A plausible reason (not tested): CCD clips are 5 s long, mostly from a different camera height and with many non-ego crashes, so the CARLA ego-collision prior misleads. With real data, the gap closes as the real fraction grows.
- **CCD leakage:** the official split gives higher AP than the grouped split for every real-data model (+0.03 to +0.07), as expected from shared source videos.
- Two seeds only; differences under ≈ 0.03 should be read as not significant.

### 7.2 Sensor fusion in the twin (E4)
CARLA test AP (CCD protocol), camera-only vs camera + sensor features:

| Inputs | Familiar towns | Town05 (held out) |
|---|---|---|
| camera | 0.912 ± 0.010 | 0.837 ± 0.006 |
| + radar | 0.959 ± 0.004 | 0.930 ± 0.014 |
| + LiDAR | 0.955 ± 0.005 | 0.926 ± 0.025 |
| + depth | 0.959 ± 0.005 | 0.940 ± 0.014 |
| + all | 0.966 ± 0.006 | **0.971 ± 0.002** |

Every range sensor helps, and most in the unseen town (+0.09 to +0.13 AP), where camera-only generalises worst. These are in-simulation results; the sensors are noise-free apart from CARLA's own models (Section 4), so they are an upper bound on real gains.

### 7.3 Twin vs real traffic kinematics (E5)
The planned crash-level comparison was dropped: the automatic crash candidates found in Accid3nD's 3D labels were 0/3 real crashes on visual check (`docs/INCIDENTS.md`). We compare car-following instead (1 Hz samples; same-lane leader within 80 m): 10,454 Accid3nD samples (roadside highway cameras) vs 2,811 CARLA samples from normal clips (437 on the Town04 highway).

| Quantity (median) | Accid3nD | CARLA Town04 highway | CARLA all towns |
|---|---|---|---|
| speed (m/s) | 18.8 | 8.1 | 5.0 |
| gap to leader (m) | 27.9 | 18.9 | 13.5 |
| time headway (s) | 1.58 | 2.15 | 2.36 |
| TTC when closing (s) | 2.1 (n = 1,288) | 6.2 (n = 30) | 3.3 (n = 430) |

All distributions differ: speed, gap and headway with KS p < 10⁻³⁰, TTC with p < 10⁻⁸ (only 30 closing samples on the CARLA highway; `outputs/results/e5_traffic_comparison.csv`). CARLA's Traffic Manager drives slower and with longer headways than real highway traffic, so the twin's *normal* driving is calmer than reality, a fidelity gap that likely makes CARLA normals too easy to separate from conflicts.

### 7.4 Robustness by condition (E6)
Video-level AP per group (`outputs/results/e6_breakdown.csv`, 95 % bootstrap CIs). CCD can't be broken down: its 3,000 normal clips have no day/night or weather labels.
- **Night (hypothesis AP night < day):** Nexar public "Dark" has only 21 videos (5 positives). Its AP is lower for every model (E1 0.37, E2 0.59, E3 0.54 vs 0.61–0.64 in normal light), but the CIs are wide. In CARLA, night is not harder (Town05 rain_night 0.97 for E1).
- **Rain (hypothesis AP rain < clear):** supported on Nexar public (E1 0.51 vs 0.66; E2 0.56 vs 0.68), but **reversed** on Nexar private (E2 0.86 vs 0.65), with 32–37 rain videos per set. Not supported overall.
- **Scene:** CARLA-only training does best on Nexar highway videos (public 0.86 vs urban 0.63), better than real-only training (0.69) — the one condition where the twin transfers clearly.
- In CARLA Town05, wet_noon is the weakest condition (E1 0.50, n = 20).

## 8. Limitations
- CARLA's visuals, vehicle set (no Indian vehicle types such as autorickshaws), behaviours and physics are simplified: no damage, and walkers don't slow vehicles on impact.
- No rural map; no snow (CCD has snowy clips).
- Side impacts often leave the 90° camera view at contact.
- Near-miss rule v1 is an operational proxy for Nexar's human labels.
- Bounding boxes (e.g. motorcycles with riders) are looser than meshes.
- Same-seed reproducibility across machines isn't guaranteed and wasn't measured.
- Accid3nD (as downloaded) has no LiDAR, and automatic crash detection from its 3D boxes failed visual checks, so E5 compares traffic, not crashes.
- Generation stopped at the 40 GB cap (3,040/3,300 slots; the missing slots are Town10HD normal driving).
- CCD normal clips carry no condition labels, so E6 covers Nexar and CARLA only.
- Two seeds per configuration (time budget).

## 9. Licences and ethics
- CCD: MIT.
- Nexar: Nexar Open Data License; attribution required. Its ethical-use clause ("simulate unsafe behavior") is flagged for supervisor review. Nexar data is used only to train and evaluate a collision-*warning* model and never drives the simulator.
- Accid3nD: CC BY-NC-SA 4.0.
- YOLO11 weights: AGPL-3.0.

## References
[1] W. Bao, Q. Yu, Y. Kong. Uncertainty-based Traffic Accident Anticipation with Spatio-Temporal Relational Learning (UString; Car Crash Dataset). ACM Multimedia 2020.
[2] F.-H. Chan, Y.-T. Chen, Y. Xiang, M. Sun. Anticipating Accidents in Dashcam Videos (DSA; DAD). ACCV 2016.
[3] D. C. Moura, O. Zvitia. Nexar Dashcam Collision Prediction Dataset and Challenge. arXiv:2503.03848, 2025.
[4] TUMTraf-A / Accid3nD roadside accident dataset. arXiv:2508.14567, 2025 [verify author list].
[5] M. M. Karim et al. A Dynamic Spatial-Temporal Attention Network for Early Anticipation of Traffic Accidents (DSTA). IEEE T-ITS 2022.
[6] T. Wang et al. DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving. AAAI 2024 (arXiv:2304.01168).
[7] Dashcam-to-CARLA crash reconstruction. arXiv:2411.16027, 2024 [verify title/authors].
[8] A. Dosovitskiy et al. CARLA: An Open Urban Driving Simulator. CoRL 2017.
[9] Y. Zhang et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV 2022.
[10] Ultralytics YOLO11, 2024.
