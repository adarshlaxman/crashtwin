# CrashTwin experiment design (v1.0, 2026-10-29)

Status: **planned. No experiment has been run.** Results are reported only after each experiment is finished (the results template is in §10).

## 1. Hypotheses
- **H1 (E1):** a crash-anticipation model trained only on CARLA clips beats chance and the no-learning baseline on held-out **real** dashcam videos.
- **H2 (E3, main claim):** at equal real-data fractions f ∈ {10, 25, 50 %}, CARLA + f real ≥ real-only f. In other words, simulation reduces the real data needed for a given AP.
- **H3 (E4):** in CARLA, adding radar/LiDAR/depth features to camera features gives earlier warnings (higher mTTA) at equal AP.
- **H4 (E5):** CARLA crash kinematics and camera detection behaviour differ measurably from Accid3nD. We report how much, and don't claim realism.

## 2. Model input: camera-only, identical for real and simulated video
All video is resampled by timestamp to **10 Hz** (Nexar ≈ 30 fps → nearest frame on a 0.1 s grid; CCD and CARLA are already 10 Hz).

| Group | Features per frame (no simulator ground truth, ever) |
|---|---|
| Detection + tracking | YOLO (**frozen COCO weights**, same model for all data) + ByteTrack; top K = 5 tracks ranked by a risk score (box bottom-centre near the image centre-bottom × area growth) |
| Per track | normalised box centre, width, height, area; d(area)/dt; scale-expansion rate → image TTC = 1/expansion; class one-hot (person, bicycle, car, motorcycle, bus, truck); track age |
| Global | number of tracks; ego-motion proxy = median Farneback optical-flow magnitude in the lower-centre region (160×90 resolution) |

- **Forbidden for real-video models:** gap, TTC, actor states, or any field from the simulator's `actors.parquet`, `labels.json` or `sensors_summary.parquet`.
- Simulated sensor features (radar/LiDAR/depth/IMU) are used **only in E4**, by separate models evaluated only on CARLA data. We make no claim that a fused model runs on real video.

## 3. Models
| Name | Description | Role |
|---|---|---|
| B0 | Constant score / random | sanity floor |
| B1 | Image-TTC rule: score = σ(−(min image TTC − τ)), τ tuned on validation | no-learning baseline |
| B2 | Logistic regression on the current frame's features (no temporal memory) | non-temporal baseline |
| **M1** | **Causal 2-layer GRU (hidden 128) over the track features → p(crash) per frame** | main model, the **same for E1, E2 and E3** |
| M1-sens (E4) | M1 + an MLP branch for sensor features, concatenated before the GRU | E4 only |

- **Loss (anticipation, as in DSA/UString work):**
  - positive clips: `L+ = −Σ_t w_t·log p_t` with `w_t = exp(−max(0, tte_t)/τ_w)`, τ_w = 2 s, only on frames where `loss_mask_t = 1` (before the event);
  - negative clips: `L− = −Σ_t log(1 − p_t)`.
- **Training:** AdamW, lr 1e-3, batch 64, up to 40 epochs, early stopping on **validation AP of the same experiment's validation set**, 3 seeds per configuration.
- Training windows: 5 s (50 frames) ending at a random point. For positive clips, the window must end before the event.

## 4. Data splits (clip/video level only; manifests are stored with their seed)
| Dataset | Train | Validation | Test | Notes |
|---|---|---|---|---|
| CARLA train pool (Town01/02/03/04/10HD) | 70 % of slots | 15 % | 15 % = **sim-test-familiar** | split by **slot** (stratified by family × outcome); every file of a slot stays together |
| CARLA Town05 | — | — | **sim-test-T05** (300) | never used for training, validation, tuning or normalisation |
| Nexar | official train (1,500) → 85 % train | 15 % of official train (stratified) | **official test-public (667) and test-private (677)**, reported separately | test labels (`solution.csv`) are read only by `eval/nexar_eval.py` at final scoring |
| CCD | 80 % | 10 % | 10 % **CCD-test** | **ASSUMPTION:** the official CCD split isn't in our download. Look for it in the CCD repository first; otherwise use a seeded split **grouped by source video id** (field 3 of `Crash-1500.txt`) so clips from one source video never cross splits, stratified by crash/normal × day/night × weather |
| Accid3nD | — | — | analysis only (E5) | no model training |

**Leakage controls:**
1. Splits are by clip, slot or source video, never by frame.
2. Normalisation statistics are fitted on the training split only.
3. YOLO is frozen COCO, never fine-tuned on any test data.
4. A test asserts that no `scenario_id`, slot, Nexar id or CCD source id appears in two splits.
5. Pilot clips never enter any split.
6. Hyper-parameters and thresholds are chosen on validation only; the threshold for false alarms per minute is fixed on validation.

## 5. Experiments
| ID | Train data | Model selection (validation) | Test sets | Question |
|---|---|---|---|---|
| **E1** | CARLA train (crash + near-miss = positive, normal = negative) | CARLA validation only (**no real data at all**) | Nexar-public, Nexar-private, CCD-test; also sim-test-familiar, sim-test-T05 | H1 |
| **E2** | Real train (Nexar train + CCD train) at f ∈ {10, 25, 50, 100 %} | Real validation | Same real tests | reference curve |
| **E3** | CARLA train + real f ∈ {10, 25, 50 %} (nested subsets: 10 ⊂ 25 ⊂ 50, the same subsets as E2) | Real validation | Same real tests | **H2**: compare E3(f) with E2(f) |
| E4 | CARLA train; modality sets {camera}, {camera + radar}, {camera + LiDAR}, {camera + depth}, {all} | CARLA validation | sim-test-familiar, sim-test-T05 | H3 |
| E4b | No training: sensor-estimate accuracy vs ground truth (radar range/range-rate, LiDAR gap, depth gap, image TTC) | — | All CARLA clips, by weather/lighting | sensor behaviour **inside the simulator** |
| E5 | No training | — | Accid3nD vs CARLA | H4 (§6) |
| E6 (optional) | E1/E2 models | — | Real tests split by Nexar light/weather/scene and CCD day/night/weather; CARLA tests by condition | robustness; hypothesis: AP(night) < AP(day) |

- **Positive class for real tests:** Nexar positives include near-misses; CCD positives are crashes. CARLA training uses crash + near-miss as positives. An ablation trains on crash-only CARLA positives.
- **3 seeds per configuration** → 1 (E1) + 4 (E2) + 3 (E3) + 5 (E4) = 13 configs × 3 = **39 training runs**.

## 6. E5 design (decision 2026-10-29: camera + 3D tracks; this download has no LiDAR)
1. **Crash kinematics:** from Accid3nD OpenLABEL cuboids (centre, quaternion, size, `speed`, `track_history`), compute per-track speed, deceleration, pairwise footprint gap and TTC (LABELS.md formulas) for the involved vehicles in the seconds before contact.
   - Compare their distributions with CARLA crashes of the matching family (Wasserstein distance; KS test).
2. **Camera perception gap:** YOLO recall/precision against projected 3D labels on Accid3nD images vs YOLO against GT 2D boxes on CARLA images.
   - Optional apples-to-apples rig: a CARLA **roadside** camera at a Town03 junction with similar height and FOV (UNVERIFIED; only if time allows).
3. **Unresolved before E5:**
   - (a) whether the labels mark the crash frame or crash participants (not seen in the sampled JSON);
   - (b) the coordinate system: `coordinate_systems` is empty in the sampled file; assume a local road frame in metres (speeds of about 14 m/s look plausible), to be confirmed from the dataset paper or devkit;
   - (c) camera calibration for projecting boxes.
   - If (a) can't be settled, E5 reduces to comparing near-crash kinematics, and we say so.
4. **Claims allowed:** we only describe measured differences in comparable quantities. Dashcam (CARLA ego) and roadside (Accid3nD) views differ, which is a stated limitation.

## 7. Metrics (each dataset's official protocol is kept)
| Test set | Primary | Secondary |
|---|---|---|
| Nexar test-public / test-private | **Official mAP** = mean AP over the groups `time_to_accident` ∈ {0.5, 1.0, 1.5} s, computed with `evaluate_submission.py`'s `score()`. Clip score = p at the **last frame** (ablation: max over the last 1 s) | AP per group |
| CCD-test | **AP** and **mTTA** (UString/DSTA protocol: a video is positive if any frame before the event passes the threshold; TTA = event time − first alarm time; mTTA averaged over thresholds) | TTA@R80; **false alarms per minute** on CCD-test normal clips at the validation-chosen threshold |
| CARLA tests | Same as CCD (AP, mTTA, TTA@R80, false alarms per minute on normal clips) | per family, per outcome |
| E4b | MAE, bias, miss rate (GT target ≤ 50 m but the sensor has no return) | by weather/lighting |
| E5 | Wasserstein / KS statistics, detection recall/precision | |

**Uncertainty:** 95 % bootstrap CIs over test videos (1,000 resamples) for every headline number; mean ± std over 3 training seeds. For H2, a paired bootstrap of AP(E3_f) − AP(E2_f) over test videos.

## 8. Ablations
- crash-only vs crash + near-miss CARLA positives;
- no optical flow;
- K ∈ {3, 5, 10};
- window 3 s vs 5 s;
- last-frame vs max-last-1 s Nexar scoring;
- B1/B2 vs M1;
- (optional, needs weights) a frozen ImageNet ResNet-18 appearance feature (decision D4).

## 9. Limitations to state in the paper
- CARLA visuals, vehicle set and behaviour differ from real traffic.
- No rural map (Town06/07 not installed).
- No snow (CCD has Snowy clips; analysed in E6, not simulated).
- No vehicle damage or rollovers.
- No Indian vehicle types (e.g. autorickshaws).
- Near-miss rule v1 is an operational proxy for Nexar's human labels.
- The same seed doesn't guarantee bit-identical clips across machines.
- Accid3nD is roadside camera only in our download (no LiDAR).
- CCD split assumption (§4).
- 2,000 clips is a target. Sufficiency is argued through per-family/condition balance and the E3 data-efficiency curve, not assumed.

**Licences:**
- CCD: MIT.
- Nexar: Nexar Open Data License (attribution required; citation given in the LICENSE). Its ethical-use clause bans "simulate unsafe behavior". Our use is safety-oriented collision *anticipation*, and Nexar data is never used to drive the simulator. **Flagged for your supervisor (decision D6).**
- Accid3nD: CC BY-NC-SA 4.0 (non-commercial; derived data shares alike).

## 10. Results template (filled only by finished runs)
| Exp | Config | Seeds done | Nexar-pub mAP [CI] | Nexar-priv mAP [CI] | CCD AP [CI] | CCD mTTA (s) | FA/min | Status |
|---|---|---|---|---|---|---|---|---|
| E1 | M1, CARLA | 0/3 | — | — | — | — | — | planned |
| E2 | M1, real 100 % | 0/3 | — | — | — | — | — | planned |
| … | | | | | | | | |
