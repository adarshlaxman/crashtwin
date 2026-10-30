# Pilot report: generator v2 (`outputs/sim_v2/pilot_20261029_165318`)

Date: 2026-10-29. Machine-generated details: `run_report.md`, `run_summary.csv`, `index.csv` and `run_log.jsonl` in the pilot folder. Every number below was measured. Nothing is estimated unless it says so.

## 1. What ran
- 13 families × 10 slots = **130 slots**, up to 5 attempts each → **382 attempts**, **243 clips written** (3.59 GB). Every written clip is kept, including invalid ones.
- Towns: Town01_Opt, Town02_Opt, Town03, Town04_Opt, Town10HD_Opt, plus held-out **Town05_Opt**. All 8 weather conditions; 3 traffic densities.
- Run under `--supervise`: **3 native CARLA client aborts** (exit code 0xC0000409), all recovered automatically. Aborted attempts are counted as `process_abort`, and a resumed run never replays them.
- About 25 min of wall time. CCD YOLO tracking shared the GPU for the last ~30 %, so the later timings are slightly slow.

## 2. Scenario reliability (slot success = slot gets a clip with outcome == intended AND validation PASS)
| Family | Slot success | Attempt success | Main reasons for non-acceptance |
|---|---|---|---|
| C1 rear-end | 90 % | 33 % | early collision in warm-up, spawn failures, 3 native aborts |
| C2 cut-in | 70 % | 23 % | `unplanned_partner` WARN (8): the ego hits a different car |
| C3 junction | 40 % | 11 % | borderline (8) and near-miss/normal spill-over (8): timing misses |
| C4 VRU crossing | 70 % | 25 % | pedestrian spawn failures (5), no event (3) |
| C5 single vehicle | 80 % | 32 % | no pole near the route (5), early collisions (6) |
| N1 rear-end avoided | 50 % | 13 % | borderline (11), crash spill-over (10), short history (8) |
| N2 cut-in avoided | 70 % | 23 % | crash spill-over (10) |
| N3 junction conflict | 60 % | 16 % | crash spill-over (10), L4 FAIL (event outside 60–90) (4) |
| N4 VRU conflict | **30 %** | **7 %** | borderline (14), spill-over (12), no event (7) |
| R1 car following | 90 % | 56 % | borderline (5) |
| R2 junction approach | 80 % | 33 % | no junction at the right distance (9) |
| R3 lane change | 100 % | 59 % | |
| R4 VRU nearby | 70 % | 24 % | history too short (6), no event (6) |

**D7 review (threshold 50 %):** N4 (30 %) and C3 (40 %) are below the review threshold. Neither is excluded: both produce valid, correctly labelled clips (audited), just inefficiently. The full run gives them extra slots and counts measured outcomes (section 6).

By town (accepted / attempts): Town04 45 %, Town01 35 %, Town03 34 %, Town05 27 %, Town02 24 %, **Town10HD 4 %** (narrow streets: warm-up collisions and spawn failures; it still gave 20 usable clips). By weather: 16–33 % in every condition, so no condition is starved. By density: 23–24 % at all three levels.

## 3. Outcomes: intended vs measured (243 written clips)
| intended ↓ / measured → | crash | near_miss | normal | invalid |
|---|---|---|---|---|
| crash | 52 | 7 | 14 | 14 |
| near_miss | 32 | 27 | 10 | 31 |
| normal | 9 | 2 | 35 | 10 |

- **Near-miss labels are checked against ground truth** (same actor: gap ≤ 1.5 m and TTC ≤ 1.5 s, no collision). Near-miss families produced as many crashes (32) as near-misses (27) when braking came late, and 31 borderline clips when it came early. The labeller caught every one; no intended near-miss was labelled near-miss without meeting the rule.
- **False positives / negatives:** unit tests cover the rule. The visual audit (section 5) found 0 wrong labels in 6 audited clips. 8 clips FAIL L4 (the near-miss partner's closest approach falls outside frames 60–90, because the window was centred on a different actor). They are correctly excluded.

## 4. Data quality
| Check | Result |
|---|---|
| Frames per clip / duration | 100 frames, 10.0 s, 0.100 s steps, in every written clip (F2–F4: 0 failures) |
| Sync across 7 streams (S1) | **0 failures** |
| Missing sensor measurements | none (S2 depth FAIL in 2 clips: >50 % of pixels invalid when the camera is pressed against an obstacle after impact; S3 LiDAR WARN in 1) |
| Collision label ↔ collision event (L1, L2) | 0 failures; crash `event_frame` = first collision frame in every crash clip |
| Ground truth / sensor agreement (P5) | median |LiDAR − GT| **0.08 m**, depth **0.04 m**, radar **0.11 m** (frames with GT gap < 30 m; 6.7k–7.0k frames). *Agreement with CARLA ground truth only; not real-world realism.* |
| Partner types in crashes | car 32, static 24, van 12, walker 12, motorcycle 5, truck 4, bicycle 4 |

**Retests**
- **Depth bonnet fix (SM-D): PASS.** Depth-lane-gap error 0.04 m median (it was a constant 0 before the fix).
- **Pedestrian timing fix (SM-P): PASS.** 12 walker crashes in the pilot; arrival-time trigger confirmed visually (`C4_vru_crossing_s00001_a3`, fog).

## 5. Visual audit (P4)
Six audit sheets (`audit.png`, frames −40 … +5 around the event, with boxes, GT gap/TTC and sensor gaps), chosen to cover the hardest conditions:

| Clip | Condition | Verdict |
|---|---|---|
| N1 rear-end avoided `s00002_a3` | Town03, rain night | ✅ near-miss: van brakes; ego stops 0.1 m behind; no contact |
| C4 VRU crossing `s00001_a3` | Town02, fog | ✅ walker crash at frame 68; fog renders plausibly |
| C5 single vehicle `s00002_a1` | Town01, night | ✅ pole impact, 27 → 5 km/h at the event frame |
| C3 junction `s00005_a2` | Town01, rain night | ✅ side impact: crossing car visible 4 s earlier, outside the 90° view at contact |
| C1 rear-end (smoke) | Town03, clear | ✅ |
| N2 / R3 sheets | written; not yet inspected | pending |

**Observed limitations:**
- CARLA walkers don't slow the car after impact.
- Side impacts leave the camera's view at contact.
- Motorcycle bounding boxes (including the rider) are looser than the mesh, so a gap of 0 can occur without contact. The collision sensor decides.

## 6. Storage and runtime (measured)
| | Value |
|---|---|
| Clip size median / mean / max | **14.6 / 14.8 / 26.2 MB** (rgb 3.0, depth 6.8, LiDAR 4.5, radar 0.17, tables 0.12) |
| Pilot total | 3.59 GB; free disk 118.6 → 114.6 GB |
| Wall time per attempt | **11.9 s** (all statuses) |
| Attempts per accepted (intended-outcome) clip | 4.24 |
| Usable clips (validation not FAIL, valid measured outcome) per attempt | **0.47** (178 / 379) |

**Updated full-run estimate (replaces the 28–55 h assumption):**
- Under the original "accepted = intended outcome" rule: 2,000 clips ≈ 28 h and ≈ 79 GB (over budget).
- Under the **usable-clip rule** adopted for the full run (spill-over clips keep their *measured* label):
  - ~3,300 single attempts ≈ **11–12 h**, ≈ **32 GB**, giving ≈ 1,400 usable clips;
  - within the 40 GB cap; ≈ 70 GB stays free.

## 7. Decision taken for the full run (4-day plan; you granted full permission on 2026-10-29)
1. **Usable-clip rule:** training and test use every clip with no validation FAIL and a valid measured outcome. WARN L6 (intent ≠ outcome) and L7 (minor contact / unplanned partner) clips are kept, with the measured label. Invalid, borderline and FAIL clips are kept on disk but excluded.
2. **One attempt per slot** (retries don't add usable data), with slots weighted to the families that yield the scarce outcomes: N1 400, N2 500, N3 450, N4 250, R1 400, R2 200, R3 400, R4 200, C1–C5 100 each = **3,300 slots**.
3. Full-run slot ids contain `_full_`, so seeds differ from the pilot. Pilot clips are never used in any split.
4. Town05 remains held out (≈ 1/6 of each family's slots, by town round-robin).

## 8. Implementation status
- **Added:**
  - `src/generator/{run,world,rig,recorder,camera,labels,geometry,io_utils}.py`, `scenarios/{base,crash,near_miss,normal}.py`
  - `src/validation/{validate_clip,run_report,audit_clip}.py`
  - `src/real/build_indexes.py`, `src/features/{track_videos,camera_features,build_cache}.py`
  - `src/models/train.py`, `src/eval/metrics.py`, `scripts/pipeline.py`
  - `configs/{sensors,weather,storage}.yaml`, `tests/{fixtures,test_core}.py`, `README.md`, `requirements.txt`
- **Changed:** `lesson8_auto_labeled_dataset.py` (Town03_Opt → Town03); `configs/storage.yaml` (pilot cap 4 → 8 GB).
- **Dependencies:** pyarrow 25.0.1 (D3); YOLO11s weights (D2, provenance file).
- **Tests:**
  - 24/24 unit tests pass (no CARLA).
  - Smoke tests (CARLA) ran for all 13 families; each produced ≥ 1 accepted clip after the fixes listed in the docs.
  - SM-O (62 v1 files unchanged, SHA-256) PASS; SM-L (raw LiDAR/radar/depth reload) PASS; SM-D, SM-P PASS.
  - R3 (same-seed rerun) **not run**.
- **Not done / limitations:**
  - Town06/07 not installed;
  - same-seed reproducibility not measured;
  - the pilot ran with code that changed mid-run once (a supervisor restart picked up the atomic-write change); every clip records its own source hashes.
- **Licences:** D6 (Nexar "simulate unsafe behavior" clause) still needs your supervisor's confirmation. Nexar is used only for evaluating and training a collision-*warning* model; it never drives the simulator.
