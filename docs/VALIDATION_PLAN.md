# CrashTwin validation plan (v1.0, 2026-10-29)

Status: **design.** Every check ID below becomes a key in `validation.json`.

Test runner: `python -m unittest discover crashtwin/tests` (standard library, so no new dependency). pytest is optional (decision D3).

## 1. Clip status rules
- **FAIL** if any FAIL-level check fails. The clip becomes outcome `invalid`, is kept on disk and is excluded from splits.
- **WARN** if no check fails but at least one warns. The clip is usable only after audit; WARN never counts as PASS. The pilot report lists every WARN.
- **PASS** if every check passes.

## 2. Unit tests (no CARLA needed)
| ID | Test |
|---|---|
| U1 | `footprint_gap`: overlapping boxes → 0; parallel boxes 2 m apart → 2.0 ± 1e-6; rotated boxes against a hand-computed value |
| U2 | `ttc`: closing at 10 m/s with a 20 m gap → 2.0 s; opening → inf; range rate −0.4 m/s (threshold −0.5) → inf |
| U3 | Outcome rule v1 on synthetic logs: a collision → crash; no collision with gap 1.0 m and TTC 1.0 s → near_miss; TTC 4 s → normal; TTC 2 s → invalid/borderline |
| U4 | Depth decode: known RGB-encoded values → expected metres; cm uint16 round-trip error ≤ 0.5 cm |
| U5 | LiDAR/radar packing: offsets reproduce the per-frame arrays exactly |
| U6 | Window cutting: event at frame 75 → `event_frame` 75; post-event frames kept |
| U7 | Seed function: deterministic; different (slot, attempt) → different seeds |
| U8 | Split builder: no id in two splits; Town05 only in sim-test-T05; CCD source groups never split; Nexar test ids never in train/val |
| U9 | Validator on hand-made good and bad clip folders (missing file, duplicate frame, time gap) → the expected status |

## 3. Synchronization and file-integrity checks (per clip)
| ID | Level | Check |
|---|---|---|
| F1 | FAIL | All required files exist and load |
| F2 | FAIL | `rgb.mp4` decodes to exactly 100 frames of 1280×720 |
| F3 | FAIL | `frames.parquet` has 100 rows; `world_frame` strictly +1 per row; no duplicates |
| F4 | FAIL | `sim_time` steps = 0.100 ± 0.001 s |
| S1 | FAIL | Each stream's frame == `world_frame` and |Δt| < 1 ms for every row |
| S2 | FAIL | depth: 100 frames, at least 50 % of pixels in (0, 655.35 m) |
| S3 | WARN | LiDAR: 100 frames; median ≥ 2,000 returns per frame; every frame within ±30 % of the **clip median**; no NaN. (Changed in implementation: rays that hit nothing within 70 m, e.g. sky, return no point, so the 10k emitted per frame isn't the expected return count. The smoke tests measured ~9.2k returns.) |
| S4 | WARN | Radar: 100 frames present (zero detections allowed, but ≥ 95 % of frames empty → WARN) |
| S5 | FAIL | IMU/GNSS/ego rows = 100; speed within [0, 60] m/s; |accel| < 200 m/s² outside the event ±0.5 s |
| S6 | WARN | Ego position jumps ≤ speed·0.1 s + 0.5 m between frames (except the event ±0.5 s) |
| S7 | WARN | Each actor's position is continuous by the same rule (teleports flagged) |

## 4. Event and label checks
| ID | Level | Check |
|---|---|---|
| L1 | FAIL | crash ⇔ ≥ 1 ego collision event inside the clip |
| L2 | FAIL | crash: `event_frame` == frame of the first collision; ∈ [60, 90] |
| L2b | WARN | crash centred on a collision: `event_frame` == the recorder's chosen `pre_frames`. (Replaces the planned "within ±2 frames of the edge" WARN: the recorder samples `pre_frames` itself, so an event at frame 61 is correct, not suspicious.) |
| L3 | WARN | crash with a non-static partner: footprint gap at `event_frame` ≤ 0.3 m |
| L4 | FAIL | near_miss: no collision; `min_gap ≤ 1.5 m` and `min_ttc ≤ 1.5 s`; `event_frame` ∈ [60, 90] |
| L5 | FAIL | normal: no collision; `min_ttc > 3.0 s` |
| L6 | WARN | outcome ≠ intended outcome → spill-over (logged, not an error) |
| L7 | WARN | `minor_contact` (intensity < 500 N·s) or `unplanned_partner` |
| L8 | FAIL | Partner id / blueprint in `labels.json` matches `collisions.parquet` and `actors.parquet` |

**What we deliberately do NOT assume:** a crash doesn't have to cause a big ego speed drop (light contacts and side hits may not), and a sensor gap reaching 0 doesn't prove a collision. Only the collision sensor decides L1.

## 5. Reproducibility checks
| ID | Level | Check |
|---|---|---|
| R1 | FAIL | `meta.json` has the seed, slot, attempt, matrix/rule/schema versions, CARLA versions, map, full weather parameters, sensor config |
| R2 | WARN | Code revision recorded: a clean git commit, **or**, while no commit has been approved, the SHA-256 of every generator source file (`meta.json → provenance.source_sha256`) |
| R3 | — (pilot test) | Re-run 5 pilot slots with the same seed: same outcome; event time within ±0.3 s; report the differences honestly (bit-identical output is not expected) |

## 6. Scenario smoke tests (Phase D, before the pilot)
One attempt per family in one feasible town, clear noon, all streams on. Each must produce a written clip with a validator status, or a logged reason.

| ID | Check |
|---|---|
| SM-C1…C5 | Crash families: L1/L2 pass |
| SM-N1…N4 | Near-miss families: L4 passes, or the measured outcome is logged |
| SM-R1…R4 | Normal families: L5 passes |
| SM-D | **Depth bonnet fix retest:** median |depth lane gap − GT gap| ≤ 1.0 m when the GT gap is < 30 m |
| SM-P | **Pedestrian timing fix retest:** at least 1 of 3 C4 attempts collides with the walker (not another actor) |
| SM-L | Raw LiDAR/radar/depth saved and reloadable (U5 on real data) |
| SM-O | **No overwrite:** running twice gives two run folders; writing into an existing `scenario_id` raises an error; files under `outputs/crashtwin_sim/` are unchanged (SHA-256 before and after) |

## 7. Pilot acceptance criteria (gate before the full run)
Pilot: 13 families × 10 slots = **130 slots**, up to 5 attempts each.

| ID | Criterion | Threshold | If it fails |
|---|---|---|---|
| P1 | Slot success rate per family (accepted / slots) | ≥ 50 % | redesign the controller, or **drop the family** (reported) |
| P2 | Validation PASS among written clips | ≥ 90 % (WARN and FAIL listed) | fix, then re-pilot that family |
| P3 | Sync failures (F3/F4/S1) | **0** | blocker |
| P4 | Visual label audit, 2 clips per family (26) | 0 wrong outcome labels; event frame within ±3 frames of the visible impact or closest approach | blocker |
| P5 | Sensor sanity in CARLA (E4b preview) | median |LiDAR gap − GT gap| ≤ 0.5 m and depth ≤ 1.0 m when GT gap < 30 m | investigate |
| P6 | Storage | median ≤ 17 MB per clip | reduction ladder (IMPLEMENTATION_PLAN.md §6) |
| P7 | Runtime | mean wall time per accepted clip measured; full-run estimate reported | informs the schedule; not a blocker |
| P8 | Balance | each feasible weather condition and town covered by ≥ 1 accepted clip per outcome | adjust the matrix |
| P9 | Retries | every attempt and reason in `run_log.jsonl`; success rates computed from all attempts | blocker |

The **pilot report** (`run_report.md`) gives success rates by family, town and weather separately; size and time per clip; every WARN/FAIL reason; the audit table; and a proposed revised matrix.

**Stop. The full run needs your approval.**
