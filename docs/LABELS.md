# CrashTwin label definitions (rule version v1, 2026-10-29)

Status: **design.** Thresholds are documented choices, not universal truths. Every measured quantity is stored, so labels can be recomputed under another rule version (`labels.json → rule_version`).

## 1. Principle
- **Scenario family** (intent, e.g. `C1_rear_end`) and **outcome** (measured: `crash` / `near_miss` / `normal` / `invalid`) are separate fields.
- The outcome is computed only from the collision-sensor log and simulator ground truth, after recording.

## 2. Quantities measured for every clip
All are computed per frame `t` for every **conflict candidate**: every vehicle, motorcycle, bicycle and walker within 50 m of the ego car. Static objects count only through collision events.

| Quantity | Definition | Units |
|---|---|---|
| `gap_t(a)` | Shortest distance between the ego's and actor a's **oriented 2D footprints** (3D box projected onto the ground plane). 0 if they overlap. | m |
| `ttc_t(a)` | **Constant-velocity predicted-overlap TTC:** the earliest t ∈ (0, 5 s], in 0.05 s steps, at which the two footprints, each moved by its own current velocity, overlap. 0 if they already overlap; **+inf** if no overlap is predicted (non-closing, diverging, side-by-side passing, or slower than the 5 s horizon). Computed only for actors with `gap ≤ 40 m`. | s |
| `min_gap` | min over t and a of `gap_t(a)`, over the **evaluation window** | m |
| `min_ttc` | min over t and a of `ttc_t(a)`, over the evaluation window | s |
| `conflict_actor` | the actor that gives `min_ttc` (or `min_gap` if `min_ttc` = inf) | id |

- **Evaluation window:** the whole 10 s clip, minus the first 0.5 s (to avoid start effects).
- Ego-relative quantities use the ego frame (DATASET_SCHEMA.md §5). `gap`, `range_rate` and `ttc` are computed from **world-frame** positions only; frames are never mixed.
- **Known simplification:** constant-velocity TTC ignores acceleration and curved paths. It's stored next to `min_gap`, so later rule versions can change it.
- **Implementation clarification (2026-10-29, before any dataset clip was made):**
  - The Phase B draft defined TTC as gap ÷ centre range-rate. That gives TTC ≈ 0 for oncoming traffic in the next lane and for parked cars being passed, which would make almost every normal clip a "near-miss".
  - v1 therefore uses the predicted-overlap TTC above (unit test `test_ttc_passing_side_by_side_is_inf`), which handles undefined TTC and non-closing actors explicitly (decision D1).
- **Bounding-box limitation:** gaps use CARLA bounding boxes, which can be looser than the meshes (e.g. motorcycles with riders). A gap of 0 without a collision event can happen; the **collision sensor alone decides `crash`**.

## 3. Outcome rules (v1, applied in this order)
1. **invalid**, if any validator check FAILs (VALIDATION_PLAN.md), or there are fewer than 100 frames, or a stream is missing. The reason goes in `invalid_reason`.
2. **crash**, if the collision sensor reports ≥ 1 event **involving the ego car** inside the clip.
   - `event_frame` = frame of the first such event; `event_time` = its simulation timestamp.
   - `partner_type` ∈ {vehicle, motorcycle, bicycle, walker, static}, from the blueprint (`vehicle.*` with base_type motorcycle/bicycle → motorcycle/bicycle).
   - `intensity` = |normal_impulse| (N·s). If `intensity < 500` and the partner is **not** a walker or bicycle → WARN `minor_contact` (still a crash, flagged for audit). Walker and bicycle impacts measured 223–353 N·s in the smoke tests, far below vehicle impacts (7,000–22,000), so the vehicle-scale threshold doesn't apply to them.
   - If the partner isn't the scenario's planned actor → WARN `unplanned_partner`.
3. **near_miss**, if there's no ego collision in the clip **and** at least one **same actor** a has `min_gap(a) ≤ 1.5 m` **and** `min_ttc(a) ≤ 1.5 s` (clarified: the two minima must come from the same actor, so a passing car's small gap can't combine with another actor's low TTC).
   - `event_frame` = the frame of that actor's minimum gap (closest approach); `partner_id` = that actor.
4. **normal**, if there's no ego collision **and** `min_ttc > 3.0 s`.
   - `event_frame` = null. The clip's anchor frame (random, 6–9 s) is stored as `anchor_frame` for windowing only.
5. Otherwise → **invalid**, with `invalid_reason = borderline_conflict` (no contact, but between the near-miss and normal thresholds). Kept for analysis and excluded from the main training and test sets.

**Threshold rationale (assumptions, to discuss in the paper):**
- 1.5 s TTC is a common conflict threshold in surrogate-safety analysis (e.g. the FHWA SSAM default).
- 1.5 m gap requires physical closeness, so fast but distant closing doesn't count.
- 3.0 s for normal leaves a margin, so normal clips have no conflict.
- Sensitivity analysis: report how many clips change class at near-miss {1.0 s / 1.0 m} and {2.0 s / 2.0 m}.

**Nexar mismatch (limitation):** Nexar's near-collisions are human-annotated with no published numeric rule. Our v1 rule is an operational proxy. Results are reported separately for crash-only and crash + near-miss positives, wherever the data allow.

## 4. Event timing inside a clip
- The recorder targets `event_time − clip_start ∈ [6.0, 9.0] s`, so `event_frame ∈ [60, 90]`.
- Crash clips keep 1–4 s **after** impact in the raw recording. Evaluation windows are cut in preprocessing, never in the raw data.
- **Validator check:** the label's `event_frame` must equal the frame of the first collision event in `collisions.parquet` (crash) or the frame of minimum gap (near-miss).

## 5. Frame-level targets (derived in preprocessing, not stored as the "truth")
For each frame t with time `s_t`:

| Target | Crash / near-miss clip | Normal clip |
|---|---|---|
| `y_clip` | 1 | 0 |
| `tte_t` (time to event, s) | `event_time − s_t` (negative after the event) | +inf |
| `y_frame` (CCD convention) | 1 if `s_t ≥ event_time` else 0 | 0 |
| `useful_warning_t` | 1 if `0 < tte_t ≤ 5.0 s` | 0 |
| `loss_mask_t` | 1 if `tte_t > 0` (frames after the event are **excluded** from anticipation training and scoring) | 1 |

**No-cheating rule:** anticipation metrics only count predictions made **before** `event_time`. A detection at or after impact is never counted as a prediction.

## 6. Real-data label mapping
| Dataset | Positive | Event time | Notes |
|---|---|---|---|
| CCD crash | `Crash-1500.txt` per-frame vector; event = first 1 | `event_frame / 10` | crash appears in roughly the last 2 s of a 5 s clip; fields: day/night, weather (Normal/Rainy/**Snowy**), ego-involved (Yes/No) |
| CCD normal | none | — | 3,000 clips, 5 s |
| Nexar train | `train/positive` (collision **or** near-miss) | `time_of_event`; `time_of_alert` also given | about 40 s, about 30 fps |
| Nexar test | `solution.csv target` (+ positive/negative folders) | videos end `time_to_accident` ∈ {0.5, 1.0, 1.5} s before the event | **one score per video**; official metric = mean AP over the 3 groups |
| Accid3nD | not used for training | crash frame **to be identified** (not confirmed in the label JSON; see EXPERIMENT_DESIGN E5) | |

## 7. How labels are verified
1. **Automatic:** VALIDATION_PLAN.md §4 checks:
   - crash ⇔ collision event;
   - near-miss ⇒ no collision and the thresholds are met;
   - event-frame consistency;
   - the gap at a crash's event frame ≤ 0.3 m with the partner (for non-static partners).
2. **Visual audit:** in the pilot, 2 clips per family (26 clips); after the full run, a random 5 % sample. The reviewer watches the RGB video with the GT overlay and records agreement in `audit.csv`.
3. **Unit tests** on synthetic trajectories with known `min_gap` / `min_ttc` (VALIDATION_PLAN.md §2): implemented in `tests/test_core.py` (crash, near-miss stop 1.0 m short, normal, borderline, oncoming traffic = normal); all pass.

Implementation: `src/generator/labels.py` (`RULE_V1`, `compute_outcome`), `src/generator/geometry.py`.
