# CrashTwin implementation plan (v1.0, 2026-10-29)

Status: **plan only. Phase C hasn't started.** Nothing below runs until you authorise it.

## 1. Ordered phases
| Step | Work | New / changed files | Depends on |
|---|---|---|---|
| **C0** | Configs + test skeleton | `configs/sensors.yaml`, `configs/weather.yaml` (8 conditions with full parameters, incl. custom fog), `configs/storage.yaml`, `tests/__init__.py` | — |
| **C1** | Sensor rig + recorder: per-stream frame/time, ring buffer, 10 s window, raw depth/LiDAR/radar, parquet/json writers, no-overwrite guard, storage guard | `src/generator/rig.py`, `recorder.py`, `io_utils.py` | C0, D3 |
| **C2** | Labeller + validator + run report; unit tests U1–U7, U9 | `src/generator/labels.py`, `src/validation/validate_clip.py`, `run_report.py`, `tests/test_labels.py`, `test_io.py`, `test_validator.py` | C1 |
| **C3** | World + run controller: traffic mix incl. motorcycles/bicycles, night lights, weather, slots, seeds, resume, `run_log.jsonl` | `src/generator/world.py`, `run.py` | C1 |
| **C4** | Port crash families C1–C4 from v1 (v1 file stays frozen); **retest depth (SM-D) and pedestrian (SM-P)** | `src/generator/scenarios/base.py`, `crash.py` | C2, C3 |
| **C5** | New families: C5 single-vehicle, N1–N4, R1–R4 | `scenarios/crash.py`, `near_miss.py`, `normal.py` | C4 |
| **D** | Smoke tests SM-* (VALIDATION_PLAN.md §6), including no-overwrite SM-O | `tests/smoke_scenarios.py` | C5 |
| **E** | Pilot: 130 slots → `outputs/sim_v2/pilot_<ts>/` + `run_report.md` | — | D passes |
| **F** | **Review gate: stop and report** | `docs/PILOT_REPORT.md` | E |
| G | Full run after approval → `outputs/sim_v2/full_<ts>/` (resumable) | — | your OK |
| H | Handoff: docs, commands, limitations | `docs/LIMITATIONS.md`, `LICENCES.md`, README | G |

**Parallel track (no CARLA needed; can be built during C3–E and run while the simulator is busy, using CPU only):**
- `src/real/ccd.py`, `nexar.py`, `accid3nd.py` → `outputs/index/*.csv`
- `src/splits/make_splits.py` + U8
- Settle D5 (CCD split) and the E5 unknowns (Accid3nD crash frames, coordinate system).

**Must be sequential:** C1 → C2 → C4 → C5 → D → E → F. Every scenario depends on the rig, recorder, labeller and validator being right.

Phase 2/3 code (`features/`, `models/`, `eval/`) comes after F, as described in EXPERIMENT_DESIGN.md. YOLO/flow extraction on the real data can start before the full CARLA run finishes, but **not at the same time as it** (both use the GPU).

## 2. Reused code
- From `src/sim/generate_crashes.py` (v1; **copied, not edited**):
  - `connect_or_start_carla`, the per-stream frame-matching loop, `radar_summary` (road filter), `lidar_summary`, `depth_summary` (bonnet fix), `side_lane` (TM direction fix), `to_ego_frame`;
  - the pedestrian ETA trigger, the rear-end/cut-in/junction set-ups, and the Town03 note.
- From `lesson8_auto_labeled_dataset.py`: the `boxes_from_instance_image` idea, for 2D boxes.
- `configs/paths.yaml`, `scripts/check_env.py`.

## 3. Commands (once implemented)
```
# unit tests
.venv\Scripts\python.exe -m unittest discover crashtwin/tests
# one smoke attempt per family
.venv\Scripts\python.exe crashtwin/src/generator/run.py --smoke --families all
# pilot
.venv\Scripts\python.exe crashtwin/src/generator/run.py --pilot --slots-per-family 10
# validate / re-report a run (read-only)
.venv\Scripts\python.exe crashtwin/src/validation/run_report.py outputs/sim_v2/pilot_<ts>
# resume an interrupted run
.venv\Scripts\python.exe crashtwin/src/generator/run.py --resume outputs/sim_v2/<run>
```

## 4. Dependencies
| Package | Status | Needed for |
|---|---|---|
| carla 0.9.15, numpy, opencv, pandas, polars, torch/cu126, ultralytics, lap, scikit-learn, open3d, PyYAML | ✅ installed | |
| **pyarrow** | ❌ not installed | parquet files (**D3**); fallback: csv.gz |
| pytest | ❌ not installed | optional (unittest works) |
| Ultralytics YOLO weights (e.g. `yolo11s.pt`, ~19 MB, auto-download) | ❌ | Phase 2 (**D2**) |
| torchvision ResNet-18 ImageNet weights (~45 MB) | ❌ | optional ablation (**D4**) |

## 5. Runtime estimates (assumptions; the pilot measures the real values)
- **v1 measured:** about 18 s wall time per 5 s attempt (RGB + depth + LiDAR + radar, low raw output).
- **v2 assumption:** 30–60 s per attempt (10 s clip, ≥ 3 s warm-up, scenario lead-in, extra instance-seg camera, raw writes), and about 1.6 attempts per accepted clip.
  - **Pilot (130 slots): about 2–4 h.** This is more than the 1–2 h stated in Phase A; that estimate assumed 5 s clips.
  - **Full run (2,000 accepted): about 28–55 h**, resumable in nightly blocks.
  - Options if too slow (decided after the pilot): lower the LiDAR rate, run a second CARLA server on another port (24 GB VRAM; unverified), or reduce the counts.
- **Phase 2 estimate:**
  - YOLO + ByteTrack: about 1.2M frames at 10 Hz (CCD 225k, Nexar about 730k, CARLA 200k) at about 150 fps → about 2–3 h GPU;
  - optical flow at 160×90: CPU, parallel workers, about 2–4 h;
  - 39 training runs × about 10 min ≈ 7 h.

## 6. Storage plan (free now: 112 GB; rule: always ≥ 30 GB free)
| Item | Budget |
|---|---|
| Pilot (`sim_v2/pilot_*`) | ≤ 3 GB (outside the 40 GB cap) |
| Full simulated set (`sim_v2/full_*`) incl. spill-over + invalid clips | **≤ 40 GB** (enforced by `run.py`: it stops if exceeded or free < 30 GB) |
| processed/ (tracks + features, real + sim) | ~6 GB |
| experiments/ (checkpoints, predictions) | ~2 GB |
| **Free space after everything** | **≈ 61 GB** ≥ 30 GB ✅ |

**Reduction ladder** (applied in order if the pilot's median is above 17 MB per clip):
1. Depth at 320×180 for train-pool clips; full resolution kept for Town05 and a 10 % audit sample.
2. LiDAR 64k points/s.
3. Reduce the target counts proportionally (reported in the paper).

## 7. Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Near-miss families rarely hit the narrow v1 window (they crash or end up normal) | high | Tune the reaction-delay range in the pilot; spill-over keeps the data; family dropped if P1 < 50 % |
| Normal clips fail L5 (min TTC ≤ 3 s in ordinary car-following) → many `borderline` | medium | Pilot measures; possible rule v1.1 (documented) or larger following gaps |
| C5 single-vehicle hard to make deterministic | high | Pilot; may be dropped (reported) |
| Bicycle / motorcycle physics or TM behaviour odd | medium | Pilot audit; restrict to scripted roles if background riding is unstable |
| Instance-seg id ≠ actor id in 0.9.15 | medium | Fallback: 3D-box projection + semantic mask |
| Fog looks unrealistic or hides everything | medium | Tune density in smoke tests; report the parameters |
| CARLA crashes during long runs | medium | Resume + reconnect on a new TM port |
| Runtime over 55 h | medium | §5 options |
| Accid3nD crash frames / coordinate system unknown | medium | Parallel-track investigation; E5 scope reduced if needed (EXPERIMENT_DESIGN §6) |
| CCD official split not available | medium | Grouped seeded split (D5) |

## 8. Decisions needing your approval before Phase C
| ID | Decision | Recommendation |
|---|---|---|
| D1 | Label rule v1 thresholds: near-miss = no contact + min gap ≤ 1.5 m + min TTC ≤ 1.5 s; normal = min TTC > 3.0 s | approve as v1; sensitivity analysis planned |
| D2 | Download Ultralytics YOLO weights (Phase 2) | yes |
| D3 | `pip install pyarrow` (parquet); pytest optional | install pyarrow; skip pytest |
| D4 | ResNet-18 appearance ablation (weights download) | optional, decide in Phase 3 |
| D5 | CCD split: official if found, otherwise a grouped seeded split | approve the fallback |
| D6 | Nexar licence "simulate unsafe behavior" clause: confirm with your supervisor that safety-oriented crash anticipation research is fine | ask your supervisor |
| D7 | Allow dropping a family (e.g. C5) if the pilot success rate < 50 % | approve |
| D8 | Full-run time could be 28–55 h | accept, or plan to reduce counts after the pilot |
