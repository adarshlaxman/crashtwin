# CrashTwin simulated dataset schema (v2 design, schema version 1.0, 2026-10-29)

Status: **design.** v1 test clips (`outputs/crashtwin_sim/run_*`) use an older, different format and are kept unchanged as history.

## 1. Directory structure
```
outputs/
  crashtwin_sim/                 v1 smoke-test runs (read-only history; see PHASE_A_INSPECTION.md)
  sim_v2/
    pilot_<YYYYmmdd_HHMMSS>/     pilot runs: never part of the final dataset
    full_<YYYYmmdd_HHMMSS>/      full dataset (only after the pilot is approved)
      run_config.json            CLI args, matrix version, versions, git commit
      run_log.jsonl              one line per ATTEMPT (accepted / spillover / no_event / invalid / error)
      index.csv                  one row per WRITTEN clip
      run_report.md              produced by validation/run_report.py
      run_summary.csv            success rates by family x town x weather
      clips/<scenario_id>/       one folder per written clip (layout in section 2)
  index/                         real-data indexes (ccd.csv, nexar.csv, accid3nd.csv)
  splits/                        split manifests (clip_id, dataset, split, group, seed, version)
  processed/                     derived and rebuildable: tracks/, features/
  experiments/<E#>/<run>/        configs, checkpoints, predictions, metrics
  reports/                       final tables and figures
```
- `scenario_id = <run_tag>_<family>_s<slot:05d>_a<attempt>`, e.g. `full_20261030_2210_C1_rear_end_s00042_a1`.
- It's unique across runs because it starts with the run tag. The recorder refuses to write into an existing folder.

## 2. Per-clip files
| File | Content | Format |
|---|---|---|
| `rgb.mp4` | 100 frames, 1280×720, 10 fps | MPEG-4 (OpenCV `mp4v`; verified in v1) |
| `frames.parquet` | One row per frame: `frame_idx` (0–99), `world_frame`, `sim_time`, and a `<stream>_frame` / `<stream>_time` pair per stream | Parquet* |
| `depth.npz` | `depth_cm`: uint16 `[100, 360, 640]`, centimetres, 0 = invalid, 65535 = ≥ 655.35 m (sky) | `np.savez_compressed` |
| `lidar.npz` | `points`: float16 `[N_total, 4]` (x, y, z, intensity); `offsets`: int32 `[101]` (frame i = `points[offsets[i]:offsets[i+1]]`) | `np.savez_compressed` |
| `radar.npz` | `detections`: float32 `[N_total, 4]` (velocity, azimuth, altitude, depth); `offsets`: int32 `[101]` | `np.savez_compressed` |
| `ego.parquet` | Per frame: world x, y, z, roll, pitch, yaw; world vx, vy, vz; speed; IMU accel x/y/z and gyro x/y/z, compass; GNSS lat/lon/alt; throttle, steer, brake, hand_brake, reverse, gear | Parquet* |
| `actors.parquet` | Per frame × actor within 50 m (section 4) | Parquet* |
| `collisions.parquet` | Every collision event: world_frame, frame_idx, sim_time, other_id, other_blueprint, impulse x/y/z, intensity | Parquet* |
| `labels.json` | Outcome, event frame/time, partner, min_gap, min_ttc, conflict actor, rule version, invalid_reason (LABELS.md) | JSON |
| `meta.json` | Family, slot, attempt, seed, intended outcome, scenario parameters (initial positions/speeds, gaps, triggers, reaction delay), town/map, **full weather parameters**, traffic density, ego and other blueprints with their roles, sensor config, versions, git commit + file SHA-256 | JSON |
| `validation.json` | Status PASS/WARN/FAIL + a result for every check (VALIDATION_PLAN.md) | JSON |
| `sensors_summary.parquet` | Derived per-frame summaries (radar nearest/closing/TTC, LiDAR lane gap, depth lane gap): **derived, not raw**; kept for E4 convenience | Parquet* |

\* Parquet (zstd) via **pyarrow 25.0.1**, installed in the project venv on 2026-10-29 (decision D3). If pyarrow can't be imported, `io_utils.write_table` writes `.csv.gz` with identical columns, and every reader (`read_table`) accepts either.

Also written: `audit.png` (only when `validation/audit_clip.py` is run on a clip) and `validation.json` (written by the validator; it never changes other files).

**Instance-segmentation camera:** used at record time to compute 2D boxes and visibility. Its images are **not stored**.

## 3. Sensor rig (`configs/sensors.yaml`, mounted relative to the ego bounding box `bb`)
| Sensor | Blueprint | Mount (ego frame) | Settings |
|---|---|---|---|
| RGB | `sensor.camera.rgb` | x = bb.loc.x + 0.3, z = roof + 0.15 | 1280×720, FOV 90°, every tick |
| Instance seg. | `sensor.camera.instance_segmentation` | same as RGB | 1280×720, FOV 90° (not stored) |
| Depth | `sensor.camera.depth` | same as RGB | 640×360, FOV 90° |
| LiDAR | `sensor.lidar.ray_cast` | x = 0, z = roof + 0.3 | 32 channels, range 70 m, **100,000 pts/s** (≈ 10k per frame), rotation 10 Hz, FOV +10°/−30°, dropoff 0 |
| Radar | `sensor.other.radar` | x = front bumper + 0.1, z = 0.7 | H-FOV 30°, V-FOV 10°, range 100 m, 1,500 pts/s |
| IMU | `sensor.other.imu` | origin | every tick, no noise |
| GNSS | `sensor.other.gnss` | origin | every tick, no noise |
| Collision | `sensor.other.collision` | origin | event-based |

- **RGB intrinsics:** fx = fy = W / (2·tan(FOV/2)) = 640, cx = 640, cy = 360 (no distortion).
- **Depth intrinsics:** fx = fy = 320, cx = 320, cy = 180.
- Sensor noise is off in v2 (recorded in meta). Noise models could be an E4 ablation.

## 4. `actors.parquet` columns
- `frame_idx`, `actor_id` (CARLA id, stable within a clip), `role` (ego_conflict / scenario_actor / background), `type` (car/van/truck/bus/motorcycle/bicycle/walker), `blueprint`
- World pose: `x, y, z, roll, pitch, yaw`; world velocity: `vx, vy, vz`
- 3D box: `bb_cx, bb_cy, bb_cz` (box centre offset in the actor frame), `bb_ex, bb_ey, bb_ez` (half-extents, m)
- Ego frame: `rel_x, rel_y, rel_z, rel_vx, rel_vy`, `dist` (centre-to-centre), `gap` (footprint gap, LABELS.md)
- 2D box (RGB image): `box_x1, box_y1, box_x2, box_y2` (px; NaN if not visible), `visible_px` (instance-mask pixel count), `proj_area` (px² of the projected 3D box, clipped to the image), `occluded` (`visible_px / proj_area < 0.1`)
- Measured by the labeller: `gap` (footprint gap, m), `ttc` (predicted-overlap TTC, s; NaN = no overlap predicted)

**2D-box method (verified 2026-10-29):** in CARLA 0.9.15, instance-segmentation ids are **not** actor ids (probe: instance 41205 for actor 1307). So `camera.match_boxes` projects each actor's 3D box into the image (nearest actor first). It takes the dominant instance id with a matching semantic tag inside that region (car/van/truck/bus 14–16, motorcycle 18, bicycle 19, walker 12), and the tight box of that instance's pixels becomes the 2D box. Visually checked on the audit sheets.

## 5. Coordinate systems and units
| Frame | Convention |
|---|---|
| World | CARLA/UE4, **left-handed**: x, y horizontal, z up, metres; rotations in degrees (pitch, yaw, roll) |
| Ego | Origin = ego actor origin; x forward, y right, z up (left-handed, CARLA convention) |
| LiDAR / radar | Sensor frame, CARLA convention (x forward, y right, z up); radar azimuth/altitude in **radians**, velocity m/s (negative = approaching), depth m |
| Camera (for projection only) | Optical frame: x right, y down, z forward; from the CARLA sensor frame: (x_c, y_c, z_c) = (y, −z, x) |
| GNSS | Degrees, from the CARLA map geo-reference (not real-world positions) |
| IMU | Accelerometer m/s² (includes gravity), gyroscope rad/s, compass rad |
| Time | `sim_time` = `world.get_snapshot().timestamp.elapsed_seconds` (s); frame = CARLA world frame number |

A column's name tells you its frame: `rel_*` = ego frame, plain `x/y/z` = world. No column mixes frames.

## 6. Synchronization rules
- Synchronous mode, `fixed_delta_seconds = 0.1`, physics substepping 0.01 s (max 10 substeps).
- At each tick, the rig waits (timeout 10 s) until **every** stream has delivered `measurement.frame == world_frame`. Older frames are discarded; a timeout aborts the attempt.
- Stored per frame: every stream's own `frame` and `timestamp`. The validator requires an identical frame number and |Δt| < 1 ms.
- "10 fps" here means **simulation time**. Saved frames = simulation ticks. Wall-clock speed is unrelated and is logged as seconds per clip.

## 7. `index.csv` (one row per written clip)
`scenario_id, run_tag, family, slot_id, attempt, intended_outcome, outcome, slot_status (accepted / spillover / invalid), validation_status, town, heldout_town (bool), weather_condition, traffic_density, ego_blueprint, other_type, other_blueprint, event_frame, event_time_s, min_gap_m, min_ttc_s, impact_speed_kmh, intensity, seed, n_frames, size_mb, wall_s, matrix_version, rule_version, schema_version, git_commit, path`

## 8. Storage estimate (to be measured in the pilot)
| Stream | Estimate per clip |
|---|---|
| rgb.mp4 | ~6 MB |
| depth.npz (640×360 uint16) | ~5–8 MB (**largest uncertainty**) |
| lidar.npz (~1M points, float16) | ~5–6 MB |
| radar.npz | < 0.2 MB |
| parquet/json | < 1 MB |
| **Target** | **≤ 17 MB per clip** |

Budget: 2,000 accepted clips plus about 15 % spill-over and invalid clips ≈ 2,300 × 17 MB ≈ 39 GB ≤ the **40 GB cap**. If the pilot's median is above 17 MB, the reduction ladder (IMPLEMENTATION_PLAN.md §6) applies.
