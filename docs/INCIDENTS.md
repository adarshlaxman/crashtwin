# Incident log (honest record of problems found and how they were handled)

| When (2026-10-29) | What happened | Impact | Fix |
|---|---|---|---|
| 14:18 | v1 generator crashed on Town03_Opt (`Assertion failed: px != 0`) | no data | 0.9.15 ships a wrong TM file for Town03_Opt → use Town03 |
| 16:2x | Phase B TTC rule (gap ÷ range-rate) would label oncoming/passing traffic as near-misses | caught before any dataset clip | predicted-overlap TTC + same-actor near-miss rule (LABELS.md) |
| 16:4x | Smoke tests: junction timing open-loop missed; lamp-pole boxes centred on the arm; walker impacts flagged as minor contact | smoke clips only | closed-loop crossing timing; slender poles only; VRU exemption |
| 17:3x–20:15 | Native CARLA client aborts (0xC0000409), 3 in the pilot and 1 in the full run | attempts lost, never replayed | `--supervise` + "started" log records |
| 19:2x | Training processes return 0xC0000409 **after** saving metrics (CUDA teardown on Windows) | none (results saved) | `os._exit(0)` after saving; experiment chains no longer stop on one failed run |
| 19:59 | A job name containing `:` created an NTFS alternate-data-stream log | log hidden | job names sanitised for files |
| 20:15 | After a native abort, the supervisor resumed the **newest folder by mtime**, which was the pilot (audit files had just been written), not the full run | full generation stopped at 431/3300; one `process_abort` line was appended to the pilot's `run_log.jsonl`; pilot clips unchanged | the supervisor fixes the run folder up front (`--run-tag`) and resumes only that folder |
| 20:15–20:26 | `sim_chain` saw the generator stop and ran on the 395 clips that existed: tracks, cache, E1_sim_s0 | preliminary results | moved (not deleted) to `outputs/experiments_prelim_n395/`, `features/archive_prelim_n395/`, `splits/archive_prelim_n395/`; the final cache is rebuilt after generation |
| 20:29 | The resume used the default 5 attempts instead of the run's 1 | 3 extra attempts on one slot (valid, logged) | a resumed run reads `max_attempts` from its own `run_config.json` |
| E5 | Automatic crash detection from Accid3nD 3D boxes gave 3 candidates; visual check of the real images: **0/3 were crashes** (label overlaps; the s05 incident is a vehicle fire) | crash-level E5 not possible without manual annotation | E5 revised to traffic-kinematics realism; frames kept in `outputs/results/e5_frames/` |
| 2026-10-30 ~01:4x | Full generation hit the 40 GB storage cap (`STORAGE STOP: run_cap`) at slot 3,040/3,300 | 260 slots never run, all Town10HD (R3 100, R1 66, R2 40, R4 40, N4 14); crash families and Town05 complete; 2,576 clips, 1,877 usable | intended guard behaviour; cap not raised (user budget). Reported in the paper |
| 2026-10-30 | All 18 sim-chain training steps returned 0xC0000409 after saving `metrics.json` (known CUDA teardown) | none: 26/26 runs have metrics | none needed |
| 2026-10-30 | E6: CCD gives no rows | CCD normal clips have no day/night/weather labels, so per-condition AP is undefined | E6 reported on Nexar + CARLA only |
