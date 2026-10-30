"""CrashTwin generator v2: run controller.

Modes
  --smoke          one slot per family (clear noon, low traffic) in one feasible town
  --pilot          13 families x --slots-per-family slots, spread over towns and weather
  --full           the full matrix; REFUSED unless --approved-full is given
  --resume DIR     continue an interrupted run (accepted slots are skipped)

Every attempt is logged in run_log.jsonl. Clips are never overwritten or deleted."""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "validation"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import carla  # noqa: E402
from io_utils import (StorageGuard, attempt_seed, dir_size_bytes, make_new_dir, provenance,  # noqa: E402
                      SCHEMA_VERSION, write_json)
from labels import RULE_V1, compute_outcome  # noqa: E402
from recorder import Episode, finish_clip, window_indices, write_clip  # noqa: E402
from rig import Rig, SensorTimeout  # noqa: E402
from scenarios import FAMILIES  # noqa: E402
from scenarios.base import WARMUP_TICKS, ScenarioError  # noqa: E402
from validate_clip import validate  # noqa: E402
from world import SimWorld, connect_or_start_carla  # noqa: E402

CFG = HERE.parents[1] / "configs"
PROJECT = Path(os.environ.get("CRASHTWIN_ROOT") or HERE.parents[2])   # folder that contains crashtwin/
MAX_TICKS = 600
N_FRAMES = 100
PRE_RANGE = (60, 90)
INDEX_COLS = ["scenario_id", "run_tag", "family", "slot_id", "attempt", "intended_outcome", "outcome",
              "slot_status", "validation_status", "town", "heldout_town", "weather_condition",
              "traffic_density", "ego_blueprint", "other_type", "other_blueprint", "event_frame",
              "event_time_s", "min_gap_m", "min_ttc_s", "impact_speed_kmh", "intensity", "seed", "n_frames",
              "size_mb", "wall_s", "matrix_version", "rule_version", "schema_version", "git_commit", "path"]
UNVERIFIED_TOWNS_FALLBACK = ["Town03", "Town10HD_Opt", "Town01_Opt", "Town02_Opt", "Town05_Opt"]


def load_cfg():
    def y(n):
        return yaml.safe_load((CFG / n).read_text())
    return y("scenario_matrix.yaml"), y("sensors.yaml"), y("weather.yaml"), y("storage.yaml")


def sim_root(storage_cfg):
    """storage.yaml's sim_root, relative to the project folder unless absolute."""
    root = Path(storage_cfg["sim_root"])
    return root if root.is_absolute() else PROJECT / root


def feasible_towns(fam):
    t = fam.get("feasible_towns")
    return list(t) if isinstance(t, list) else list(UNVERIFIED_TOWNS_FALLBACK)


def expand_slots(matrix, mode, families, slots_per_family, smoke_town=None):
    conds = matrix["weather_conditions"]
    dens = list(matrix["traffic_density"])
    slots = []
    for fi, name in enumerate(families):
        fam = matrix["families"][name]
        towns = feasible_towns(fam)
        if mode == "smoke":
            town = smoke_town if smoke_town in towns else next(t for t in towns if t != "Town05_Opt")
            slots.append({"slot_id": f"{name}_s00000", "family": name, "town": town,
                          "weather": "clear_noon", "density": "low"})
            continue
        n_slots = slots_per_family.get(name, 0) if isinstance(slots_per_family, dict) else slots_per_family
        tag = "_full" if mode == "full" else ""      # full-run seeds differ from the pilot's
        for s in range(n_slots):
            slots.append({"slot_id": f"{name}{tag}_s{s:05d}", "family": name, "town": towns[s % len(towns)],
                          "weather": conds[(s + fi) % len(conds)], "density": dens[(s + fi) % len(dens)]})
    return sorted(slots, key=lambda s: (s["town"], s["slot_id"]))


def other_type_of(sw, other_id, blueprint):
    if other_id in sw.registry:
        return sw.registry[other_id]["type"]
    if blueprint.startswith("walker"):
        return "walker"
    if blueprint.startswith("vehicle"):
        return "vehicle"
    return "static"


class Runner:
    def __init__(self, args):
        self.args = args
        self.matrix, self.sensors_cfg, self.weather_cfg, self.storage_cfg = load_cfg()
        root = sim_root(self.storage_cfg)
        if args.resume:
            self.run_dir = Path(args.resume)
            self.run_tag = self.run_dir.name
            self.run_cfg = json.loads((self.run_dir / "run_config.json").read_text())
            self.mode = self.run_cfg["mode"]
        else:
            self.mode = args.mode
            self.run_tag = args.run_tag or f"{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.run_dir = make_new_dir(root / self.run_tag)
            make_new_dir(self.run_dir / "clips")
        cap = self.storage_cfg["full_run_cap_gb"] if self.mode == "full" else self.storage_cfg["pilot_run_cap_gb"]
        self.guard = StorageGuard(self.run_dir, cap, self.storage_cfg["min_free_gb"],
                                  self.storage_cfg["per_attempt_reserve_gb"])
        self.log_path = self.run_dir / "run_log.jsonl"
        self.index_path = self.run_dir / "index.csv"
        if not self.index_path.exists():
            with open(self.index_path, "w", newline="") as f:
                csv.writer(f).writerow(INDEX_COLS)
        self.depth_res = (320, 180) if self.storage_cfg["depth_resolution"] == "reduced" else None

    # ---------------- logging
    def log(self, rec):
        rec["time"] = datetime.now().isoformat(timespec="seconds")
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def previous_attempts(self):
        done, counts = set(), {}
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                r = json.loads(line)
                if "slot_id" not in r:
                    continue
                counts[r["slot_id"]] = max(counts.get(r["slot_id"], 0), r.get("attempt", 0))
                if r.get("status") == "accepted":
                    done.add(r["slot_id"])
        return done, counts

    # ---------------- main
    def run(self):
        a = self.args
        if self.args.resume:
            slots = self.run_cfg["slots"]
        else:
            fams = list(self.matrix["families"]) if a.families == ["all"] else a.families
            per = a.slots_per_family
            if a.family_slots:
                per = {k: int(v) for k, v in (x.split("=") for x in a.family_slots.split(","))}
                fams = list(per)
            slots = expand_slots(self.matrix, self.mode, fams, per, a.smoke_town)
            self.client = connect_or_start_carla()
            self.client.set_timeout(60.0)
            write_json(self.run_dir / "run_config.json", {
                "mode": self.mode, "args": vars(a), "slots": slots, "matrix_version": self.matrix["version"],
                "sensors": self.sensors_cfg, "weather": self.weather_cfg, "storage": self.storage_cfg,
                "label_rule": RULE_V1, "provenance": provenance(self.client),
                "free_gb_at_start": round(shutil.disk_usage(self.run_dir.anchor or ".").free / 1e9, 2)})
        if not hasattr(self, "client"):
            self.client = connect_or_start_carla()
            self.client.set_timeout(60.0)
        done, counts = self.previous_attempts()
        self.sw = SimWorld(self.client, a.tm_port, self.sensors_cfg, self.weather_cfg, self.matrix)
        self.prov = provenance(self.client)
        print(f"Run: {self.run_dir}  slots: {len(slots)}  already accepted: {len(done)}")
        t_run = time.time()
        try:
            for i, slot in enumerate(slots):
                if slot["slot_id"] in done:
                    continue
                for attempt in range(counts.get(slot["slot_id"], 0) + 1, a.max_attempts + 1):
                    stop = self.guard.check()
                    if stop:
                        self.log({"status": "storage_stop", "reason": stop})
                        print("STORAGE STOP:", stop)
                        return
                    res = self.attempt(slot, attempt)
                    print(f"[{i + 1}/{len(slots)}] {slot['slot_id']} a{attempt} {slot['town']} "
                          f"{slot['weather']}: {res['status']} {res.get('outcome') or ''} "
                          f"{res.get('reason') or ''} ({res['wall_s']:.0f}s, "
                          f"{(time.time() - t_run) / 60:.1f} min total)", flush=True)
                    if res["status"] == "accepted":
                        break
        finally:
            try:
                self.sw.cleanup()
                self.sw.release_sync()
            except Exception:
                pass
            print(f"Finished. Run folder: {self.run_dir}")

    def attempt(self, slot, attempt):
        t0 = time.time()
        fam_name = slot["family"]
        fam_cfg = self.matrix["families"][fam_name]
        seed = attempt_seed(self.matrix["version"], slot["slot_id"], attempt)
        rec = {"slot_id": slot["slot_id"], "family": fam_name, "attempt": attempt, "seed": seed,
               "town": slot["town"], "weather": slot["weather"], "density": slot["density"],
               "intended": fam_cfg["outcome"], "status": None, "reason": None, "outcome": None,
               "validation": None, "scenario_id": None, "size_bytes": 0, "ticks": 0}
        sc = None
        # logged first, so a native abort mid-attempt is never replayed with the same seed
        self.log({"slot_id": slot["slot_id"], "family": fam_name, "attempt": attempt, "status": "started"})
        rig = None
        try:
            self.sw.load_town(slot["town"])
            self.sw.tm.set_random_device_seed(seed)
            rng = random.Random(seed)
            np.random.seed(seed % (2 ** 32))
            weather = self.sw.set_weather(slot["weather"])
            sc = FAMILIES[fam_name](self.sw, rng, fam_cfg)
            ego = sc.setup()
            keep = [ego.get_location()] + [x.get_location() for x in self.sw.scenario_actors]
            keep += [w.transform.location for w in sc.route[::10][:12]]
            bg = self.sw.spawn_background(slot["density"], rng, keep_clear=keep)
            rig = Rig(self.sw.world, ego, self.sensors_cfg, self.depth_res,
                      self.storage_cfg.get("lidar_points_per_second"))
            sc.ep_mount = rig.mount
            ep = Episode(self.sw.world, rig, ego, self.sw.registry)
            rec.update(self.run_episode(sc, ep, rig, rng))
            rec["ticks"] = ep.ticks
            if rec["status"] is None:
                rec.update(self.write(slot, attempt, seed, sc, ep, rig, weather, bg, rec, t0))
        except ScenarioError as e:
            rec.update(status="setup_error", reason=str(e))
        except SensorTimeout as e:
            rec.update(status="error", reason=f"sensor_timeout:{e}")
        except (RuntimeError, FileExistsError) as e:
            rec.update(status="error", reason=f"{type(e).__name__}:{e}")
            traceback.print_exc()
            self.recover()
        finally:
            try:
                if rig is not None:
                    rig.destroy(self.client)
                self.sw.cleanup()
            except Exception as e:
                rec["cleanup_error"] = str(e)
                self.recover()
        rec["wall_s"] = round(time.time() - t0, 2)
        if sc is not None:
            rec["params"] = {k: v for k, v in sc.params.items() if k != "ego_spawn"}
        self.log(rec)
        return rec

    def recover(self):
        """Reconnect after a CARLA error; use a fresh Traffic Manager port."""
        try:
            self.sw.release_sync()
        except Exception:
            pass
        self.args.tm_port += 1
        self.client = connect_or_start_carla()
        self.client.set_timeout(60.0)
        self.sw = SimWorld(self.client, self.args.tm_port, self.sensors_cfg, self.weather_cfg, self.matrix)

    def run_episode(self, sc, ep, rig, rng):
        """Tick until the anchor (first collision or scenario anchor) plus enough post-event frames."""
        anchor, source, post_until, pre = None, None, None, None
        for _ in range(MAX_TICKS):
            sc.on_tick(ep, ep.ticks)
            ep.tick()
            sc.after_tick(ep, ep.ticks)
            if source == "scenario" and rig.collisions:
                anchor = None   # a collision during the post-event frames: re-centre on it
            if anchor is None:
                if rig.collisions:
                    c = rig.collisions[0]
                    if c["world_frame"] - ep.spawn_frame < WARMUP_TICKS:
                        return {"status": "early_collision", "reason": f"collision_in_warmup:{c['other_blueprint']}"}
                    anchor, source = c["world_frame"], "collision"
                elif sc.anchor is not None:
                    anchor, source = sc.anchor, "scenario"
                if anchor is not None:
                    available = anchor - ep.spawn_frame - WARMUP_TICKS
                    if available < PRE_RANGE[0]:
                        return {"status": "insufficient_history", "reason": f"anchor_after_{anchor - ep.spawn_frame}_ticks"}
                    pre = rng.randint(PRE_RANGE[0], min(PRE_RANGE[1], available))
                    post_until = anchor + (N_FRAMES - pre - 1)
            if post_until is not None and ep.buffer[-1]["world_frame"] >= post_until:
                break
        if anchor is None:
            return {"status": "no_event", "reason": f"no_anchor_in_{MAX_TICKS}_ticks"}
        if ep.buffer[-1]["world_frame"] < post_until:
            return {"status": "no_event", "reason": "timeout_before_post_frames"}
        start = window_indices(ep.frames, anchor, pre, N_FRAMES)
        if start is None:
            return {"status": "insufficient_history", "reason": "window_outside_buffer"}
        self._window = list(ep.buffer)[start:start + N_FRAMES]
        sc.params.update(anchor_source=source, pre_frames=pre)
        return {"status": None}

    def write(self, slot, attempt, seed, sc, ep, rig, weather, bg, rec, t0):
        window = self._window
        fam = slot["family"]
        sid = f"{self.run_tag}_{fam}_s{int(slot['slot_id'].rsplit('_s', 1)[1]):05d}_a{attempt}"
        clip_dir = self.run_dir / "clips" / sid
        if sc.conflict is not None:
            sc.params["conflict_actor_id"] = sc.conflict.id
        meta = {
            "scenario_id": sid, "run_tag": self.run_tag, "family": fam, "slot_id": slot["slot_id"],
            "attempt": attempt, "seed": seed, "intended_outcome": sc.intended, "town": slot["town"],
            "heldout_town": slot["town"].startswith("Town05"), "weather": weather,
            "traffic_density": slot["density"], "background_spawned": bg, "scenario_params": sc.params,
            "matrix_version": self.matrix["version"], "rule_version": RULE_V1["version"],
            "schema_version": SCHEMA_VERSION, "fps": 10, "n_frames": N_FRAMES,
            "sensor_config": {"rgb": {k: self.sensors_cfg["rgb"][k] for k in ("width", "height", "fov")},
                              "rig": rig.config_used, "mount": rig.mount},
            "provenance": self.prov,
        }
        if clip_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing {clip_dir}")
        # atomic: everything is written into <id>.partial and renamed only when complete,
        # so an interrupted run never leaves a half-written clip that looks finished
        tmp_dir = clip_dir.with_name(sid + ".partial")
        ego_df, actors_df, coll = write_clip(tmp_dir, window, rig.collisions, rig, meta)
        for c in coll:
            c["other_type"] = other_type_of(self.sw, c["other_id"], c["other_blueprint"])
        labels, gaps, ttcs = compute_outcome(ego_df.set_index("frame_idx", drop=False), actors_df, coll)
        finish_clip(tmp_dir, actors_df, gaps, ttcs, labels)
        val = validate(tmp_dir)
        os.replace(tmp_dir, clip_dir)
        size = dir_size_bytes(clip_dir)
        self.guard.add(size)
        outcome = labels["outcome"]
        if outcome == sc.intended and val["status"] == "PASS":
            status = "accepted"
        elif val["status"] == "FAIL" or outcome == "invalid":
            status = "invalid"
        elif outcome != sc.intended:
            status = "spillover"
        else:
            status = "warn_hold"      # right outcome but WARN: kept for audit, not counted
        impact = None
        if outcome == "crash" and labels["event_frame"]:
            impact = round(ego_df["speed"].iloc[max(labels["event_frame"] - 1, 0)] * 3.6, 1)
        other_bp = sc.params.get("other_blueprint", "")
        row = {"scenario_id": sid, "run_tag": self.run_tag, "family": fam, "slot_id": slot["slot_id"],
               "attempt": attempt, "intended_outcome": sc.intended, "outcome": outcome, "slot_status": status,
               "validation_status": val["status"], "town": slot["town"],
               "heldout_town": slot["town"].startswith("Town05"), "weather_condition": slot["weather"],
               "traffic_density": slot["density"], "ego_blueprint": sc.params.get("ego_blueprint"),
               "other_type": labels.get("partner_type") or sc.params.get("other_kind"),
               "other_blueprint": labels.get("partner_blueprint") or other_bp,
               "event_frame": labels["event_frame"], "event_time_s": labels["event_time_s"],
               "min_gap_m": labels["min_gap_m"], "min_ttc_s": labels["min_ttc_s"], "impact_speed_kmh": impact,
               "intensity": labels["intensity"], "seed": seed, "n_frames": len(window),
               "size_mb": round(size / 1e6, 2), "wall_s": round(time.time() - t0, 1),
               "matrix_version": self.matrix["version"], "rule_version": RULE_V1["version"],
               "schema_version": SCHEMA_VERSION, "git_commit": self.prov["git"]["commit"],
               "path": str(clip_dir)}
        with open(self.index_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=INDEX_COLS).writerow(row)
        return {"status": status, "outcome": outcome, "validation": val["status"],
                "validation_failed": val["failed"], "validation_warned": val["warned"],
                "reason": labels.get("invalid_reason"), "scenario_id": sid, "size_bytes": size,
                "event_frame": labels["event_frame"], "min_gap_m": labels["min_gap_m"],
                "min_ttc_s": labels["min_ttc_s"], "partner_type": labels.get("partner_type")}


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", dest="mode", action="store_const", const="smoke")
    g.add_argument("--pilot", dest="mode", action="store_const", const="pilot")
    g.add_argument("--full", dest="mode", action="store_const", const="full")
    g.add_argument("--resume", default=None)
    p.add_argument("--approved-full", action="store_true", help="required for --full (pilot approval)")
    p.add_argument("--families", nargs="+", default=["all"])
    p.add_argument("--slots-per-family", type=int, default=10)
    p.add_argument("--smoke-town", default="Town03")
    p.add_argument("--family-slots", default="", help="per-family slot counts, e.g. N2_cut_in_avoided=500,R1_car_following=400")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--run-tag", default=None, help="explicit run folder name (set by --supervise)")
    p.add_argument("--supervise", action="store_true",
                   help="restart with --resume (new TM port) if the CARLA client aborts natively")
    p.add_argument("--max-restarts", type=int, default=20)
    a = p.parse_args()
    if a.mode == "full" and not a.approved_full:
        sys.exit("--full needs --approved-full (the pilot report must be approved first)")
    if a.supervise:
        supervise(a)
        return
    if a.resume:
        a.mode = None
        # a resumed run keeps its original attempt limit (bug 2026-10-29: default 5 overrode 1)
        cfg_args = json.loads((Path(a.resume) / "run_config.json").read_text())["args"]
        a.max_attempts = cfg_args.get("max_attempts", a.max_attempts)
    Runner(a).run()


def supervise(a):
    """Native CARLA client aborts (e.g. 'Assertion failed: px != 0') kill Python without a
    traceback. Run the generator as a child process and resume the same run after an abort."""
    import subprocess
    base = [sys.executable, "-u", str(Path(__file__).resolve())]
    passthrough = ["--max-attempts", str(a.max_attempts), "--families", *a.families,
                   "--slots-per-family", str(a.slots_per_family), "--smoke-town", a.smoke_town]
    if a.family_slots:
        passthrough += ["--family-slots", a.family_slots]
    # the supervisor fixes the run folder up front, so a restart always resumes THIS run
    # (bug 2026-10-29: "newest folder by mtime" picked the pilot after audit files were written)
    root = sim_root(yaml.safe_load((CFG / "storage.yaml").read_text()))
    if a.resume:
        run_dir = Path(a.resume)
        cmd = base + ["--resume", a.resume]
    else:
        tag = f"{a.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = root / tag
        cmd = base + [f"--{a.mode}", "--run-tag", tag] + (["--approved-full"] if a.approved_full else []) + passthrough
    port = a.tm_port
    for restart in range(a.max_restarts + 1):
        proc = subprocess.run(cmd + ["--tm-port", str(port)])
        if proc.returncode == 0:
            return
        if not (run_dir / "run_config.json").exists():
            sys.exit(f"run folder {run_dir} was never initialised; nothing to resume")
        with open(run_dir / "run_log.jsonl", "a") as f:
            f.write(json.dumps({"status": "process_abort", "returncode": proc.returncode, "restart": restart + 1,
                                "time": datetime.now().isoformat(timespec="seconds")}) + "\n")
        print(f"SUPERVISOR: generator exited with {proc.returncode}; resuming {run_dir.name} "
              f"(restart {restart + 1}/{a.max_restarts})", flush=True)
        port += 1
        cmd = base + ["--resume", str(run_dir)]
        time.sleep(10)
    sys.exit("too many restarts")


if __name__ == "__main__":
    main()
