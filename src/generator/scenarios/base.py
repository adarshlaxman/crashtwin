"""Scenario base class and shared helpers (routes, spawning, emergency braking, closest approach).

A scenario only sets up actors and controls them; it never labels the clip. Its `anchor`
(world frame) tells the run controller where to centre the 10 s window when no collision happened."""
import math

import carla
import numpy as np

from geometry import box_corners, footprint_gap, predicted_ttc

WARMUP_TICKS = 30          # 3 s at 10 Hz before the clip can start
RESOLVE_TICKS = 60         # after the trigger, wait 6 s before taking the closest approach as anchor


class ScenarioError(RuntimeError):
    """Set-up could not be completed (logged as the attempt's failure reason)."""


def yaw_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build_route(start_wp, length_m, step=2.0):
    """Waypoints every `step` m, taking the straightest option at every fork."""
    route, dist = [start_wp], 0.0
    while dist < length_m:
        nxt = route[-1].next(step)
        if not nxt:
            break
        yaw = route[-1].transform.rotation.yaw
        route.append(min(nxt, key=lambda w: yaw_diff(w.transform.rotation.yaw, yaw)))
        dist += step
    return route


def route_locations(route, every=5):
    return [w.transform.location for w in route[::every]]


def first_junction_index(route):
    return next((i for i, w in enumerate(route) if w.is_junction), None)


def no_junction(route, i0, i1):
    return not any(w.is_junction for w in route[max(i0, 0):i1])


def raised(tf, dz=0.5):
    return carla.Transform(tf.location + carla.Location(z=dz), tf.rotation)


def pick_kind(rng, mix):
    kinds = list(mix)
    return rng.choices(kinds, [mix[k] for k in kinds])[0]


class Scenario:
    family = "base"
    intended = None
    route_length = 200.0

    def __init__(self, sw, rng, fam_cfg):
        self.sw, self.rng, self.cfg = sw, rng, fam_cfg
        self.params = {}
        self.anchor = None          # world frame to centre the window on (no-collision case)
        self.trigger_frame = None
        self.conflict = None        # planned conflict actor
        self.ego = None
        self.brake = None

    # ---------------- helpers
    @property
    def tm(self):
        return self.sw.tm

    def spawn_ego(self, require):
        """Try spawn points until `require(route) -> dict|None` accepts one."""
        sps = self.sw.map.get_spawn_points()
        self.rng.shuffle(sps)
        for sp in sps[:80]:
            wp = self.sw.map.get_waypoint(sp.location)
            if wp is None or wp.is_junction:
                continue
            route = build_route(wp, self.route_length)
            if len(route) * 2.0 < self.route_length * 0.9:
                continue
            info = require(route)
            if info is None:
                continue
            ego = self.sw.spawn(self.rng.choice(self.sw.ego_bps), raised(sp, 0.3), "ego")
            if ego is None:
                continue
            self.ego, self.route = ego, route
            self.params.update(ego_blueprint=ego.type_id, ego_spawn=[sp.location.x, sp.location.y,
                                                                    sp.rotation.yaw])
            self.sw.ego_lights(ego)
            return info
        raise ScenarioError("no_suitable_spawn")

    def spawn_on_route(self, idx, kind, role="scenario_actor", lateral=0.0):
        pool = self.sw.bps.get(kind) or self.sw.bps["car"]
        wp = self.route[idx]
        tf = wp.transform
        if lateral:
            r = tf.get_right_vector()
            tf = carla.Transform(tf.location + carla.Location(x=r.x * lateral, y=r.y * lateral), tf.rotation)
        a = self.sw.spawn(self.rng.choice(pool), raised(tf), role)
        if a is None:
            raise ScenarioError(f"spawn_failed:{kind}")
        if self.sw.night and a.type_id.startswith("vehicle"):
            self.tm.update_vehicle_lights(a, True)
        return a

    def follow_route(self, actor, start_idx=0):
        self.tm.set_path(actor, route_locations(self.route[start_idx:]))

    def ego_reckless(self, vehicles=True, lights=False, walkers=False, speed_diff=-10):
        port = self.sw.tm_port
        self.ego.set_autopilot(True, port)
        self.tm.auto_lane_change(self.ego, False)
        if vehicles:
            self.tm.ignore_vehicles_percentage(self.ego, 100)
            self.tm.distance_to_leading_vehicle(self.ego, 0.0)
        if lights:
            self.tm.ignore_lights_percentage(self.ego, 100)
            self.tm.ignore_signs_percentage(self.ego, 100)
        if walkers:
            self.tm.ignore_walkers_percentage(self.ego, 100)
        self.tm.vehicle_percentage_speed_difference(self.ego, speed_diff)
        self.follow_route(self.ego)

    def last_rows(self, ep):
        if not ep.buffer:
            return None, {}
        b = ep.buffer[-1]
        return b, {a["actor_id"]: a for a in b["actors"]}

    def ego_box(self, ego_row):
        m = self.ep_mount
        return box_corners(ego_row["x"], ego_row["y"], ego_row["yaw"], m["bb_ex"], m["bb_ey"],
                           m["bb_cx"], m["bb_cy"])

    @staticmethod
    def actor_box(a):
        return box_corners(a["x"], a["y"], a["yaw"], a["bb_ex"], a["bb_ey"], a["bb_cx"], a["bb_cy"])

    def gap_ttc_to(self, ep, actor_id):
        b, rows = self.last_rows(ep)
        if b is None or actor_id not in rows:
            return math.inf, math.inf
        e, a = b["ego"], rows[actor_id]
        eb, ab = self.ego_box(e), self.actor_box(a)
        return footprint_gap(eb, ab), predicted_ttc(eb, (e["vx"], e["vy"]), ab, (a["vx"], a["vy"]))

    def closest_approach_frame(self, ep, actor_id, since_frame):
        best, best_f = math.inf, None
        for b in ep.buffer:
            if b["world_frame"] < since_frame:
                continue
            rows = {a["actor_id"]: a for a in b["actors"]}
            if actor_id not in rows:
                continue
            g = footprint_gap(self.ego_box(b["ego"]), self.actor_box(rows[actor_id]))
            if g < best:
                best, best_f = g, b["world_frame"]
        return best_f

    def resolve_after_trigger(self, ep, tick):
        """Default anchor when no collision happens: closest approach to the conflict actor."""
        if self.anchor is None and self.trigger_tick is not None and tick >= self.trigger_tick + RESOLVE_TICKS \
                and self.conflict is not None:
            self.anchor = self.closest_approach_frame(ep, self.conflict.id, self.trigger_frame or 0)
            self.params["anchor_rule"] = "closest_approach_to_conflict_actor"

    # ---------------- interface
    trigger_tick = None

    def setup(self):
        raise NotImplementedError

    def on_tick(self, ep, tick):
        """Called BEFORE each world tick. tick = number of ticks recorded so far."""

    def after_tick(self, ep, tick):
        """Called AFTER each world tick (state of the new frame is in ep.buffer[-1])."""
        if self.brake is not None:
            self.brake.update(self, ep, tick)
        self.resolve_after_trigger(ep, tick)

    def mark_trigger(self, ep, tick, **info):
        self.trigger_tick = tick
        self.trigger_frame = ep.buffer[-1]["world_frame"] if ep.buffer else None
        self.params["trigger_tick"] = tick
        self.params.update(info)


class EmergencyBrake:
    """Ego emergency braking for near-miss families: once the predicted TTC to the conflict
    actor drops below `brake_ttc`, full braking starts after `reaction_s`. The outcome is still
    measured afterwards; this controller does not guarantee a near-miss."""

    def __init__(self, rng, ttc_range=(1.0, 1.8), reaction_range=(0.2, 0.5)):
        self.brake_ttc = round(rng.uniform(*ttc_range), 3)
        self.reaction_ticks = max(1, int(round(rng.uniform(*reaction_range) * 10)))
        self.armed_tick = None
        self.active = False

    def info(self):
        return {"brake_ttc_s": self.brake_ttc, "reaction_s": self.reaction_ticks / 10}

    def update(self, sc, ep, tick):
        if sc.conflict is None or sc.trigger_tick is None:
            return
        if self.armed_tick is None:
            _, ttc = sc.gap_ttc_to(ep, sc.conflict.id)
            if ttc <= self.brake_ttc:
                self.armed_tick = tick + self.reaction_ticks
                sc.params["brake_armed_tick"] = tick
        elif tick >= self.armed_tick:
            if not self.active:
                sc.ego.set_autopilot(False, sc.sw.tm_port)
                self.active = True
                sc.params["brake_start_tick"] = tick
            sc.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
