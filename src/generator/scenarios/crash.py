"""Crash scenario families C1-C5. The ego car is set up to NOT avoid the conflict.
The measured outcome (labels.py) decides the label; a crash family can end as a near-miss
or normal clip (kept as spill-over)."""
import math

import carla

from geometry import to_ego_frame
from scenarios.base import (Scenario, ScenarioError, build_route, first_junction_index, no_junction,
                            pick_kind, raised, route_locations, yaw_diff)

VEHICLE_MIX_REAR = {"car": 0.50, "van": 0.15, "truck": 0.10, "bus": 0.05, "motorcycle": 0.20}
VEHICLE_MIX_CUT = {"car": 0.55, "van": 0.15, "truck": 0.10, "motorcycle": 0.20}
CROSS_MIX = {"car": 0.60, "van": 0.15, "truck": 0.10, "motorcycle": 0.15}
VRU_MIX = {"pedestrian": 0.65, "bicycle": 0.35}


def side_lane(wp):
    """Same-direction neighbour lane and the TM lane-change direction (True = right) needed
    for an actor in that lane to move into wp's lane."""
    for lane, move_right in ((wp.get_left_lane(), True), (wp.get_right_lane(), False)):
        if lane is not None and lane.lane_type == carla.LaneType.Driving and lane.lane_id * wp.lane_id > 0:
            return lane, move_right
    return None


def is_rightmost(wp):
    r = wp.get_right_lane()
    return r is None or r.lane_type != carla.LaneType.Driving or r.lane_id * wp.lane_id < 0


# ============================================================ C1 rear-end
class RearEnd(Scenario):
    family, intended = "C1_rear_end", "crash"

    def setup(self):
        self.spawn_ego(lambda route: {})
        kind = pick_kind(self.rng, self.cfg.get("other_actor_mix", VEHICLE_MIX_REAR))
        idx = int(self.rng.uniform(20.0, 35.0) / 2.0)
        self.lead = self.spawn_on_route(idx, kind)
        self.conflict = self.lead
        self.ego_reckless(vehicles=True, lights=True, speed_diff=-10)
        port = self.sw.tm_port
        self.lead.set_autopilot(True, port)
        self.tm.auto_lane_change(self.lead, False)
        self.tm.ignore_lights_percentage(self.lead, 100)
        self.tm.ignore_signs_percentage(self.lead, 100)
        self.follow_route(self.lead, idx)
        self.planned_trigger = self.rng.randint(110, 140)
        self.brake_level = round(self.rng.uniform(0.6, 1.0), 3)
        self.params.update(other_kind=kind, other_blueprint=self.lead.type_id, initial_gap_m=idx * 2.0,
                           planned_trigger_tick=self.planned_trigger, lead_brake=self.brake_level)
        return self.ego

    def on_tick(self, ep, tick):
        if tick == self.planned_trigger:
            self.lead.set_autopilot(False, self.sw.tm_port)
            self.mark_trigger(ep, tick)
        if self.trigger_tick is not None:
            self.lead.apply_control(carla.VehicleControl(throttle=0.0, brake=self.brake_level))


# ============================================================ C2 cut-in
class CutIn(Scenario):
    family, intended = "C2_cut_in", "crash"

    def setup(self):
        def need(route):
            for k in range(10, 80, 10):
                if k >= len(route) or side_lane(route[k]) is None:
                    return None
            return {}
        self.spawn_ego(need)
        k = int(self.rng.uniform(25.0, 40.0) / 2.0)
        lane, self.move_right = side_lane(self.route[k])
        kind = pick_kind(self.rng, self.cfg.get("other_actor_mix", VEHICLE_MIX_CUT))
        pool = self.sw.bps.get(kind) or self.sw.bps["car"]
        self.cutter = self.sw.spawn(self.rng.choice(pool), raised(lane.transform), "scenario_actor")
        if self.cutter is None:
            raise ScenarioError(f"spawn_failed:{kind}")
        if self.sw.night:
            self.tm.update_vehicle_lights(self.cutter, True)
        self.conflict = self.cutter
        self.ego_reckless(vehicles=True, lights=True, speed_diff=0)
        port = self.sw.tm_port
        self.cutter.set_autopilot(True, port)
        self.tm.auto_lane_change(self.cutter, False)
        self.tm.ignore_lights_percentage(self.cutter, 100)
        self.tm.ignore_vehicles_percentage(self.cutter, 100)
        self.band = round(self.rng.uniform(6.0, 12.0), 2)
        self.params.update(other_kind=kind, other_blueprint=self.cutter.type_id,
                           initial_ahead_m=k * 2.0, cut_in_ahead_max_m=self.band,
                           lane_change_right=self.move_right)
        return self.ego

    def on_tick(self, ep, tick):
        if tick == 100:  # start closing: ego faster, cutter slower
            self.tm.vehicle_percentage_speed_difference(self.ego, -25)
            self.tm.vehicle_percentage_speed_difference(self.cutter, 20)
        if tick > 100 and self.trigger_tick is None and ep.buffer:
            b, rows = self.last_rows(ep)
            a = rows.get(self.cutter.id)
            if a is None:
                return
            fx, _ = to_ego_frame(b["ego"]["x"], b["ego"]["y"], b["ego"]["yaw"], a["x"], a["y"])
            ahead = float(fx) - self.ep_mount["bb_ex"] - a["bb_ex"]
            if 2.0 <= ahead <= self.band:
                self.tm.force_lane_change(self.cutter, self.move_right)
                self.mark_trigger(ep, tick, cut_in_ahead_m=round(ahead, 2))


# ============================================================ C3 junction
class Junction(Scenario):
    family, intended = "C3_junction", "crash"
    route_length = 220.0
    ego_ignores_vehicles = True

    def setup(self):
        self.spawn_ego(self._need_junction)
        info = self.jinfo
        kind = pick_kind(self.rng, CROSS_MIX)
        pool = self.sw.bps.get(kind) or self.sw.bps["car"]
        self.cross = self.sw.spawn(self.rng.choice(pool), raised(info["cross_start"].transform), "scenario_actor")
        if self.cross is None:
            raise ScenarioError(f"spawn_failed:{kind}")
        if self.sw.night:
            self.tm.update_vehicle_lights(self.cross, True)
        self.cross.apply_control(carla.VehicleControl(hand_brake=True))
        self.conflict = self.cross
        self.ego_reckless(vehicles=self.ego_ignores_vehicles, lights=True, speed_diff=-10)
        # smoke test: |jitter| >= 0.19 s missed by 2-4 m, 0.03 s collided -> keep it narrow
        self.jitter = round(self.rng.uniform(-0.15, 0.15), 3)
        self.params.update(other_kind=kind, other_blueprint=self.cross.type_id,
                           junction_id=info["junction_id"], ego_to_conflict_m=info["ego_s"],
                           cross_to_conflict_m=info["cross_s"], crossing_angle_deg=info["angle"],
                           eta_jitter_s=self.jitter)
        return self.ego

    def _need_junction(self, route):
        # any junction that STARTS 90-160 m ahead (earlier junctions are passed on the way;
        # Town10HD's short blocks never have the first junction that far away)
        starts = [i for i in range(45, min(81, len(route))) if route[i].is_junction and not route[i - 1].is_junction]
        for j in starts:
            info = self._try_junction(route, j)
            if info is not None:
                return info
        return None

    def _try_junction(self, route, j):
        ego_jpath = [w for w in route[j:j + 40] if w.is_junction]
        if len(ego_jpath) < 3:
            return None
        ego_yaw = route[j].transform.rotation.yaw
        junction = route[j].get_junction()
        for entry, exit_ in junction.get_waypoints(carla.LaneType.Driving):
            ang = yaw_diff(entry.transform.rotation.yaw, ego_yaw)
            if not 60.0 <= ang <= 120.0 or yaw_diff(entry.transform.rotation.yaw, exit_.transform.rotation.yaw) > 20:
                continue
            path = entry.next_until_lane_end(1.0)
            best = None
            for ei, ew in enumerate(ego_jpath):
                for ci, cw in enumerate(path):
                    d = ew.transform.location.distance(cw.transform.location)
                    if d < 1.5 and (best is None or d < best[0]):
                        best = (d, ei, ci)
            if best is None:
                continue
            ups = entry.previous(self.rng.uniform(30.0, 40.0))
            if not ups or ups[0].is_junction:
                continue
            upstream = ups[0]
            conflict_idx = j + best[1]
            cross_s = upstream.transform.location.distance(entry.transform.location) + best[2] * 1.0
            cross_route = build_route(upstream, cross_s + 50.0)
            conflict_loc = route[conflict_idx].transform.location
            c_idx = min(range(len(cross_route)),
                        key=lambda i: cross_route[i].transform.location.distance(conflict_loc))
            if cross_route[c_idx].transform.location.distance(conflict_loc) > 2.0:
                continue   # the straightest crossing route does not pass the conflict point
            self.conflict_idx = conflict_idx
            self.cross_route, self.cross_conflict_idx = cross_route, c_idx
            self.cross_path = route_locations(cross_route)
            self.jinfo = {"junction_id": junction.id, "cross_start": upstream, "ego_s": conflict_idx * 2.0,
                          "cross_s": round(cross_s, 2), "angle": round(ang, 1)}
            return self.jinfo
        return None

    @staticmethod
    def _remaining(route, conflict_idx, loc):
        near = min(range(len(route)), key=lambda i: route[i].transform.location.distance(loc))
        return max((conflict_idx - near) * 2.0, 0.0)

    def on_tick(self, ep, tick):
        """Closed-loop timing (smoke test: open-loop release missed by 15-22 m). The crossing
        vehicle waits, then its desired speed is reset every tick so it reaches the conflict
        point at the ego's ETA + jitter."""
        if tick < 30 or not ep.buffer:
            return
        e = ep.buffer[-1]["ego"]
        ego_rem = self._remaining(self.route, self.conflict_idx, carla.Location(e["x"], e["y"], e["z"]))
        ego_eta = ego_rem / max(e["speed"], 2.0)
        cross_rem = self._remaining(self.cross_route, self.cross_conflict_idx, self.cross.get_location())
        if self.trigger_tick is None:
            if ego_eta > cross_rem / 8.0 + 2.5:     # releasing now would need a crawl: keep waiting
                return
            port = self.sw.tm_port
            self.cross.apply_control(carla.VehicleControl(hand_brake=False))
            self.cross.set_autopilot(True, port)
            self.tm.auto_lane_change(self.cross, False)
            self.tm.ignore_lights_percentage(self.cross, 100)
            self.tm.ignore_signs_percentage(self.cross, 100)
            self.tm.ignore_vehicles_percentage(self.cross, 100)
            self.tm.set_path(self.cross, self.cross_path)
            self.mark_trigger(ep, tick, ego_eta_at_release_s=round(ego_eta, 2))
        if cross_rem > 1.0:
            v = min(max(cross_rem / max(ego_eta + self.jitter, 0.3), 2.0), 14.0)
            self.tm.set_desired_speed(self.cross, v * 3.6)
        self.ego_rem = ego_rem

    def resolve_after_trigger(self, ep, tick):
        """The TM slows the ego before junctions, so wait until the ego has passed the
        conflict point (or 20 s after release) before taking the closest approach."""
        if self.anchor is None and self.trigger_tick is not None and \
                (getattr(self, "ego_rem", 1.0) <= 0.0 or tick >= self.trigger_tick + 200):
            if getattr(self, "passed_tick", None) is None:
                self.passed_tick = tick
            if tick >= self.passed_tick + 10:
                self.anchor = self.closest_approach_frame(ep, self.conflict.id, self.trigger_frame or 0)
                self.params["anchor_rule"] = "closest_approach_after_ego_passed_conflict_point"


# ============================================================ C4 VRU crossing
class VRUCrossing(Scenario):
    family, intended = "C4_vru_crossing", "crash"
    ego_ignores = True          # crash family: ego does not yield to the VRU
    eta_offset = (-0.3, 0.2)    # VRU start timing relative to the car's arrival (s)

    def setup(self):
        def need(route):
            for k in range(45, 66):
                if k + 5 < len(route) and no_junction(route, k - 5, k + 6) and is_rightmost(route[k]):
                    return {"k": k}
            return None
        info = self.spawn_ego(need)
        k = info["k"]
        wp = self.route[k]
        self.kind = pick_kind(self.rng, VRU_MIX)
        right = wp.transform.get_right_vector()
        self.side = carla.Vector3D(right.x, right.y, 0.0)
        if self.kind == "pedestrian":
            self.offset = wp.lane_width / 2.0 + 1.5
            loc = wp.transform.location + carla.Location(x=right.x * self.offset, y=right.y * self.offset, z=1.0)
            bp = self.rng.choice(self.sw.walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            self.vru = self.sw.spawn(bp, carla.Transform(loc), "scenario_actor")
            self.vru_speed = round(self.rng.uniform(2.0, 3.5), 3)
            lag = 0.3
        else:
            self.offset = wp.lane_width / 2.0 - 0.3
            loc = wp.transform.location + carla.Location(x=right.x * self.offset, y=right.y * self.offset, z=0.3)
            rot = carla.Rotation(yaw=wp.transform.rotation.yaw - 90.0)
            self.vru = self.sw.spawn(self.rng.choice(self.sw.bps["bicycle"]), carla.Transform(loc, rot),
                                     "scenario_actor")
            self.vru_speed = 4.0
            lag = 1.0
        if self.vru is None:
            raise ScenarioError(f"spawn_failed:{self.kind}")
        if self.kind == "bicycle":
            self.vru.apply_control(carla.VehicleControl(hand_brake=True))
        self.conflict = self.vru
        self.vru_eta = self.offset / self.vru_speed + lag
        self.jitter = round(self.rng.uniform(*self.eta_offset), 3)
        self.setup_ego()
        self.params.update(other_kind=self.kind, other_blueprint=self.vru.type_id, vru_route_m=k * 2.0,
                           vru_offset_m=round(self.offset, 2), vru_speed=self.vru_speed,
                           eta_jitter_s=self.jitter)
        return self.ego

    def setup_ego(self):
        self.ego_reckless(vehicles=self.kind == "bicycle", lights=False, walkers=True, speed_diff=0)

    def on_tick(self, ep, tick):
        if self.trigger_tick is None and tick > 5 and ep.buffer:
            e = ep.buffer[-1]["ego"]
            vl = self.vru.get_location()
            ahead, _ = to_ego_frame(e["x"], e["y"], e["yaw"], vl.x, vl.y)
            speed = e["speed"]
            car_eta = (float(ahead) - self.ep_mount["bb_ex"]) / max(speed, 0.1)
            if speed > 3.0 and car_eta <= self.vru_eta + self.jitter:
                self.mark_trigger(ep, tick, car_eta_at_start_s=round(car_eta, 2))
        if self.trigger_tick is not None:
            if self.kind == "pedestrian":
                self.vru.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(-self.side.x, -self.side.y, 0.0), speed=self.vru_speed))
            else:
                v = self.vru.get_velocity().length()
                self.vru.apply_control(carla.VehicleControl(throttle=0.6 if v < self.vru_speed else 0.2,
                                                            hand_brake=False))


# ============================================================ C5 single vehicle
class SingleVehicle(Scenario):
    family, intended = "C5_single_vehicle", "crash"

    def setup(self):
        if not hasattr(self.sw, "_poles") or self.sw._poles_town != self.sw.town:
            # Only slender vertical poles: a street lamp's box is centred on its overhanging arm
            # (smoke test 2026-10-29: the 'target' was above the road). Aim at the base.
            self.sw._poles = [o.bounding_box.location for o in
                              self.sw.world.get_environment_objects(carla.CityObjectLabel.Poles)
                              if o.bounding_box.extent.x < 0.6 and o.bounding_box.extent.y < 0.6
                              and o.bounding_box.extent.z > 1.0]
            self.sw._poles_town = self.sw.town

        def need(route):
            for k in range(50, min(76, len(route))):
                wp = route[k]
                t = wp.transform
                for p in self.sw._poles:
                    if abs(p.x - t.location.x) > 8 or abs(p.y - t.location.y) > 8:
                        continue
                    fx, ry = to_ego_frame(t.location.x, t.location.y, t.rotation.yaw, p.x, p.y)
                    if abs(fx) < 2.0 and wp.lane_width / 2 + 0.5 <= ry <= wp.lane_width / 2 + 4.0:
                        return {"k": k, "target": p}
            return None
        info = self.spawn_ego(need)
        self.target = info["target"]
        self.ego_reckless(vehicles=False, lights=True, speed_diff=-10)
        self.params.update(other_kind="static_pole", target_route_m=info["k"] * 2.0,
                           target=[self.target.x, self.target.y])
        return self.ego

    def on_tick(self, ep, tick):
        if not ep.buffer:
            return
        e = ep.buffer[-1]["ego"]
        dist = math.hypot(self.target.x - e["x"], self.target.y - e["y"])
        if self.trigger_tick is None and tick > 40 and dist < max(20.0, e["speed"] * 2.5):
            self.ego.set_autopilot(False, self.sw.tm_port)
            self.hold_speed = max(e["speed"], 6.0)
            self.mark_trigger(ep, tick, trigger_dist_m=round(dist, 2))
        if self.trigger_tick is not None:
            fx, ry = to_ego_frame(e["x"], e["y"], e["yaw"], self.target.x, self.target.y)
            steer = max(-0.7, min(0.7, 1.5 * math.atan2(float(ry), max(float(fx), 0.5))))
            self.ego.apply_control(carla.VehicleControl(
                throttle=0.6 if e["speed"] < self.hold_speed else 0.3, steer=steer))

    def resolve_after_trigger(self, ep, tick):
        if self.anchor is None and self.trigger_tick is not None and tick >= self.trigger_tick + 60:
            best = min((b for b in ep.buffer if b["world_frame"] >= self.trigger_frame),
                       key=lambda b: math.hypot(self.target.x - b["ego"]["x"], self.target.y - b["ego"]["y"]))
            self.anchor = best["world_frame"]
            self.params["anchor_rule"] = "closest_approach_to_static_target"
