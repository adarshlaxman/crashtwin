"""Normal-driving families R1-R4: the ego car follows traffic rules (Traffic Manager defaults)."""
import carla

from scenarios.base import Scenario, first_junction_index, pick_kind
from scenarios.crash import VEHICLE_MIX_REAR, VRUCrossing, side_lane


class _Normal(Scenario):
    intended = "normal"

    def ego_normal(self, follow=True):
        self.ego.set_autopilot(True, self.sw.tm_port)
        self.tm.auto_lane_change(self.ego, False)
        gap = round(self.rng.uniform(3.0, 6.0), 2)
        self.tm.distance_to_leading_vehicle(self.ego, gap)
        self.params["ego_following_gap_m"] = gap
        if follow:
            self.follow_route(self.ego)


class CarFollowing(_Normal):
    family = "R1_car_following"

    def setup(self):
        self.spawn_ego(lambda route: {})
        kind = pick_kind(self.rng, VEHICLE_MIX_REAR)
        idx = int(self.rng.uniform(20.0, 35.0) / 2.0)
        self.lead = self.spawn_on_route(idx, kind)
        self.conflict = self.lead
        self.lead.set_autopilot(True, self.sw.tm_port)
        self.tm.auto_lane_change(self.lead, False)
        self.follow_route(self.lead, idx)
        self.ego_normal()
        self.anchor_tick = self.rng.randint(120, 160)
        self.params.update(other_kind=kind, other_blueprint=self.lead.type_id, initial_gap_m=idx * 2.0,
                           anchor_tick=self.anchor_tick, anchor_rule="random_tick")
        return self.ego

    def after_tick(self, ep, tick):
        if tick == self.anchor_tick:
            self.anchor = ep.buffer[-1]["world_frame"]


class JunctionApproach(_Normal):
    family = "R2_junction_approach"
    route_length = 220.0

    def setup(self):
        def need(route):
            j = first_junction_index(route)
            return {"j": j} if j is not None and 50 <= j <= 85 else None
        info = self.spawn_ego(need)
        self.ego_normal()
        self.params.update(junction_route_m=info["j"] * 2.0, anchor_rule="ego_enters_junction")
        return self.ego

    def after_tick(self, ep, tick):
        if self.anchor is None and tick > 30:
            wp = self.sw.map.get_waypoint(self.ego.get_location())
            if wp is not None and wp.is_junction:
                self.anchor = ep.buffer[-1]["world_frame"]
                self.params["anchor_tick"] = tick


class LaneChange(_Normal):
    family = "R3_lane_change"

    def setup(self):
        def need(route):
            return {} if all(k < len(route) and side_lane(route[k]) for k in range(10, 80, 10)) else None
        self.spawn_ego(need)
        self.ego_normal(follow=False)   # a fixed path would undo the lane change
        self.start_lane = None
        self.params["anchor_rule"] = "ego_lane_id_changes"
        return self.ego

    def after_tick(self, ep, tick):
        wp = self.sw.map.get_waypoint(self.ego.get_location())
        if wp is None:
            return
        if self.trigger_tick is None and tick >= 100:
            sl = side_lane(wp)
            if sl is None:
                return
            lane, move_right = sl
            # direction for the EGO to move into `lane` is the opposite of side_lane's
            ego_right = not move_right
            b, rows = self.last_rows(ep)
            clear = True
            for a in rows.values():
                awp = self.sw.map.get_waypoint(carla.Location(a["x"], a["y"], a["z"]))
                if awp is not None and awp.lane_id == lane.lane_id and awp.road_id == lane.road_id \
                        and abs(a["x"] - b["ego"]["x"]) + abs(a["y"] - b["ego"]["y"]) < 28.0:
                    clear = False
                    break
            if clear:
                self.start_lane = (wp.road_id, wp.lane_id)
                self.tm.force_lane_change(self.ego, ego_right)
                self.mark_trigger(ep, tick, lane_change_right=ego_right)
        elif self.trigger_tick is not None and self.anchor is None and self.start_lane is not None:
            if wp.lane_id != self.start_lane[1]:
                self.anchor = ep.buffer[-1]["world_frame"]
                self.params["anchor_tick"] = tick
            elif tick > self.trigger_tick + 80:
                self.params["lane_change_failed"] = True
                self.anchor = None


class VRUNearby(VRUCrossing):
    """A VRU crosses well before the ego car arrives; the ego yields if needed."""
    family, intended = "R4_vru_nearby", "normal"
    eta_offset = (3.0, 5.0)     # VRU starts 3-5 s earlier than a collision course would need

    def setup_ego(self):
        self.ego.set_autopilot(True, self.sw.tm_port)
        self.tm.auto_lane_change(self.ego, False)
        self.tm.distance_to_leading_vehicle(self.ego, 4.0)
        self.follow_route(self.ego)
