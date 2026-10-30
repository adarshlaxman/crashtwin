"""CARLA world management: connection, town loading, synchronous settings, weather,
background traffic (incl. motorcycles and bicycles), night lights and an actor registry."""
import os
import subprocess
import time

import carla

from recorder import actor_type

# CARLA install folder: set CARLA_ROOT on a new machine (default = the original lab PC)
CARLA_ROOT = os.environ.get("CARLA_ROOT", r"C:\Users\PESU-RF\Downloads\CARLA_0.9.15\WindowsNoEditor")
CARLA_EXE = os.path.join(CARLA_ROOT, "CarlaUE4.exe" if os.name == "nt" else "CarlaUE4.sh")
WEATHER_FIELDS = ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity",
                  "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "fog_distance",
                  "fog_falloff", "wetness", "scattering_intensity", "mie_scattering_scale",
                  "rayleigh_scattering_scale", "dust_storm")


def connect_or_start_carla(host="127.0.0.1", port=2000, wait_s=300):
    """Connect to CARLA; if it isn't running, start it and wait until ready (v1 logic)."""
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    try:
        client.get_world()
        return client
    except RuntimeError:
        pass
    print("CARLA is not running. Starting it now...")
    subprocess.Popen([CARLA_EXE], cwd=os.path.dirname(CARLA_EXE))
    deadline = time.time() + wait_s
    while time.time() < deadline:
        time.sleep(5)
        try:
            client.get_world()
            time.sleep(5)
            return client
        except RuntimeError:
            print("  waiting for CARLA to finish loading...")
    raise RuntimeError("CARLA did not start in time")


def base_type(bp):
    return bp.get_attribute("base_type").as_str().lower() if bp.has_attribute("base_type") else ""


class SimWorld:
    def __init__(self, client, tm_port, sensors_cfg, weather_cfg, matrix):
        self.client = client
        self.tm_port = tm_port
        self.tm = client.get_trafficmanager(tm_port)
        self.sensors_cfg, self.weather_cfg, self.matrix = sensors_cfg, weather_cfg, matrix
        self.world = None
        self.town = None
        self.original_settings = None
        self.registry = {}            # actor_id -> info (type, blueprint, role, bb, bbox)
        self.background, self.controllers, self.scenario_actors = [], [], []

    # ---------------- town / settings
    def load_town(self, town):
        if self.town == town:
            return
        self.release_sync()
        self.world = self.client.load_world(town)
        self.town = town
        self.map = self.world.get_map()
        self.lib = self.world.get_blueprint_library()
        self.original_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.sensors_cfg["fixed_delta_seconds"]
        s.substepping = True
        s.max_substep_delta_time = self.sensors_cfg["substep_delta"]
        s.max_substeps = self.sensors_cfg["max_substeps"]
        self.world.apply_settings(s)
        self.tm.set_synchronous_mode(True)
        excl = set(self.matrix["ego"]["exclude"])
        vehicles = self.lib.filter("vehicle.*")
        self.bps = {k: [] for k in ("car", "van", "truck", "bus", "motorcycle", "bicycle")}
        for bp in vehicles:
            bt = base_type(bp) or "car"   # bmw.grandtourer, mini.cooper_s have no base_type
            if bt in self.bps:
                self.bps[bt].append(bp)
        self.ego_bps = [bp for bp in self.bps["car"] if bp.id.replace("vehicle.", "") not in excl]
        self.walker_bps = list(self.lib.filter("walker.pedestrian.*"))

    def release_sync(self):
        if self.world is not None and self.original_settings is not None:
            self.tm.set_synchronous_mode(False)
            self.world.apply_settings(self.original_settings)

    # ---------------- weather
    def set_weather(self, condition):
        c = self.weather_cfg["conditions"][condition]
        wp = getattr(carla.WeatherParameters, c["preset"])
        wp = carla.WeatherParameters(**{f: getattr(wp, f) for f in WEATHER_FIELDS})
        for k, v in c.get("override", {}).items():
            setattr(wp, k, v)
        self.world.set_weather(wp)
        self.night = bool(c["night"])
        got = self.world.get_weather()
        return {"condition": condition, "preset": c["preset"], "night": self.night,
                "parameters": {f: round(getattr(got, f), 4) for f in WEATHER_FIELDS}}

    # ---------------- registry
    def register(self, actor, role):
        bp_id = actor.type_id
        bt = ""
        if bp_id.startswith("vehicle"):
            bt = actor.attributes.get("base_type", "").lower()
        bb = actor.bounding_box
        self.registry[actor.id] = {
            "type": actor_type(bp_id, bt), "blueprint": bp_id, "role": role,
            "bbox": carla.BoundingBox(bb.location, bb.extent),
            "bb": {"bb_cx": bb.location.x, "bb_cy": bb.location.y, "bb_cz": bb.location.z,
                   "bb_ex": bb.extent.x, "bb_ey": bb.extent.y, "bb_ez": bb.extent.z},
        }

    def spawn(self, bp, transform, role, autopilot=False):
        a = self.world.try_spawn_actor(bp, transform)
        if a is None:
            return None
        if role == "background":
            self.background.append(a)
        else:
            self.scenario_actors.append(a)
        self.register(a, role)
        if autopilot:
            a.set_autopilot(True, self.tm_port)
        return a

    # ---------------- traffic
    def spawn_background(self, density, rng, keep_clear=None, clear_radius=25.0):
        d = self.matrix["traffic_density"][density]
        mix = self.matrix["background_vehicle_mix"]
        kinds, weights = list(mix), [mix[k] for k in mix]
        sps = self.map.get_spawn_points()
        rng.shuffle(sps)
        n = 0
        for sp in sps:
            if n >= d["vehicles"]:
                break
            if keep_clear is not None and any(sp.location.distance(c) < clear_radius for c in keep_clear):
                continue
            kind = rng.choices(kinds, weights)[0]
            pool = self.bps.get(kind) or self.bps["car"]
            a = self.spawn(rng.choice(pool), sp, "background", autopilot=True)
            if a is not None:
                n += 1
                if self.night:
                    self.tm.update_vehicle_lights(a, True)
        walkers = []
        ctrl_bp = self.lib.find("controller.ai.walker")
        self.world.set_pedestrians_cross_factor(0.05)
        for _ in range(d["walkers"]):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = rng.choice(self.walker_bps)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            w = self.spawn(bp, carla.Transform(loc + carla.Location(z=1.0)), "background")
            if w is not None:
                walkers.append(w)
        self.world.tick()
        for w in walkers:
            c = self.world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            self.controllers.append(c)
        self.world.tick()
        for c in self.controllers:
            c.start()
            dest = self.world.get_random_location_from_navigation()
            if dest is not None:
                c.go_to_location(dest)
            c.set_max_speed(1.0 + rng.random())
        return {"vehicles": n, "walkers": len(walkers)}

    def ego_lights(self, ego):
        if self.night:
            ego.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))

    def cleanup(self):
        for c in self.controllers:
            try:
                c.stop()
            except RuntimeError:
                pass
        actors = self.controllers + self.background + self.scenario_actors
        if actors:
            self.client.apply_batch_sync([carla.command.DestroyActor(a) for a in actors])
        self.controllers, self.background, self.scenario_actors = [], [], []
        self.registry = {}
        if self.world is not None:
            for _ in range(3):
                self.world.tick()
