"""
CrashTwin step 1.1: generate multi-sensor crash and normal-driving clips in CARLA.

Every clip matches the CCD format so real and simulated data share one pipeline:
  - 50 frames at 10 fps (5 s), dashcam-style RGB video (1280x720)
  - per-frame crash label (0 before the crash frame, 1 from it on)
  - day/night and weather tags

On top of the video, every frame also stores what real crash datasets lack:
  radar, LiDAR, depth, GNSS/IMU, the collision event and ground-truth actor states.

Crash scenarios:
  rear_end    a lead car brakes hard; the ego car ignores it
  cut_in      a car in the next lane swerves in front of the ego car
  junction    the ego car runs lights and ignores other cars in dense traffic
  pedestrian  a pedestrian runs across the road; the ego car ignores walkers
Normal clips: the ego car drives safely with the same traffic and weather.

Output (new folder each run):
  outputs/crashtwin_sim/run_<timestamp>/
      index.csv                 one row per clip
      clips/<clip_id>/video.mp4 RGB video, 10 fps
      clips/<clip_id>/sensors.csv per-frame sensor summary + labels
      clips/<clip_id>/actors.csv  ground-truth positions of nearby actors
      clips/<clip_id>/meta.json

Example:
  python crashtwin/src/sim/generate_crashes.py --crashes-per-town 2 --normals-per-town 1 --towns Town03_Opt
"""
import argparse
import collections
import csv
import json
import math
import os
import queue
import random
import shutil
import subprocess
import threading
import time
from datetime import datetime

import carla
import cv2
import numpy as np

CARLA_EXE = r"C:\Users\PESU-RF\Downloads\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe"
OUT_ROOT = r"C:\CARLA_Project\outputs\crashtwin_sim"
# Town03 (not Town03_Opt): 0.9.15 ships a wrong Traffic Manager file for Town03_Opt
# (a copy of Town04's), which crashes the client with "Assertion failed: px != 0".
TOWNS = ["Town03", "Town10HD_Opt", "Town05_Opt", "Town04_Opt", "Town02_Opt"]
SCENARIOS = ["rear_end", "cut_in", "junction", "pedestrian"]

FIXED_DT = 0.1               # 10 fps, same as CCD
CLIP_FRAMES = 50             # 5 s clips, same as CCD
CRASH_INDEX_RANGE = (30, 45) # crash frame position inside the clip
WIDTH, HEIGHT = 1280, 720
DEPTH_W, DEPTH_H = 640, 360
NUM_VEHICLES = 40
NUM_WALKERS = 30
RESPAWN_TRAFFIC_EVERY = 20   # crashed cars get stuck, so refresh traffic regularly
MAX_EPISODE_TICKS = 600      # give up on a crash attempt after 60 s
MIN_FREE_GB = 20
GT_RADIUS = 50.0             # actors closer than this go into actors.csv

WEATHERS = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "CloudyNoon": carla.WeatherParameters.CloudyNoon,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
    "HardRainNoon": carla.WeatherParameters.HardRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
    "ClearNight": carla.WeatherParameters.ClearNight,
    "CloudyNight": carla.WeatherParameters.CloudyNight,
    "SoftRainNight": carla.WeatherParameters.SoftRainNight,
}

SENSOR_COLUMNS = [
    "frame_idx", "sim_time", "label",
    # ego state (GNSS / IMU / vehicle)
    "speed_mps", "accel_x", "accel_y", "gyro_z", "lat", "lon",
    "throttle", "steer", "brake",
    # radar: nearest return in front
    "radar_n", "radar_min_depth", "radar_closing_speed", "radar_min_ttc",
    # LiDAR: free space in the ego lane
    "lidar_front_gap", "lidar_front_points", "lidar_points",
    # depth camera: free space in the ego lane
    "depth_front_gap", "depth_front_pixels",
    # ground truth from the simulator
    "gt_front_gap", "gt_closing_speed", "gt_ttc", "gt_partner_dist",
    "collision",
]
NO_VALUE = 99.0              # "nothing detected" / "no closing" for distances and TTC


def connect_or_start_carla():
    """Connect to CARLA; if it isn't running, start it and wait until ready."""
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    try:
        client.get_world()
        return client
    except RuntimeError:
        pass

    print("CARLA is not running. Starting it now...")
    subprocess.Popen([CARLA_EXE], cwd=os.path.dirname(CARLA_EXE))
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(5)
        try:
            client.get_world()
            time.sleep(5)
            return client
        except RuntimeError:
            print("  waiting for CARLA to finish loading...")
    print("CARLA did not start within 5 minutes.")
    raise SystemExit


# ---------------------------------------------------------------- sensor maths

def radar_summary(meas, radar_z):
    """Nearest radar return and its closing speed (radar velocity < 0 = approaching)."""
    det = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)
    # drop returns from the road surface (lower than 0.3 m above the ground)
    det = det[radar_z + det[:, 3] * np.sin(det[:, 2]) > 0.3]
    if len(det) == 0:
        return 0, NO_VALUE, 0.0, NO_VALUE
    vel, depth = det[:, 0], det[:, 3]
    i = int(np.argmin(depth))
    closing = np.maximum(-vel, 0.0)
    with np.errstate(divide="ignore"):
        ttc = np.where(closing > 0.5, depth / np.maximum(closing, 1e-3), NO_VALUE)
    return len(det), float(depth[i]), float(closing[i]), float(min(ttc.min(), NO_VALUE))


def lidar_summary(meas, half_len, sensor_z):
    """Gap to the first obstacle in the ego lane, ignoring the road surface."""
    pts = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    above_road = z > -(sensor_z - 0.3)
    below_roof = z < 0.5
    lane = (x > half_len) & (np.abs(y) < 1.2) & above_road & below_roof
    gap = float(x[lane].min() - half_len) if lane.any() else NO_VALUE
    return min(gap, NO_VALUE), int(lane.sum()), len(pts)


def depth_summary(image, cam_height, cam_to_bumper):
    """Gap (m) to the nearest non-road pixel in the ego lane, from the depth camera."""
    bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4)).astype(np.float32)
    depth = (bgra[:, :, 2] + bgra[:, :, 1] * 256.0 + bgra[:, :, 0] * 65536.0) \
        / 16777215.0 * 1000.0
    h, w = depth.shape
    f = w / 2.0  # 90 degree field of view
    v, u = np.mgrid[0:h, 0:w]
    height = cam_height - depth * (v - h / 2.0) / f   # above the road
    lateral = depth * (u - w / 2.0) / f
    # anything closer than the front bumper is the ego car's own bonnet
    lane = (np.abs(lateral) < 1.2) & (height > 0.3) & (height < cam_height + 0.5) \
        & (depth > cam_to_bumper + 0.05) & (depth < 200.0)
    if not lane.any():
        return NO_VALUE, 0
    gap = float(depth[lane].min()) - cam_to_bumper
    return min(max(gap, 0.0), NO_VALUE), int(lane.sum())


def to_ego_frame(ego_tf, loc):
    """World location -> (forward, right) metres relative to the ego car."""
    yaw = math.radians(ego_tf.rotation.yaw)
    dx, dy = loc.x - ego_tf.location.x, loc.y - ego_tf.location.y
    return (dx * math.cos(yaw) + dy * math.sin(yaw),
            -dx * math.sin(yaw) + dy * math.cos(yaw))


# ---------------------------------------------------------------- the generator

class CrashGenerator:
    def __init__(self, client, out_dir, save_lidar, tm_port):
        self.client = client
        self.out_dir = out_dir
        self.save_lidar = save_lidar
        self.tm = client.get_trafficmanager(tm_port)
        self.world = None
        self.background = []
        self.walker_controllers = []
        self.episode_actors = []
        self.sensors = []
        self.collisions = []
        self.write_queue = queue.Queue(maxsize=8)
        self.writer = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer.start()
        self.index_path = os.path.join(out_dir, "index.csv")
        with open(self.index_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "clip_id", "town", "scenario", "label", "crash_index",
                "weather", "daynight", "condition", "partner_type",
                "impact_speed_kmh", "intensity", "seed"])
        self.saved = 0

    # ---------- world setup

    def load_town(self, town, seed):
        print(f"\n=== Loading {town} ===")
        self.world = self.client.load_world(town)
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(seed)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        self.car_bps = [bp for bp in self.bp_lib.filter("vehicle.*")
                        if bp.has_attribute("base_type")
                        and bp.get_attribute("base_type").as_str() == "car"]
        self.vehicle_bps = [bp for bp in self.bp_lib.filter("vehicle.*")
                            if int(bp.get_attribute("number_of_wheels")) == 4]
        self.spawn_traffic()

    def spawn_traffic(self):
        self.destroy_traffic()
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        for sp in spawn_points[:NUM_VEHICLES]:
            v = self.world.try_spawn_actor(random.choice(self.vehicle_bps), sp)
            if v is not None:
                v.set_autopilot(True, self.tm.get_port())
                self.background.append(v)
        walker_bps = self.bp_lib.filter("walker.pedestrian.*")
        controller_bp = self.bp_lib.find("controller.ai.walker")
        self.world.set_pedestrians_cross_factor(0.1)
        walkers = []
        for _ in range(NUM_WALKERS):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            wbp = random.choice(walker_bps)
            if wbp.has_attribute("is_invincible"):
                wbp.set_attribute("is_invincible", "false")
            w = self.world.try_spawn_actor(wbp, carla.Transform(loc))
            if w is not None:
                walkers.append(w)
        self.world.tick()
        for w in walkers:
            c = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=w)
            self.walker_controllers.append(c)
        self.world.tick()
        for c in self.walker_controllers:
            c.start()
            c.go_to_location(self.world.get_random_location_from_navigation())
            c.set_max_speed(1.0 + random.random())
        self.background += walkers
        for _ in range(20):
            self.world.tick()
        print(f"  traffic: {len(self.background) - len(walkers)} cars, {len(walkers)} walkers")

    def destroy_traffic(self):
        for c in self.walker_controllers:
            c.stop()
        self.client.apply_batch([carla.command.DestroyActor(a)
                                 for a in self.walker_controllers + self.background])
        self.walker_controllers, self.background = [], []

    def restore(self):
        self.destroy_episode()
        self.destroy_traffic()
        if self.world is not None:
            self.tm.set_synchronous_mode(False)
            self.world.apply_settings(self.original_settings)

    # ---------- ego car and sensors

    def spawn_ego(self, scenario):
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        for sp in spawn_points[:30]:
            wp = self.map.get_waypoint(sp.location)
            if scenario in ("rear_end", "pedestrian", "cut_in"):
                # need a straight-ish stretch ahead without a junction
                ahead = wp.next(45.0)
                if wp.is_junction or not ahead or ahead[0].is_junction \
                        or ahead[0].road_id != wp.road_id:
                    continue
            if scenario == "cut_in" and self.side_lane(wp) is None:
                continue
            ego = self.world.try_spawn_actor(random.choice(self.car_bps), sp)
            if ego is not None:
                self.episode_actors.append(ego)
                return ego, wp
        return None, None

    @staticmethod
    def side_lane(wp):
        # Traffic Manager lane-change direction: True = right, False = left.
        for lane, move_right in ((wp.get_left_lane(), True), (wp.get_right_lane(), False)):
            if lane is not None and lane.lane_type == carla.LaneType.Driving \
                    and lane.lane_id * wp.lane_id > 0:
                return lane, move_right
        return None

    def attach_sensors(self, ego):
        bb = ego.bounding_box
        roof = bb.location.z + bb.extent.z
        self.half_len = bb.extent.x
        self.lidar_z = roof + 0.3
        self.radar_z = 0.7
        self.cam_z = roof + 0.15
        self.cam_to_bumper = bb.extent.x - 0.3
        cam_tf = carla.Transform(carla.Location(x=bb.location.x + 0.3, z=self.cam_z))

        def make(name, tf, **attrs):
            bp = self.bp_lib.find(name)
            for k, v in attrs.items():
                bp.set_attribute(k, str(v))
            s = self.world.spawn_actor(bp, tf, attach_to=ego)
            self.sensors.append(s)
            return s

        rgb = make("sensor.camera.rgb", cam_tf,
                   image_size_x=WIDTH, image_size_y=HEIGHT, fov=90)
        depth = make("sensor.camera.depth", cam_tf,
                     image_size_x=DEPTH_W, image_size_y=DEPTH_H, fov=90)
        lidar = make("sensor.lidar.ray_cast",
                     carla.Transform(carla.Location(z=self.lidar_z)),
                     channels=32, range=70, points_per_second=300000,
                     rotation_frequency=1.0 / FIXED_DT, upper_fov=10, lower_fov=-30,
                     dropoff_general_rate=0.0)
        radar = make("sensor.other.radar",
                     carla.Transform(carla.Location(x=bb.location.x + bb.extent.x + 0.1,
                                                    z=self.radar_z)),
                     horizontal_fov=30, vertical_fov=10, range=100, points_per_second=1500)
        gnss = make("sensor.other.gnss", carla.Transform())
        imu = make("sensor.other.imu", carla.Transform())
        coll = make("sensor.other.collision", carla.Transform())

        self.queues = {}
        for name, s in (("rgb", rgb), ("depth", depth), ("lidar", lidar),
                        ("radar", radar), ("gnss", gnss), ("imu", imu)):
            q = queue.Queue()
            s.listen(q.put)
            self.queues[name] = q
        self.collisions = []
        coll.listen(self.collisions.append)

    def destroy_episode(self):
        for s in self.sensors:
            if s.is_alive:
                s.stop()
        self.client.apply_batch([carla.command.DestroyActor(a)
                                 for a in self.sensors + self.episode_actors])
        self.sensors, self.episode_actors = [], []

    # ---------- one simulation step

    def step(self, ego):
        frame = self.world.tick()
        data = {}
        for name, q in self.queues.items():
            item = q.get(timeout=10.0)
            while item.frame < frame:
                item = q.get(timeout=10.0)
            data[name] = item

        rgb = np.frombuffer(data["rgb"].raw_data, dtype=np.uint8).reshape(
            (HEIGHT, WIDTH, 4))[:, :, :3].copy()
        ego_tf = ego.get_transform()
        vel = ego.get_velocity()
        ctrl = ego.get_control()
        imu, gnss = data["imu"], data["gnss"]
        radar_n, radar_d, radar_c, radar_ttc = radar_summary(data["radar"], self.radar_z)
        lidar_gap, lidar_front, lidar_n = lidar_summary(data["lidar"], self.half_len, self.lidar_z)
        depth_gap, depth_px = depth_summary(data["depth"], self.cam_z, self.cam_to_bumper)

        # Ground truth: every nearby car or walker, in the ego frame.
        ego_v = np.array([vel.x, vel.y])
        yaw = math.radians(ego_tf.rotation.yaw)
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        right = np.array([-math.sin(yaw), math.cos(yaw)])
        actors, gt_gap, gt_closing = [], NO_VALUE, 0.0
        snapshot = self.world.get_snapshot()
        for aid, (kind, ext) in self.tracked.items():
            if aid == ego.id:
                continue
            snap = snapshot.find(aid)
            if snap is None:
                continue
            loc = snap.get_transform().location
            fx, ry = to_ego_frame(ego_tf, loc)
            dist = math.hypot(fx, ry)
            if dist > GT_RADIUS:
                continue
            v = snap.get_velocity()
            rel_v = np.array([v.x, v.y]) - ego_v
            rvx, rvy = float(rel_v @ fwd), float(rel_v @ right)
            actors.append((aid, kind, round(fx, 2), round(ry, 2), round(rvx, 2), round(rvy, 2)))
            if fx > 0 and abs(ry) < 1.0 + ext:
                gap = fx - self.half_len - ext
                if gap < gt_gap:
                    gt_gap, gt_closing = max(gap, 0.0), max(-rvx, 0.0)
        gt_ttc = gt_gap / gt_closing if gt_closing > 0.5 else NO_VALUE

        row = {
            "sim_time": round(data["imu"].timestamp, 3),
            "speed_mps": round(math.hypot(vel.x, vel.y), 3),
            "accel_x": round(imu.accelerometer.x, 3), "accel_y": round(imu.accelerometer.y, 3),
            "gyro_z": round(imu.gyroscope.z, 4),
            "lat": gnss.latitude, "lon": gnss.longitude,
            "throttle": round(ctrl.throttle, 3), "steer": round(ctrl.steer, 3),
            "brake": round(ctrl.brake, 3),
            "radar_n": radar_n, "radar_min_depth": round(radar_d, 3),
            "radar_closing_speed": round(radar_c, 3), "radar_min_ttc": round(radar_ttc, 3),
            "lidar_front_gap": round(lidar_gap, 3), "lidar_front_points": lidar_front,
            "lidar_points": lidar_n,
            "depth_front_gap": round(depth_gap, 3), "depth_front_pixels": depth_px,
            "gt_front_gap": round(min(gt_gap, NO_VALUE), 3),
            "gt_closing_speed": round(gt_closing, 3), "gt_ttc": round(min(gt_ttc, NO_VALUE), 3),
            "gt_partner_dist": NO_VALUE,  # filled in by save() once the partner is known
        }
        lidar_pts = None
        if self.save_lidar:
            lidar_pts = np.frombuffer(data["lidar"].raw_data, dtype=np.float32) \
                .reshape(-1, 4).astype(np.float16)
        return frame, {"rgb": rgb, "row": row, "actors": actors, "lidar": lidar_pts}

    def track(self, actors):
        """Remember the type and half-width of every actor for ground truth."""
        # No is_alive check: actors spawned this tick report False until the next tick.
        self.tracked = {}
        for a in actors:
            if a is None:
                continue
            kind = "walker" if a.type_id.startswith("walker") else "vehicle"
            self.tracked[a.id] = (kind, a.bounding_box.extent.y)

    # ---------- episodes

    def warm_up(self, ego, ticks):
        for _ in range(ticks):
            self.step(ego)

    def run_crash(self, scenario, crash_index):
        ego, wp = self.spawn_ego(scenario)
        if ego is None:
            return None
        self.attach_sensors(ego)
        port = self.tm.get_port()
        ego.set_autopilot(True, port)
        self.tm.auto_lane_change(ego, False)
        if scenario != "pedestrian":
            # the pedestrian scenario must hit the walker, not a car in front
            self.tm.ignore_vehicles_percentage(ego, 100)
            self.tm.distance_to_leading_vehicle(ego, 0.0)

        other = None
        trigger_tick = random.randint(30, 50)
        if scenario == "rear_end":
            ahead = wp.next(random.uniform(20.0, 35.0))[0].transform
            ahead.location.z += 0.3
            other = self.world.try_spawn_actor(random.choice(self.vehicle_bps), ahead)
            self.tm.vehicle_percentage_speed_difference(ego, -15)
        elif scenario == "cut_in":
            lane, move_right = self.side_lane(wp)
            beside = lane.next(random.uniform(6.0, 12.0))[0].transform
            beside.location.z += 0.3
            other = self.world.try_spawn_actor(random.choice(self.vehicle_bps), beside)
            self.tm.vehicle_percentage_speed_difference(ego, -20)
        elif scenario == "junction":
            self.tm.ignore_lights_percentage(ego, 100)
            self.tm.ignore_signs_percentage(ego, 100)
            self.tm.vehicle_percentage_speed_difference(ego, -40)
        elif scenario == "pedestrian":
            target = wp.next(random.uniform(35.0, 50.0))[0].transform
            side = target.get_right_vector()
            offset = wp.lane_width / 2 + 1.5
            loc = target.location + carla.Location(x=side.x * offset, y=side.y * offset, z=1.0)
            walker_bp = random.choice(self.bp_lib.filter("walker.pedestrian.*"))
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            other = self.world.try_spawn_actor(walker_bp, carla.Transform(loc))
            self.tm.ignore_walkers_percentage(ego, 100)
            walk_speed = random.uniform(2.0, 3.5)
            # start early/late by up to a few tenths of a second so impacts vary
            timing_jitter = random.uniform(-0.3, 0.2)
            walking = False

        if scenario != "junction":
            if other is None:
                return None
            self.episode_actors.append(other)
            if other.type_id.startswith("vehicle"):
                other.set_autopilot(True, port)
                self.tm.auto_lane_change(other, False)
                self.tm.vehicle_percentage_speed_difference(other, 30 if scenario == "cut_in" else 0)

        self.track([ego] + self.background + self.episode_actors)
        buffer = collections.deque(maxlen=CLIP_FRAMES + 10)
        crash_frame, crash_event, tick = None, None, 0
        while tick < MAX_EPISODE_TICKS:
            tick += 1
            if scenario == "rear_end" and tick == trigger_tick:
                other.set_autopilot(False, port)
                other.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            elif scenario == "cut_in" and tick == trigger_tick:
                self.tm.force_lane_change(other, move_right)
            elif scenario == "pedestrian" and tick > 5 and not walking:
                # Start walking so the walker reaches the lane centre when the car arrives.
                ahead, _ = to_ego_frame(ego.get_transform(), other.get_location())
                speed = ego.get_velocity().length()
                car_eta = (ahead - self.half_len) / max(speed, 0.1)
                walker_eta = offset / walk_speed + 0.3  # +0.3 s to get going
                if speed > 3.0 and car_eta <= walker_eta + timing_jitter:
                    walking = True
                    other.apply_control(carla.WalkerControl(
                        direction=carla.Vector3D(-side.x, -side.y, 0.0), speed=walk_speed))

            buffer.append(self.step(ego))

            if crash_frame is None:
                hits = [e for e in self.collisions if tick > 10]
                if hits:
                    crash_event = hits[0]
                    crash_frame = crash_event.frame
                    impact_speed = buffer[-2][1]["row"]["speed_mps"] if len(buffer) > 1 else 0.0
                    self.track([ego] + self.background + self.episode_actors
                               + [crash_event.other_actor])
                    post_needed = CLIP_FRAMES - crash_index - 1
                    for _ in range(post_needed):
                        buffer.append(self.step(ego))
                    break
                self.collisions.clear()

        if crash_frame is None:
            return None
        frames = [f for f, _ in buffer]
        if crash_frame not in frames:
            crash_frame = min(frames, key=lambda f: abs(f - crash_frame))
        start = frames.index(crash_frame) - crash_index
        if start < 0 or start + CLIP_FRAMES > len(buffer):
            return None  # crash came too early to have enough history
        clip = [item for _, item in list(buffer)[start:start + CLIP_FRAMES]]

        other_type = crash_event.other_actor.type_id
        imp = crash_event.normal_impulse
        info = {
            "partner_type": ("walker" if other_type.startswith("walker") else
                             "vehicle" if other_type.startswith("vehicle") else "static"),
            "partner_blueprint": other_type,
            "impact_speed_kmh": round(impact_speed * 3.6, 1),
            "intensity": round(math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2), 1),
            "ego_blueprint": ego.type_id,
            "partner_id": crash_event.other_actor.id,
        }
        return clip, info

    def run_normal(self):
        ego, _ = self.spawn_ego("normal")
        if ego is None:
            return None
        self.attach_sensors(ego)
        ego.set_autopilot(True, self.tm.get_port())
        self.tm.distance_to_leading_vehicle(ego, 4.0)
        self.track([ego] + self.background)
        self.warm_up(ego, random.randint(30, 80))
        for _ in range(4):
            self.collisions.clear()
            clip = [self.step(ego)[1] for _ in range(CLIP_FRAMES)]
            mean_speed = np.mean([c["row"]["speed_mps"] for c in clip])
            if not self.collisions and mean_speed > 2.0:
                return clip, {"partner_type": "", "impact_speed_kmh": "", "intensity": "",
                              "ego_blueprint": ego.type_id}
        return None

    # ---------- saving

    def save(self, clip, info, town, scenario, label, crash_index, weather_name, seed):
        self.saved += 1
        clip_id = f"{town}_{scenario}_{self.saved:05d}"
        daynight = "Night" if "Night" in weather_name else "Day"
        condition = "Rainy" if ("Rain" in weather_name or "Wet" in weather_name) else "Normal"
        partner_id = info.pop("partner_id", None)
        for i, c in enumerate(clip):
            c["row"]["frame_idx"] = i
            c["row"]["label"] = int(label == 1 and i >= crash_index)
            c["row"]["collision"] = int(label == 1 and i == crash_index)
            for a in c["actors"]:
                if a[0] == partner_id:
                    c["row"]["gt_partner_dist"] = round(math.hypot(a[2], a[3]), 3)
        meta = dict(info, clip_id=clip_id, town=town, scenario=scenario, label=label,
                    crash_index=crash_index if label else -1, weather=weather_name,
                    daynight=daynight, condition=condition, fps=round(1 / FIXED_DT),
                    frames=CLIP_FRAMES, seed=seed)
        self.write_queue.put((clip_id, clip, meta))
        return clip_id

    def _writer_loop(self):
        while True:
            job = self.write_queue.get()
            if job is None:
                break
            clip_id, clip, meta = job
            d = os.path.join(self.out_dir, "clips", clip_id)
            os.makedirs(d)
            video = cv2.VideoWriter(os.path.join(d, "video.mp4"),
                                    cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (WIDTH, HEIGHT))
            for c in clip:
                video.write(c["rgb"])
            video.release()
            with open(os.path.join(d, "sensors.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=SENSOR_COLUMNS)
                w.writeheader()
                for c in clip:
                    w.writerow(c["row"])
            with open(os.path.join(d, "actors.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame_idx", "actor_id", "kind", "rel_x", "rel_y", "rel_vx", "rel_vy"])
                for i, c in enumerate(clip):
                    for a in c["actors"]:
                        w.writerow((i,) + a)
            if clip[0]["lidar"] is not None:
                np.savez_compressed(os.path.join(d, "lidar.npz"),
                                    **{f"f{i:02d}": c["lidar"] for i, c in enumerate(clip)})
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            with open(self.index_path, "a", newline="") as f:
                csv.writer(f).writerow([meta[k] for k in (
                    "clip_id", "town", "scenario", "label", "crash_index", "weather",
                    "daynight", "condition", "partner_type", "impact_speed_kmh",
                    "intensity", "seed")])

    def close(self):
        self.write_queue.put(None)
        self.writer.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--towns", nargs="+", default=TOWNS)
    parser.add_argument("--crashes-per-town", type=int, default=200)
    parser.add_argument("--normals-per-town", type=int, default=200)
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-lidar", action="store_true",
                        help="also store full LiDAR point clouds (about 15 MB per clip)")
    parser.add_argument("--tm-port", type=int, default=8000,
                        help="use another port if a crashed run left a Traffic Manager behind")
    args = parser.parse_args()

    free_gb = shutil.disk_usage("C:\\").free / 1e9
    if free_gb < MIN_FREE_GB:
        print(f"Only {free_gb:.1f} GB free on C:. Need at least {MIN_FREE_GB} GB.")
        raise SystemExit
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = os.path.join(OUT_ROOT, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(out_dir, "clips"))
    with open(os.path.join(out_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Output: {out_dir}")

    client = connect_or_start_carla()
    client.set_timeout(60.0)
    gen = CrashGenerator(client, out_dir, args.save_lidar, args.tm_port)
    start = time.time()
    try:
        for t_i, town in enumerate(args.towns):
            gen.load_town(town, args.seed + t_i)
            crashes = normals = attempts = episodes = 0
            while crashes < args.crashes_per_town or normals < args.normals_per_town:
                episodes += 1
                if episodes % RESPAWN_TRAFFIC_EVERY == 0:
                    gen.spawn_traffic()
                weather_name = random.choice(list(WEATHERS))
                gen.world.set_weather(WEATHERS[weather_name])
                seed = random.randrange(1 << 30)
                random.seed(seed)
                want_crash = crashes < args.crashes_per_town and (
                    normals >= args.normals_per_town or random.random() < 0.5)
                try:
                    if want_crash:
                        attempts += 1
                        scenario = random.choice(args.scenarios)
                        crash_index = random.randint(*CRASH_INDEX_RANGE)
                        result = gen.run_crash(scenario, crash_index)
                        if result:
                            gen.save(*result, town, scenario, 1, crash_index, weather_name, seed)
                            crashes += 1
                    else:
                        result = gen.run_normal()
                        if result:
                            gen.save(*result, town, "normal", 0, -1, weather_name, seed)
                            normals += 1
                except RuntimeError as e:
                    print(f"  episode failed: {e}")
                finally:
                    gen.destroy_episode()
                    for _ in range(3):
                        gen.world.tick()
                mins = (time.time() - start) / 60
                print(f"  {town}: crashes {crashes}/{args.crashes_per_town} "
                      f"(attempts {attempts}) | normal {normals}/{args.normals_per_town} "
                      f"| {mins:.1f} min")
            gen.destroy_traffic()
    finally:
        print("\nRestoring CARLA settings and finishing writes...")
        gen.restore()
        gen.close()
        print(f"Saved {gen.saved} clips to: {out_dir}")


if __name__ == "__main__":
    main()
