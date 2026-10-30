"""Synchronous sensor rig: spawns the sensors from configs/sensors.yaml and, at every tick,
waits until each stream has delivered the measurement for exactly that world frame."""
import queue

import carla
import numpy as np

from io_utils import depth_bgra_to_metres, metres_to_cm_u16

STREAMS = ("rgb", "instance", "depth", "lidar", "radar", "imu", "gnss")


class SensorTimeout(RuntimeError):
    pass


class Rig:
    def __init__(self, world, ego, cfg, depth_res=None, lidar_pps=None):
        self.world, self.ego, self.cfg = world, ego, cfg
        self.sensors, self.queues, self.collisions = {}, {}, []
        bb = ego.bounding_box
        roof = bb.location.z + bb.extent.z
        self.mount = {
            "roof_z": roof, "bb_ex": bb.extent.x, "bb_ey": bb.extent.y, "bb_ez": bb.extent.z,
            "bb_cx": bb.location.x, "bb_cy": bb.location.y, "bb_cz": bb.location.z,
        }
        lib = world.get_blueprint_library()
        r = cfg["rgb"]
        cam_tf = carla.Transform(carla.Location(x=bb.location.x + r["mount"]["x_from_bb_center"],
                                                z=roof + r["mount"]["z_above_roof"]))
        d = dict(cfg["depth"])
        if depth_res:
            d["width"], d["height"] = depth_res
        li = cfg["lidar"]
        ra = cfg["radar"]
        specs = {
            "rgb": (r["blueprint"], cam_tf, {"image_size_x": r["width"], "image_size_y": r["height"],
                                             "fov": r["fov"]}),
            "instance": (cfg["instance"]["blueprint"], cam_tf,
                         {"image_size_x": r["width"], "image_size_y": r["height"], "fov": r["fov"]}),
            "depth": (d["blueprint"], cam_tf, {"image_size_x": d["width"], "image_size_y": d["height"],
                                               "fov": d["fov"]}),
            "lidar": (li["blueprint"],
                      carla.Transform(carla.Location(x=li["mount"]["x"], z=roof + li["mount"]["z_above_roof"])),
                      {k: li[k] for k in ("channels", "range", "rotation_frequency", "upper_fov",
                                          "lower_fov", "dropoff_general_rate")}
                      | {"points_per_second": lidar_pps or li["points_per_second"]}),
            "radar": (ra["blueprint"],
                      carla.Transform(carla.Location(x=bb.location.x + bb.extent.x + ra["mount"]["x_beyond_front"],
                                                     z=ra["mount"]["z"])),
                      {k: ra[k] for k in ("horizontal_fov", "vertical_fov", "range", "points_per_second")}),
            "imu": (cfg["imu"]["blueprint"], carla.Transform(), {}),
            "gnss": (cfg["gnss"]["blueprint"], carla.Transform(), {}),
        }
        self.config_used = {}
        for name, (bp_id, tf, attrs) in specs.items():
            bp = lib.find(bp_id)
            for k, v in attrs.items():
                bp.set_attribute(k, str(v))
            s = world.spawn_actor(bp, tf, attach_to=ego)
            q = queue.Queue()
            s.listen(q.put)
            self.sensors[name], self.queues[name] = s, q
            self.config_used[name] = {"blueprint": bp_id, "attributes": {k: str(v) for k, v in attrs.items()},
                                      "location": [tf.location.x, tf.location.y, tf.location.z]}
        self.radar_z = ra["mount"]["z"]
        self.depth_size = (d["width"], d["height"])
        coll = world.spawn_actor(lib.find(cfg["collision"]["blueprint"]), carla.Transform(), attach_to=ego)
        coll.listen(self._on_collision)
        self.sensors["collision"] = coll
        self.timeout = cfg.get("sensor_timeout_s", 10.0)

    def _on_collision(self, e):
        imp = e.normal_impulse
        self.collisions.append({
            "world_frame": e.frame, "sim_time": e.timestamp, "other_id": e.other_actor.id,
            "other_blueprint": e.other_actor.type_id,
            "impulse": (imp.x, imp.y, imp.z),
            "intensity": float(np.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2)),
        })

    def read(self, frame):
        """Collect every stream's measurement for this world frame (older ones are dropped)."""
        out = {}
        for name in STREAMS:
            q = self.queues[name]
            try:
                m = q.get(timeout=self.timeout)
                while m.frame < frame:
                    m = q.get(timeout=self.timeout)
            except queue.Empty:
                raise SensorTimeout(name)
            out[name] = m
        return out

    def decode(self, meas):
        """Raw measurements -> numpy (copied out of CARLA buffers)."""
        rgb = meas["rgb"]
        bgr = np.frombuffer(rgb.raw_data, np.uint8).reshape(rgb.height, rgb.width, 4)[:, :, :3].copy()
        inst = np.frombuffer(meas["instance"].raw_data, np.uint8).reshape(
            meas["instance"].height, meas["instance"].width, 4).copy()
        dp = meas["depth"]
        depth_cm = metres_to_cm_u16(depth_bgra_to_metres(
            np.frombuffer(dp.raw_data, np.uint8).reshape(dp.height, dp.width, 4)))
        lidar = np.frombuffer(meas["lidar"].raw_data, np.float32).reshape(-1, 4).astype(np.float16)
        radar = np.frombuffer(meas["radar"].raw_data, np.float32).reshape(-1, 4).copy()
        imu, gnss = meas["imu"], meas["gnss"]
        return {
            "rgb": bgr, "instance": inst, "depth_cm": depth_cm, "lidar": lidar, "radar": radar,
            "imu": {"acc_x": imu.accelerometer.x, "acc_y": imu.accelerometer.y, "acc_z": imu.accelerometer.z,
                    "gyro_x": imu.gyroscope.x, "gyro_y": imu.gyroscope.y, "gyro_z": imu.gyroscope.z,
                    "compass": imu.compass},
            "gnss": {"lat": gnss.latitude, "lon": gnss.longitude, "alt": gnss.altitude},
            "stream_frames": {n: (meas[n].frame, meas[n].timestamp) for n in STREAMS},
        }

    def camera_world_matrix_inv(self, snapshot):
        snap = snapshot.find(self.sensors["rgb"].id)
        tf = snap.get_transform() if snap is not None else self.sensors["rgb"].get_transform()
        return np.array(tf.get_inverse_matrix())

    def destroy(self, client):
        for s in self.sensors.values():
            try:
                if s.is_listening:
                    s.stop()
            except RuntimeError:
                pass
        client.apply_batch_sync([carla.command.DestroyActor(s) for s in self.sensors.values()])
        self.sensors = {}
