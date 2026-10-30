"""Unit tests U1-U7, U9 + storage guard + no-overwrite (docs/VALIDATION_PLAN.md). No CARLA needed.
Run: .venv\\Scripts\\python.exe -m unittest discover crashtwin/tests"""
import math
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

from fixtures import make_clip  # also puts src/generator and src/validation on sys.path
from geometry import box_corners, footprint_gap, predicted_ttc, to_ego_frame
from io_utils import (StorageGuard, attempt_seed, depth_bgra_to_metres, make_new_dir, metres_to_cm_u16,
                      pack_ragged, unpack_ragged)
from labels import compute_outcome
from recorder import window_indices
from validate_clip import validate


def ego_track(n=100, x0=0.0, v=8.0):
    x = x0 + np.arange(n) * v * 0.1
    return pd.DataFrame({"frame_idx": np.arange(n), "x": x, "y": 0.0, "yaw": 0.0, "vx": v, "vy": 0.0,
                         "bb_ex": 2.3, "bb_ey": 1.0, "bb_cx": 0.0, "bb_cy": 0.0}).set_index("frame_idx", drop=False)


def actor_track(xs, ys, vx, vy, aid=7, yaw=0.0):
    n = len(xs)
    return pd.DataFrame({"frame_idx": np.arange(n), "actor_id": aid, "type": "car", "x": xs, "y": ys,
                         "yaw": yaw, "vx": vx, "vy": vy, "bb_ex": 2.0, "bb_ey": 0.9, "bb_cx": 0.0, "bb_cy": 0.0})


class TestGeometry(unittest.TestCase):          # U1, U2
    def test_gap_overlap_is_zero(self):
        a = box_corners(0, 0, 0, 2, 1)
        self.assertEqual(footprint_gap(a, box_corners(1, 0.5, 30, 2, 1)), 0.0)

    def test_gap_parallel(self):
        a = box_corners(0, 0, 0, 2, 1)
        b = box_corners(0, 4, 0, 2, 1)       # edges 2 m apart
        self.assertAlmostEqual(footprint_gap(a, b), 2.0, places=6)

    def test_gap_rotated(self):
        a = box_corners(0, 0, 0, 1, 1)
        b = box_corners(4, 0, 45, 1, 1)      # diamond: nearest vertex at x = 4 - sqrt(2)
        self.assertAlmostEqual(footprint_gap(a, b), 4 - math.sqrt(2) - 1, places=6)

    def test_ttc_closing(self):
        a, b = box_corners(0, 0, 0, 2, 1), box_corners(24, 0, 0, 2, 1)   # gap 20 m
        self.assertAlmostEqual(predicted_ttc(a, (10, 0), b, (0, 0)), 2.0, places=2)

    def test_ttc_opening_and_slow(self):
        a, b = box_corners(0, 0, 0, 2, 1), box_corners(24, 0, 0, 2, 1)
        self.assertTrue(math.isinf(predicted_ttc(a, (0, 0), b, (5, 0))))
        self.assertTrue(math.isinf(predicted_ttc(a, (0.4, 0), b, (0, 0))))   # meets after the horizon

    def test_ttc_passing_side_by_side_is_inf(self):
        # oncoming car in the next lane (1.5 m lateral gap): closing fast but never overlapping
        a, b = box_corners(0, 0, 0, 2.3, 1), box_corners(10, 3.5, 180, 2.3, 1)
        self.assertTrue(math.isinf(predicted_ttc(a, (8, 0), b, (-8, 0))))

    def test_to_ego_frame(self):
        fx, ry = to_ego_frame(0, 0, 90, 0, 10)   # yaw 90: forward = +y
        self.assertAlmostEqual(float(fx), 10)
        self.assertAlmostEqual(float(ry), 0, places=6)


class TestLabels(unittest.TestCase):            # U3
    def test_crash(self):
        ego = ego_track()
        a = actor_track(ego["x"].to_numpy() + 30, np.zeros(100), 8, 0)
        lab, _, _ = compute_outcome(ego, a, [{"frame_idx": 70, "other_id": 7, "other_blueprint": "vehicle.x",
                                              "other_type": "car", "intensity": 3000.0}])
        self.assertEqual(lab["outcome"], "crash")
        self.assertEqual(lab["event_frame"], 70)

    def test_near_miss(self):
        # 8 m/s until frame 70, then 8 m/s^2 braking to a stop 1.0 m behind a stopped car
        n, v, x, xs, vs = 100, 8.0, 0.0, [], []
        for i in range(n):
            xs.append(x)
            vs.append(v)
            if i >= 70:
                v = max(v - 0.8, 0.0)
            x += v * 0.1
        stop_front = xs[-1] + 2.3
        lead_x = stop_front + 1.0 + 2.0          # lead centre: rear edge 1.0 m ahead of the ego's front
        ego = pd.DataFrame({"frame_idx": np.arange(n), "x": xs, "y": 0.0, "yaw": 0.0, "vx": vs, "vy": 0.0,
                            "bb_ex": 2.3, "bb_ey": 1.0, "bb_cx": 0.0, "bb_cy": 0.0}).set_index("frame_idx", drop=False)
        lab, _, _ = compute_outcome(ego, actor_track(np.full(n, lead_x), np.zeros(n), 0.0, 0.0), [])
        self.assertEqual(lab["outcome"], "near_miss", lab)
        self.assertAlmostEqual(lab["min_gap_m"], 1.0, places=1)
        self.assertLessEqual(lab["min_ttc_s"], 1.5)

    def test_normal_and_borderline(self):
        ego = ego_track()
        far = actor_track(ego["x"].to_numpy() + 30, np.zeros(100), 8, 0)     # same speed, never closing
        self.assertEqual(compute_outcome(ego, far, [])[0]["outcome"], "normal")
        # approaching a stopped car, ending 15 m short: TTC < 3 s but gap > 1.5 m -> borderline
        stop = actor_track(np.full(100, ego["x"].iloc[-1] + 4.3 + 15), np.zeros(100), 0, 0)
        lab = compute_outcome(ego, stop, [])[0]
        self.assertEqual(lab["outcome"], "invalid")
        self.assertEqual(lab["invalid_reason"], "borderline_conflict")

    def test_oncoming_traffic_is_normal(self):
        ego = ego_track()
        on = actor_track(ego["x"].to_numpy()[::-1] + 20, np.full(100, 3.5), -8, 0, yaw=180)
        self.assertEqual(compute_outcome(ego, on, [])[0]["outcome"], "normal")


class TestIO(unittest.TestCase):                # U4, U5, U6, U7
    def test_depth_roundtrip(self):
        m = np.array([[0.5, 12.34, 300.0, 999.0]])
        cm = metres_to_cm_u16(m)
        self.assertTrue(np.all(np.abs(cm[0, :3] / 100.0 - m[0, :3]) <= 0.005))
        self.assertEqual(cm[0, 3], 65535)

    def test_depth_decode(self):
        bgra = np.zeros((1, 1, 4), np.uint8)
        bgra[0, 0, 2] = 255   # R only -> 255 / (256^3 - 1) * 1000 m
        self.assertAlmostEqual(depth_bgra_to_metres(bgra)[0, 0], 255 / 16777215 * 1000, places=9)

    def test_ragged(self):
        arrs = [np.random.rand(k, 4).astype(np.float32) for k in (3, 0, 5)]
        data, off = pack_ragged(arrs, np.float32, 4)
        for a, b in zip(arrs, unpack_ragged(data, off)):
            np.testing.assert_array_equal(a, b)

    def test_window(self):
        frames = list(range(500, 700))
        self.assertEqual(window_indices(frames, 600, 75), 25)
        self.assertIsNone(window_indices(frames, 520, 75))   # not enough history
        self.assertIsNone(window_indices(frames, 690, 60))   # not enough future

    def test_seed(self):
        self.assertEqual(attempt_seed("1.0", "C1_s00001", 1), attempt_seed("1.0", "C1_s00001", 1))
        self.assertNotEqual(attempt_seed("1.0", "C1_s00001", 1), attempt_seed("1.0", "C1_s00001", 2))


class TestGuards(unittest.TestCase):
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p = make_new_dir(Path(t) / "clip")
            (p / "keep.txt").write_text("original")
            with self.assertRaises(FileExistsError):
                make_new_dir(p)
            self.assertEqual((p / "keep.txt").read_text(), "original")

    def test_storage_guard(self):
        Usage = namedtuple("Usage", "total used free")
        with tempfile.TemporaryDirectory() as t:
            g = StorageGuard(t, cap_gb=1.0, min_free_gb=30, reserve_gb=0.15,
                             disk_usage=lambda p: Usage(0, 0, 100e9))
            self.assertIsNone(g.check())
            g.add(0.9e9)
            self.assertIn("run_cap", g.check())
            g2 = StorageGuard(t, cap_gb=40, min_free_gb=30, reserve_gb=0.15,
                              disk_usage=lambda p: Usage(0, 0, 30.1e9))
            self.assertIn("min_free", g2.check())


class TestValidator(unittest.TestCase):         # U9
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_good_clip_passes(self):
        res = validate(make_clip(self.root / "good"))
        self.assertEqual(res["status"], "PASS", res)

    def test_missing_file_fails(self):
        res = validate(make_clip(self.root / "missing", drop="depth"))
        self.assertEqual(res["status"], "FAIL")
        self.assertIn("F1", res["failed"])

    def test_duplicate_frame_fails(self):
        res = validate(make_clip(self.root / "dup", dup_frame=True))
        self.assertIn("F3", res["failed"])

    def test_time_gap_fails(self):
        res = validate(make_clip(self.root / "gap", time_gap=True))
        self.assertIn("F4", res["failed"])

    def test_crash_label_without_collision_fails(self):
        res = validate(make_clip(self.root / "nocoll", outcome="crash", intended="crash"))
        self.assertIn("L1", res["failed"])

    def test_outcome_mismatch_warns(self):
        res = validate(make_clip(self.root / "spill", intended="crash"))
        self.assertEqual(res["status"], "WARN")
        self.assertIn("L6", res["warned"])


if __name__ == "__main__":
    unittest.main()
