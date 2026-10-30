"""Near-miss families N1-N4: the crash set-ups plus ego emergency braking (EmergencyBrake).
Whether the result is really a near-miss is measured afterwards by labels.py."""
from scenarios.base import EmergencyBrake
from scenarios.crash import CutIn, Junction, RearEnd, VRUCrossing


class _WithBrake:
    brake_ttc = (1.0, 1.8)      # predicted TTC at which braking is armed (s)

    def setup(self):
        ego = super().setup()
        self.brake = EmergencyBrake(self.rng, ttc_range=self.brake_ttc)
        self.params.update(self.brake.info())
        return ego


class RearEndAvoided(_WithBrake, RearEnd):
    family, intended = "N1_rear_end_avoided", "near_miss"


class CutInAvoided(_WithBrake, CutIn):
    family, intended = "N2_cut_in_avoided", "near_miss"
    brake_ttc = (0.5, 1.1)      # smoke test: 1.55 s stopped 4 m short (too early)


class JunctionConflict(_WithBrake, Junction):
    family, intended = "N3_junction_conflict", "near_miss"
    brake_ttc = (0.6, 1.2)


class VRUConflict(_WithBrake, VRUCrossing):
    family, intended = "N4_vru_conflict", "near_miss"
