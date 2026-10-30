from scenarios.crash import CutIn, Junction, RearEnd, SingleVehicle, VRUCrossing
from scenarios.near_miss import CutInAvoided, JunctionConflict, RearEndAvoided, VRUConflict
from scenarios.normal import CarFollowing, JunctionApproach, LaneChange, VRUNearby

FAMILIES = {cls.family: cls for cls in (
    RearEnd, CutIn, Junction, VRUCrossing, SingleVehicle,
    RearEndAvoided, CutInAvoided, JunctionConflict, VRUConflict,
    CarFollowing, JunctionApproach, LaneChange, VRUNearby)}
