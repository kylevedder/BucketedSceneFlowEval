BACKGROUND_CATEGORIES = ["background"]

# These catagories are ignored because of labeling oddities
STATIC_OBJECTS = [
    "movable_object.barrier",
    "movable_object.debris",
    "movable_object.pushable_pullable",
    "movable_object.trafficcone",
    "static_object.bicycle_rack",
]

PEDESTRIAN_CATEGORIES = [
    "animal",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
]

WHEELED_VRU = ["vehicle.bicycle", "vehicle.motorcycle"]

CAR = ["vehicle.car"]

OTHER_VEHICLES = [
    "vehicle.bus.bendy",
    "vehicle.bus.rigid",
    "vehicle.construction",
    "vehicle.emergency.ambulance",
    "vehicle.emergency.police",
    "vehicle.trailer",
    "vehicle.truck",
]

BUCKETED_METACATAGORIES = {
    "BACKGROUND": BACKGROUND_CATEGORIES,
    "CAR": CAR,
    "PEDESTRIAN": PEDESTRIAN_CATEGORIES,
    "WHEELED_VRU": WHEELED_VRU,
    "OTHER_VEHICLES": OTHER_VEHICLES,
}

THREEWAY_EPE_METACATAGORIES = {
    "BACKGROUND": BACKGROUND_CATEGORIES,
    "FOREGROUND": PEDESTRIAN_CATEGORIES + WHEELED_VRU + CAR + OTHER_VEHICLES,
}
