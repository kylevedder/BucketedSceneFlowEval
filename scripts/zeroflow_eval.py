# load the result pickle
import pickle

from bucketed_scene_flow_eval.eval import BucketedEPEEvaluator

# read "eval_frame_results.pkl"


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


BACKGROUND_CATEGORIES = ["-1BACKGROUND"]

STUFF_CATEGORIES = [
    "BOLLARD",
    "CONSTRUCTION_BARREL",
    "CONSTRUCTION_CONE",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "SIGN",
    "STOP_SIGN",
]

PEDESTRIAN_CATEGORIES = ["PEDESTRIAN", "STROLLER", "WHEELCHAIR", "OFFICIAL_SIGNALER"]
SMALL_VEHICLE_CATEGORIES = [
    "BICYCLE",
    "BICYCLIST",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
]
VEHICLE_CATEGORIES = [
    "ARTICULATED_BUS",
    "BOX_TRUCK",
    "BUS",
    "LARGE_VEHICLE",
    "RAILED_VEHICLE",
    "REGULAR_VEHICLE",
    "SCHOOL_BUS",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "TRAFFIC_LIGHT_TRAILER",
    "MESSAGE_BOARD_TRAILER",
]
ANIMAL_CATEGORIES = ["ANIMAL", "DOG"]

METACATAGORIES = {
    "BACKGROUND": BACKGROUND_CATEGORIES,
    "STUFF": STUFF_CATEGORIES,
    "PEDESTRIAN": PEDESTRIAN_CATEGORIES,
    "SMALL_MOVERS": SMALL_VEHICLE_CATEGORIES,
    "LARGE_MOVERS": VEHICLE_CATEGORIES,
}

evaluator = BucketedEPEEvaluator(meta_class_lookup=METACATAGORIES)
eval_frame_results = load_pickle("/tmp/frame_results/bucketed_epe/eval_frame_results.pkl")
evaluator.eval_frame_results = eval_frame_results
evaluator.compute_results(save_results=False)
