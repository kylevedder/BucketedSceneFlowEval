"""A file for generating flow from detection + tracker results"""

import os

os.environ["OMP_NUM_THREADS"] = "12"
import multiprocessing
import os
import pickle
from argparse import ArgumentParser
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.se3 import SE3
from av2.structures.cuboid import Cuboid
from av2.structures.sweep import Sweep
from tqdm import tqdm

from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import (
    CATEGORY_MAP_INV,
)
from bucketed_scene_flow_eval.datasets.argoverse2.av2_metacategories import *
from bucketed_scene_flow_eval.utils.loaders import save_feather

# TRACKER_RESULTS_PATH = "/efs/lt3d_weights/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_50m_wide_hierarchy_tta_20e_av2/val_tracker_at3dmot_results/track_predictions.pkl"
TRACKER_RESULTS_PATH = "/efs/bevfusion_av2_leaderboard_submissions/bevfusion_track_predictions.pkl"
# TRACKER_RESULTS_PATH = "/efs/le3de2e_av2_leaderboard_submissions/le3de2e_track_predictions.pkl"

"""
Track data is a dict with the following structure:
{
      <log_id>: [
            {
                  "timestamp_ns": <timestamp_ns>,
                  "track_id": <track_id>
                  "score": <score>,
                  "label": <label>,
                  "name": <name>,
                  "translation_m": <translation_m>,
                  "size": <size>,
                  "yaw": <yaw>,
                  "velocity_m_per_s": <velocity_m_per_s>,
            }
      ] ... # list of length number of timestamps in the log
}
"""


def convert_lt3d_track_info_to_av2_cuboid(
    track_info: dict[str, Any],
    score_mask: tuple[np.ndarray, ...] | np.ndarray,
    box_index: int,
    cuboid_dest_SE3_transform: SE3,
) -> Cuboid:
    numpy_cuboid = np.array(
        [
            track_info["translation_m"][score_mask][box_index, 0],
            track_info["translation_m"][score_mask][box_index, 1],
            track_info["translation_m"][score_mask][box_index, 2]
            + (0.5 * track_info["size"][score_mask][box_index, 2]),
            track_info["size"][score_mask][box_index, 0],
            track_info["size"][score_mask][box_index, 1],
            track_info["size"][score_mask][box_index, 2],
            np.cos(track_info["yaw"][score_mask][box_index] / 2),
            0,
            0,
            np.sin(track_info["yaw"][score_mask][box_index] / 2),
        ]
    )
    cuboid = Cuboid.from_numpy(
        numpy_cuboid,
        category=track_info["name"][score_mask][box_index],
        timestamp_ns=track_info["timestamp_ns"],
    )
    cuboid = cuboid.transform(cuboid_dest_SE3_transform)
    return cuboid


def get_ids_and_cuboids_at_lidar_timestamps(
    track_data: dict,
    log_id: str,
    lidar_timestamps_ns: Sequence[int],
    ego_to_city_SE3s: Sequence[SE3],
    confidence_threshold: float,
) -> list[dict[str, Cuboid]]:
    """Load the sweep annotations at the provided timestamp with unique ids.
    Args:
        log_id: Log unique id.
        lidar_timestamp_ns: Nanosecond timestamp.
    Returns:
        dict mapping ids to cuboids
    """
    tracks_for_log = track_data[log_id]
    cuboids_and_ids_list = []
    for track_info in tracks_for_log:
        for pose_index, timestamp in enumerate(lidar_timestamps_ns):
            if track_info["timestamp_ns"] == timestamp:
                cuboid_id_dict = {}
                score_mask = np.nonzero(track_info["score"] > confidence_threshold)
                for box_index in range(track_info["track_id"][score_mask].shape[0]):
                    cuboid = convert_lt3d_track_info_to_av2_cuboid(
                        track_info, score_mask, box_index, ego_to_city_SE3s[pose_index].inverse()
                    )
                    cuboid_id_dict[track_info["track_id"][score_mask][box_index]] = cuboid
                cuboids_and_ids_list.append(cuboid_id_dict)
    return cuboids_and_ids_list


def compute_sceneflow(
    dataset: AV2SensorDataLoader,
    track_data: dict,
    log_id: str,
    timestamps: tuple[int, int],
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sceneflow between the sweeps at the given timestamps.
    Args:
      dataset: Sensor dataset.
      log_id: unique id.
      timestamps: the timestamps of the lidar sweeps to compute flow between
    Returns:
      dictionary with fields:
        pcl_0: Nx3 array containing the points at time 0
        pcl_1: Mx3 array containing the points at time 1
        flow_0_1: Nx3 array containing flow from timestamp 0 to 1
        classes_0: Nx1 array containing the class ids for each point in sweep 0
        valid_0: Nx1 array indicating if the returned flow from 0 to 1 is valid (1 for valid, 0 otherwise)
        ego_motion: SE3 motion from sweep 0 to sweep 1
    """

    def compute_flow(sweeps, cuboids, poses):
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0])
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)

        flow_0_1 = np.zeros_like(sweeps[0].xyz, dtype=np.float32)

        valid_0 = np.ones(len(sweeps[0].xyz), dtype=bool)
        classes_0 = np.ones(len(sweeps[0].xyz), dtype=np.int8) * CATEGORY_MAP_INV["BACKGROUND"]

        for id in cuboids[0]:
            c0 = cuboids[0][id]
            c0.length_m += (
                0.2  # the bounding boxes are a little too tight and some points are missed
            )
            c0.width_m += 0.2
            obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
            classes_0[obj_mask] = CATEGORY_MAP_INV[c0.category]

            min_clamp = 0.05 if c0.category in PEDESTRIAN_CATEGORIES else 0.2

            if id in cuboids[1]:
                c1 = cuboids[1][id]
                c1_SE3_c0_ego_frame = ego1_SE3_ego0.inverse().compose(
                    c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                )
                obj_flow = c1_SE3_c0_ego_frame.transform_point_cloud(obj_pts) - obj_pts
                if obj_flow.shape[0] > 0:
                    per_point_vel = np.linalg.norm(obj_flow, axis=1)
                    avg_obj_vel = np.mean(per_point_vel, axis=0)
                    # If the object is moving unreasonably fast, our tracker made a mistake, zero out the flow
                    if avg_obj_vel > 3.5 or avg_obj_vel < min_clamp:
                        # print(f"Object speed > 78mph, speed is: {avg_obj_vel}m/s")
                        obj_flow = np.zeros_like(obj_flow)
                flow_0_1[obj_mask] = obj_flow.astype(np.float32)
            else:
                valid_0[obj_mask] = 0
        return flow_0_1, classes_0, valid_0, ego1_SE3_ego0

    def zero_flow(sweeps, cuboids, poses):
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0])
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
        flow_0_1 = np.zeros_like(sweeps[0].xyz, dtype=np.float32)
        valid_0 = np.ones(len(sweeps[0].xyz), dtype=bool)
        classes_0 = np.ones(len(sweeps[0].xyz), dtype=np.int8) * CATEGORY_MAP_INV["BACKGROUND"]
        return flow_0_1, valid_0, classes_0, ego1_SE3_ego0

    sweeps = [Sweep.from_feather(dataset.get_lidar_fpath(log_id, ts)) for ts in timestamps]
    poses = [dataset.get_city_SE3_ego(log_id, ts) for ts in timestamps]
    cuboids = get_ids_and_cuboids_at_lidar_timestamps(
        track_data, log_id, timestamps, poses, confidence_threshold
    )
    if len(cuboids) != 2:
        print(f"Length of id_cuboid_map = {len(cuboids)}, log_id: {log_id} has no detections at timestamps: {timestamps}")
        flow_0_1, classes_0, valid_0, _ = zero_flow(sweeps, cuboids, poses)
    else:
        flow_0_1, classes_0, valid_0, _ = compute_flow(sweeps, cuboids, poses)
    return flow_0_1, classes_0, valid_0


def process_log(
    dataset: AV2SensorDataLoader,
    track_data: dict,
    log_id: str,
    output_dir: Path,
    confidence_threshold: float,
    n: Optional[int] = None,
):
    """Outputs sceneflow and auxillary information for each pair of pointclouds in the
    dataset. Output files have the format <output_dir>/<log_id>_<sweep_1_timestamp>.npz
     Args:
       dataset: Sensor dataset to process.
       log_id: Log unique id.
       output_dir: Output_directory.
       n: the position to use for the progress bar
     Returns:
       None
    """
    timestamps = dataset.get_ordered_log_lidar_timestamps(log_id)

    iter_bar = zip(timestamps, timestamps[1:])
    if n is not None:
        iter_bar = tqdm(
            iter_bar,
            leave=False,
            total=len(timestamps) - 1,
            position=n,
            desc=f"Log {log_id}",
        )

    for ts0, ts1 in iter_bar:
        flow_0_1, classes_0, valid_0 = compute_sceneflow(
            dataset, track_data, log_id, (ts0, ts1), confidence_threshold
        )
        df = pd.DataFrame(
            {
                "flow_tx_m": flow_0_1[:, 0],
                "flow_ty_m": flow_0_1[:, 1],
                "flow_tz_m": flow_0_1[:, 2],
                "is_valid": valid_0,
                "classes_0": classes_0,
            }
        )
        save_feather(output_dir / log_id / f"{ts0}.feather", df, verbose=False)


def process_log_wrapper(x, ignore_current_process=False):
    if not ignore_current_process:
        current = current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)


def process_logs(data_dir: Path, output_dir: Path, nproc: int, confidence_threshold: float):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
    Args:
      data_dir: Argoverse 2.0 directory
      output_dir: Output directory.
    """

    if not data_dir.exists():
        print(f"{data_dir} not found")
        return

    split_output_dir = output_dir
    split_output_dir.mkdir(exist_ok=True, parents=True)

    dataset = AV2SensorDataLoader(data_dir=data_dir, labels_dir=data_dir)
    with open(TRACKER_RESULTS_PATH, "rb") as openfile:
        track_data = pickle.load(openfile)
    logs = dataset.get_log_ids()
    args = sorted(
        [(dataset, track_data, log, split_output_dir, confidence_threshold) for log in logs]
    )

    print(f"Using {nproc} processes")
    if nproc <= 1:
        for x in tqdm(args):
            process_log_wrapper(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(process_log_wrapper, args), total=len(logs)))


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="create",
        description="Create a LiDAR sceneflow dataset from Argoverse 2.0 Sensor",
    )
    parser.add_argument(
        "--argo_dir",
        type=str,
        help="The top level directory contating the input dataset",
    )
    parser.add_argument(
        "--output_dir", type=str, help="The location to output the sceneflow files to"
    )
    parser.add_argument("--nproc", type=int, default=(multiprocessing.cpu_count() - 1))
    parser.add_argument("--confidence", type=float, default=0.5)

    args = parser.parse_args()
    data_root = Path(args.argo_dir)
    output_dir = Path(args.output_dir)

    process_logs(data_root, output_dir, args.nproc, args.confidence)
