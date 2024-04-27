import os

os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing
from argparse import ArgumentParser
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from bucketed_scene_flow_eval.datastructures.dataclasses import PoseInfo
from bucketed_scene_flow_eval.datastructures.pointcloud import PointCloud
from bucketed_scene_flow_eval.datasets.nuscenes.nuscenes_scene_flow import (
    CATEGORY_MAP_INV,
)
from bucketed_scene_flow_eval.utils.loaders import save_feather
from bucketed_scene_flow_eval.datasets.nuscenes import NuScenesRawSequenceLoader
from bucketed_scene_flow_eval.datasets.nuscenes.nuscenes_utils import InstanceBox

def compute_sceneflow(
    dataset: NuScenesRawSequenceLoader, log_id: str, timestamps: tuple[int, int]
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

    def compute_flow(sweeps, cuboids: list[list[InstanceBox]], poses: list[PoseInfo]):
        ego1_SE3_ego0 = poses[1].ego_to_global.inverse().compose(poses[0].ego_to_global)

        flow_0_1 = np.zeros_like(sweeps[0].points, dtype=np.float32)

        valid_0 = np.ones(len(sweeps[0].points), dtype=bool)
        classes_0 = np.ones(len(sweeps[0].points), dtype=np.int8) * CATEGORY_MAP_INV["background"]

        c1_instance_tokens = {c.instance_token: i for i, c in enumerate(cuboids[1])}

        for c0 in cuboids[0]:
            c0.wlh += np.array([0.2, 0.2, 0.0]) # the bounding boxes are a little too tight, so some points are missed, expand width and length by 0.2m
            obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].points)
            classes_0[obj_mask] = CATEGORY_MAP_INV[c0.name]

            if c0.instance_token in c1_instance_tokens:
                c1 = cuboids[1][c1_instance_tokens[c0.instance_token]]
                c1_SE3_c0_ego_frame = ego1_SE3_ego0.inverse().compose(
                    c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                )
                obj_flow = c1_SE3_c0_ego_frame.transform_points(obj_pts) - obj_pts
                flow_0_1[obj_mask] = obj_flow.astype(np.float32)
            else:
                valid_0[obj_mask] = 0

        # Convert flow from ego -> sensor frame for storage
        flow_0_1 = PointCloud(flow_0_1)
        flow_0_1_sensor = poses[0].sensor_to_ego.inverse().transform_flow(flow_0_1)

        return flow_0_1_sensor, classes_0, valid_0, ego1_SE3_ego0

    sequence = dataset._load_sequence_uncached(log_id)

    pc_objects = [sequence.load(ts, 0)[0].pc for ts in timestamps]

    # Sweeps are stored in sensor frame so we must transform to ego frame
    sweeps = [pc_obj.full_ego_pc for pc_obj in pc_objects]

    # Ego to map poses, used for computing ego motion
    poses = [sequence._load_pose(ts) for ts in timestamps]

    # Cuboids are fetched in global frame initially
    lidar_sensor_tokens = [sequence.synced_sensors[ts].lidar_top['token'] for ts in timestamps]
    cuboids = [sequence.nusc.get_boxes_with_instance_token(lidar_token) for lidar_token in lidar_sensor_tokens]
    # Here we convert cuboids from global frame to ego frame
    for cuboid_list, pose in zip(cuboids, poses):
        for c in cuboid_list:
            c.transform(pose.ego_to_global.inverse())

    flow_0_1, classes_0, valid_0, _ = compute_flow(sweeps, cuboids, poses)
    return flow_0_1, classes_0, valid_0


def process_log(
    dataset: NuScenesRawSequenceLoader, log_id: str, output_dir: Path, n: Optional[int] = None
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
    timestamps = range(len(dataset._load_sequence_uncached(log_id)))

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
        flow_0_1, classes_0, valid_0 = compute_sceneflow(dataset, log_id, (ts0, ts1))
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


def process_logs(data_dir: Path, nusc_ver: str, output_dir: Path, nproc: int):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
    Args:
      data_dir: NuScenes directory
      output_dir: Output directory.
    """

    if not data_dir.exists():
        print(f"{data_dir} not found")
        return

    split_output_dir = output_dir
    split_output_dir.mkdir(exist_ok=True, parents=True)

    dataset = NuScenesRawSequenceLoader(version=nusc_ver, sequence_dir=str(data_dir), point_cloud_range=None)
    logs = dataset.get_sequence_ids()
    args = sorted([(dataset, log, split_output_dir) for log in logs])

    print(f"Using {nproc} processes")
    if nproc <= 1:
        for x in tqdm(args):
            process_log_wrapper(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(process_log_wrapper, args), total=len(logs)))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = ArgumentParser(
        prog="create",
        description="Create a LiDAR sceneflow dataset from NuScenes",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="The top level directory contating the input dataset",
    )
    parser.add_argument(
        "--nusc_ver",
        type=str,
        help="The version of nuscenes to use.",
    )
    parser.add_argument(
        "--output_dir", type=str, help="The location to output the sceneflow files to"
    )
    parser.add_argument("--nproc", type=int, default=(multiprocessing.cpu_count() - 1))

    args = parser.parse_args()
    data_root = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    process_logs(data_root, args.nusc_ver, output_dir, args.nproc)
