# Set omp_num_threads=1 to avoid slamming the CPU
import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import multiprocessing
import tqdm
import time

from bucketed_scene_flow_eval.datasets import Argoverse2SceneFlow
from bucketed_scene_flow_eval.eval import Evaluator
from bucketed_scene_flow_eval.datastructures import (
    QuerySceneSequence,
    GroundTruthParticleTrajectories,
    EstimatedParticleTrajectories,
    RawSceneItem,
)
from bucketed_scene_flow_eval.datastructures import PointCloud, O3DVisualizer


def read_feather_file(zip_ref: zipfile.ZipFile, file: Path):
    with zip_ref.open(str(file)) as file:
        return pd.read_feather(file)


def load_feather_files(
        zip_path: Path) -> Dict[str, List[Tuple[int, pd.DataFrame]]]:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        names = [Path(name) for name in zip_ref.namelist()]
        names = [name for name in names if name.suffix == ".feather"]
        # Create dictionary mapping sequence names to list of files by joining on the name.parent
        sequence_dirs = {}
        for name in names:
            sequence_dirs.setdefault(name.parent.name, []).append(
                (int(name.stem), read_feather_file(zip_ref, name)))
        for k in sequence_dirs.keys():
            sequence_dirs[k] = sorted(sequence_dirs[k], key=lambda x: x[0])
        return sequence_dirs


def perform_evaluate(
    mask: pd.DataFrame,
    result: pd.DataFrame,
    query: QuerySceneSequence,
    gt: GroundTruthParticleTrajectories,
    evaluator: Evaluator,
):
    mask_array = mask["mask"].to_numpy()
    gt._mask_entries(mask_array)

    xs = result["flow_tx_m"].to_numpy()
    ys = result["flow_ty_m"].to_numpy()
    zs = result["flow_tz_m"].to_numpy()

    assert (
        len(xs) == len(ys) == len(zs)
    ), f"Lengths do not match. xs: {len(xs)}, ys: {len(ys)}, zs: {len(zs)}"

    assert (
        len(query.scene_sequence) == 2
    ), f"Query sequence length is not 2 as expected for Scene Flow; it's {len(query.scene_sequence)}"

    uncompensated_flow_array = np.stack([xs, ys, zs], axis=1)

    # To move the flow to the correct frame, add the uncompensated_flow_array to the ego
    # frame PC1, then transform it to the PC2 global frame to get the estimated PC2.

    raw_scene_item_pc1: RawSceneItem = query.scene_sequence[0]
    raw_scene_item_pc2: RawSceneItem = query.scene_sequence[1]
    pc1_frame = raw_scene_item_pc1.pc_frame
    pc2_frame = raw_scene_item_pc2.pc_frame

    ego_frame_pc1 = pc1_frame.pc
    global_pc1 = ego_frame_pc1.transform(pc1_frame.global_pose)
    ego_flowed_pc2 = ego_frame_pc1.flow_masked(uncompensated_flow_array,
                                               mask_array)
    global_flowed_pc2 = ego_flowed_pc2.transform(pc2_frame.global_pose)

    # visualizer = O3DVisualizer()
    # visualizer.add_pointcloud(global_pc1, color=[0, 0, 1])
    # visualizer.add_pointcloud(global_flowed_pc2, color=[1, 0, 0])
    # gt.visualize(visualizer)
    # visualizer.run()

    stacked_points = np.stack([global_pc1.points, global_flowed_pc2.points],
                              axis=1)

    masked_array_idxes = np.arange(len(mask_array))[mask_array]

    lookup = EstimatedParticleTrajectories(len(gt), gt.trajectory_timestamps)
    lookup[masked_array_idxes] = (
        stacked_points[masked_array_idxes],
        [0, 1],
        np.zeros((masked_array_idxes.shape[0], 2), dtype=bool),
    )

    evaluator.eval(lookup, gt, 0)


def process_problem(
    input: Tuple[int, str, List[Tuple[int, pd.DataFrame]],
                 List[Tuple[int, pd.DataFrame]], Argoverse2SceneFlow, ]
) -> Evaluator:
    idx, sequence_name, mask_sequence, result_sequence, dataset = input
    evaluator = dataset.evaluator()
    mask_timestamp_to_data = {e[0]: e[1] for e in mask_sequence}
    result_timestamp_to_data = {e[0]: e[1] for e in result_sequence}

    # All entries in the mask sequence need to be in the result sequence
    assert set(mask_timestamp_to_data.keys()) <= set(
        result_timestamp_to_data.keys()
    ), f"Mask sequence {sequence_name} has timestamps that are not in the result sequence: {set(mask_timestamp_to_data.keys()) - set(result_timestamp_to_data.keys())}"

    def timestamp_to_dataset_idx(timestamp):
        mask_data = mask_timestamp_to_data[timestamp]
        result_data = result_timestamp_to_data[timestamp]

        return (
            dataset._av2_sequence_id_and_timestamp_to_idx(
                sequence_name, timestamp),
            mask_data,
            result_data,
        )

    dataset_idxes, mask_datas, result_datas = zip(*[
        timestamp_to_dataset_idx(timestamp)
        for timestamp in sorted(mask_timestamp_to_data.keys())
    ])

    for dataset_idx, mask_data, result_data in tqdm.tqdm(
            list(zip(dataset_idxes, mask_datas, result_datas)),
            disable=True,
            position=idx // 10 + 1,
            desc=f"Seq {idx:04d}",
    ):
        query, gt = dataset[dataset_idx]
        perform_evaluate(mask_data, result_data, query, gt, evaluator)
    return evaluator


def build_process_problems(
    mask_sequence_map: Dict[str, List[Tuple[int, pd.DataFrame]]],
    result_sequence_map: Dict[str, List[Tuple[int, pd.DataFrame]]],
    dataset: Argoverse2SceneFlow,
) -> List[Tuple[int, str, List[Tuple[int, pd.DataFrame]], List[Tuple[
        int, pd.DataFrame]], Argoverse2SceneFlow, ]]:
    mask_key_set = set(mask_sequence_map.keys())
    result_key_set = set(result_sequence_map.keys())

    # All the sequences need to be the same
    assert (
        mask_key_set == result_key_set
    ), f"Mask and result keys do not match. Mask has {mask_key_set - result_key_set} more keys, Result has {result_key_set - mask_key_set} more keys."

    for idx, sequence_name in enumerate(sorted(mask_key_set)):
        mask_sequence = mask_sequence_map[sequence_name]
        result_sequence = result_sequence_map[sequence_name]

        yield (idx, sequence_name, mask_sequence, result_sequence, dataset)


def main_loop(mask_zip: Path, result_zip: Path, dataset: Argoverse2SceneFlow,
              multiprocessor):
    mask_sequence_map, result_sequence_map = multiprocessor(
        load_feather_files,
        [mask_zip, result_zip],
        leave=False,
        desc="Loading zip files",
    )

    problems = list(
        build_process_problems(mask_sequence_map, result_sequence_map,
                               dataset))
    evaluators: List[Evaluator] = multiprocessor(process_problem,
                                                 problems,
                                                 desc="Problems")
    sum(evaluators).compute_results()


def build_multiprocessor(cpu_count: int):
    def single_threaded_process(
        worker,
        problems,
        leave: bool = True,
        verbose: bool = True,
        desc: str = None,
        bar_format: str = "\033[91m{l_bar}{bar}{r_bar}\033[0m",
    ):
        return [
            worker(problem) for problem in tqdm.tqdm(
                problems,
                disable=not verbose,
                leave=leave,
                desc=desc,
                bar_format=bar_format,
            )
        ]

    def multi_threaded_process(
        worker,
        problems,
        leave: bool = True,
        verbose: bool = True,
        desc: str = None,
        bar_format: str = "\033[91m{l_bar}{bar}{r_bar}\033[0m",
    ):
        cpus_to_use = min(cpu_count, len(problems))
        with multiprocessing.Pool(cpus_to_use) as pool:
            return list(
                tqdm.tqdm(
                    pool.imap(worker, problems),
                    total=len(problems),
                    disable=not verbose,
                    leave=leave,
                    bar_format=bar_format,
                    desc=desc,
                ))

    if cpu_count <= 1:
        print("Using single threaded process")
        return single_threaded_process
    print("Using multi threaded process")
    return multi_threaded_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iterate over .feather files in a result zip file.")
    parser.add_argument("root_dir",
                        type=Path,
                        help="Path to the root directory of the dataset")
    parser.add_argument("mask_zip_file",
                        type=Path,
                        help="Path to the mask zip file")
    parser.add_argument("result_zip_file",
                        type=Path,
                        help="Path to the result zip file")
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use",
    )
    args = parser.parse_args()

    assert args.root_dir.exists(
    ), f"Root directory {args.root_dir} does not exist."

    assert (args.mask_zip_file.exists()
            ), f"Mask zip file {args.mask_zip_file} does not exist."

    assert (args.result_zip_file.exists()
            ), f"Result zip file {args.result_zip_file} does not exist."

    dataset = Argoverse2SceneFlow(args.root_dir,
                                  with_ground=True,
                                  with_rgb=False)

    multiprocessor = build_multiprocessor(args.cpu_count)

    main_loop(args.mask_zip_file, args.result_zip_file, dataset,
              multiprocessor)
