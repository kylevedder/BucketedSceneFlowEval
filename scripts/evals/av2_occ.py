# Set OMP_NUM_THREADS=1 to avoid slamming the CPU
import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from bucketed_scene_flow_eval.datasets import Argoverse2NonCausalSceneFlow
from bucketed_scene_flow_eval.eval import Evaluator
from bucketed_scene_flow_eval.utils import load_feather


def _make_range_shards(total_len: int, num_shards: int) -> list[tuple[int, int]]:
    """
    Return a list of tuples of (start, end) indices for each shard.

    The function divides the range specified by total_len into num_shards shards.
    Each shard is represented by a tuple of (start, end) indices.
    The division tries to distribute the elements as evenly as possible among the shards.
    """
    shards = []
    shard_len = total_len // num_shards
    remainder = total_len % num_shards

    start = 0
    for _ in range(num_shards):
        end = start + shard_len + (1 if remainder > 0 else 0)
        shards.append((start, min(end, total_len)))
        start = end
        remainder -= 1

    return shards


def _make_index_shards(
    dataset: Argoverse2NonCausalSceneFlow, num_shards: int, every_kth_in_sequence: int
) -> list[list[int]]:
    dataset_valid_indices: list[int] = [
        dataset_idx
        for (
            _,
            (subsequence_start_idx, subsequence_end_idx),
        ), dataset_idx in dataset.sequence_subsequence_idx_to_dataset_idx.items()
        if (subsequence_start_idx % every_kth_in_sequence) == 0
    ]

    tuple_shards = _make_range_shards(len(dataset_valid_indices), num_shards)
    return [dataset_valid_indices[start:end] for start, end in tuple_shards]


def _work(
    shard_idx: int,
    shard_list: list[int],
    gt_dataset: Argoverse2NonCausalSceneFlow,
    occ_folder: Path,
    verbose: bool = True,
) -> list[float]:
    # Set tqdm bar on the row of the terminal corresponding to the shard index
    iterator = shard_list
    if verbose:
        iterator = tqdm.tqdm(shard_list, position=shard_idx + 1, desc=f"Shard {shard_idx}")

    per_frame_l1_errors = []

    for idx in iterator:
        gt_lst = gt_dataset[idx]
        assert len(gt_lst) == 2, f"GT list must have length 2."
        source_frame = gt_lst[0]
        log_id = source_frame.log_id
        log_idx = source_frame.log_idx
        est_occ_path = occ_folder / f"{log_id}/{log_idx:010d}_occ.feather"

        assert est_occ_path.exists(), f"Estimated occ file {est_occ_path} does not exist."
        est_occ_df = load_feather(est_occ_path, verbose=False)
        est_occ_is_valid = est_occ_df["is_valid"].values
        est_occ_distances = est_occ_df["distances_m"].values
        est_occ_is_colliding = est_occ_df["is_colliding"].values

        gt_pc = source_frame.pc.full_ego_pc
        # Convert to a set of distances using L2 norm
        gt_distances = np.linalg.norm(gt_pc.points, axis=1)
        gt_is_valid_mask = source_frame.pc.mask

        # Ensure that the est_occ_df is the same length as the ground truth point cloud
        assert len(est_occ_df) == len(
            gt_pc
        ), "Estimated occ and ground truth point cloud must have the same length."

        # Ensure that for all entries, if gt_mask is true, then is_valid is true in the df
        assert np.all(
            np.logical_and(gt_is_valid_mask, est_occ_is_valid) == gt_is_valid_mask
        ), f"If gt_mask is true, then is_valid must be true in the estimated occ. Num differences: {np.sum(np.logical_and(gt_is_valid_mask, est_occ_is_valid) != gt_is_valid_mask)}"

        valid_gt_distances = gt_distances[gt_is_valid_mask]
        valid_est_distances = est_occ_distances[gt_is_valid_mask]

        l1_differences = np.abs(valid_gt_distances - valid_est_distances)
        per_frame_l1_errors.append(np.mean(l1_differences))

    return per_frame_l1_errors


def _work_wrapper(
    args: tuple[int, list[int], Argoverse2NonCausalSceneFlow, Path, bool]
) -> list[float]:
    return _work(*args)


def run_eval(
    data_dir: Path,
    est_occ_dir: Path,
    output_path: Path,
    cpu_count: int,
    every_kth: int = 5,
    eval_type: str = "bucketed_epe",
    verbose: bool = True,
) -> None:
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."
    assert est_occ_dir.exists(), f"Estimated occ directory {est_occ_dir} does not exist."

    # Make the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    gt_dataset = Argoverse2NonCausalSceneFlow(
        root_dir=data_dir,
        with_ground=False,
        with_rgb=False,
        load_flow=False,
        eval_type=eval_type,
        eval_args=dict(output_path=output_path),
        use_cache=False,
        range_crop_type="ego",
    )

    # Shard the dataset into pieces for each CPU
    shard_lists = _make_index_shards(gt_dataset, cpu_count, every_kth)
    args_list = [
        (shard_idx, shard_list, gt_dataset, est_occ_dir, verbose)
        for shard_idx, shard_list in enumerate(shard_lists)
    ]

    if cpu_count > 1:
        print(f"Running evaluation on {len(gt_dataset)} scenes using {cpu_count} CPUs.")
        # Run the evaluation in parallel
        with multiprocessing.Pool(cpu_count) as pool:
            sharded_results = pool.map(_work_wrapper, args_list)
    else:
        print(f"Running evaluation on {len(gt_dataset)} scenes using 1 CPU.")
        # Run the evaluation serially
        sharded_results = [_work_wrapper(args) for args in args_list]

    # Combine the results from each shard
    all_l1_error_results = []
    for shard_results in sharded_results:
        all_l1_error_results.extend(shard_results)

    mean_l1_error = np.mean(all_l1_error_results)
    std_l1_error = np.std(all_l1_error_results)

    # Save mean and std output to a CSV file
    csv_file_path = output_path / "l1_error_results.csv"
    df = pd.DataFrame(
        {
            "mean_l1_error": [mean_l1_error],
            "std_l1_error": [std_l1_error],
        }
    )
    df.to_csv(csv_file_path, index=False)
    print(f"Saved results to {csv_file_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Iterate over .feather files in a result zip file."
    )
    parser.add_argument("data_dir", type=Path, help="Path to the data_dir directory of the dataset")
    parser.add_argument("est_occ_dir", type=Path, help="Path to the estimated flow directory")
    parser.add_argument("output_path", type=Path, help="Path to save the results")
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use",
    )
    parser.add_argument(
        "--every_kth", type=int, default=5, help="Only evaluate every kth scene in a sequence"
    )
    parser.add_argument("--eval_type", type=str, default="bucketed_epe", help="Type of evaluation")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    run_eval(
        data_dir=args.data_dir,
        est_occ_dir=args.est_occ_dir,
        output_path=args.output_path,
        cpu_count=args.cpu_count,
        every_kth=args.every_kth,
        eval_type=args.eval_type,
        verbose=not args.quiet,
    )
