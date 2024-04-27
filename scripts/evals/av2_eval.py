# Set OMP_NUM_THREADS=1 to avoid slamming the CPU
import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import multiprocessing
from pathlib import Path

import tqdm

from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from bucketed_scene_flow_eval.eval import Evaluator


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
    dataset: Argoverse2CausalSceneFlow, num_shards: int, every_kth_in_sequence: int
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
    gt_dataset: Argoverse2CausalSceneFlow,
    est_dataset: Argoverse2CausalSceneFlow,
    evaluator: Evaluator,
    verbose: bool = True,
) -> Evaluator:
    # Set tqdm bar on the row of the terminal corresponding to the shard index
    iterator = shard_list
    if verbose:
        iterator = tqdm.tqdm(shard_list, position=shard_idx + 1, desc=f"Shard {shard_idx}")

    for idx in iterator:
        gt_lst = gt_dataset[idx]
        est_lst = est_dataset[idx]
        assert len(gt_lst) == len(est_lst) == 2, f"GT and estimated lists must have length 2."
        gt_frame0, gt_frame1 = gt_lst
        est_frame0, est_frame1 = est_lst
        evaluator.eval(est_frame0.flow, gt_frame0)

    return evaluator


def _work_wrapper(
    args: tuple[
        int, list[int], Argoverse2CausalSceneFlow, Argoverse2CausalSceneFlow, Evaluator, bool
    ]
) -> Evaluator:
    return _work(*args)


def run_eval(
    data_dir: Path,
    gt_flow_dir: Path,
    est_flow_dir: Path,
    output_path: Path,
    cpu_count: int,
    cache_root: Path,
    every_kth: int = 5,
    eval_type: str = "bucketed_epe",
    verbose: bool = True,
) -> None:
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."
    assert gt_flow_dir.exists(), f"GT flow directory {gt_flow_dir} does not exist."
    assert est_flow_dir.exists(), f"Estimated flow directory {est_flow_dir} does not exist."

    # Make the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    gt_dataset = Argoverse2CausalSceneFlow(
        root_dir=data_dir,
        flow_data_path=gt_flow_dir,
        with_ground=False,
        with_rgb=False,
        use_gt_flow=True,
        eval_type=eval_type,
        eval_args=dict(output_path=output_path),
        cache_root=cache_root,
    )

    est_dataset = Argoverse2CausalSceneFlow(
        root_dir=data_dir,
        flow_data_path=est_flow_dir,
        with_ground=False,
        with_rgb=False,
        use_gt_flow=False,
        use_cache=False,
        eval_type=eval_type,
        cache_root=cache_root,
    )

    dataset_evaluator = gt_dataset.evaluator()

    assert len(gt_dataset) == len(
        est_dataset
    ), f"GT and estimated datasets must be the same length, but are {len(gt_dataset)} and {len(est_dataset)} respectively."

    # Shard the dataset into pieces for each CPU
    shard_lists = _make_index_shards(gt_dataset, cpu_count, every_kth)
    args_list = [
        (shard_idx, shard_list, gt_dataset, est_dataset, dataset_evaluator, verbose)
        for shard_idx, shard_list in enumerate(shard_lists)
    ]

    if cpu_count > 1:
        print(f"Running evaluation on {len(gt_dataset)} scenes using {cpu_count} CPUs.")
        # Run the evaluation in parallel
        with multiprocessing.Pool(cpu_count) as pool:
            sharded_evaluators = pool.map(_work_wrapper, args_list)
    else:
        print(f"Running evaluation on {len(gt_dataset)} scenes using 1 CPU.")
        # Run the evaluation serially
        sharded_evaluators = [_work_wrapper(args) for args in args_list]

    # Combine the sharded evaluators
    gathered_evaluator: Evaluator = sum(sharded_evaluators)
    gathered_evaluator.compute_results()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Iterate over .feather files in a result zip file."
    )
    parser.add_argument("data_dir", type=Path, help="Path to the data_dir directory of the dataset")
    parser.add_argument("gt_flow_dir", type=Path, help="Path gt flow directory")
    parser.add_argument("est_flow_dir", type=Path, help="Path to the estimated flow directory")
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
        "--cache_root",
        type=Path,
        default=Path("/tmp/av2_eval_cache/"),
        help="Path to the cache root directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    run_eval(
        data_dir=args.data_dir,
        gt_flow_dir=args.gt_flow_dir,
        est_flow_dir=args.est_flow_dir,
        output_path=args.output_path,
        cpu_count=args.cpu_count,
        every_kth=args.every_kth,
        eval_type=args.eval_type,
        cache_root=args.cache_root,
        verbose=not args.quiet,
    )
