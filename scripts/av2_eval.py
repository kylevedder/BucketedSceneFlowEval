# Set OMP_NUM_THREADS=1 to avoid slamming the CPU
import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import multiprocessing
from pathlib import Path

import tqdm

from bucketed_scene_flow_eval.datasets import Argoverse2SceneFlow
from bucketed_scene_flow_eval.eval import Evaluator


def _make_shards(total_len: int, num_shards: int) -> list[tuple[int, int]]:
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


def _work(
    shard_idx: int,
    shard: tuple[int, int],
    gt_dataset: Argoverse2SceneFlow,
    est_dataset: Argoverse2SceneFlow,
    evaluator: Evaluator,
) -> Evaluator:
    start_idx, end_idx = shard

    # Set tqdm bar on the row of the terminal corresponding to the shard index
    for idx in tqdm.tqdm(
        range(start_idx, end_idx), position=shard_idx + 1, desc=f"Shard {shard_idx}"
    ):
        (gt_query, gt_flow), (_, est_flow) = gt_dataset[idx], est_dataset[idx]
        evaluator.eval(
            predictions=est_flow,
            ground_truth=gt_flow,
            query_timestamp=gt_query.query_particles.query_init_timestamp,
        )

    return evaluator


def _work_wrapper(
    args: tuple[int, tuple[int, int], Argoverse2SceneFlow, Argoverse2SceneFlow, Evaluator]
) -> Evaluator:
    return _work(*args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Iterate over .feather files in a result zip file."
    )
    parser.add_argument("data_dir", type=Path, help="Path to the data_dir directory of the dataset")
    parser.add_argument("gt_flow_dir", type=Path, help="Path gt flow directory")
    parser.add_argument("est_flow_dir", type=Path, help="Path to the estimated flow directory")
    parser.add_argument(
        "--cpu_count",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of CPUs to use",
    )
    args = parser.parse_args()

    assert args.data_dir.exists(), f"Data directory {args.data_dir} does not exist."
    assert args.gt_flow_dir.exists(), f"GT flow directory {args.gt_flow_dir} does not exist."
    assert (
        args.est_flow_dir.exists()
    ), f"Estimated flow directory {args.est_flow_dir} does not exist."

    gt_dataset = Argoverse2SceneFlow(
        root_dir=args.data_dir,
        flow_data_path=args.gt_flow_dir,
        with_ground=False,
        with_rgb=False,
        use_gt_flow=True,
        eval_args=dict(output_path="eval_results/bucketed_epe/nsfp_distillation_1x/"),
    )

    est_dataset = Argoverse2SceneFlow(
        root_dir=args.data_dir,
        flow_data_path=args.est_flow_dir,
        with_ground=False,
        with_rgb=False,
        use_gt_flow=False,
    )

    dataset_evaluator = gt_dataset.evaluator()

    assert len(gt_dataset) == len(
        est_dataset
    ), f"GT and estimated datasets must be the same length, but are {len(gt_dataset)} and {len(est_dataset)} respectively."

    # Shard the dataset into pieces for each CPU
    shards = _make_shards(len(gt_dataset), args.cpu_count)
    args_list = [
        (shard_idx, shard, gt_dataset, est_dataset, dataset_evaluator)
        for shard_idx, shard in enumerate(shards)
    ]

    if args.cpu_count > 1:
        print(f"Running evaluation on {len(gt_dataset)} scenes using {args.cpu_count} CPUs.")
        # Run the evaluation in parallel
        with multiprocessing.Pool(args.cpu_count) as pool:
            sharded_evaluators = pool.map(_work_wrapper, args_list)
    else:
        print(f"Running evaluation on {len(gt_dataset)} scenes using 1 CPU.")
        # Run the evaluation serially
        sharded_evaluators = [_work_wrapper(args) for args in args_list]

    # Combine the sharded evaluators
    gathered_evaluator: Evaluator = sum(sharded_evaluators)
    gathered_evaluator.compute_results()
