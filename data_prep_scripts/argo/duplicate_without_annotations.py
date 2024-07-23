import argparse
from pathlib import Path

import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_root",
        type=Path,
        help="Path to the root directory of the source dataset.",
    )
    parser.add_argument(
        "target_root",
        type=Path,
        help="Path to the root directory of the target dataset.",
    )
    return parser.parse_args()


def symlink_raw_data_split(source_root: Path, target_root: Path):
    assert source_root.is_dir(), f"{source_root} is not a directory"
    target_root.mkdir(exist_ok=True, parents=True)

    sequence_dirs = sorted(source_root.glob("*"))
    for sequence_dir in tqdm.tqdm(sequence_dirs):
        sequence_id = sequence_dir.name
        target_sequence_dir = target_root / sequence_id
        target_sequence_dir.mkdir(exist_ok=True)

        # Relevant files / folders are:
        # - calibration
        # - map
        # - sensors
        # - city_SE3_egovehicle.feather

        symlink_targets = [
            "calibration",
            "map",
            "sensors",
            "city_SE3_egovehicle.feather",
        ]
        for symlink_target in symlink_targets:
            source_path = sequence_dir / symlink_target
            target_path = target_sequence_dir / symlink_target
            assert source_path.exists(), f"{source_path} not found"
            # If the target path already exists, remove it
            if target_path.exists():
                target_path.unlink()
            target_path.symlink_to(source_path)


def main(source_root: Path, target_root: Path):
    subdirs = ["train", "val", "test"]
    for subdir in subdirs:
        source_subdir = source_root / subdir
        target_subdir = target_root / subdir

        if not source_subdir.exists():
            print(f"{source_subdir} not found")
            continue

        symlink_raw_data_split(source_subdir, target_subdir)

    label_subdirs = ["train_sceneflow_feather", "val_sceneflow_feather"]
    for label_subdir in label_subdirs:
        source_label_subdir = source_root / label_subdir
        target_label_subdir = target_root / label_subdir

        if not source_label_subdir.exists():
            print(f"{source_label_subdir} not found")
            continue

        # Symlink directly
        # If the target path already exists, remove it
        if target_label_subdir.exists():
            target_label_subdir.unlink()
        target_label_subdir.symlink_to(source_label_subdir)


if __name__ == "__main__":
    args = parse_args()
    main(args.source_root, args.target_root)
