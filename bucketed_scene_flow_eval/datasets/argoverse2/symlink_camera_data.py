import argparse
import shutil
from pathlib import Path

import tqdm

# Get path to missing_cam_frames AV2 and with_cam_frames AV2 copies.
parser = argparse.ArgumentParser()
parser.add_argument("missing_cam_frames", type=Path)
parser.add_argument("with_cam_frames", type=Path)
args = parser.parse_args()

assert args.missing_cam_frames.is_dir(), f"{args.missing_cam_frames} is not a directory"
assert args.with_cam_frames.is_dir(), f"{args.with_cam_frames} is not a directory"

split_names = ["train", "val", "test"]

for split in split_names:
    missing_frames_dir = args.missing_cam_frames / split
    with_frames_dir = args.with_cam_frames / split

    # iterate through directories in missing_frames_dir
    for missing_dir in tqdm.tqdm(
        list(missing_frames_dir.iterdir()), desc=f"Processing {split} split"
    ):
        # Corresponding data dir
        with_dir = with_frames_dir / missing_dir.name
        assert missing_dir.is_dir(), f"{missing_dir} is not a directory"
        assert with_dir.is_dir(), f"{with_dir} is not a directory"

        # Symlink the "sensors/cameras" directory from with_dir to missing_dir.
        # Remove the "sensors/cameras" directory from missing_dir if it exists.
        missing_cameras_dir = missing_dir / "sensors/cameras"
        with_cameras_dir = with_dir / "sensors/cameras"
        assert with_cameras_dir.is_dir(), f"{with_cameras_dir} is not a directory"

        if missing_cameras_dir.is_dir():
            shutil.rmtree(missing_cameras_dir)

        missing_cameras_dir.symlink_to(with_cameras_dir, target_is_directory=True)
