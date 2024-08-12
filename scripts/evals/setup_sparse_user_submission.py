import argparse
import math
import shutil
import time
import zipfile
from pathlib import Path

import tqdm


def _unzip_submission(working_dir: Path, submission_zip: Path, every_kth: int) -> Path:
    assert submission_zip.exists(), f"Submission zip {submission_zip} does not exist."
    submission_dir = working_dir / "submission"

    submission_dir.mkdir(parents=True, exist_ok=False)

    # If the submission zip is actually a directory, symlink it to the submission dir
    if submission_zip.is_dir():
        print("Submission zip is a directory, symlinking it to the submission dir.")
        # Iterate over every sequence folder
        for sequence_folder in tqdm.tqdm(sorted(submission_zip.glob("*"))):
            sequence_folder_name = sequence_folder.name
            user_sequence_folder = submission_dir / sequence_folder_name
            user_sequence_folder.mkdir(parents=True, exist_ok=False)
            for idx, user_file in enumerate(sorted(sequence_folder.glob("*.feather"))):
                if idx % every_kth == 0:
                    # Symlink the file to the user sequence folder
                    user_file_symlink = user_sequence_folder / user_file.name
                    user_file_symlink.symlink_to(user_file)

        return submission_dir

    print(f"Unzipping {submission_zip} to {submission_dir}")
    before_unzip = time.time()
    with zipfile.ZipFile(submission_zip, "r") as zip_ref:
        zip_ref.extractall(submission_dir)
    after_unzip = time.time()
    print(
        f"Unzipped {submission_zip} to {submission_dir} in {after_unzip - before_unzip:.2f} seconds."
    )
    return submission_dir


def _validate_sequence_folder_and_create_dummy_entries(
    user_sequence_folder: Path, ground_truth_sequence_folder: Path, divisor: int = 5
):
    assert (
        user_sequence_folder.exists()
    ), f"User sequence folder {user_sequence_folder} does not exist."
    assert (
        ground_truth_sequence_folder.exists()
    ), f"Ground truth sequence folder {ground_truth_sequence_folder} does not exist."

    # Check that they have the same name
    assert (
        user_sequence_folder.name == ground_truth_sequence_folder.name
    ), f"User sequence folder {user_sequence_folder} and ground truth sequence folder {ground_truth_sequence_folder} do not have the same name."

    # Check that the user sequence folder has // divisor fewer feather files vs the ground truth sequence folder
    user_sequence_files = sorted(user_sequence_folder.glob("*.feather"))
    gt_sequence_files = sorted(ground_truth_sequence_folder.glob("*.feather"))

    expected_num_user_files = int(math.ceil(len(gt_sequence_files) / divisor))
    assert (
        len(user_sequence_files) == expected_num_user_files
    ), f"User sequence folder {user_sequence_folder} has {len(user_sequence_files)} files, expected {expected_num_user_files} files."

    # Ensure that all user files are 10 characters long to match the 010d expected format and are all integers mod divisor.
    for user_file in user_sequence_files:
        assert len(user_file.stem) == 10, f"User file {user_file} does not have 10 characters."
        assert (
            int(user_file.stem) % divisor == 0
        ), f"User file int {int(user_file.stem)} is not divisible by {divisor}."

    for idx in range(len(gt_sequence_files)):
        user_file = user_sequence_folder / f"{idx:010d}.feather"
        if idx % divisor == 0:
            # Check that file exists
            assert user_file.exists(), f"User file {user_file} does not exist."
        else:
            # Create dummy file
            with open(user_file, "w") as f:
                f.write("")

    # Check that the user sequence folder has the same number of files as the ground truth sequence folder
    user_sequence_files = sorted(user_sequence_folder.glob("*.feather"))
    assert len(user_sequence_files) == len(
        gt_sequence_files
    ), f"User sequence folder {user_sequence_folder} has {len(user_sequence_files)} files, expected {len(gt_sequence_files)} files."


def run_setup_sparse_user_submission(
    working_dir: Path,
    user_submission_zip: Path,
    ground_truth_root_folder: Path,
    every_kth_entry: int,
) -> Path:
    working_dir = Path(working_dir)
    user_submission_zip = Path(user_submission_zip)
    ground_truth_root_folder = Path(ground_truth_root_folder)

    working_dir.mkdir(parents=True, exist_ok=True)
    assert (
        user_submission_zip.exists()
    ), f"User submission zip {user_submission_zip} does not exist."
    assert (
        ground_truth_root_folder.exists()
    ), f"Ground truth root folder {ground_truth_root_folder} does not exist."

    unziped_submission_dir = _unzip_submission(working_dir, user_submission_zip, every_kth_entry)

    # Iterate over the sequence folders and validate and create dummy entries
    for gt_sequence_folder in tqdm.tqdm(sorted(ground_truth_root_folder.glob("*"))):
        user_sequence_folder = unziped_submission_dir / gt_sequence_folder.name
        _validate_sequence_folder_and_create_dummy_entries(
            user_sequence_folder, gt_sequence_folder, divisor=every_kth_entry
        )

    return unziped_submission_dir


if __name__ == "__main__":
    # Get arguments for the script
    parser = argparse.ArgumentParser(
        description="Setup a sparse user submission for the Argoverse 2.0 Scene Flow Prediction Challenge."
    )
    parser.add_argument(
        "working_dir",
        type=Path,
        help="The working directory to unzip the user submission and create dummy entries.",
    )
    parser.add_argument(
        "user_submission_zip",
        type=Path,
        help="The user submission zip file to unzip and create dummy entries.",
    )
    parser.add_argument(
        "ground_truth_root_folder",
        type=Path,
        help="The root folder containing the ground truth sequence folders.",
    )
    parser.add_argument(
        "--every_kth_entry",
        type=int,
        default=5,
        help="The number of entries to skip in the user submission.",
    )
    args = parser.parse_args()

    run_setup_sparse_user_submission(
        args.working_dir,
        args.user_submission_zip,
        args.ground_truth_root_folder,
        args.every_kth_entry,
    )
