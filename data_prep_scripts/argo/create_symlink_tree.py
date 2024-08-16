import os
import shutil
from pathlib import Path


def create_symlink_tree(input_folder: str, target_folder: str, total_length: int | None = None):
    # Convert to Path objects for easier manipulation
    input_folder = Path(input_folder)
    target_folder = Path(target_folder)

    # Iterate through the sequence subfolders in the input folder
    for sequence_folder in input_folder.iterdir():
        if sequence_folder.is_dir():
            # Create corresponding folder in the target directory
            target_sequence_folder = target_folder / sequence_folder.name
            target_sequence_folder.mkdir(parents=True, exist_ok=True)

            # List all feather files in the sequence folder, sorted by name
            feather_files = sorted(sequence_folder.glob("*.feather"))

            modified_feather_files = feather_files
            if total_length is not None:
                modified_feather_files = feather_files[:total_length]
            else:
                modified_feather_files = feather_files[:-1]

            # Create symlinks for all but the last file in the sequence
            for file in modified_feather_files:
                symlink_target = target_sequence_folder / file.name
                symlink_target.symlink_to(file.resolve())


if __name__ == "__main__":
    import argparse

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Create a folder tree with symlinks to all but the last file in each sequence."
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to the input folder containing sequence subfolders."
    )
    parser.add_argument(
        "target_folder",
        type=str,
        help="Path to the target folder where the symlink tree will be created.",
    )
    parser.add_argument(
        "--total_length",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # Run the symlink creation
    create_symlink_tree(args.input_folder, args.target_folder, args.total_length)
