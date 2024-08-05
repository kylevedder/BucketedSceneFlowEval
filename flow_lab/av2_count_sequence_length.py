import argparse
import json
from pathlib import Path

from bucketed_scene_flow_eval.utils import load_feather, save_json


def count_sequence_lengths_in_subfolders(root_folder_path):
    # Convert the input path to a Path object for easy handling
    root_folder = Path(root_folder_path)

    # Check if the provided root folder path exists and is a directory
    assert root_folder.is_dir(), f"The path {root_folder_path} is not a valid directory."

    sequence_lengths_lookup = {}

    for subfolder in sorted(root_folder.iterdir()):
        if subfolder.is_dir():
            # Count the sequence length in the "annotations.feather" file
            annotations_file = subfolder / "annotations.feather"
            if annotations_file.exists():
                # Load the annotations.feather file and count unique timestamps
                data = load_feather(annotations_file)
                timestamps = data["timestamp_ns"].unique()
                sequence_length = len(timestamps)
                sequence_lengths_lookup[subfolder.name] = sequence_length
            else:
                print(f"annotations.feather not found in {subfolder}")
    return sequence_lengths_lookup


def save_counts_to_file(counts_lookup, output_file_path):
    with open(output_file_path, "w") as f:
        json.dump(counts_lookup, f, indent=4, sort_keys=True)


def main(root_folder_path: Path):
    sequence_lengths_lookup = count_sequence_lengths_in_subfolders(root_folder_path)

    # Construct output file name based on root folder name
    output_file_name = f"{Path(root_folder_path).name}_sequence_lengths.json"
    # Construct full output path
    output_file_path = root_folder_path.parent / output_file_name
    save_json(output_file_path, sequence_lengths_lookup, indent=4)

    print(f"Saved sequence lengths to {output_file_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Count .feather files in subfolders of a given directory."
    )
    parser.add_argument("root_folder_path", type=Path, help="Path to the root folder.")
    args = parser.parse_args()
    main(args.root_folder_path)
