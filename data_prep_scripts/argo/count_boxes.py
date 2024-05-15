import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_cuboid_metadata(json_file: Path):
    """Load cuboid metadata from a JSON file."""
    with open(json_file, "r") as f:
        cuboid_metadata = json.load(f)
    return cuboid_metadata


def plot_histogram(cuboid_metadata, output_file: Path):
    """Plot a histogram of boxes by volume, colored by class name."""
    volumes_by_class = defaultdict(list)

    # Collect volumes by class
    for entry in cuboid_metadata:
        class_name = entry["class_name"]
        volume = entry["volume"]
        volumes_by_class[class_name].append(volume)

    # Prepare data for the histogram
    classes = list(volumes_by_class.keys())
    volumes = [volumes_by_class[class_name] for class_name in classes]

    # Create histogram bins
    all_volumes = np.concatenate(volumes)
    bin_edges = np.histogram_bin_edges(all_volumes, bins="auto")

    # Compute histogram data for each class
    histogram_data = []
    for class_volumes in volumes:
        counts, _ = np.histogram(class_volumes, bins=bin_edges)
        histogram_data.append(counts)

    histogram_data = np.array(histogram_data)

    # Plot stacked histogram
    plt.figure(figsize=(10, 6), dpi=300)
    bottom = np.zeros(len(bin_edges) - 1)

    for class_name, class_counts in zip(classes, histogram_data):
        plt.bar(
            bin_edges[:-1],
            class_counts,
            width=np.diff(bin_edges),
            bottom=bottom,
            label=class_name,
            align="edge",
        )
        bottom += class_counts

    plt.xlabel("Volume")
    plt.ylabel("Count")
    plt.title("Histogram of Boxes by Volume and Class")
    plt.legend()
    plt.grid(True)

    # Save plot as high-resolution PNG
    plt.savefig(output_file, format="png")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot histogram of cuboid volumes by class")
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the JSON file containing cuboid metadata",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output PNG file",
    )

    args = parser.parse_args()
    json_file = Path(args.json_file)
    output_file = Path(args.output_file)

    cuboid_metadata = load_cuboid_metadata(json_file)
    plot_histogram(cuboid_metadata, output_file)
