import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Define the category mappings
BACKGROUND_CATEGORIES = ["BACKGROUND"]

ROAD_SIGNS = [
    "BOLLARD",
    "CONSTRUCTION_BARREL",
    "CONSTRUCTION_CONE",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "SIGN",
    "STOP_SIGN",
    "MESSAGE_BOARD_TRAILER",
    "TRAFFIC_LIGHT_TRAILER",
]

PEDESTRIAN_CATEGORIES = ["PEDESTRIAN", "STROLLER", "WHEELCHAIR", "OFFICIAL_SIGNALER"]

WHEELED_VRU = [
    "BICYCLE",
    "BICYCLIST",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
]

CAR = ["REGULAR_VEHICLE"]

OTHER_VEHICLES = [
    "BOX_TRUCK",
    "LARGE_VEHICLE",
    "RAILED_VEHICLE",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "ARTICULATED_BUS",
    "BUS",
    "SCHOOL_BUS",
]

BUCKETED_METACATAGORIES = {
    "BACKGROUND": BACKGROUND_CATEGORIES,
    "CAR": CAR,
    "PEDESTRIAN": PEDESTRIAN_CATEGORIES,
    "WHEELED_VRU": WHEELED_VRU,
    "OTHER_VEHICLES": OTHER_VEHICLES,
}

# Reverse mapping from specific category to meta category
CATEGORY_TO_META = {
    category: meta
    for meta, categories in BUCKETED_METACATAGORIES.items()
    for category in categories
}


def load_cuboid_metadata(json_file: Path):
    """Load cuboid metadata from a JSON file."""
    print("Loading cuboid metadata from", json_file)
    with open(json_file, "r") as f:
        cuboid_metadata = json.load(f)
    print("Loaded", len(cuboid_metadata), "cuboids")
    return cuboid_metadata


def plot_histogram(cuboid_metadata, output_file: Path):
    """Plot a histogram of boxes by volume, colored by class name."""
    volumes_by_class = defaultdict(list)

    # Collect volumes by class
    for entry in cuboid_metadata:
        class_name = entry["class_name"]
        if class_name in CATEGORY_TO_META:
            meta_class_name = CATEGORY_TO_META[class_name]
            volume = entry["volume"]
            volumes_by_class[meta_class_name].append(volume)

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

    plt.yscale("log")
    plt.xlabel("Volume")
    plt.ylabel("Count")
    plt.title("Histogram of Boxes by Volume and Class")
    # Set x-axis to limit of 0 to 60
    plt.xlim(0, 60)
    plt.legend()
    plt.grid(True, which="both", ls="--")

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
