import json
from pathlib import Path

import matplotlib.pyplot as plt

path = Path("/tmp/frame_results/scaled_epe")

subdirs = [x for x in path.iterdir() if x.is_dir() and "_01" not in x.name]
subdirs.sort()


def load_data(path: Path):
    path = Path(path)
    assert path.exists(), f"Path {path} does not exist!"
    with open(path, "r") as f:
        data = json.load(f)

    keys, values = zip(*sorted(data.items()))
    return keys, values


data_entries = [(e.name, load_data(e / "metric_table_35.json")) for e in subdirs]


def data_index_to_bar_width_location(data_index: int) -> tuple[float, float]:
    num_data_entries = len(data_entries)
    bar_width = 1 / num_data_entries

    bar_location = bar_width * data_index - 0.5 + bar_width / 2
    return bar_width, bar_location


for i, (name, (keys, values)) in enumerate(data_entries):
    bar_width, bar_location = data_index_to_bar_width_location(i)
    # Plot the bar chart with the data offset by i
    plt.bar(
        [x + bar_location for x in range(len(keys))],
        values,
        label=name,
        width=bar_width,
    )

# Set the x-axis labels
plt.xticks(range(len(keys)), keys)
# Rotate the x-axis labels
plt.xticks(rotation=90)
# Set the y-axis label
plt.ylabel("Scaled EPE")
plt.legend()
# Tight fit to the figure
plt.tight_layout()
plt.show()
