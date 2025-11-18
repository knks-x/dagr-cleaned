import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

# --- CONFIG ---
# Root folder containing sequences (e.g., 'train' or 'test')
ROOT_DIR = "/data/sbaumann/nature-gnn/dagr/dsec-det/test"
save_path= "/data/sbaumann/nature-gnn/dagr/dsec-det"
# COCO thresholds (area in pixels)
SMALL_MAX = 32 * 32         # <= 1,024 px²
MEDIUM_MAX = 96 * 96        # <= 9,216 px²
# Large: > 9,216 px²

def categorize_bbox(area):
    """Return size category for a given bounding box area."""
    if area <= SMALL_MAX:
        return "small"
    elif area <= MEDIUM_MAX:
        return "medium"
    else:
        return "large"

def process_tracks_file(file_path):
    """Load a tracks.npy file and return a list of size categories + areas."""
    data = np.load(file_path)
    widths = data["w"]
    heights = data["h"]
    areas = widths * heights
    categories = [categorize_bbox(a) for a in areas]
    return categories, areas

def main():
    all_sizes = []
    all_areas = []
    save_path= "/data/sbaumann/nature-gnn/dagr/dsec-det"

    # Walk through all subfolders to find tracks.npy files
    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file == "tracks.npy":
                file_path = os.path.join(root, file)
                categories, areas = process_tracks_file(file_path)
                all_sizes.extend(categories)
                all_areas.extend(areas)

    # Count occurrences per category
    count = Counter(all_sizes)
    total = sum(count.values())

    print("\n--- Object Size Distribution (COCO thresholds) ---")
    for k in ["small", "medium", "large"]:
        pct = (count[k] / total * 100) if total > 0 else 0
        print(f"{k.capitalize():<7}: {count[k]:>8} ({pct:5.2f}%)")

    categories = ["small", "medium", "large"]
    values = [count[c] for c in categories]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(categories, values)
    ax.set_title("DSEC Detection: Object Size Distribution in Test Set")
    ax.set_xlabel("Object size category")
    ax.set_ylabel("Count")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()

    
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


    """fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(all_areas, bins=100, log=True)
    ax.set_title("Distribution of Bounding Box Areas (log scale) in Train Set")
    ax.set_xlabel("Bounding box area (px²)")
    ax.set_ylabel("Frequency (log scale)")
    fig.tight_layout()


    hist_save_path = save_path.replace(".png", "_hist.png")
    fig.savefig(hist_save_path, dpi=150)
    plt.close(fig)"""

if __name__ == "__main__":
    main()
