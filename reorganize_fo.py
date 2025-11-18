import os
from pathlib import Path
import shutil

# === CONFIGURATION ===
LOG_DIR = Path("/data/sbaumann/nature-gnn/dagr/logs/dsec/detection")                         # Root where your detection folders live
OUTPUT_DIR = Path("/data/sbaumann/nature-gnn/dagr/logs/dsec/detection/comparisons")          # Where to store reorganized structure

RUNS = {
    "front": "test_interframe_front_det",
    "back": "test_interframe_back_det"
}

# Make sure output dir exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Go through each run type and copy files
for run_name, run_folder in RUNS.items():
    run_path = LOG_DIR / run_folder
    if not run_path.exists():
        print(f"[WARNING] Run folder not found: {run_path}")
        continue

    for det_file in run_path.glob("detections_*.npy"):
        seq_name = det_file.stem.replace("detections_", "")
        seq_folder = OUTPUT_DIR / seq_name
        seq_folder.mkdir(parents=True, exist_ok=True)

        dest_file = seq_folder / f"{run_name}_detections.npy"
        shutil.copy(det_file, dest_file)
        print(f"Copied {det_file} -> {dest_file}")

print("\n✅ Reorganization complete!")
