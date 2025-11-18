import sys
import os
import h5py
import csv
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt



#h5_path = "/data/sbaumann/nature-gnn/datasets-b/ball-inc/data/00000/00000_events.hdf5"

#check structure
"""with h5py.File(h5_path, "r") as f:
    def print_structure(g, indent=0):
        for k, v in g.items():
            if isinstance(v, h5py.Group):
                print("  " * indent + f"[Group] {k}")
                print_structure(v, indent + 1)
            else:
                print("  " * indent + f"[Dataset] {k}, shape={v.shape}, dtype={v.dtype}")
    print_structure(f)
"""

def cleanup_empty_cropped_files(root_dir):
    """
    Recursively scans the dataset folders and deletes cropped HDF5 files
    that contain 1 or fewer events.
    """
    deleted_files = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if "cropped" in fname.lower() and fname.endswith(".hdf5"):
                fpath = os.path.join(dirpath, fname)
                try:
                    with h5py.File(fpath, "r") as f:
                        os.remove(fpath)
                        deleted_files += 1
                        print(f"Deleted cropped file: {fpath}")
                except Exception as e:
                    print(f"⚠️ Could not read {fpath}: {e}")

    print(f"\nCleanup finished. Deleted {deleted_files} empty/tiny cropped files.")


root = "/data/sbaumann/nature-gnn/datasets-b/ball-inc/data"  
out_csv = "ballinc_summary_short.csv"

cleanup_empty_cropped_files(root)

