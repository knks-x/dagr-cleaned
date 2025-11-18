import sys
import os
import h5py
import csv
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from libs.ball.inspect_folders import event_dist, count_events, count_events_in_folder, event_dist_sorted


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
                        n_events = len(f["events"]["x"])
                    if n_events <= 1:
                        os.remove(fpath)
                        deleted_files += 1
                        print(f"Deleted empty or tiny file: {fpath} ({n_events} events)")
                except Exception as e:
                    print(f"⚠️ Could not read {fpath}: {e}")

    print(f"\nCleanup finished. Deleted {deleted_files} empty/tiny cropped files.")


root = "/data/sbaumann/nature-gnn/datasets-b/ball-inc/data"  
out_csv = "ballinc_summary_short.csv"

cleanup_cropped_files(root)

"""rows = []
for folder in tqdm(sorted(os.listdir(root))):
    folder_path = os.path.join(root, folder)                                                #.../data/00001
    if not os.path.isdir(folder_path):
        continue
    try:
        n_events = count_events_in_folder(folder_path)
    except Exception as e:
        print("error counting events in", folder_path, e)
        n_events = None"""

    """meta = {}
    #/data/sbaumann/nature-gnn/datasets-b/ball-inc/data/00000/00000_metadata.csv
    meta_path = os.path.join(folder_path, f"{folder}_metadata.csv")                             #../data/00001/00001_metadata.csv
    if os.path.exists(meta_path):
        try:
            # assume single-line header CSV like the example
            dfm = pd.read_csv(meta_path)
            # take first row
            meta = dfm.iloc[0].to_dict()
        except Exception as e:
            print("bad metadata read", folder_path, e)"""

    """row = {"folder": folder, "n_events": n_events}
    #row.update(meta)
    rows.append(row)

pd.DataFrame(rows).to_csv(out_csv, index=False)
print("wrote", out_csv)

out_dir=f"/data/sbaumann/nature-gnn/dagr/{out_csv}"
df = pd.read_csv(out_dir)
print(f"Minimum number of events: {df['n_events'].min()}")
print(f"Maximum number of events: {df['n_events'].max()}")


save_histogram=True
if save_histogram:
    event_dist_sorted(out_dir, plot=False)
"""
