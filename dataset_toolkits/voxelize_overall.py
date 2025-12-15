import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict


if __name__ == "__main__":
    dataset_utils = importlib.import_module(f"datasets.{sys.argv[1]}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the metadata")
    dataset_utils.add_args(parser)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--filter_small_part_th", type=int, default=5)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "voxels"), exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, "parts_ids"), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, "metadata.csv")):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(os.path.join(opt.output_dir, "metadata.csv"))

    selected_metadata = metadata[metadata["name"] == "overall"]

    start = len(selected_metadata) * opt.rank // opt.world_size
    end = len(selected_metadata) * (opt.rank + 1) // opt.world_size
    selected_metadata = selected_metadata[start:end]

    print(f"Processing {len(selected_metadata)} objects...")

    # process objects
    voxelize_overall_df = dataset_utils.voxelize_overall(selected_metadata, metadata, **opt)
    voxelize_overall_df.to_csv(os.path.join(opt.output_dir, f"voxelized_overall.csv"), index=False)
