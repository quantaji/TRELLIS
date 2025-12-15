import argparse
import copy
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import utils3d
from easydict import EasyDict as edict
from tqdm import tqdm


def _voxelize_overall_worker(args) -> Union[None, Dict[str, Any]]:
    sha256, parts_sha256, part_ids, output_dir, filter_small_part_th = args

    part_ps = [Path(output_dir) / "voxels" / f"{s}.ply" for s in parts_sha256]
    if not all(p.exists() and p.is_file() for p in part_ps):
        print(f"Error: Not all parts exist for {sha256}")
        return None

    part_voxel_list = []
    new_part_id_list = []

    for dir_path, part_id in zip(part_ps, part_ids):
        voxel = utils3d.io.read_ply(str(dir_path))[0]
        if len(voxel) <= filter_small_part_th:
            continue
        part_voxel_list.append(voxel)
        new_part_id_list.append(part_id)

    if len(part_voxel_list) == 0:
        print(f"Error: No valid parts found for {sha256}")
        return None

    combined = list(zip(part_voxel_list, new_part_id_list))
    sorted_combined = sorted(combined, key=lambda x: x[0].min(axis=0)[2])
    part_voxel_list, new_part_id_list = zip(*sorted_combined)

    overall_voxel = np.vstack(part_voxel_list)
    overall_voxel = np.unique(overall_voxel, axis=0)

    vox_dst = output_dir / "voxels" / f"{sha256}.ply"
    utils3d.io.write_ply(str(vox_dst), overall_voxel)

    part_id_dst = output_dir / "parts_ids" / f"{sha256}.txt"
    with open(part_id_dst, "w") as f:
        for part_id in new_part_id_list:
            f.write(f"{part_id}\n")

    return {
        "sha256": sha256,
        "voxelized": True,
        "num_voxels": len(overall_voxel),
        "parts_listed": True,
    }


def voxelize_overall(
    selected_metadata,
    metadata,
    output_dir,
    max_workers: Optional[int] = None,
    filter_small_part_th: int = 5,
    **kwargs,
) -> pd.DataFrame:

    output_dir = Path(output_dir)

    # step 1 filter out overall and get a list of parts by uuid
    cond = metadata["name"].ne("overall") & metadata["sha256"].notna()
    df = metadata.loc[cond, ["uuid", "name", "sha256"]].sort_index()
    agg = df.groupby("uuid", sort=False).agg(
        partid_list=("name", list),
        sha256_list=("sha256", list),
    )
    partid_list_by_uuid = agg["partid_list"].to_dict()
    sha256_list_by_uuid = agg["sha256_list"].to_dict()

    # get a list of overall uuid
    overall_df = selected_metadata[(selected_metadata["name"].eq("overall")) & (selected_metadata["sha256"].notna())]

    args = []
    for sha256, uuid in zip(overall_df["sha256"], overall_df["uuid"]):
        args.append(
            (
                sha256,
                sha256_list_by_uuid[uuid],
                partid_list_by_uuid[uuid],
                output_dir,
                filter_small_part_th,
            )
        )

    max_workers = max_workers or os.cpu_count()
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for r in tqdm(
            ex.map(_voxelize_overall_worker, args, chunksize=64),
            total=len(args),
        ):
            if r is not None:
                rows.append(r)

    return pd.DataFrame(rows)


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
    voxelize_overall_df = voxelize_overall(selected_metadata, metadata, **opt)
    voxelize_overall_df.to_csv(os.path.join(opt.output_dir, f"voxelized_overall.csv"), index=False)
