# a modified version of voxelize.py for partverse
# only parts are voxelized
# the overal voxel should be computed after all parts
import argparse
import copy
import importlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import pandas as pd
import utils3d
from easydict import EasyDict as edict
from tqdm import tqdm


def _voxelize_part_worker(args: Tuple[str, str, str]) -> Union[None, Dict[str, Any]]:
    sha256, overall_sha256, output_dir = args
    output_dir = Path(output_dir)

    mesh: o3d.geometry = o3d.io.read_triangle_mesh(output_dir / f"renders/{sha256}/mesh.ply")

    # transform back
    with open(output_dir / f"renders/{sha256}/transforms.json") as f:
        metadata = json.load(f)

    scale_part = metadata["scale"]
    offset_part = np.array(metadata["offset"])

    mesh.scale(1.0 / scale_part, center=(0.0, 0.0, 0.0))
    mesh.translate(-offset_part, relative=True)

    # transform to overall
    with open(output_dir / f"renders/{overall_sha256}/transforms.json") as f:
        overall_metadata = json.load(f)

    scale_overall = overall_metadata["scale"]
    offset_overall = np.array(overall_metadata["offset"])

    mesh.translate(offset_overall, relative=True)
    mesh.scale(scale_overall, center=(0.0, 0.0, 0.0))

    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    vertices = np.array(
        [voxel.grid_index for voxel in voxel_grid.get_voxels()],
    )
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5

    utils3d.io.write_ply(
        output_dir / f"voxels/{sha256}_overall_centered.ply",
        vertices,
    )

    # convert to self centered for dino feature
    vertices_self_centered = vertices / scale_overall - offset_overall
    vertices_self_centered = (vertices_self_centered + offset_part) * scale_part

    utils3d.io.write_ply(
        output_dir / f"voxels/{sha256}.ply",
        vertices_self_centered,
    )

    return {
        "sha256": sha256,
        "voxelized": True,
        "num_voxels": len(vertices),
    }


def voxelize_part(
    selected_metadata,
    metadata,
    output_dir,
    max_workers: Optional[int] = None,
    **kwargs,
):

    # uuid to overall sha256
    uuid_to_overall_sha256 = metadata.loc[
        metadata["name"].eq("overall") & metadata["sha256"].notna(),
        ["uuid", "sha256"],
    ].set_index(
        "uuid"
    )["sha256"]
    m = metadata.assign(overall_sha256=metadata["uuid"].map(uuid_to_overall_sha256))
    part_sha256_to_overall_sha256 = dict(m.loc[m["name"].ne("overall") & m["sha256"].notna(), ["sha256", "overall_sha256"]].dropna().drop_duplicates("sha256", keep="last").itertuples(index=False, name=None))

    args = []
    for sha256 in selected_metadata["sha256"]:
        args.append(
            (
                sha256,
                part_sha256_to_overall_sha256[sha256],
                output_dir,
            )
        )

    max_workers = max_workers or os.cpu_count()
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for r in tqdm(
            ex.map(_voxelize_part_worker, args, chunksize=64),
            total=len(args),
        ):
            if r is not None:
                rows.append(r)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=None)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "voxels").mkdir(exist_ok=True, parents=True)

    # get file list
    if not ((output_dir / "metadata.csv").exists() and (output_dir / "metadata.csv").is_file()):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(output_dir / "metadata.csv")

    # filter out all metadata that is overall
    selected_metadata = metadata[metadata["name"] != "overall"]
    start = len(selected_metadata) * opt.rank // opt.world_size
    end = len(selected_metadata) * (opt.rank + 1) // opt.world_size
    selected_metadata = selected_metadata[start:end]

    # filter out objects that are already processed
    records = []
    for sha256 in copy.copy(metadata["sha256"].values):
        if (output_dir / f"voxels/{sha256}.ply").exists():
            pts = utils3d.io.read_ply(output_dir / f"voxels/{sha256}.ply")[0]
            records.append(
                {
                    "sha256": sha256,
                    "voxelized": True,
                    "num_voxels": len(pts),
                }
            )
            selected_metadata = selected_metadata[selected_metadata["sha256"] != sha256]

    print(f"Processing {len(metadata)} objects...")

    # process objects
    voxelize_part_df = voxelize_part(
        selected_metadata=selected_metadata,
        metadata=metadata,
        **opt,
    )
    voxelize_part_df = pd.concat(
        [voxelize_part_df, pd.DataFrame.from_records(records)],
    )
    voxelize_part_df.to_csv(
        output_dir / f"voxelized_part_{opt.rank}.csv",
        index=False,
    )
