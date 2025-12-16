import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from easydict import EasyDict as edict
from tqdm import tqdm


def merge_part_latents_worker(args: Tuple[str, List[str], List[str], str, str, str, str]):
    sha256, parts_sha256, part_ids, preprocess_dir, omnipart_dir, tgt_rel, latent_folder = args

    preprocess_dir = Path(preprocess_dir)
    tgt_dir = Path(omnipart_dir) / tgt_rel
    tgt_dir.mkdir(exist_ok=True, parents=True)

    # load valid part ids
    val_part_id_p = preprocess_dir / "parts_ids" / f"{sha256}.txt"
    with open(val_part_id_p) as f:
        val_part_ids = f.read().strip().splitlines()

    all_data_coord = []
    all_data_feat = []
    all_data_offset = [0]

    # overall latent
    overall_latent = np.load(preprocess_dir / latent_folder / f"{sha256}.npz")
    all_data_coord.append(overall_latent["coords"])
    all_data_feat.append(overall_latent["feats"])
    all_data_offset.append(overall_latent["coords"].shape[0])

    # part latent
    for psha256, pid in sorted(zip(parts_sha256, part_ids), key=lambda x: x[1]):
        if not pid in val_part_ids:
            continue
        part_latent = np.load(preprocess_dir / latent_folder / f"{psha256}.npz")
        all_data_coord.append(part_latent["coords"])
        all_data_feat.append(part_latent["feats"])
        all_data_offset.append(all_data_offset[-1] + part_latent["coords"].shape[0])

    all_data_coord = np.concatenate(all_data_coord, axis=0)
    all_data_feat = np.concatenate(all_data_feat, axis=0)
    all_data_offset = np.array(all_data_offset)

    save_dict = {
        "coords": all_data_coord,
        "feats": all_data_feat,
        "offsets": all_data_offset,
    }
    save_path = tgt_dir / "all_latent.npz"
    np.savez_compressed(str(save_path), **save_dict)
    return {
        "sha256": sha256,
        "latents_merged": True,
    }


def merge_part_latents(
    selected_metadata,
    metadata,
    preprocess_dir,
    omnipart_dir,
    latent_folder,
    max_workers: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:

    preprocess_dir = Path(preprocess_dir)
    omnipart_dir = Path(omnipart_dir)
    omnipart_dir.mkdir(exist_ok=True, parents=True)

    # step 1 filter out overall and get a list of parts by uuid
    cond = metadata["name"].ne("overall") & metadata["sha256"].notna()
    df = metadata.loc[cond, ["uuid", "name", "sha256"]].sort_values(["uuid", "name"], kind="mergesort")
    agg = df.groupby("uuid", sort=False).agg(
        partid_list=("name", list),
        sha256_list=("sha256", list),
    )
    partid_list_by_uuid = agg["partid_list"].to_dict()
    sha256_list_by_uuid = agg["sha256_list"].to_dict()

    # get a list of overall uuid
    overall_df = selected_metadata[(selected_metadata["name"].eq("overall")) & (selected_metadata["sha256"].notna())]

    args = []
    for sha256, uuid, rel in zip(overall_df["sha256"], overall_df["uuid"], overall_df["omnipart_rel"]):
        _rel = "/".join(rel.split("/")[:2])
        args.append(
            (
                sha256,
                sha256_list_by_uuid[uuid],
                partid_list_by_uuid[uuid],
                str(preprocess_dir),
                str(omnipart_dir),
                _rel,
                latent_folder,
            )
        )

    max_workers = max_workers or os.cpu_count()
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for r in tqdm(
            ex.map(merge_part_latents_worker, args, chunksize=64),
            total=len(args),
        ):
            if r is not None:
                rows.append(r)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument("--omnipart_dir", type=str, required=True, help="Directory to save the metadata")

    parser.add_argument("--latent_folder", type=str, default="latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16", help="Checkpoint to load")

    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    # get file list
    if not os.path.exists(os.path.join(opt.preprocess_dir, "metadata.csv")):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(os.path.join(opt.preprocess_dir, "metadata.csv"))
    selected_metadata = metadata[metadata["name"] == "overall"]

    start = len(selected_metadata) * opt.rank // opt.world_size
    end = len(selected_metadata) * (opt.rank + 1) // opt.world_size
    selected_metadata = selected_metadata[start:end]

    print(f"Processing {len(selected_metadata)} objects...")

    # process objects
    merge_part_latents_df = merge_part_latents(
        selected_metadata,
        metadata,
        opt.preprocess_dir,
        opt.omnipart_dir,
        opt.latent_folder,
        opt.max_workers,
    )
    merge_part_latents_df.to_csv(
        os.path.join(opt.preprocess_dir, f"mege_part_latents.csv"),
        index=False,
    )
