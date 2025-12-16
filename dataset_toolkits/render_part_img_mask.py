import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm

BLENDER_LINK = "https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz"
BLENDER_INSTALLATION_PATH = "/tmp"
BLENDER_TMP_PATH = "/tmp/blender-3.0.1-linux-x64/blender"
BLENDER_DOCKER_PATH = "/opt/blender-3.0.1-linux-x64/blender"
BLENDER_PATH = BLENDER_DOCKER_PATH if os.path.exists(BLENDER_DOCKER_PATH) else BLENDER_TMP_PATH


def _install_blender() -> None:
    if not os.path.exists(BLENDER_PATH):
        os.system(f"wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}")
        os.system(f"tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}")


def render_worker(args: Tuple[List[str], str, Optional[int], float]) -> dict:

    glb_paths, output_dir, seed, force_rotation_deg = args
    shm_dir = Path("/dev/shm")
    tmp_base = shm_dir if shm_dir.is_dir() else None
    tmp_output_folder = Path(tempfile.mkdtemp(prefix=f"tmp_omnipart_render", dir=tmp_base))

    views = []

    default_camera_lens = 50.0
    default_camera_sensor_width = 36.0
    fov = 2.0 * math.atan(default_camera_sensor_width / (2.0 * default_camera_lens))

    # fixed view
    elev_deg = 25.0
    for azim_deg in (0, 45, 90, 135, 180, 225, 270, 315):
        views.append(
            {
                "yaw": math.radians(float(azim_deg)),
                "pitch": math.radians(float(elev_deg)),
                "radius": 1.0,  # 占位：你说 distance/radius 在后端结合 bbox 计算后替换
                "fov": float(fov),
            }
        )

    # random view
    random.seed(seed)
    for _ in range(8):
        theta = random.uniform(0.0, 2.0 * math.pi)
        phi = random.uniform(0.0, 0.5 * math.pi)
        flag = -1.0 if bool(random.randint(0, 1)) else 1.0
        pitch = flag * (0.5 * math.pi - phi)

        views.append(
            {
                "yaw": float(theta),
                "pitch": float(pitch),
                "radius": 1.0,  # 占位：后端替换
                "fov": float(fov),
            }
        )

    bargs = [
        BLENDER_PATH,
        "-b",
        "-P",
        os.path.join(os.path.dirname(__file__), "blender_script", "part_mask_render.py"),
        "--",
        "--views",
        json.dumps(views),
        "--objects",
        json.dumps([os.path.expanduser(p) for p in glb_paths]),
        "--resolution",
        "512",
        "--output_folder",
        tmp_output_folder,
        "--engine",
        "CYCLES",
        "--save_mask",
        "--force_rotation_deg",
        str(force_rotation_deg),
    ]

    subprocess.run(bargs, check=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    for src in tmp_output_folder.glob("color_*"):
        dst = str(output_dir / f"{src.stem}.webp")
        Image.open(src).save(dst, format="WEBP", lossless=True, method=6)

    for src in tmp_output_folder.glob("mask_*.exr"):
        shutil.copy2(src, output_dir / src.name)

    shutil.rmtree(tmp_output_folder, ignore_errors=True)

    return {"rendered": True}


def render(
    selected_metadata,
    metadata,
    preprocess_dir,
    omnipart_dir,
    seed: int = 42,
    force_rotation_deg: float = 0.0,
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
        # collect valid part's glb paths
        val_part_id_p = preprocess_dir / "parts_ids" / f"{sha256}.txt"
        with open(val_part_id_p) as f:
            val_part_ids = f.read().strip().splitlines()

        parts_sha256 = sha256_list_by_uuid[uuid]
        parts_id = partid_list_by_uuid[uuid]

        glb_paths = []
        for s, idx in zip(parts_sha256, parts_id):
            if idx in val_part_ids:
                glb_paths.append(str(preprocess_dir / f"raw/{s}.glb"))

        _rel = "/".join(rel.split("/")[:2])
        output_dir = str(omnipart_dir / _rel)

        args.append((glb_paths, output_dir, seed, force_rotation_deg))

    max_workers = max_workers or os.cpu_count()
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for r in tqdm(
            ex.map(render_worker, args, chunksize=64),
            total=len(args),
        ):
            if r is not None:
                rows.append(r)

    return pd.DataFrame(rows)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument("--omnipart_dir", type=str, required=True, help="Directory to save the metadata")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_rotation_deg", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    if not os.path.exists(os.path.join(opt.preprocess_dir, "metadata.csv")):
        raise ValueError("metadata.csv not found")
    metadata = pd.read_csv(os.path.join(opt.preprocess_dir, "metadata.csv"))
    selected_metadata = metadata[metadata["name"] == "overall"]

    # install blender
    print("Checking blender...", flush=True)
    _install_blender()

    start = len(selected_metadata) * opt.rank // opt.world_size
    end = len(selected_metadata) * (opt.rank + 1) // opt.world_size
    selected_metadata = selected_metadata[start:end]

    records = []

    print(f"Processing {len(metadata)} objects...")

    # process objects
    render_part_img_mask_df = render(
        selected_metadata,
        metadata,
        opt.preprocess_dir,
        opt.omnipart_dir,
        opt.seed,
        opt.force_rotation_deg,
        opt.max_workers,
    )
    render_part_img_mask_df.to_csv(
        os.path.join(opt.preprocess_dir, f"render_part_img_mask_df.csv"),
        index=False,
    )
