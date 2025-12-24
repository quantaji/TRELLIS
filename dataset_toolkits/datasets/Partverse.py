import argparse
import hashlib
import io
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--partverse_dir",
        type=str,
        help="Data folder that contains all glb)",
    )


def validate_mesh_as_single_trimesh(data: bytes):
    try:
        mesh = trimesh.load(
            file_obj=io.BytesIO(data),
            file_type="glb",
            force="mesh",
        )
        return isinstance(mesh, trimesh.Trimesh)
    except:
        return False


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata_worker(input_args: Tuple[str, str, Path, Path, bool]):
    uuid, target_rel_folder, p, partverse_dir, is_part = input_args

    name = f"{int(p.stem):04d}" if is_part else "overall"
    data = p.read_bytes()
    if validate_mesh_as_single_trimesh(data):
        return {
            "sha256": sha256_bytes(data),
            "original_path": str(p),
            "omnipart_rel": str(Path(target_rel_folder) / f"{name}"),
            "name": name,
            "uuid": uuid,
            "aesthetic_score": None,
            "captions": None,
        }
    else:
        return None


def get_metadata(
    partverse_dir: str,
    max_workers: Optional[int] = 8,
    **kwargs,
):
    part_level_dir = Path(partverse_dir) / "textured_part_glbs"
    assert part_level_dir.exists() and part_level_dir.is_dir()

    part_uuid_dirs = sorted(part_level_dir.glob("*"))
    input_args = []

    for d in part_uuid_dirs:
        uuid = d.name
        target_rel_folder = Path(uuid[:2]) / uuid
        for p in sorted(d.glob("*.glb")):
            input_args.append((uuid, target_rel_folder, p, Path(partverse_dir), True))

    merged_dir = Path(partverse_dir) / "normalized_glbs"
    assert merged_dir.exists() and merged_dir.is_dir()

    for p in merged_dir.glob("*.glb"):
        uuid = p.stem
        target_rel_folder = Path(uuid[:2]) / uuid
        input_args.append((uuid, target_rel_folder, p, Path(partverse_dir), False))

    max_workers = max_workers or os.cpu_count()
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for r in tqdm(
            ex.map(metadata_worker, input_args, chunksize=64),
            total=len(input_args),
        ):
            if r is not None:
                rows.append(r)
                # print(r)

    return pd.DataFrame(rows)


def download(metadata, output_dir, max_workers: int = 8, **kwargs,) -> pd.DataFrame:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    os.makedirs(raw_dir, exist_ok=True)

    downloaded = {}

    max_workers = max_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(metadata), desc="Copying") as pbar:

        def worker(args: Tuple[str, str]):
            src, sha256_val = args
            try:
                sha256 = get_file_hash(src)
                assert sha256 == sha256_val
                dst = raw_dir / f"{sha256}.glb"
                shutil.copy2(src, dst)

                return (sha256, dst.relative_to(output_dir))
            except Exception as e:
                print(f"Error copying {src}: {e}")
                return None
            finally:
                pbar.update()

        results = list(executor.map(worker, zip(metadata["original_path"], metadata["sha256"])))

    for r in results:
        if r is not None:
            sha256, relpath = r
            downloaded[sha256] = relpath

    return pd.DataFrame(downloaded.items(), columns=["sha256", "local_path"])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc="Processing objects") -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    # load metadata
    metadata = metadata.to_dict("records")

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(metadata), desc=desc) as pbar:

            def worker(metadatum):
                try:
                    local_path = metadatum["local_path"]
                    sha256 = metadatum["sha256"]
                    file = os.path.join(output_dir, local_path)
                    record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()

            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")

    return pd.DataFrame.from_records(records)

