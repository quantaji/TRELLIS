import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import huggingface_hub
import pandas as pd
from tqdm import tqdm
from utils import get_file_hash


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Data folder that contains all glb)",
    )


def get_metadata(
    input_dir: str,
    **kwargs,
):
    root = Path(input_dir).expanduser().resolve()
    glbs = sorted(root.glob("*.glb"))
    rows = [
        {
            "sha256": sha256_file(p),
            "file_identifier": str(p.resolve()),
            "aesthetic_score": None,
            "captions": "[]",
        }
        for p in glbs
    ]
    return pd.DataFrame(rows)


def download(metadata, output_dir, **kwargs):
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True, parents=True)

    downloaded = {}
    metadata = metadata.set_index("file_identifier")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, tqdm(total=len(metadata), desc="Copying") as pbar:

        def worker(src: str):
            try:
                sha256 = get_file_hash(src)
                dst = raw_dir / f"{sha256}{Path(src).suffix}"
                shutil.copy2(src, dst)  # 会覆盖同名文件

                return (sha256, dst.relative_to(output_dir))
            except Exception as e:
                print(f"Error copying {src}: {e}")
                return None
            finally:
                pbar.update()

        results = list(executor.map(worker, metadata.index))

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
