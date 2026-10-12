"""Download the selfie2anime dataset into the layout cyclegan.py expects.

selfie2anime is the standard benchmark for unpaired real-face -> anime-face
translation, introduced with U-GAT-IT: 3,400 training and 100 test images per
domain, all 256x256. This pulls the Hugging Face parquet mirror, which needs no
authentication, and writes:

    <output>/
    |-- trainA/  real selfies     trainB/  anime faces
    `-- testA/                    testB/

Usage:
    python download_selfie2anime.py --output datasets/selfie2anime
"""

import argparse
import io
import pathlib
import urllib.request

import pyarrow.parquet as pq
from PIL import Image

PARQUET_URL = (
    "https://huggingface.co/api/datasets/huggan/selfie2anime/parquet"
    "/default/{split}/0.parquet"
)


def download(url, destination):
    if destination.exists():
        print(f"using cached {destination.name}")
        return

    print(f"downloading {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download to a .part file so an interrupted transfer is never mistaken
    # for a complete cache entry on the next run.
    partial = destination.with_suffix(".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as out:
        total = int(response.headers.get("content-length", 0))
        read = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            read += len(chunk)
            print(f"\r  {read / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    partial.rename(destination)


def extract(parquet_path, out_a, out_b):
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)

    index = 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            for column, out_dir in (("imageA", out_a), ("imageB", out_b)):
                image = Image.open(io.BytesIO(row[column]["bytes"])).convert("RGB")
                image.save(out_dir / f"{index:05d}.jpg", quality=95)
            index += 1
    return index


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/selfie2anime")
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="keep the downloaded parquet files under <output>/.cache",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = pathlib.Path(args.output).expanduser()
    cache = root / ".cache"

    for split, dir_a, dir_b in (
        ("train", "trainA", "trainB"),
        ("test", "testA", "testB"),
    ):
        parquet_path = cache / f"{split}.parquet"
        download(PARQUET_URL.format(split=split), parquet_path)
        count = extract(parquet_path, root / dir_a, root / dir_b)
        print(f"{split}: {count} images each into {dir_a}/ and {dir_b}/")

    if not args.keep_cache:
        for parquet_path in cache.glob("*.parquet"):
            parquet_path.unlink()
        cache.rmdir()

    print(f"\ndataset ready: {root}")
    print(f"next: python cyclegan.py --data-dir {root}")


if __name__ == "__main__":
    main()
