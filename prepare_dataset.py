"""Build a CycleGAN dataset from two folders of unpaired images.

Collects images from two source folders (searched recursively), splits each
into train/test, and writes them in the layout cyclegan.py expects:

    <output>/
    |-- trainA/  trainB/
    `-- testA/   testB/

Images are center-cropped to a square before resizing, so faces are not
stretched by the non-square resize the training pipeline would otherwise do.

Usage:
    python prepare_dataset.py \
        --domain-a ~/Downloads/celeba \
        --domain-b preprocess/images \
        --output datasets/photo2anime \
        --limit 5000
"""

import argparse
import pathlib
import random
import statistics

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(directory):
    directory = pathlib.Path(directory).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def square_crop(image):
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def convert(source, destination, size):
    with Image.open(source) as image:
        side = min(image.size)
        converted = square_crop(image.convert("RGB")).resize(
            (size, size), Image.LANCZOS
        )
        converted.save(destination, quality=95)
    return side


def write_split(files, out_dir, size):
    out_dir.mkdir(parents=True, exist_ok=True)
    sides = []
    for index, source in enumerate(files):
        try:
            sides.append(convert(source, out_dir / f"{index:06d}.jpg", size))
        except OSError as error:
            print(f"  skipped {source.name}: {error}")
    return sides


def prepare_domain(source_dir, label, output_root, test_count, limit, size, rng):
    files = find_images(source_dir)
    if len(files) <= test_count:
        raise ValueError(
            f"domain {label}: found {len(files)} images in {source_dir}, "
            f"need more than --test-count ({test_count})"
        )

    rng.shuffle(files)
    if limit:
        files = files[:limit]

    print(f"domain {label}: {len(files)} images from {source_dir}")
    sides = write_split(files[:test_count], output_root / f"test{label}", size)
    sides += write_split(files[test_count:], output_root / f"train{label}", size)

    median_side = int(statistics.median(sides))
    print(
        f"  wrote {len(files) - test_count} train / {test_count} test "
        f"at {size}x{size} (median source side: {median_side}px)"
    )
    if median_side < size:
        print(
            f"  WARNING: sources are smaller than {size}px, so they are being "
            f"upscaled. Expect blurry output in this domain."
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain-a", required=True, help="source folder for domain A")
    parser.add_argument("--domain-b", required=True, help="source folder for domain B")
    parser.add_argument("--output", required=True, help="dataset root to create")
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument(
        "--limit", type=int, default=0, help="max images per domain (0 = all)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=286,
        help="output size; 286 matches the random-jitter size used in training",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = pathlib.Path(args.output).expanduser()
    rng = random.Random(args.seed)

    for source_dir, label in ((args.domain_a, "A"), (args.domain_b, "B")):
        prepare_domain(
            source_dir,
            label,
            output_root,
            args.test_count,
            args.limit,
            args.size,
            rng,
        )

    print(f"\ndataset ready: {output_root}")


if __name__ == "__main__":
    main()
