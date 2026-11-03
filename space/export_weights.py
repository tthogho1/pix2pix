"""Export generator-only weights and upload them to a Hugging Face model repo.

The training checkpoints also hold the discriminators and every optimizer's
state, none of which inference needs. This restores just the generator each
app model loads, under the same checkpoint object name infer.py restores
(`generator` for pix2pix, `generator_g` for CycleGAN), and saves it alone, so
the Space downloads ~265 MB instead of ~1 GB.

Usage (from the repo root):
    python space/export_weights.py
    python space/export_weights.py --no-upload        # export only
"""

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf  # noqa: E402


def export(generator_factory, object_name, source_dir, output_dir):
    latest = tf.train.latest_checkpoint(source_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint in {source_dir}")

    generator = generator_factory()
    tf.train.Checkpoint(**{object_name: generator}).restore(
        latest
    ).expect_partial().assert_existing_objects_matched()

    # A manager, not Checkpoint.write, because it also writes the `checkpoint`
    # state file tf.train.latest_checkpoint looks for.
    manager = tf.train.CheckpointManager(
        tf.train.Checkpoint(**{object_name: generator}), output_dir, max_to_keep=1
    )
    path = manager.save()
    print(f"{latest} -> {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="tthogho1/face2anime-weights",
                        help="Hugging Face model repo to upload to")
    parser.add_argument("--pix2pix-checkpoint-dir", default=str(REPO_ROOT / "training_checkpoints"))
    parser.add_argument("--cyclegan-checkpoint-dir", default=str(REPO_ROOT / "cyclegan_checkpoints"))
    parser.add_argument("--output", default=str(REPO_ROOT / "space_weights"))
    parser.add_argument("--no-upload", action="store_true", help="export locally only")
    return parser.parse_args()


def main():
    args = parse_args()
    output = pathlib.Path(args.output)

    import cyclegan
    import pix2pix

    export(pix2pix.Generator, "generator", args.pix2pix_checkpoint_dir, str(output / "pix2pix"))
    export(cyclegan.Generator, "generator_g", args.cyclegan_checkpoint_dir, str(output / "cyclegan"))

    if args.no_upload:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(output),
        commit_message="Upload generator-only weights",
    )
    print(f"uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
