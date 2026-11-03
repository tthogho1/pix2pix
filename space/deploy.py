"""Push the inference app to a Hugging Face Docker Space.

The Space gets the files in space/ at its root (README.md carries the Space
config, Dockerfile builds it) plus the code infer.py needs, at the same
relative paths it has here, since infer.py imports pix2pix.py and cyclegan.py
from its parent directory. Weights are not uploaded here: the Dockerfile
downloads them from the model repo export_weights.py fills.

Usage (from the repo root):
    python space/deploy.py
"""

import argparse
import pathlib

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# local path (relative to the repo root) -> path in the Space
FILES = {
    "space/README.md": "README.md",
    "space/Dockerfile": "Dockerfile",
    "space/requirements.txt": "requirements.txt",
    "pix2pix.py": "pix2pix.py",
    "cyclegan.py": "cyclegan.py",
    "application/infer.py": "application/infer.py",
    "application/static/index.html": "application/static/index.html",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="tthogho1/face2anime", help="Space repo id")
    parser.add_argument("--message", default="Deploy the inference app")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.create_commit(
        repo_id=args.repo,
        repo_type="space",
        commit_message=args.message,
        operations=[
            CommitOperationAdd(path_in_repo=target, path_or_fileobj=str(REPO_ROOT / source))
            for source, target in FILES.items()
        ],
    )
    owner, name = args.repo.split("/")
    print(f"pushed https://huggingface.co/spaces/{args.repo}")
    print(f"app: https://{owner}-{name}.hf.space (once the build finishes)")


if __name__ == "__main__":
    main()
