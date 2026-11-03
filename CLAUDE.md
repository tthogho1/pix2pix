# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A face photo → anime experiment comparing four image-translation approaches, served side by side
by one inference app. Two models are **trained here** (pix2pix on paired edge→image data, CycleGAN
on unpaired photo/anime data); two are **pretrained references** (AnimeGANv2 via `torch.hub`, and an
anime SD 1.5 + ControlNet via Hugging Face `diffusers`). `README.md` has the full walkthrough and the
reasoning for why each model exists — read it before changing model behaviour.

There are no tests, linter config, or build scripts. Verification is running the scripts.

## Commands

Python deps live in `.venv` (`source .venv/bin/activate`). Two separate requirement sets:
`requirements.txt` (training + dataset tools, TensorFlow) and `application/requirements.txt`
(inference app: TF + torch + diffusers + FastAPI).

```bash
# pix2pix dataset: Go + GoCV (needs `brew install opencv@4` and PKG_CONFIG_PATH set, see README)
cd preprocess && go run main.go -input <photos> -output <pairs dir> [-edge-threshold 40]

# CycleGAN dataset
python download_selfie2anime.py --output datasets/selfie2anime
python prepare_dataset.py --domain-a <photos> --domain-b <anime> --output datasets/mine

# Training (quick sanity run: use a small --steps, e.g. 1000)
python pix2pix.py  --data-dir datasets/anime_pairs  --steps 40000 --buffer-size 3400
python cyclegan.py --data-dir datasets/selfie2anime --steps 20000 --total-steps 40000 --restore

# Inference (run from application/ — default checkpoint paths are relative: ../training_checkpoints, ../cyclegan_checkpoints)
cd application
python infer.py --serve                              # http://127.0.0.1:8000
python infer.py --input face.jpg --model pix2pix,cyclegan   # explicit list; fails loudly instead of skipping
```

GPU training runs on Kubernetes: `Dockerfile` (pix2pix) and `Dockerfile.cyclegan` bake the dataset
into the image, and `k8s/job.yaml` / `k8s/cyclegan-job.yaml` run them, writing to a PVC at
`/workspace/output`. Note the Job `args` override the Dockerfile `CMD` defaults.

The inference app is also deployed publicly on Google Cloud Run (`space/deploy_cloudrun.sh`, CPU,
all four models, diffusion at 12 steps), built from `space/Dockerfile`, which was written for an HF Docker Space. That
Space is blocked because the free CPU tier needs HF PRO; `space/deploy.py` still pushes it.
`space/export_weights.py` uploads generator-only weights to `tthogho1/face2anime-weights`, which the
image downloads at build time. Any change to `infer.py`, the training scripts' `Generator`, or the
UI needs a redeploy. x86 TensorFlow won't run under emulation on Apple Silicon (no AVX), so test
the image locally as a native arm64 build with `tensorflow` in place of `tensorflow-cpu`.

## Architecture notes that span files

- **`application/infer.py` imports the training scripts.** It adds the repo root to `sys.path` and
  does `from pix2pix import Generator` / `from cyclegan import Generator`, then restores checkpoints
  into them. Any change to a generator's architecture in `pix2pix.py` or `cyclegan.py` breaks
  loading of existing checkpoints in `training_checkpoints*/` / `cyclegan_checkpoints*/`. Both training
  scripts must stay importable without side effects (all work under `main()`).
- **Checkpoint object names are a contract.** pix2pix restores `tf.train.Checkpoint(generator=...)`;
  CycleGAN restores `generator_g=...` (the A→B generator). Renaming the attributes in the training
  scripts' `tf.train.Checkpoint(...)` breaks inference.
- **Preprocessing must match between training and inference.**
  - pix2pix pairs are 512×256 JPEGs: **left half = target image, right half = Canny edges (input)**;
    `load()` in `pix2pix.py` splits them. `Pix2PixModel.detect_edges` in `infer.py` reimplements
    `preprocess/main.go`'s Canny step (threshold 40, high = 3×low) — keep them in sync.
  - `infer.py`'s `square_crop` mirrors the center-crop in `prepare_dataset.py`.
  - Models are run with `training=True` at inference on purpose (pix2pix: dropout as noise;
    CycleGAN: instance-norm statistics as during training).
- **CycleGAN direction follows the folders**: G maps `trainA` → `trainB`, so real photos go in `trainA`.
- **CycleGAN LR schedule**: `--total-steps` defines the linear-decay schedule (decays over the second
  half) and is driven by the optimizers' checkpointed iteration counts; `--steps` is only how far the
  current run goes. With `--restore`, the loop's step counter and sample filenames restart at 0, but
  the schedule resumes correctly. The current `Generator` is a ResNet (9 residual blocks, reflection
  padding, LSGAN loss, image buffer); old U-Net CycleGAN checkpoints cannot be loaded. (The README's
  sentence about CycleGAN sharing pix2pix's U-Net predates this.)
- **`load_models` skips failing models** when `--model all`, so a missing/mismatched checkpoint shows
  up only as a `skipping ...` log line, not an error.

## Repo hygiene

Checkpoints (~686 MB pix2pix, ~1.4 GB CycleGAN each), samples, logs, `datasets/`, and
`preprocess/{images,outdata}` are gitignored. `aws-cluster.conf` and kubeconfigs are cluster
credentials — never commit or print them.
