# pix2pix — edges to face

An end-to-end pix2pix (conditional GAN) pipeline that turns Canny edge maps into
face images: dataset preparation in Go, GPU training on Kubernetes, and a FastAPI
web app for inference.

Based on the [TensorFlow pix2pix tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix).

## Pipeline

```
raw photos ──▶ preprocess/ ──▶ outdata/ ──▶ pix2pix.py ──▶ checkpoint ──▶ application/
               (Go + OpenCV)   512x256      (TF, on K8s)                  (FastAPI)
                               pairs
```

Each training sample is a single 512×256 JPEG holding two 256×256 halves:
the **left** half is the original photo (target) and the **right** half is its
Canny edge map (input). `load()` in `pix2pix.py` splits them back apart, so the
generator learns *edges → photo*.

## Layout

| Path | What it is |
|---|---|
| `preprocess/` | Go tool that turns raw photos into paired training images |
| `pix2pix.py` | Training script (U-Net generator + PatchGAN discriminator) |
| `Dockerfile` | GPU training image, with the prepared dataset baked in |
| `k8s/` | Kubernetes manifests: training Jobs, PVCs, storage and GPU plugins |
| `application/` | FastAPI inference app with a drag & drop web UI |
| `cyclegan.py` | CycleGAN training for unpaired translation — see [CycleGAN](#cyclegan-photo--anime) |
| `prepare_dataset.py`, `download_selfie2anime.py` | CycleGAN dataset tooling |

## 1. Prepare the dataset

The Go tool reads every image under `images/`, resizes it to 256×256, runs Canny
edge detection, and writes the combined pair to `outdata/`, preserving the
subfolder structure.

Place your raw photos like this:

```
preprocess/
├── images/
│   ├── train/   raw photos for training
│   └── test/    raw photos for evaluation (or "val/")
└── outdata/     generated automatically
```

`pix2pix.py` looks for `train/*.jpg` and `test/*.jpg` (falling back to `val/*.jpg`),
so those folder names matter.

Requires OpenCV 4.x for [GoCV](https://gocv.io):

```bash
brew install opencv@4
export PKG_CONFIG_PATH="/opt/homebrew/opt/opencv@4/lib/pkgconfig:$PKG_CONFIG_PATH"

cd preprocess
go run main.go
```

### How many images?

Training length is set by `--steps`, not dataset size — the dataset repeats —
so more images buys variety, not longer training. Roughly:

- 400–1,000 pairs: comparable to the original paper's `facades` dataset
- thousands: better generalization for varied subjects
- 20–50 test images is plenty; they are only used to render progress samples

## 2. Train

### Locally

```bash
pip install -r requirements.txt
python pix2pix.py --data-dir preprocess/outdata --steps 40000 --buffer-size 1000
```

Without `--data-dir` the script downloads a public dataset instead
(`--dataset facades`, `cityscapes`, `edges2shoes`, …).

Outputs go to `--checkpoint-dir`, `--sample-dir` and `--log-dir`. A checkpoint is
saved every 5,000 steps and `CheckpointManager` keeps the latest 3 (~686 MB each,
weights plus Adam optimizer state).

### On Kubernetes with a GPU

The image bakes `preprocess/outdata` in, so the Job needs no dataset volume:

```bash
docker build -t docker.io/<user>/pix2pix:latest .
docker push docker.io/<user>/pix2pix:latest

kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/job.yaml
kubectl logs -f job/pix2pix-train
```

The Job requests `nvidia.com/gpu: 1` and writes all output to the `pix2pix-output`
PVC at `/workspace/output`.

Cluster prerequisites:

- **NVIDIA device plugin** — `kubectl apply -f k8s/nvidia-device-plugin.yml`, plus
  the NVIDIA driver and `nvidia-container-toolkit` installed on each GPU node
  (`nvidia-ctk runtime configure --runtime=containerd --set-as-default`).
- **A StorageClass** — bare kubeadm clusters have none, so PVCs stay `Pending`.
  `kubectl apply -f k8s/local-path-storage.yml` installs one.
- **cuDNN** — the base image ships cuDNN 8.9.6, which this TensorFlow build
  rejects during its internal version check, silently falling back to CPU. The
  Dockerfile installs the current pip-distributed cuDNN 9.x to fix GPU detection.

On a Tesla T4, 1,000 steps take about 110 seconds, so a 40,000-step run is roughly
75 minutes.

### Retrieving the results

`kubectl cp` is unreliable for the ~686 MB checkpoint files. Copying straight off
the node backing the PVC is faster and does not truncate:

```bash
kubectl get pv $(kubectl get pvc pix2pix-output -o jsonpath='{.spec.volumeName}') \
  -o jsonpath='{.spec.hostPath.path}'

scp -i <key> <node>:<that path>/training_checkpoints/ckpt-8.* ./training_checkpoints/
```

Only the newest checkpoint is needed: its `.index` and `.data-00000-of-00001` pair.

## 3. Run the inference app

The app runs two models on the same upload and shows the results side by side:

- **cyclegan** — the generator trained in this repo, from a local checkpoint
- **animegan** — pretrained [AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch)
  (`face_paint_512_v2`), fetched through `torch.hub`

Running both is the point: the pretrained model is a reference to judge our own
training against, which sample images alone cannot provide. Note that
`torch.hub` downloads and executes model code from that GitHub repo.

Both see the same center-cropped square — the framing `prepare_dataset.py`
applies — resized to whatever each model was trained at (256 and 512). Neither
extracts edges, so the photo's own colors carry through.

If the CycleGAN checkpoint is missing or was trained with a different
architecture, that model is skipped with a message and the app serves the other
one. Checkpoints from the earlier U-Net generator cannot be loaded by the
current ResNet one.

```bash
cd application
pip install -r requirements.txt

# Web UI at http://127.0.0.1:8000 — upload by file picker or drag & drop
python infer.py --serve

# Or convert a single file
python infer.py --input face.jpg --output output/result.png
```

| Option | Default | Meaning |
|---|---|---|
| `--input` | — | source photo (required unless `--serve`) |
| `--output` | `output/result.png` | output path; the model name is appended to the stem |
| `--model` | `both` | `both`, `cyclegan` or `animegan` |
| `--checkpoint-dir` | `../cyclegan_checkpoints` | directory holding the trained CycleGAN checkpoint |
| `--serve` / `--host` / `--port` | `127.0.0.1:8000` | run the web UI |

Models load once at startup and are reused across requests.
`POST /api/convert` takes a multipart `file` field and returns the cropped
source plus one entry per model, all as base64 data URIs.

Note that the generator runs with `training=True` at inference, keeping dropout
active — pix2pix uses it as its noise source, so the same input yields slightly
different outputs each time.

## CycleGAN: photo → anime

The pix2pix pipeline above cannot turn a real photo into an anime one, for a
reason no amount of training fixes: Canny preprocessing discards all color, so
the generator invents it from its training prior instead of keeping the photo's
own. Translating between two domains that have no paired data is what CycleGAN
is for, and `cyclegan.py` implements it.

It shares pix2pix's U-Net generator and PatchGAN discriminator shapes, but uses
instance normalization and a single-image discriminator (there is no
conditioning pair to judge against), so the networks are defined separately.
Four networks train at once — two generators and two discriminators — which
costs roughly 3–4× pix2pix per step.

### Get a dataset

`download_selfie2anime.py` fetches **selfie2anime**, the standard benchmark for
this task (introduced with U-GAT-IT): 3,400 train + 100 test images per domain,
256×256 in both. It pulls the Hugging Face parquet mirror, so no login is needed.

```bash
pip install -r requirements.txt
python download_selfie2anime.py --output datasets/selfie2anime
```

To use your own images instead, `prepare_dataset.py` builds the same layout from
two folders. It center-crops to square before resizing, so faces are not
stretched, and warns when the sources are smaller than the target size:

```bash
python prepare_dataset.py \
    --domain-a <real photos> \
    --domain-b <anime faces> \
    --output datasets/mine
```

Either way the result is the layout `cyclegan.py` expects:

```
<data-dir>/
├── trainA/  domain A — the input side
├── trainB/  domain B — the output side
├── testA/
└── testB/
```

**The direction follows the folders.** Generator G translates A → B, and that is
what the progress samples and `final_*.png` show. For photo → anime, real photos
must be in `trainA`.

### Train

```bash
python cyclegan.py --data-dir datasets/selfie2anime --steps 40000 --buffer-size 3400
```

On a Tesla T4, expect roughly 400–450 seconds per 1,000 steps, so 40,000 steps
is about 5 hours. Start at 10,000 to confirm it is learning before committing to
a long run — samples land in `--sample-dir` every 1,000 steps.

Checkpoints hold all four networks and their optimizers, so each save is ~1.4 GB
and `CheckpointManager` keeps the newest 3. `--restore` resumes from the newest
one, and is a no-op when there is none.

On Kubernetes, with the same cluster prerequisites as above:

```bash
docker build -f Dockerfile.cyclegan -t docker.io/<user>/cyclegan:latest .
docker push docker.io/<user>/cyclegan:latest

kubectl apply -f k8s/cyclegan-pvc.yaml
kubectl apply -f k8s/cyclegan-job.yaml
kubectl logs -f job/cyclegan-train
```

The Job passes `--restore`, so if a pod dies it resumes from the last checkpoint
rather than starting over. Note that the step counter restarts at 0 on resume:
the run does `--steps` more steps from the restored weights, and sample
filenames begin again at `step_0000000.png`.
