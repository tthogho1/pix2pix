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
| `k8s/` | Kubernetes manifests: training Job, PVC, storage and GPU plugins |
| `application/` | FastAPI inference app with a drag & drop web UI |

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

The app applies the same preprocessing as the Go tool (resize → grayscale →
Canny), then runs the generator on the resulting edge map.

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
| `--output` | `output/result.png` | where to write the generated photo |
| `--checkpoint-dir` | `../training_checkpoints` | directory holding the trained checkpoint |
| `--save-edges` | — | also save the intermediate edge map |
| `--serve` / `--host` / `--port` | `127.0.0.1:8000` | run the web UI |

The model loads once at startup and is reused across requests.
`POST /api/convert` takes a multipart `file` field and returns the edge map and
the generated image as base64 data URIs.

Note that the generator runs with `training=True` at inference, keeping dropout
active — pix2pix uses it as its noise source, so the same input yields slightly
different outputs each time.
