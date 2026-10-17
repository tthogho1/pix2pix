"""Face photo -> anime inference app, serving two models side by side.

- cyclegan: the generator trained in this repo, from a local checkpoint.
- animegan: pretrained AnimeGANv2 (face_paint_512_v2), loaded via torch.hub.

Running both on the same input is the point: the pretrained model gives a
reference to judge our own training against, which sample images alone cannot.

Both see the same center-cropped square, resized to whatever each was trained
at. Neither extracts edges — the photo's own colors carry through.

Usage:
    python infer.py --input face.jpg
    python infer.py --serve                 # web UI on http://127.0.0.1:8000
"""

import argparse
import base64
import io
import pathlib
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(REPO_ROOT))

DISPLAY_SIZE = 512


def square_crop(image_bgr):
    """Match prepare_dataset.py, so inference sees the framing training did."""
    height, width = image_bgr.shape[:2]
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image_bgr[top : top + side, left : left + side]


def to_rgb_input(cropped_bgr, size):
    resized = cv2.resize(cropped_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def read_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    return image


def normalize(image):
    return (image.astype(np.float32) / 127.5) - 1.0


def denormalize(image):
    image = (image * 0.5 + 0.5) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CycleGanModel:
    name = "cyclegan"
    label = "CycleGAN (trained here)"
    size = 256

    def __init__(self, checkpoint_dir):
        import tensorflow as tf

        from cyclegan import Generator

        self.tf = tf
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")

        generator = Generator()
        checkpoint = tf.train.Checkpoint(generator_g=generator)
        try:
            checkpoint.restore(latest).expect_partial().assert_existing_objects_matched()
        except (AssertionError, ValueError) as error:
            raise RuntimeError(
                f"{latest} does not fit the current generator. Checkpoints from "
                f"the old U-Net architecture cannot be loaded by the ResNet one; "
                f"retrain or point --checkpoint-dir elsewhere. ({error})"
            ) from error

        self.generator = generator
        print(f"restored generator_g from {latest}")

    def translate(self, image_rgb):
        tensor = self.tf.expand_dims(normalize(image_rgb), axis=0)
        # training=True keeps instance-norm using batch statistics, which is
        # how the generator ran during training and in its samples.
        prediction = self.generator(tensor, training=True)
        return denormalize(prediction[0].numpy())


class AnimeGanModel:
    name = "animegan"
    label = "AnimeGANv2 (pretrained)"
    size = 512

    def __init__(self):
        import torch

        self.torch = torch
        # trust_repo skips the interactive prompt. Note what it consents to:
        # torch.hub downloads and executes model code from this GitHub repo.
        self.model = torch.hub.load(
            "bryandlee/animegan2-pytorch:main",
            "generator",
            pretrained="face_paint_512_v2",
            trust_repo=True,
        )
        self.model.eval()
        print("loaded AnimeGANv2 face_paint_512_v2")

    def translate(self, image_rgb):
        torch = self.torch
        tensor = torch.from_numpy(normalize(image_rgb)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
        return denormalize(output.squeeze(0).permute(1, 2, 0).numpy())


def load_models(selection, checkpoint_dir):
    """Load the requested models, skipping ones that are unavailable.

    A missing CycleGAN checkpoint is expected while retraining, so serving
    only AnimeGANv2 beats refusing to start — unless it was asked for by name.
    """
    wanted = ["cyclegan", "animegan"] if selection == "both" else [selection]

    models = []
    for name in wanted:
        try:
            models.append(
                CycleGanModel(checkpoint_dir) if name == "cyclegan" else AnimeGanModel()
            )
        except Exception as error:
            if selection != "both":
                raise
            print(f"skipping {name}: {error}")

    if not models:
        raise RuntimeError("no models could be loaded")
    return models


def to_data_uri(image_rgb):
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def convert(models, image_bgr):
    cropped = square_crop(image_bgr)
    return [
        {
            "name": model.name,
            "label": model.label,
            "image": model.translate(to_rgb_input(cropped, model.size)),
        }
        for model in models
    ]


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------

def create_app(models):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="face to anime")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/convert")
    async def api_convert(file: UploadFile = File(...)):
        data = await file.read()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(
                status_code=400, detail="could not decode the upload as an image"
            )

        results = convert(models, image)
        return {
            "source": to_data_uri(to_rgb_input(square_crop(image), DISPLAY_SIZE)),
            "results": [
                {
                    "name": r["name"],
                    "label": r["label"],
                    "image": to_data_uri(r["image"]),
                }
                for r in results
            ],
        }

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="path to a face photo")
    parser.add_argument(
        "--output",
        default="output/result.png",
        help="output path; the model name is appended to the stem",
    )
    parser.add_argument(
        "--model",
        default="both",
        choices=["both", "cyclegan", "animegan"],
        help="which model(s) to run",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "cyclegan_checkpoints"),
        help="directory holding the trained CycleGAN checkpoint",
    )
    parser.add_argument(
        "--serve", action="store_true", help="run the web UI instead of converting once"
    )
    parser.add_argument("--host", default="127.0.0.1", help="web UI bind address")
    parser.add_argument("--port", type=int, default=8000, help="web UI port")

    args = parser.parse_args()
    if not args.serve and not args.input:
        parser.error("--input is required unless --serve is given")
    return args


def main():
    args = parse_args()
    models = load_models(args.model, args.checkpoint_dir)

    if args.serve:
        import uvicorn

        uvicorn.run(create_app(models), host=args.host, port=args.port)
        return

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for result in convert(models, read_image(args.input)):
        path = output.with_name(f"{output.stem}_{result['name']}{output.suffix}")
        Image.fromarray(result["image"]).save(path)
        print(f"saved {result['label']} output to {path}")


if __name__ == "__main__":
    main()
