"""Face photo -> pix2pix (edges2face) inference app.

Preprocesses an input face photo the same way preprocess/main.go builds
training pairs (resize -> grayscale -> Canny edges), then runs the trained
generator to synthesize a photo from that edge map.

Usage:
    python infer.py --input face.jpg --output output/result.png
    python infer.py --serve                 # web UI on http://127.0.0.1:8000
"""

import argparse
import base64
import io
import pathlib
import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(REPO_ROOT))

from pix2pix import Generator  # noqa: E402

IMG_SIZE = 256
EDGE_THRESH = 40


def detect_edges(image_bgr, img_size=IMG_SIZE, edge_thresh=EDGE_THRESH):
    """Mirror preprocess/main.go: resize -> grayscale -> Canny edges."""
    resized = cv2.resize(
        image_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_thresh, edge_thresh * 3)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def preprocess(image_path, img_size=IMG_SIZE, edge_thresh=EDGE_THRESH):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    return detect_edges(image, img_size, edge_thresh)


def normalize(image):
    return (image.astype(np.float32) / 127.5) - 1.0


def denormalize(image):
    image = (image * 0.5 + 0.5) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def load_generator(checkpoint_dir):
    generator = Generator()
    checkpoint = tf.train.Checkpoint(generator=generator)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")
    status = checkpoint.restore(latest).expect_partial()
    status.assert_existing_objects_matched()
    print(f"restored generator from {latest}")
    return generator


def translate(generator, edges):
    input_tensor = tf.expand_dims(normalize(edges), axis=0)
    # training=True keeps dropout active, which pix2pix uses as its noise
    # source at inference time too (same as generate_images in pix2pix.py).
    prediction = generator(input_tensor, training=True)
    return denormalize(prediction[0].numpy())


def to_data_uri(image_rgb):
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def create_app(checkpoint_dir):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    generator = load_generator(checkpoint_dir)

    app = FastAPI(title="pix2pix face translator")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/convert")
    async def convert(file: UploadFile = File(...)):
        data = await file.read()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(
                status_code=400, detail="could not decode the upload as an image"
            )

        edges = detect_edges(image)
        result = translate(generator, edges)
        return {"edges": to_data_uri(edges), "result": to_data_uri(result)}

    return app


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="path to a face photo")
    parser.add_argument(
        "--output", default="output/result.png", help="path to write the generated photo"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "training_checkpoints"),
        help="directory containing the trained checkpoint",
    )
    parser.add_argument(
        "--save-edges",
        default=None,
        help="optional path to also save the intermediate edge map",
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

    if args.serve:
        import uvicorn

        uvicorn.run(create_app(args.checkpoint_dir), host=args.host, port=args.port)
        return

    edges = preprocess(args.input)

    if args.save_edges:
        pathlib.Path(args.save_edges).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(edges).save(args.save_edges)
        print(f"saved edge map to {args.save_edges}")

    generator = load_generator(args.checkpoint_dir)
    output_image = translate(generator, edges)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_image).save(output_path)
    print(f"saved generated photo to {output_path}")


if __name__ == "__main__":
    main()
