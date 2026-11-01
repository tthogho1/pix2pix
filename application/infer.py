"""Face photo -> anime inference app, serving four models side by side.

- pix2pix:   trained here on paired data; takes Canny edges, so colour is
             discarded and reinvented.
- cyclegan:  trained here on unpaired data; takes the photo directly, keeping
             its colours.
- animegan:  pretrained AnimeGANv2 (face_paint_512_v2), via torch.hub.
- diffusion: an anime Stable Diffusion checkpoint steered by a ControlNet —
             the same edge conditioning as pix2pix, but on a pretrained prior.

Running them together on one input is the point: the pretrained model is a
reference to judge our own training against, and the two trained here differ
in method, so the comparison shows what each approach buys.

All three see the same center-cropped square, resized to whatever each was
trained at; pix2pix then reduces it to edges.

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

class Pix2PixModel:
    """Edges to anime, the paired model.

    Unlike the other two this does not read the photo directly: it was trained
    on Canny edge maps, so the photo is reduced to edges first, the same way
    preprocess/main.go builds its training pairs. Colour is discarded along the
    way and invented again by the generator.
    """

    name = "pix2pix"
    label = "pix2pix (edges, trained here)"
    size = 256
    edge_threshold = 40

    def __init__(self, checkpoint_dir):
        import tensorflow as tf

        from pix2pix import Generator

        self.tf = tf
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(f"no checkpoint in {checkpoint_dir}")

        generator = Generator()
        checkpoint = tf.train.Checkpoint(generator=generator)
        checkpoint.restore(latest).expect_partial().assert_existing_objects_matched()

        self.generator = generator
        print(f"restored pix2pix generator from {latest}")

    def detect_edges(self, image_rgb):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 3)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def translate(self, image_rgb):
        edges = self.detect_edges(image_rgb)
        tensor = self.tf.expand_dims(normalize(edges), axis=0)
        # training=True keeps dropout active, which pix2pix uses as its noise
        # source at inference time too.
        prediction = self.generator(tensor, training=True)
        return denormalize(prediction[0].numpy())


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


class DiffusionModel:
    """An anime diffusion model steered by a ControlNet.

    This is the same idea as pix2pix — generate an image from an edge map —
    and the comparison is the point of including it. pix2pix had to learn
    every colour and texture from 3,400 anime frames; this reads the edge map
    as a constraint on a model pretrained on billions of images, so the things
    absent from our training set, which is most of what a real photo holds,
    are still drawable.

    `conditioning_scale` is the dial between the two ways this goes wrong. At
    1.0 the photo's structure is held so tightly that the face stays
    photo-shaped; near 0 the anime style takes over but the likeness goes.
    """

    name = "diffusion"
    label = "ControlNet + anime diffusion"
    size = 512

    # SD 1.5 anime checkpoints in diffusers layout, neither gated.
    BASE_MODELS = {
        "counterfeit": "gsdf/Counterfeit-V2.5",
        "anything": "stablediffusionapi/anything-v5",
    }
    # Canny is the default because it is the very edge map pix2pix is fed, so
    # the two panels differ only in what sits behind the edges. Lineart traces
    # photos more cleanly and usually looks better, but needs controlnet_aux.
    CONTROLNETS = {
        "canny": "lllyasviel/control_v11p_sd15_canny",
        "lineart": "lllyasviel/control_v11p_sd15_lineart",
    }

    DEFAULT_PROMPT = (
        "masterpiece, best quality, anime style portrait, detailed face, "
        "clean lineart, cel shading, vibrant colors"
    )
    DEFAULT_NEGATIVE = (
        "photorealistic, realistic, 3d render, lowres, bad anatomy, "
        "worst quality, low quality, blurry, watermark, text, signature"
    )

    def __init__(
        self,
        base="counterfeit",
        control="canny",
        prompt=None,
        negative_prompt=None,
        steps=25,
        guidance=7.5,
        conditioning_scale=0.7,
        edge_threshold=40,
    ):
        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        self.torch = torch
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.negative_prompt = negative_prompt or self.DEFAULT_NEGATIVE
        self.steps = steps
        self.guidance = guidance
        self.conditioning_scale = conditioning_scale
        self.edge_threshold = edge_threshold
        self.control = control
        self.detector = None

        base_id = self.BASE_MODELS.get(base, base)
        control_id = self.CONTROLNETS[control]
        self.label = f"{control} ControlNet + {base_id.split('/')[-1]}"

        if torch.cuda.is_available():
            device, dtype = "cuda", torch.float16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float16
        else:
            device, dtype = "cpu", torch.float32

        controlnet = ControlNetModel.from_pretrained(control_id, torch_dtype=dtype)
        # These checkpoints ship no safety checker, so the pipeline has to be
        # told not to look for one.
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            base_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        pipeline.set_progress_bar_config(disable=True)
        self.pipeline = pipeline.to(device)

        if control == "lineart":
            from controlnet_aux import LineartDetector

            self.detector = LineartDetector.from_pretrained("lllyasviel/Annotators")

        print(f"loaded {control_id} + {base_id} on {device}")

    def control_image(self, image_rgb):
        if self.detector is not None:
            return self.detector(Image.fromarray(image_rgb), detect_resolution=self.size,
                                 image_resolution=self.size)
        # The same thresholds preprocess/main.go uses, so pix2pix and this see
        # the same lines.
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 3)
        return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

    def translate(self, image_rgb):
        output = self.pipeline(
            self.prompt,
            image=self.control_image(image_rgb),
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            controlnet_conditioning_scale=self.conditioning_scale,
        ).images[0]
        return np.array(output)


MODEL_NAMES = ["pix2pix", "cyclegan", "animegan", "diffusion"]


def load_models(selection, pix2pix_dir, cyclegan_dir, diffusion_options=None):
    """Load the requested models, skipping ones that are unavailable.

    A checkpoint can easily be missing or belong to an older architecture while
    a model is being retrained, so serving the rest beats refusing to start —
    unless that model was the one asked for by name.
    """
    wanted = MODEL_NAMES if selection == "all" else [selection]
    builders = {
        "pix2pix": lambda: Pix2PixModel(pix2pix_dir),
        "cyclegan": lambda: CycleGanModel(cyclegan_dir),
        "animegan": AnimeGanModel,
        "diffusion": lambda: DiffusionModel(**(diffusion_options or {})),
    }

    models = []
    for name in wanted:
        try:
            models.append(builders[name]())
        except Exception as error:
            if selection != "all":
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
        default="all",
        choices=["all"] + MODEL_NAMES,
        help="which model(s) to run",
    )
    parser.add_argument(
        "--pix2pix-checkpoint-dir",
        default=str(REPO_ROOT / "training_checkpoints"),
        help="directory holding the trained pix2pix checkpoint",
    )
    parser.add_argument(
        "--cyclegan-checkpoint-dir",
        default=str(REPO_ROOT / "cyclegan_checkpoints"),
        help="directory holding the trained CycleGAN checkpoint",
    )
    parser.add_argument(
        "--diffusion-base",
        default="counterfeit",
        help="anime SD 1.5 checkpoint: a key of DiffusionModel.BASE_MODELS, "
        "or any Hugging Face repo id in diffusers layout",
    )
    parser.add_argument(
        "--diffusion-control",
        default="canny",
        choices=list(DiffusionModel.CONTROLNETS),
        help="edge detector and matching ControlNet; lineart needs controlnet_aux",
    )
    parser.add_argument(
        "--diffusion-prompt",
        default=None,
        help=f"style prompt (default: {DiffusionModel.DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--diffusion-steps", type=int, default=25, help="denoising steps"
    )
    parser.add_argument(
        "--diffusion-conditioning-scale",
        type=float,
        default=0.7,
        help="how tightly the edges hold the output: 1.0 keeps the photo's "
        "shape, lower lets the anime style reshape it",
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
    models = load_models(
        args.model,
        args.pix2pix_checkpoint_dir,
        args.cyclegan_checkpoint_dir,
        diffusion_options={
            "base": args.diffusion_base,
            "control": args.diffusion_control,
            "prompt": args.diffusion_prompt,
            "steps": args.diffusion_steps,
            "conditioning_scale": args.diffusion_conditioning_scale,
        },
    )

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
