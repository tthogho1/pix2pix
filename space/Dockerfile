FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels: the free Space has no GPU, and the CUDA ones are several GB.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Spaces run the container as uid 1000.
# The app directory is created and chowned here because the classic Docker
# builder (which Cloud Build uses) makes a missing WORKDIR as root, even after
# USER, and the downloads below then cannot write into it.
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown user:user /home/user/app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Downloaded at build time so the Space starts without waiting on them. New
# weights in the model repo need a "Factory rebuild" to be picked up.
RUN hf download tthogho1/face2anime-weights --local-dir weights
# trust_repo: torch.hub downloads and executes this GitHub repo's model code.
RUN python -c "import torch; torch.hub.load('bryandlee/animegan2-pytorch:main', 'generator', pretrained='face_paint_512_v2', trust_repo=True)"

# The diffusion model's files, into the Hugging Face cache from_pretrained reads.
# Only what DiffusionModel loads: the fp32 safetensors of each component (the
# repos also hold .bin copies, fp16 variants and whole-model checkpoints, which
# would add ~20 GB). About 5.7 GB in all.
RUN python -c "from huggingface_hub import snapshot_download as d; \
d('gsdf/Counterfeit-V2.5', allow_patterns=['model_index.json', 'scheduler/*', 'tokenizer/*', \
  'text_encoder/config.json', 'text_encoder/model.safetensors', \
  'unet/config.json', 'unet/diffusion_pytorch_model.safetensors', \
  'vae/config.json', 'vae/diffusion_pytorch_model.safetensors']); \
d('lllyasviel/control_v11p_sd15_canny', allow_patterns=['config.json', 'diffusion_pytorch_model.safetensors'])"
# Everything is in the image now; a missing file should fail startup rather
# than quietly download gigabytes on every cold start.
ENV HF_HUB_OFFLINE=1

COPY --chown=user pix2pix.py cyclegan.py ./
COPY --chown=user application ./application

EXPOSE 7860
CMD ["python", "application/infer.py", "--serve", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--model", "pix2pix,cyclegan,animegan,diffusion", \
     "--diffusion-steps", "12", \
     "--pix2pix-checkpoint-dir", "weights/pix2pix", \
     "--cyclegan-checkpoint-dir", "weights/cyclegan"]
