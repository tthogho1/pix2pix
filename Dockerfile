FROM tensorflow/tensorflow:2.21.0-gpu

WORKDIR /workspace

COPY requirements.txt .
# TensorFlow (and a compatible numpy) is already provided by the base image;
# matplotlib and tensorboard (needed for tf.summary.scalar) are missing.
# The numpy/tensorboard version pins target Python 3.12+ and don't match this
# image's Python 3.11, so let pip resolve compatible versions instead.
RUN pip install --no-cache-dir matplotlib==3.11.2 tensorboard

# The base image's system cuDNN (8.9.6, Oct 2023) is too old for this
# TensorFlow build: cudnnCreate() succeeds when called directly, but TF's own
# internal version check rejects it, silently falling back to CPU. Installing
# the current pip-distributed cuDNN (9.x) alongside it fixes GPU detection.
RUN pip install --no-cache-dir nvidia-cudnn-cu12

COPY pix2pix.py .
COPY preprocess/outdata /workspace/data/outdata

ENTRYPOINT ["python", "pix2pix.py"]
CMD ["--data-dir", "/workspace/data/outdata", "--steps", "40000", \
     "--checkpoint-dir", "/workspace/output/training_checkpoints", \
     "--log-dir", "/workspace/output/logs/", \
     "--sample-dir", "/workspace/output/samples"]
