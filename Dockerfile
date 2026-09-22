FROM tensorflow/tensorflow:2.21.0-gpu

WORKDIR /workspace

COPY requirements.txt .
# TensorFlow (and a compatible numpy) is already provided by the base image;
# only matplotlib is missing. Its numpy version pin targets Python 3.12+ and
# doesn't match this image's Python 3.11, so let pip resolve a compatible numpy.
RUN pip install --no-cache-dir matplotlib==3.11.2

COPY pix2pix.py .
COPY preprocess/outdata /workspace/data/outdata

ENTRYPOINT ["python", "pix2pix.py"]
CMD ["--data-dir", "/workspace/data/outdata", "--steps", "40000", \
     "--checkpoint-dir", "/workspace/output/training_checkpoints", \
     "--log-dir", "/workspace/output/logs/", \
     "--sample-dir", "/workspace/output/samples"]
