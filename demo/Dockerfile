# BallHawk on a Hugging Face Space (Docker SDK, free CPU tier) or any Docker host.
# Build context: the folder tools/make_space.sh assembles (app.py, the ballhawk_*.py modules, weights/,
# examples/, web/). The app listens on 7860, the port Spaces expect.
FROM python:3.12-slim

# ultralytics pulls in full opencv-python, which needs these shared libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/tmp/Ultralytics PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image about 2 GB smaller than the default CUDA build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
