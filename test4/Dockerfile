FROM rocm/pytorch:latest

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torchvision matplotlib pillow

ENV HSA_OVERRIDE_GFX_VERSION=12.0.0

