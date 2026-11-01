FROM rocm/pytorch:latest

RUN pip install ultralytics opencv-python-headless
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libxcb1

ENV HSA_OVERRIDE_GFX_VERSION=12.0.0
