import torch
from ultralytics import YOLO

print("Running on GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Load YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# Run prediction explicitly targeting ROCm GPU
results = model.predict(
    source='https://ultralytics.com/images/bus.jpg',
    device='cuda:0',
    save=True,
    project='/app',
    name='output'
)

print("\nInference finished! Results saved to /app/output.")
