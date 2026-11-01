import torch
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("ROCm PyTorch GPU Training Test")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA / ROCm Available: {torch.cuda.is_available()}")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
    else:
        print("WARNING: GPU not detected, falling back to CPU.")
    print("=" * 60)

    # 1. Load base YOLOv8 model
    print("\n[1/3] Loading YOLOv8 nano model (yolov8n.pt)...")
    model = YOLO('yolov8n.pt')

    # 2. Train on sample mini-dataset (coco8 is a built-in 8-image dataset for quick verification)
    print("\n[2/3] Starting training on sample dataset (coco8)...")
    results = model.train(
        data='coco8.yaml',       # Auto-downloads Ultralytics' mini test dataset (8 images)
        epochs=5,                # Quick 5-epoch test to verify forward/backward pass & GPU kernels
        imgsz=640,
        batch=4,
        device=device,
        project='/app',
        name='train_output',
        exist_ok=True,           # Overwrite train_output folder instead of creating train_output2, etc.
        workers=4                # Takes advantage of host IPC & shared memory
    )

    # 3. Validate trained weights
    print("\n[3/3] Validating the fine-tuned model...")
    val_results = model.val()

    print("\n" + "=" * 60)
    print("Training test finished successfully!")
    print("Trained weights and metrics saved to: /app/train_output")
    print("=" * 60)

if __name__ == '__main__':
    main()

