import torch
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("ROCm YOLO Heavy Training Test (YOLOv8 Medium)")
    print("=" * 60)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # YOLOv8m (Medium model, ~25.9 million parameters)
    print("\nLoading YOLOv8m model...")
    model = YOLO('yolov8m.pt')

    print("\nStarting heavy training run (50 epochs)...")
    results = model.train(
        data='coco8.yaml',
        epochs=50,               # 50 epochs will sustain load and give time to warm up
        imgsz=640,
        batch=8,
        device=device,
        project='/app',
        name='heavy_train_output',
        exist_ok=True,
        workers=4
    )

    print("\nValidation...")
    model.val()
    print("\nHeavy YOLO training complete!")

if __name__ == '__main__':
    main()

