import os
import torch
from ultralytics import YOLO

def main():
    print("=" * 65)
    print("     BEFORE vs AFTER TRAINING COMPARISON (INFERENCE)      ")
    print("=" * 65)

    base_model_path = 'yolov8n.pt'
    trained_model_path = 'train_output/weights/best.pt'

    if not os.path.exists(trained_model_path):
        print(f"Error: Trained weights not found at {trained_model_path}.")
        print("Please run `python3 train.py` first to generate trained weights.")
        return

    # Choose a test image (from validation set or sample bus image)
    test_img = 'datasets/coco8/images/val/000000000036.jpg'
    if not os.path.exists(test_img):
        test_img = 'bus.jpg'

    print(f"Test Image: {test_img}")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 1. Run inference with BASE (BEFORE) model
    print("\n[1/2] Running inference with BASE model (Before Training)...")
    base_model = YOLO(base_model_path)
    res_before = base_model.predict(
        source=test_img,
        device=device,
        save=True,
        project='/app/compare_output',
        name='before',
        exist_ok=True
    )[0]

    # 2. Run inference with TRAINED (AFTER) model
    print("[2/2] Running inference with TRAINED model (After Training)...")
    trained_model = YOLO(trained_model_path)
    res_after = trained_model.predict(
        source=test_img,
        device=device,
        save=True,
        project='/app/compare_output',
        name='after',
        exist_ok=True
    )[0]

    # Print Comparison Table
    print("\n" + "=" * 65)
    print("DETECTION COMPARISON RESULTS:")
    print("=" * 65)

    def print_detections(res, title):
        print(f"\n--- {title} ---")
        boxes = res.boxes
        if len(boxes) == 0:
            print("  No objects detected.")
            return
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = res.names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]
            print(f"  [{i+1}] Class: {cls_name:<15} Confidence: {conf:.2%}   Box: {xyxy}")

    print_detections(res_before, "BEFORE TRAINING (Original yolov8n.pt)")
    print_detections(res_after, "AFTER TRAINING (train_output/weights/best.pt)")

    print("\n" + "=" * 65)
    print("VISUAL IMAGES SAVED:")
    print("  Before: /app/compare_output/before/" + os.path.basename(test_img))
    print("  After:  /app/compare_output/after/" + os.path.basename(test_img))
    print("=" * 65)

if __name__ == '__main__':
    main()

