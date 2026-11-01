import os
import torch
from ultralytics import YOLO

def main():
    print("=" * 65)
    print("    LEARNING FROM SCRATCH DEMO (UNTRAINED vs TRAINED)    ")
    print("=" * 65)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    test_img = 'bus.jpg'
    output_dir = '/app/scratch_demo_output'
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # STEP 1: Load a completely UNTRAINED model (Random Weights)
    # Note: .yaml creates random weights, unlike .pt which has pre-trained weights
    # -------------------------------------------------------------
    print("\n[STEP 1] Creating an UNTRAINED model with random weights (yolov8n.yaml)...")
    untrained_model = YOLO('yolov8n.yaml')

    print("\nRunning inference with UNTRAINED model (BEFORE TRAINING)...")
    res_untrained = untrained_model.predict(
        source=test_img,
        device=device,
        save=True,
        project=output_dir,
        name='1_before_training',
        exist_ok=True
    )[0]

    print(f"-> Detected {len(res_untrained.boxes)} objects (Untrained AI is blind / random).")

    # -------------------------------------------------------------
    # STEP 2: Train the model from scratch on GPU for 30 epochs
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("[STEP 2] Training the model on ROCm GPU for 30 epochs...")
    print("=" * 65)

    untrained_model.train(
        data='coco8.yaml',
        epochs=30,               # 30 epochs gives enough time for learning rate to train weights
        imgsz=640,
        batch=4,
        device=device,
        project=output_dir,
        name='training_run',
        exist_ok=True,
        workers=4
    )

    # -------------------------------------------------------------
    # STEP 3: Load the newly trained weights and test again
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("[STEP 3] Running inference with TRAINED model (AFTER TRAINING)...")
    print("=" * 65)

    trained_weights = os.path.join(output_dir, 'training_run', 'weights', 'best.pt')
    trained_model = YOLO(trained_weights)

    res_trained = trained_model.predict(
        source=test_img,
        device=device,
        save=True,
        project=output_dir,
        name='2_after_training',
        exist_ok=True
    )[0]

    print(f"-> Detected {len(res_trained.boxes)} objects!")

    # -------------------------------------------------------------
    # STEP 4: Compare results
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("SUMMARY COMPARISON:")
    print("=" * 65)
    print(f"BEFORE TRAINING detections: {len(res_untrained.boxes)}")
    print(f"AFTER TRAINING detections:  {len(res_trained.boxes)}")
    
    if len(res_trained.boxes) > 0:
        print("\nTrained model discovered:")
        for i, box in enumerate(res_trained.boxes):
            cls_id = int(box.cls[0].item())
            cls_name = res_trained.names[cls_id]
            conf = float(box.conf[0].item())
            print(f"  [{i+1}] {cls_name} (Confidence: {conf:.1%})")

    print("\n" + "=" * 65)
    print("Inspect the two images side by side:")
    print(f"  Before: {output_dir}/1_before_training/{test_img}")
    print(f"  After:  {output_dir}/2_after_training/{test_img}")
    print("=" * 65)

if __name__ == '__main__':
    main()

