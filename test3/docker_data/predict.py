import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def main():
    print("=" * 60)
    print("  Test 3: ResNet18 Fine-Tuned Model Inference Tool")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_path = '/app/resnet18_cifar10_finetuned.pt'
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        print("Please run `python train_transfer.py` first!")
        return

    # Load ResNet18 structure matching the fine-tuned model
    model = models.resnet18()
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 10)
    )
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Loaded fine-tuned ResNet18 weights successfully.")

    # ImageNet normalization with resize to 224x224
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_set = torchvision.datasets.CIFAR10(root='/app/data', train=False, download=True, transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Run predictions on 5 random samples
    indices = torch.randint(0, len(test_set), (5,)).tolist()
    print(f"\nEvaluating 5 random test samples with fine-tuned ResNet18...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, true_label = test_set[idx]
            img_batch = img_tensor.unsqueeze(0).to(device)

            logits = model(img_batch)
            probs = torch.softmax(logits, dim=1)[0]
            top3_prob, top3_idx = torch.topk(probs, 3)

            true_name = classes[true_label]
            top_pred = classes[top3_idx[0].item()]
            is_correct = (top_pred == true_name)

            status = "CORRECT [✓]" if is_correct else "WRONG [✗]"
            print(f"\n--- Sample {i+1} (Index {idx}) | True Label: {true_name.upper()} | {status} ---")
            for rank in range(3):
                cls_name = classes[top3_idx[rank].item()]
                prob_val = top3_prob[rank].item() * 100
                bar_len = int(prob_val // 5)
                bar = '█' * bar_len + ' ' * (20 - bar_len)
                print(f"  {rank+1}. {cls_name:<7} [{bar}] {prob_val:5.1f}%")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
