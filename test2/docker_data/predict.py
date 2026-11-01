import os
import torch
import torchvision
import torchvision.transforms as transforms
from train_cifar10 import SimpleVisionCNN, unnormalize

def main():
    print("=" * 60)
    print("  PyTorch ROCm Model Prediction & Inspection Tool")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_path = '/app/cifar10_cnn.pt'
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        print("Please run `python train_cifar10.py` first to train the model!")
        return

    # Load model architecture & saved weights
    model = SimpleVisionCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print("Loaded trained model weights successfully.")

    # Load test dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_set = torchvision.datasets.CIFAR10(root='/app/data', train=False, download=True, transform=transform_test)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Run predictions on 5 random samples
    indices = torch.randint(0, len(test_set), (5,)).tolist()
    print(f"\nEvaluating 5 random test samples from test set...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            img_tensor, true_label = test_set[idx]
            img_batch = img_tensor.unsqueeze(0).to(device)  # Add batch dimension [1, 3, 32, 32]
            
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

