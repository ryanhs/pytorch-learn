import time
import torch
import torch.nn as nn
import torch.optim as optim

def test_pytorch_training():
    print("=" * 60)
    print("Pure PyTorch ROCm Synthetic Training & Benchmark Test")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA/ROCm Available: {cuda_available}")
    
    device = torch.device("cuda:0" if cuda_available else "cpu")
    if cuda_available:
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Running on CPU.")
    print("=" * 60)

    # Simple ConvNet model
    class SimpleConvNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    print("\n[1/3] Initializing model and moving to device...")
    model = SimpleConvNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Benchmark parameters (scaled for high-RAM & GPU testing)
    batch_size = 64
    image_size = 224
    num_batches = 150
    print(f"[2/3] Simulating training loop: {num_batches} batches (Batch: {batch_size}, Image: {image_size}x{image_size})...")

    # Warmup
    for _ in range(5):
        dummy_inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
        dummy_targets = torch.randint(0, 10, (batch_size,), device=device)
        optimizer.zero_grad()
        outputs = model(dummy_inputs)
        loss = criterion(outputs, dummy_targets)
        loss.backward()
        optimizer.step()

    if cuda_available:
        torch.cuda.synchronize()

    start_time = time.time()

    for step in range(1, num_batches + 1):
        # Generate synthetic batch
        inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == num_batches:
            if cuda_available:
                torch.cuda.synchronize()
            mem_mb = torch.cuda.memory_allocated(0) / (1024**2) if cuda_available else 0
            print(f"  Step [{step:3d}/{num_batches}] - Loss: {loss.item():.4f} - GPU Memory: {mem_mb:.1f} MB")

    if cuda_available:
        torch.cuda.synchronize()
    total_time = time.time() - start_time
    total_samples = batch_size * num_batches
    fps = total_samples / total_time

    print("\n[3/3] Performance Summary:")
    print(f"Total Images Processed: {total_samples}")
    print(f"Total Time:             {total_time:.2f} s")
    print(f"Throughput:             {fps:.1f} images/sec")
    print("=" * 60)
    print("PyTorch ROCm training & backpropagation test passed successfully!")

if __name__ == "__main__":
    test_pytorch_training()

