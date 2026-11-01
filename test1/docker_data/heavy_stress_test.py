import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim

def run_stress_test(duration_seconds=60, batch_size=64, img_size=256):
    print("=" * 65)
    print("       ROCm GPU HEAVY STRESS & SUSTAINED LOAD TEST        ")
    print("=" * 65)
    
    if not torch.cuda.is_available():
        print("ERROR: ROCm / CUDA GPU is not available! Exiting.")
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"Device:       {gpu_name}")
    print(f"Total VRAM:   {total_mem:.2f} GB")
    print(f"Duration:     {duration_seconds} seconds of continuous training load")
    print(f"Batch Size:   {batch_size}")
    print(f"Input Tensor: {batch_size} x 3 x {img_size} x {img_size}")
    print("=" * 65)

    # Deep Residual Network block for heavy compute
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(x + self.block(x))

    class HeavyStressNet(nn.Module):
        def __init__(self, in_channels=3, base_channels=128, num_blocks=12, num_classes=1000):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            
            blocks = []
            channels = base_channels
            for i in range(num_blocks):
                blocks.append(ResidualBlock(channels))
                # Double channels twice
                if i in [3, 7]:
                    next_channels = channels * 2
                    blocks.append(nn.Conv2d(channels, next_channels, kernel_size=3, stride=2, padding=1))
                    blocks.append(nn.BatchNorm2d(next_channels))
                    blocks.append(nn.ReLU(inplace=True))
                    channels = next_channels

            self.body = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channels, num_classes)
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.body(x)
            return self.head(x)

    print("\n[1/3] Building deep residual network (~30M params)...")
    model = HeavyStressNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler() # Tests AMP FP16 matrix compute units

    # Pre-allocate input/target tensors to maximize GPU compute saturation without CPU memory bottlenecks
    input_data = torch.randn(batch_size, 3, img_size, img_size, device=device)
    target_data = torch.randint(0, 1000, (batch_size,), device=device)

    print("[2/3] Warming up GPU kernels and memory pipelines...")
    for _ in range(10):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input_data)
            loss = criterion(output, target_data)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    torch.cuda.synchronize()
    print("\n[3/3] Sustained Stress Loop started! Watch your power & fans spin up...")
    print("-" * 65)

    start_time = time.time()
    last_print = start_time
    total_samples = 0
    step = 0

    while True:
        now = time.time()
        elapsed = now - start_time
        if elapsed >= duration_seconds:
            break

        step += 1
        total_samples += batch_size

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(input_data)
            loss = criterion(output, target_data)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Print stats every 2 seconds
        if now - last_print >= 2.0:
            torch.cuda.synchronize()
            mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            current_fps = total_samples / elapsed
            progress = min(100.0, (elapsed / duration_seconds) * 100)
            
            print(f"[{progress:4.1f}% | {elapsed:4.1f}s/{duration_seconds}s] "
                  f"Step: {step:5d} | "
                  f"FPS: {current_fps:6.1f} img/s | "
                  f"VRAM: {mem_allocated:4.2f}/{mem_reserved:4.2f} GB | "
                  f"Loss: {loss.item():.4f}")
            last_print = now

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    final_fps = total_samples / total_time

    print("-" * 65)
    print("STRESS TEST COMPLETED!")
    print(f"Total Time:      {total_time:.2f} seconds")
    print(f"Total Steps:     {step} full forward+backward iterations")
    print(f"Total Images:    {total_samples} images processed")
    print(f"Avg Throughput:  {final_fps:.1f} images/sec")
    print(f"Peak VRAM:       {torch.cuda.max_memory_allocated(0) / (1024**3):.2f} GB allocated")
    print("=" * 65)

if __name__ == "__main__":
    duration = 60
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            pass
    run_stress_test(duration_seconds=duration)

