"""
Quick GPU diagnostic — run in a separate terminal while training is active.
"""
import subprocess, sys

# 1. nvidia-smi snapshot
print("=" * 70)
print("  NVIDIA-SMI  (GPU hardware status)")
print("=" * 70)
try:
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    print(r.stdout)
except FileNotFoundError:
    print("ERROR: nvidia-smi not found. NVIDIA drivers may not be installed.")
    sys.exit(1)

# 2. PyTorch CUDA check
print("=" * 70)
print("  PyTorch CUDA diagnostic")
print("=" * 70)
try:
    import torch
    print(f"  PyTorch version   : {torch.__version__}")
    print(f"  CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version      : {torch.version.cuda}")
        print(f"  Device count      : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            alloc = torch.cuda.memory_allocated(i) / 1024**2
            resrv = torch.cuda.memory_reserved(i) / 1024**2
            total = props.total_memory / 1024**2
            print(f"  GPU {i}: {props.name}")
            print(f"    Total VRAM      : {total:.0f} MB")
            print(f"    Allocated now   : {alloc:.0f} MB")
            print(f"    Reserved now    : {resrv:.0f} MB")
            print(f"    Compute cap.    : {props.major}.{props.minor}")

        # Quick bench: measure actual GPU compute speed
        print()
        print("  Running quick GPU speed test...")
        dev = torch.device("cuda")
        x = torch.randn(64, 256, 256, device=dev)

        # Warmup
        for _ in range(5):
            _ = torch.nn.functional.conv2d(
                x.unsqueeze(1),
                torch.randn(32, 1, 3, 3, device=dev),
                padding=1
            )
        torch.cuda.synchronize()

        import time
        t0 = time.perf_counter()
        for _ in range(50):
            _ = torch.nn.functional.conv2d(
                x.unsqueeze(1),
                torch.randn(32, 1, 3, 3, device=dev),
                padding=1
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  50x conv2d (64,1,256,256)->32ch : {elapsed:.3f}s  ({elapsed/50*1000:.1f} ms/op)")
        print(f"  GPU compute: {'OK — working normally' if elapsed < 5 else 'SLOW — possible issue'}")
    else:
        print("  WARNING: CUDA not available to PyTorch!")
        print("  Training is running on CPU (very slow).")
except ImportError:
    print("  PyTorch not installed in this environment.")

print()
print("=" * 70)
print("  TIP: To monitor GPU live during training, run:")
print('    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 2')
print("=" * 70)
