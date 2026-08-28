"""Benchmark inference speed of a trained model.

Usage:
  python benchmark.py --weights runs/final/weights/best.pt --imgsz 640
"""

import argparse
import time

import torch
from pathlib import Path
from models.yolo_builder import CustomYOLO


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference speed.")
    parser.add_argument("--weights", type=str, required=True, help="Model weights path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device.")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=300, help="Benchmark iterations.")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model = CustomYOLO(args.weights)
    model = model.model.to(device).eval()

    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    # Warmup
    print(f"Warming up ({args.warmup} iterations)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            model(dummy_input)
    torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({args.iterations} iterations)...")
    latencies = []
    with torch.no_grad():
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy_input)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    # Trim 5% outliers on each side
    latencies.sort()
    trim = int(len(latencies) * 0.05)
    trimmed = latencies[trim:-trim] if trim > 0 else latencies

    avg_ms = sum(trimmed) / len(trimmed)
    fps = 1000.0 / avg_ms

    print(f"\n{'='*50}")
    print(f"Device:     {torch.cuda.get_device_name(device)}")
    print(f"Image size: {args.imgsz}x{args.imgsz}")
    print(f"Precision:  FP32")
    print(f"Latency:    {avg_ms:.2f} ms")
    print(f"FPS:        {fps:.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
