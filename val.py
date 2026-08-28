"""Validate a trained model on DsPCBSD+ dataset.

Usage:
  python val.py --weights runs/final/weights/best.pt
"""

import argparse
from pathlib import Path

from models.yolo_builder import CustomYOLO
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Validate trained model.")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt).")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    model = CustomYOLO(args.weights)
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    print(f"\n{'='*50}")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
