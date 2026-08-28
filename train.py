"""Training entry point for GASE-DSG-SAF PCB defect detection.

Usage:
  # Final method (GASE + DSG + SAF-IoU):
  python train.py --model configs/model_final.yaml --name final --iou-type saf_ciou --saf-kappa 1.5

  # Baseline:
  python train.py --model configs/model_baseline.yaml --name baseline
"""

import argparse
from pathlib import Path

from models.yolo_builder import CustomYOLO
from models.saf_iou import reset_saf_state
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Train GASE-DSG-SAF for PCB defect detection.")
    parser.add_argument("--model", type=str, required=True, help="Model YAML config path.")
    parser.add_argument("--name", type=str, required=True, help="Experiment name.")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Dataset YAML.")
    parser.add_argument("--batch", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device.")
    parser.add_argument("--workers", type=int, default=6, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cls-gain", type=float, default=0.8, help="Classification loss gain.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for last N epochs.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")

    # SAF-IoU options
    parser.add_argument("--iou-type", type=str, default="ciou",
                        choices=["ciou", "saf_ciou"],
                        help="IoU loss type. Use 'saf_ciou' for SAF-IoU.")
    parser.add_argument("--saf-kappa", type=float, default=1.5,
                        help="SAF scale correction strength (κ).")
    return parser


def main():
    args = build_parser().parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    # Determine if custom modules are needed
    use_custom = args.iou_type == "saf_ciou" or "GASE" in model_path.read_text() or "P2Guided" in model_path.read_text()

    if use_custom:
        model = CustomYOLO(str(model_path))
        reset_saf_state()
    else:
        model = YOLO(str(model_path))

    # Training arguments
    train_args = dict(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        name=args.name,
        project=str(ROOT / "runs"),
        optimizer="SGD",
        lr0=args.lr0,
        cls=args.cls_gain,
        close_mosaic=args.close_mosaic,
        patience=args.epochs,  # No early stopping
        amp=True,
    )

    model.train(**train_args)
    print(f"\nTraining complete. Results saved to: runs/{args.name}/")


if __name__ == "__main__":
    main()
