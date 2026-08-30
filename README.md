# YOLO-CE: Lightweight PCB Defect Detection

Official implementation of *"Lightweight PCB Defect Detection via Accuracy-Efficiency Co-optimization"* (submitted to Journal of Real-Time Image Processing).

## Highlights

- **GASE** (Geometric-Adaptive Sparse Enhancement): Lightweight deformable convolution with depthwise-separable offset prediction and geometric gating, deployed at a single P4 position (+0.03G FLOPs).
- **DSG** (Denoised Selective Guidance): P2-to-P3 detail injection with denoising, gated selection, and residual constraint — 14% cost of a full P2 detection head.
- **SAF-IoU** (Scale-Adaptive Focusing IoU): Corrects the outlier-degree threshold in dynamic focusing losses to protect small-target gradients — zero inference cost.

## Results on DsPCBSD+

| Method | Params (M) | FLOPs (G) | mAP50 | mAP50-95 |
|--------|-----------|-----------|-------|----------|
| YOLO26n (baseline) | 2.51 | 5.79 | 84.28 | 51.75 |
| + GASE | 2.74 | 5.82 | — | 52.16 |
| + GASE + DSG | 2.74 | 6.08 | — | 52.52 |
| **+ GASE + DSG + SAF-IoU (Ours)** | **2.74** | **6.08** | **85.24** | **52.81** |

Total improvement: **+1.06 mAP50-95** with only **+0.29G FLOPs (+5.0%)** overhead.

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/GASE-DSG-SAF.git
cd GASE-DSG-SAF

# Create conda environment (recommended)
conda create -n gase-dsg-saf python=3.10 -y
conda activate gase-dsg-saf

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

This project uses the [DsPCBSD+ dataset](https://doi.org/10.1038/s41597-024-03651-x) (10,259 images, 9 defect classes).

1. Download `DsPCBSD+.zip` from the [official source](https://figshare.com/articles/dataset/DsPCBSD_/25697498)
2. Place it under `datasets/`
3. Run the conversion script:

```bash
python scripts/prepare_dspcbsd.py --zip-path datasets/DsPCBSD+.zip
```

This converts COCO annotations to YOLO format and creates the train/val split.

## Training

**Reproduce the final result (GASE + DSG + SAF-IoU):**

```bash
python train.py \
    --model configs/model_final.yaml \
    --name final_gase_dsg_saf \
    --epochs 300 \
    --batch 32 \
    --cls-gain 0.8 \
    --close-mosaic 10 \
    --iou-type saf_ciou \
    --saf-kappa 1.5
```

**Train the baseline:**

```bash
python train.py \
    --model configs/model_baseline.yaml \
    --name baseline \
    --epochs 300 \
    --batch 32 \
    --cls-gain 0.8 \
    --close-mosaic 10
```

## Inference & Benchmark

```bash
# Validate a trained model
python val.py --weights runs/final_gase_dsg_saf/weights/best.pt

# Benchmark inference speed
python benchmark.py --weights runs/final_gase_dsg_saf/weights/best.pt --imgsz 640
```

## Project Structure

```
├── configs/
│   ├── model_final.yaml      # GASE + DSG + SAF-IoU architecture
│   ├── model_baseline.yaml   # YOLO26n baseline
│   └── dataset.yaml          # Dataset configuration template
├── models/
│   ├── gase.py               # GASE module
│   ├── dsg.py                # DSG module
│   ├── saf_iou.py            # SAF-IoU loss function
│   └── yolo_builder.py       # Custom YOLO model builder
├── scripts/
│   └── prepare_dspcbsd.py    # Dataset preparation
├── train.py                  # Training entry point
├── val.py                    # Validation script
└── benchmark.py              # Inference speed benchmark
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{author2026gase,
  title={Lightweight PCB Defect Detection via Accuracy-Efficiency Co-optimization},
  author={Author Name},
  journal={Journal of Real-Time Image Processing},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the detection framework
- [DsPCBSD+](https://doi.org/10.1038/s41597-024-03651-x) for the dataset
