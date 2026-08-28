"""Prepare DsPCBSD+ dataset for YOLO training.

Downloads/extracts the dataset and converts COCO annotations to YOLO format.

Usage:
  python scripts/prepare_dspcbsd.py --zip-path datasets/DsPCBSD+.zip
"""

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "datasets" / "DsPCBSD+.zip"
DEFAULT_EXTRACT = ROOT / "datasets" / "dspcbsd" / "raw"
DEFAULT_OUTPUT = ROOT / "datasets" / "dspcbsd" / "yolo"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DsPCBSD+ to YOLO format.")
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-extract", action="store_true")
    return parser.parse_args()


def extract_zip(zip_path: Path, extract_dir: Path, force: bool = False):
    if extract_dir.exists() and not force:
        print(f"Already extracted: {extract_dir}")
        return
    print(f"Extracting {zip_path}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("Done.")


def convert_coco_to_yolo(coco_json: Path, images_src: Path,
                         images_dst: Path, labels_dst: Path):
    """Convert a single COCO annotation file to YOLO label files."""
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r") as f:
        coco = json.load(f)

    # Build image id -> filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    # Build category id -> index mapping (0-based)
    cat_ids = sorted([c["id"] for c in coco["categories"]])
    cat_map = {cid: idx for idx, cid in enumerate(cat_ids)}

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    converted = 0
    for img_id, filename in id_to_file.items():
        w, h = id_to_size[img_id]
        src_img = images_src / filename
        dst_img = images_dst / filename
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Write YOLO label
        label_name = Path(filename).stem + ".txt"
        label_path = labels_dst / label_name
        lines = []
        for ann in img_anns.get(img_id, []):
            cls_idx = cat_map[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]  # COCO: x,y,w,h (top-left)
            # Convert to YOLO: cx, cy, w, h (normalized)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        converted += 1

    return converted


def main():
    args = parse_args()

    if not args.zip_path.exists():
        print(f"ERROR: {args.zip_path} not found.")
        print("Download DsPCBSD+ from: https://figshare.com/articles/dataset/DsPCBSD_/25697498")
        print(f"Place the zip file at: {args.zip_path}")
        return

    extract_zip(args.zip_path, args.extract_dir, args.force_extract)

    # Find COCO annotation files
    coco_dir = args.extract_dir / "Data_COCO"
    ann_dir = coco_dir / "annotations"

    for split, ann_name, img_folder in [
        ("train", "instances_train2017.json", "train2017"),
        ("val", "instances_val2017.json", "val2017"),
    ]:
        ann_path = ann_dir / ann_name
        if not ann_path.exists():
            print(f"Warning: {ann_path} not found, skipping {split} split.")
            continue

        images_src = coco_dir / img_folder
        images_dst = args.output_dir / "images" / split
        labels_dst = args.output_dir / "labels" / split

        n = convert_coco_to_yolo(ann_path, images_src, images_dst, labels_dst)
        print(f"Converted {split}: {n} images")

    print(f"\nDataset ready at: {args.output_dir}")
    print("You can now train with: python train.py --model configs/model_final.yaml --name final")


if __name__ == "__main__":
    main()
