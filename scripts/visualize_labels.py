"""
Visualize bounding box labels overlaid on images.

Supports multiple label formats so you can visually confirm which one your labels use:
  - xyxy   : class_id xmin ymin xmax ymax  (normalized, corner format — this repo's format)
  - xywh   : class_id cx cy w h            (normalized, YOLO center format — Roboflow export)
  - xyxy_px: class_id xmin ymin xmax ymax  (pixel coordinates, not normalized)

Usage:
  # Show one image with its label file (auto-side-by-side comparison of xyxy vs xywh)
  python scripts/visualize_labels.py --image path/to/image.jpg --label path/to/label.txt

  # Show all images in a folder (picks matching .txt from same folder or sibling labels/)
  python scripts/visualize_labels.py --folder path/to/images/

  # Specify format explicitly — no side-by-side, just draws in one format
  python scripts/visualize_labels.py --folder path/to/images/ --format xywh

  # Save output images to a folder instead of displaying
  python scripts/visualize_labels.py --folder path/to/images/ --save-dir out/
"""

import argparse
import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def load_labels(label_path: Path):
    """Returns list of (class_id, x1, y1, x2, y2_or_h) raw values."""
    lines = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                vals = list(map(float, parts[1:]))
                lines.append((cls, *vals))
    return lines


def draw_boxes(ax, img_w, img_h, labels, fmt, title):
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    for cls, a, b, c, d in labels:
        color = COLORS[cls % len(COLORS)]

        if fmt == "xyxy":
            # normalized xmin ymin xmax ymax
            x1, y1, x2, y2 = a * img_w, b * img_h, c * img_w, d * img_h
        elif fmt == "xywh":
            # normalized cx cy w h
            cx, cy, w, h = a * img_w, b * img_h, c * img_w, d * img_h
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        elif fmt == "xyxy_px":
            # pixel xmin ymin xmax ymax
            x1, y1, x2, y2 = a, b, c, d
        else:
            continue

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, str(cls), color=color, fontsize=8,
                fontweight="bold", va="bottom")


def find_label(image_path: Path) -> Path | None:
    """Look for matching .txt in same folder or sibling labels/ folder."""
    candidates = [
        image_path.with_suffix(".txt"),
        image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def visualize(image_path: Path, label_path: Path, fmt: str, save_dir: Path | None):
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    labels = load_labels(label_path)

    if not labels:
        print(f"  No labels in {label_path.name}, skipping.")
        return

    if fmt == "compare":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(image_path.name, fontsize=11)
        for ax, f, title in zip(
            axes,
            ["xyxy", "xywh"],
            ["xyxy — corner format (this repo)", "xywh — YOLO center format (Roboflow)"],
        ):
            ax.imshow(img)
            draw_boxes(ax, img_w, img_h, labels, f, title)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(image_path.name, fontsize=11)
        ax.imshow(img)
        draw_boxes(ax, img_w, img_h, labels, fmt, fmt)

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / (image_path.stem + "_vis.jpg")
        plt.savefig(out_path, bbox_inches="tight", dpi=120)
        print(f"  Saved: {out_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay bbox labels on images to verify format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Single image file")
    group.add_argument("--folder", type=Path, help="Folder of images")

    parser.add_argument("--label", type=Path, default=None,
                        help="Label .txt file (only with --image; auto-detected if omitted)")
    parser.add_argument("--format", choices=["compare", "xyxy", "xywh", "xyxy_px"],
                        default="compare",
                        help="Label format (default: compare — shows xyxy and xywh side by side)")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Save visualizations here instead of displaying")
    parser.add_argument("--max", type=int, default=10,
                        help="Max images to process when using --folder (default: 10)")
    args = parser.parse_args()

    if args.image:
        label_path = args.label or find_label(args.image)
        if label_path is None:
            print(f"No label file found for {args.image}")
            return
        print(f"Image : {args.image}")
        print(f"Label : {label_path}")
        visualize(args.image, label_path, args.format, args.save_dir)

    elif args.folder:
        images = sorted(
            p for p in args.folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )[: args.max]
        if not images:
            print(f"No images found in {args.folder}")
            return
        print(f"Found {len(images)} image(s) in {args.folder}")
        for img_path in images:
            label_path = find_label(img_path)
            if label_path is None:
                print(f"  No label for {img_path.name}, skipping.")
                continue
            print(f"  {img_path.name} + {label_path.name}")
            visualize(img_path, label_path, args.format, args.save_dir)


if __name__ == "__main__":
    main()
