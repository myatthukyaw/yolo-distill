#!/usr/bin/env python3
"""
Prepare a Roboflow YOLO-format dataset for use with this repo.

Roboflow exports datasets in this layout:
    <root>/
        train/
            images/  img001.jpg ...
            labels/  img001.txt ...  (cx cy w h, normalized)
        valid/
            images/
            labels/
        test/          (optional)
            images/
            labels/

This repo's data loader expects either:
  (A) images/<phase>/ + labels/<phase>/  structure, OR
  (B) a <phase>.txt index file listing image paths (one per line),
      with labels inferred by replacing "images" -> "labels" in each path.

This script uses approach (B):
  - Writes train.txt, val.txt (and test.txt if present) at the dataset root.
  - Converts all label files in-place from cx cy w h (normalized)
    to xmin ymin xmax ymax (normalized, [0,1]), which is what this repo expects.
  - Backs up original labels to .bak files before converting.

Usage:
    python scripts/prepare_roboflow_dataset.py --root /path/to/roboflow-dataset
    python scripts/prepare_roboflow_dataset.py --root /path/to/dataset --dry-run
    python scripts/prepare_roboflow_dataset.py --root /path/to/dataset --no-backup
"""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PHASE_MAP = [
    ("train", "train"),   # (roboflow folder name, txt file name)
    ("valid", "val"),
    ("test",  "test"),
]


def write_index(root: Path, rf_phase: str, txt_name: str, dry_run: bool) -> int:
    """Write <txt_name>.txt listing all image paths under <root>/<rf_phase>/images/."""
    images_dir = root / rf_phase / "images"
    if not images_dir.is_dir():
        return 0

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        return 0

    # Paths are relative to dataset root so the data loader can prepend it.
    lines = [str(p.relative_to(root)) for p in image_paths]
    out_path = root / f"{txt_name}.txt"

    print(f"  Writing {out_path.name}  ({len(lines)} images)")
    if not dry_run:
        out_path.write_text("\n".join(lines) + "\n")
    return len(lines)


def convert_label_file(path: Path, dry_run: bool, backup: bool):
    """Convert one label file from cx cy w h to xmin ymin xmax ymax in-place."""
    lines = path.read_text().splitlines()
    out_lines = []
    changed = False

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls = parts[0]
            try:
                cx, cy, w, h = map(float, parts[1:])
                x1 = max(0.0, min(1.0, cx - w / 2.0))
                y1 = max(0.0, min(1.0, cy - h / 2.0))
                x2 = max(0.0, min(1.0, cx + w / 2.0))
                y2 = max(0.0, min(1.0, cy + h / 2.0))
                out_lines.append(f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}")
                changed = True
                continue
            except ValueError:
                pass
        out_lines.append(line)

    if changed and not dry_run:
        if backup:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)
        path.write_text("\n".join(out_lines) + "\n")

    return changed


def convert_labels(root: Path, rf_phase: str, dry_run: bool, backup: bool) -> int:
    """Convert all label files under <root>/<rf_phase>/labels/."""
    labels_dir = root / rf_phase / "labels"
    if not labels_dir.is_dir():
        return 0

    label_files = sorted(labels_dir.rglob("*.txt"))
    converted = sum(
        convert_label_file(p, dry_run, backup)
        for p in label_files
    )
    print(f"  Converted {converted}/{len(label_files)} label files in {rf_phase}/labels/")
    return converted


def main():
    ap = argparse.ArgumentParser(description="Prepare a Roboflow dataset for yolo-distill")
    ap.add_argument("--root", required=True, help="Roboflow dataset root directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview actions without writing any files")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip .bak backups before converting labels")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    print(f"\nDataset root: {root}")
    print(f"Dry run: {args.dry_run}\n")

    found_phases = []
    for rf_phase, txt_name in PHASE_MAP:
        phase_dir = root / rf_phase
        if not phase_dir.is_dir():
            continue
        found_phases.append((rf_phase, txt_name))

        print(f"[{rf_phase}]")
        write_index(root, rf_phase, txt_name, args.dry_run)
        convert_labels(root, rf_phase, args.dry_run, backup=not args.no_backup)

    if not found_phases:
        raise SystemExit("No train/valid/test folders found. Is this a Roboflow dataset root?")

    # Print the dataset config to use
    txt_names = {rf: txt for rf, txt in found_phases}
    print("\n" + "=" * 60)
    print("Dataset preparation complete.")
    if args.dry_run:
        print("(Dry run — no files were written.)")
    print("\nCreate yolo/config/dataset/<your-dataset>.yaml with:\n")
    print(f"  path: {root}")
    print(f"  train: {txt_names.get('train', 'train')}")
    print(f"  validation: {txt_names.get('valid', 'val')}")
    if "test" in txt_names:
        print(f"  test: {txt_names['test']}")
    print()
    print("  class_num: <number of classes>")
    print("  class_list: ['Class1', 'Class2', ...]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
