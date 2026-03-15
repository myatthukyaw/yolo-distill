#!/usr/bin/env python3
"""
Bulk-convert YOLO center-format labels (class cx cy w h in [0,1])
to corner-format (class xmin ymin xmax ymax in [0,1]) for the YOLO-MIT repo.

Directory layout expected (same as your repo):
<dataset_root>/
  images/<phase>/...(.jpg/.png)
  labels/<phase>/...(.txt)

Usage:
  python convert_labels.py --root /path/to/dataset --phase train2017
  # add --dry-run to preview without writing
"""

import argparse
import shutil
from pathlib import Path

def convert_line_to_corners(line: str):
    parts = line.strip().split()
    if len(parts) == 0:
        return None, "empty"
    try:
        cls = parts[0]
        vals = list(map(float, parts[1:]))

        # If it's already polygon/corners (not 4 numbers), keep as-is
        if len(vals) != 4:
            return line.rstrip("\n"), "skipped-non-center"

        cx, cy, w, h = vals
        # convert center -> corners
        x1 = max(0.0, min(1.0, cx - w / 2.0))
        y1 = max(0.0, min(1.0, cy - h / 2.0))
        x2 = max(0.0, min(1.0, cx + w / 2.0))
        y2 = max(0.0, min(1.0, cy + h / 2.0))

        # format with consistent precision
        out = f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}"
        return out, "converted"

    except Exception:
        # Non-numeric or malformed; keep original
        return line.rstrip("\n"), "skipped-malformed"

def process_label_file(path: Path, dry_run=False, backup=True):
    text = path.read_text().splitlines()
    out_lines = []
    stats = {"converted":0, "skipped-non-center":0, "skipped-malformed":0, "empty":0}

    for line in text:
        res, tag = convert_line_to_corners(line)
        stats[tag] = stats.get(tag, 0) + 1
        # keep original line if None (empty) so we don't drop lines silently
        out_lines.append(res if res is not None else line)

    if not dry_run:
        if backup:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(path, bak)
        path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains images/ and labels/)")
    ap.add_argument("--phase", default="train2017", help="Phase folder (e.g., train2017/val2017)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write any files")
    ap.add_argument("--no-backup", action="store_true", help="Do not write .bak backups")
    args = ap.parse_args()

    root = Path(args.root)
    labels_dir = root / "labels" / args.phase
    if not labels_dir.exists():
        raise SystemExit(f"Labels dir not found: {labels_dir}")

    txts = sorted(labels_dir.rglob("*.txt"))
    if not txts:
        raise SystemExit(f"No label files found under {labels_dir}")

    total = {"converted":0, "skipped-non-center":0, "skipped-malformed":0, "empty":0, "files":0}
    for p in txts:
        stats = process_label_file(p, dry_run=args.dry_run, backup=not args.no_backup)
        total["files"] += 1
        for k,v in stats.items():
            total[k] += v

    print(f"\nDone. Files: {total['files']}")
    print(f"  converted:           {total['converted']}")
    print(f"  skipped-non-center:  {total['skipped-non-center']}  (already polygon/corners)")
    print(f"  skipped-malformed:   {total['skipped-malformed']}")
    print(f"  empty:               {total['empty']}")
    if args.dry_run:
        print("\n(Dry run: no files were modified.)")
    else:
        print("\nBackups (.bak) were created next to each label (unless --no-backup).")

if __name__ == "__main__":
    main()