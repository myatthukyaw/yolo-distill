"""
Comparison runner: trains teacher, baseline, CWD, and MGD sequentially.

Correct distillation workflow:
  1. Train teacher (larger model) on the target dataset first.
  2. Train student baseline (no distillation) for reference.
  3. Train student with CWD using the trained teacher.
  4. Train student with MGD using the trained teacher.

The teacher must be trained on the same dataset so it has task-specific
experience to distill. Using a generic pre-trained teacher only provides
general visual feature guidance, not task-specific knowledge transfer.

All runs are logged to W&B under the same group for easy side-by-side comparison.

Prerequisites:
  - Dataset must be configured in yolo/config/dataset/
  - W&B logged in: wandb login

Usage:
  python scripts/run_comparison.py
  python scripts/run_comparison.py --dataset coins --epochs 50 --teacher-epochs 50
  python scripts/run_comparison.py --epochs 5 --dataset mock  # quick smoke test
  python scripts/run_comparison.py --skip-teacher --teacher-weight runs/train/v9s-teacher/checkpoints/epoch=49.ckpt
"""

import argparse
import glob
import os
import subprocess
import sys


DEFAULTS = {
    "dataset": "face",
    "student_model": "v9-t",
    "teacher_model": "v9-s",
    "epochs": 50,
    "teacher_epochs": 50,
    "group": "distill-comparison",
    "image_size": 640,
}


def find_latest_checkpoint(run_name: str) -> str:
    """Find the latest checkpoint produced by a training run."""
    pattern = os.path.join("runs", "train", run_name, "**", "checkpoints", "*.ckpt")
    checkpoints = glob.glob(pattern, recursive=True)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found under runs/train/{run_name}/. "
            "Make sure the teacher training completed successfully."
        )
    return max(checkpoints, key=os.path.getmtime)


def run_experiment(cmd: list, name: str):
    print(f"\n{'='*60}")
    print(f"Starting: {name}")
    print(f"Command:  {' '.join(cmd)}")
    print("=" * 60 + "\n")
    subprocess.run(cmd, check=True)


def run(args):
    teacher_name = f"{args.teacher_model}-teacher"

    # ------------------------------------------------------------------ #
    # Step 1: Train teacher on the target dataset                         #
    # ------------------------------------------------------------------ #
    image_size_arg = [f"image_size=[{args.image_size},{args.image_size}]"] if args.image_size else []

    if not args.skip_teacher:
        teacher_cmd = [
            sys.executable, "yolo/lazy.py",
            "task=train",
            f"model={args.teacher_model}",
            f"dataset={args.dataset}",
            f"name={teacher_name}",
            f"task.epoch={args.teacher_epochs}",
            "use_wandb=True",
            f"wandb_group={args.group}",
        ] + image_size_arg
        run_experiment(teacher_cmd, teacher_name)
        teacher_weight = find_latest_checkpoint(teacher_name)
        print(f"\nTeacher checkpoint: {teacher_weight}\n")
    else:
        if not args.teacher_weight:
            print("Error: --skip-teacher requires --teacher-weight to be set.")
            sys.exit(1)
        teacher_weight = args.teacher_weight
        if not os.path.exists(teacher_weight):
            print(f"Error: Teacher weight not found at {teacher_weight}")
            sys.exit(1)
        print(f"\nUsing existing teacher weight: {teacher_weight}\n")

    # ------------------------------------------------------------------ #
    # Step 2: Train student baseline (no distillation)                   #
    # Step 3: Train student with CWD distillation                        #
    # Step 4: Train student with MGD distillation                        #
    # ------------------------------------------------------------------ #
    student_experiments = [
        {
            "name": f"{args.student_model}-baseline",
            "distill": False,
            "extra_args": [],
        },
        {
            "name": f"{args.student_model}-cwd",
            "distill": True,
            "extra_args": ["task.loss.distiller_type=cwd"],
        },
        {
            "name": f"{args.student_model}-mgd",
            "distill": True,
            "extra_args": ["task.loss.distiller_type=mgd"],
        },
    ]

    for exp in student_experiments:
        cmd = [
            sys.executable, "yolo/lazy.py",
            "task=train",
            f"model={args.student_model}",
            f"dataset={args.dataset}",
            f"name={exp['name']}",
            f"task.epoch={args.epochs}",
            "use_wandb=True",
            f"wandb_group={args.group}",
        ] + image_size_arg
        if exp["distill"]:
            # Wrap teacher_weight in single quotes so Hydra treats the path as
            # a quoted string literal — checkpoint filenames contain '=' characters
            # (e.g. epoch=49-step=1350.ckpt) which Hydra would otherwise misparse
            # as part of its key=value override grammar.
            cmd += [
                f"task.teacher_weight='{teacher_weight}'",
                f"task.teacher_model={args.teacher_model}",
            ]
        cmd += exp["extra_args"]
        run_experiment(cmd, exp["name"])

    print(f"\nAll experiments complete.")
    print(f"Runs logged under W&B group '{args.group}':")
    print(f"  - {teacher_name}       (teacher, reference)")
    for exp in student_experiments:
        print(f"  - {exp['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run teacher training + baseline vs CWD vs MGD distillation comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", default=DEFAULTS["dataset"],
                        help="Dataset config name (default: coins)")
    parser.add_argument("--student-model", default=DEFAULTS["student_model"],
                        help="Student model size (default: v9-t)")
    parser.add_argument("--teacher-model", default=DEFAULTS["teacher_model"],
                        help="Teacher model size (default: v9-s)")
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"],
                        help="Student training epochs (default: 50)")
    parser.add_argument("--teacher-epochs", type=int, default=DEFAULTS["teacher_epochs"],
                        help="Teacher training epochs (default: 50)")
    parser.add_argument("--group", default=DEFAULTS["group"],
                        help="W&B group name (default: distill-comparison)")
    parser.add_argument("--image-size", type=int, default=DEFAULTS["image_size"],
                        help="Input image size (square). Overrides config default, e.g. --image-size 416")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip teacher training (use with --teacher-weight)")
    parser.add_argument("--teacher-weight", default=None,
                        help="Path to existing teacher checkpoint (use with --skip-teacher)")
    run(parser.parse_args())
