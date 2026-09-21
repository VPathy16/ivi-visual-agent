"""VisionLaya CLI: export training data (GPU-free) and train the head.

    visionlaya export --runs runs --out data/visionlaya.jsonl
    visionlaya train  --data data/visionlaya.jsonl --task verify --out models/verify.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .dataset import export_dataset, write_jsonl


def _parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="visionlaya", description="VisionLaya data + training")
    commands = root.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="Export a training set from agent runs")
    export.add_argument("--runs", default="runs", help="Run directory root")
    export.add_argument("--out", required=True, help="Output .jsonl path")

    train = commands.add_parser("train", help="Train a VisionLaya head (needs [visionlaya] extra)")
    train.add_argument("--data", required=True, help="Exported .jsonl dataset")
    train.add_argument("--task", default="verify", choices=["verify"], help="Which head to train")
    train.add_argument("--out", required=True, help="Output model path (.pt)")
    train.add_argument("--backbone", default="mobilevit_xs", help="timm backbone (frozen)")
    train.add_argument("--epochs", type=int, default=20)
    return root


def main() -> None:
    args = _parser().parse_args()
    if args.command == "export":
        examples = export_dataset(Path(args.runs))
        write_jsonl(examples, Path(args.out))
        by_task = Counter(e.task for e in examples)
        by_label = Counter(
            ("verify:pass" if e.satisfied else "verify:fail")
            for e in examples if e.task == "verify" and e.satisfied is not None
        )
        print(json.dumps({
            "exported": len(examples),
            "by_task": dict(by_task),
            "verify_labels": dict(by_label),
            "out": args.out,
        }, indent=2))
        raise SystemExit(0)
    if args.command == "train":
        from .train import train_verify_head  # lazy: needs torch

        summary = train_verify_head(
            Path(args.data), Path(args.out), backbone=args.backbone, epochs=args.epochs
        )
        print(json.dumps(summary, indent=2))
        raise SystemExit(0)


if __name__ == "__main__":
    main()
