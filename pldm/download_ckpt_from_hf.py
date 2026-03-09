#!/usr/bin/env python3
"""Download a model checkpoint from Hugging Face into the local pretrained folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a checkpoint from HF Hub into ./pretrained"
    )
    parser.add_argument(
        "--repo-id",
        default="kevinghst/pldm-maze2d-large-diverse",
        help="HF repo id, e.g. user/repo",
    )
    parser.add_argument(
        "--filename",
        default="epoch=3_sample_step=15465472.ckpt",
        help="Checkpoint filename in the HF repo",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="HF repo type",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch/tag/commit hash",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <repo_root>/pldm/pretrained)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "pretrained"
    out_dir.mkdir(parents=True, exist_ok=True)

    cached_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        repo_type=args.repo_type,
        revision=args.revision,
    )

    dst_path = out_dir / Path(args.filename).name
    if Path(cached_path).resolve() != dst_path.resolve():
        shutil.copy2(cached_path, dst_path)

    print(f"Downloaded: {args.repo_id}/{args.filename}")
    print(f"Saved to: {dst_path}")


if __name__ == "__main__":
    main()
