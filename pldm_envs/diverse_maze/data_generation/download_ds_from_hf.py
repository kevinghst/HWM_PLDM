#!/usr/bin/env python3
"""Download a dataset snapshot from Hugging Face into a local dataset folder."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a dataset from HF Hub into ./pldm_envs/diverse_maze/datasets"
    )
    parser.add_argument(
        "--repo-id",
        default="kevinghst/maze2d-large-diverse-25maps",
        help="HF dataset repo id, e.g. user/repo",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
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
        help="Output directory (default: <repo_root>/pldm_envs/diverse_maze/datasets)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "pldm_envs" / "diverse_maze" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )

    print(f"Downloaded dataset: {args.repo_id}")
    print(f"Saved to: {out_dir}")
    print(f"Snapshot cache path: {snapshot_path}")


if __name__ == "__main__":
    main()
