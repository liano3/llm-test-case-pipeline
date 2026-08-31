#!/usr/bin/env python3
"""Download Qwen3.5-27B into the local model directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ID = "Qwen/Qwen3.5-27B"
MODEL_DIR = Path(__file__).resolve().parent / "Qwen3.5-27B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install the downloader with: pip install huggingface_hub") from exc

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        revision=args.revision,
        local_dir=MODEL_DIR,
        token=os.environ.get("HF_TOKEN"),
        max_workers=args.max_workers,
    )
    print(path)


if __name__ == "__main__":
    main()
