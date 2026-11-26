#!/usr/bin/env python3
"""Download a frozen S3 prefix to a new directory and verify it before publication."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from verify_frozen_dataset import DEFAULT_PROCESSED, verify_dataset


DEFAULT_SOURCE = "s3://data298-team5-artifacts-2026/frozen/amazon2023-skincare-v1/processed/"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--aws-cli", default="aws", help="AWS CLI executable or absolute path")
    args = parser.parse_args()
    destination = args.processed.expanduser().absolute()
    if not args.source.startswith("s3://"):
        parser.error("--source must be an s3:// prefix")
    if destination.exists() or destination.is_symlink():
        parser.error("destination already exists; use verify_frozen_dataset.py or a new directory")
    # AWS CLI uses the EC2 instance role. This script never configures credentials
    # or uploads/deletes S3 objects. Use one writer per destination.
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".frozen-download-", dir=destination.parent) as temp:
            staged = Path(temp) / "processed"
            staged.mkdir()
            subprocess.run([
                args.aws_cli, "s3", "cp", args.source.rstrip("/") + "/", str(staged) + "/",
                "--recursive", "--region", args.region, "--only-show-errors", "--no-cli-pager",
            ], check=True)
            count = verify_dataset(staged)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"Destination appeared during download: {destination}")
            staged.rename(destination)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: manifest and all {count} dataset files verified; saved to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
