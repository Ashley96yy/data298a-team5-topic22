#!/usr/bin/env python3
"""Verify the frozen EC2 dataset against the project's pinned manifest (Python 3.11+)."""

import argparse
import hashlib
import json
from pathlib import Path


MANIFEST_SHA256 = "e4b7801c0d9ea27d0b39a6af253ac154d6f4a7181d8a7c1dc31d651a050af0e5"
DEFAULT_PROCESSED = Path.home() / "data298-deployment/data/processed"


def verify_dataset(folder: Path, expected_manifest_sha256: str = MANIFEST_SHA256) -> int:
    """Return the verified file count, or raise ValueError/OSError on failure.

    The default hash identifies amazon2023-skincare-v1. A new frozen version
    requires a reviewed code change; never recompute the pin to accept bad input.
    Extra files are not verified or executed.
    """
    folder = folder.expanduser().resolve()
    manifest_bytes = (folder / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256:
        raise ValueError("manifest differs from the verified project manifest")
    entries = json.loads(manifest_bytes)["processed_files"]
    if not entries:
        raise ValueError("manifest contains no dataset files")
    errors = []
    for name, info in sorted(entries.items()):
        relative = Path(name)
        path = (folder / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(folder):
            errors.append(f"Unsafe path: {name}")
        elif not path.is_file():
            errors.append(f"Missing: {name}")
        else:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != info["bytes"] or digest != info["sha256"]:
                errors.append(f"Mismatch: {name}")
    if errors:
        raise ValueError("\n".join(errors))
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    args = parser.parse_args()
    try:
        count = verify_dataset(args.processed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: manifest and all {count} dataset files verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
