"""Integrity regression checks using synthetic files; no AWS calls."""

import hashlib
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/aws/verify_frozen_dataset.py"
SPEC = importlib.util.spec_from_file_location("aws_dataset_verifier", SCRIPT)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
DOWNLOAD_SPEC = importlib.util.spec_from_file_location(
    "aws_dataset_downloader", SCRIPT.with_name("download_frozen_dataset.py")
)
DOWNLOADER = importlib.util.module_from_spec(DOWNLOAD_SPEC)
with patch.dict(sys.modules, {"verify_frozen_dataset": VERIFIER}):
    DOWNLOAD_SPEC.loader.exec_module(DOWNLOADER)


class FrozenDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "sample.bin").write_bytes(b"verified")
        self.manifest = {"processed_files": {"sample.bin": {
            "bytes": 8, "sha256": hashlib.sha256(b"verified").hexdigest(),
        }}}
        self.pin = self.write_manifest()

    def write_manifest(self):
        payload = json.dumps(self.manifest).encode()
        (self.root / "manifest.json").write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def test_matching_file(self):
        self.assertEqual(VERIFIER.verify_dataset(self.root, self.pin), 1)

    def test_same_size_corruption(self):
        (self.root / "sample.bin").write_bytes(b"corrupt!")
        with self.assertRaisesRegex(ValueError, "Mismatch"):
            VERIFIER.verify_dataset(self.root, self.pin)

    def test_missing_file(self):
        (self.root / "sample.bin").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            VERIFIER.verify_dataset(self.root, self.pin)

    def test_replaced_manifest(self):
        (self.root / "manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            VERIFIER.verify_dataset(self.root, self.pin)

    def test_path_traversal(self):
        entry = self.manifest["processed_files"].pop("sample.bin")
        self.manifest["processed_files"]["../outside.bin"] = entry
        pin = self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            VERIFIER.verify_dataset(self.root, pin)

    def test_symlink_escape(self):
        (self.root / "sample.bin").unlink()
        (self.root / "sample.bin").symlink_to(self.root.parent / "outside.bin")
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            VERIFIER.verify_dataset(self.root, self.pin)


class DatasetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.destination = self.parent / "processed"
        self.payload = b"test dataset"
        self.manifest_bytes = json.dumps({"processed_files": {"sample.bin": {
            "bytes": len(self.payload),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }}}).encode()
        self.pin = hashlib.sha256(self.manifest_bytes).hexdigest()
        self.argv = ["download_frozen_dataset.py", "--processed", str(self.destination)]

    def fake_copy(self, args, check):
        self.assertTrue(check)
        self.assertEqual(args[:3], ["aws", "s3", "cp"])
        self.assertEqual(args[3], DOWNLOADER.DEFAULT_SOURCE)
        staged = Path(args[4])
        (staged / "sample.bin").write_bytes(self.payload)
        (staged / "manifest.json").write_bytes(self.manifest_bytes)

    def run_download(self, copy=None):
        # Exercise real hash verification with a synthetic manifest trust pin.
        with patch.object(sys, "argv", self.argv), \
                patch.object(DOWNLOADER.subprocess, "run", side_effect=copy or self.fake_copy), \
                patch.object(DOWNLOADER, "verify_dataset",
                             side_effect=lambda folder: VERIFIER.verify_dataset(folder, self.pin)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return DOWNLOADER.main()

    def test_verified_download_published_and_staging_removed(self):
        self.assertEqual(self.run_download(), 0)
        self.assertEqual((self.destination / "sample.bin").read_bytes(), self.payload)
        self.assertEqual(list(self.parent.iterdir()), [self.destination])

    def test_failed_transfer_does_not_publish(self):
        def fail(args, check):
            self.fake_copy(args, check)
            raise subprocess.CalledProcessError(1, args)
        self.assertEqual(self.run_download(fail), 1)
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_corrupt_download_does_not_publish(self):
        def corrupt(args, check):
            self.fake_copy(args, check)
            (Path(args[4]) / "sample.bin").write_bytes(b"wrong")
        self.assertEqual(self.run_download(corrupt), 1)
        self.assertEqual(list(self.parent.iterdir()), [])

    def test_existing_destination_untouched_without_aws_call(self):
        self.destination.mkdir()
        marker = self.destination / "keep.txt"
        marker.write_text("existing data")
        def unexpected_call(args, check):
            self.fail("AWS must not be called for an existing destination")
        with self.assertRaises(SystemExit) as result:
            self.run_download(unexpected_call)
        self.assertEqual(result.exception.code, 2)
        self.assertEqual(marker.read_text(), "existing data")

    def test_destination_created_during_download_preserved(self):
        def competing_writer(args, check):
            self.fake_copy(args, check)
            self.destination.mkdir()
            (self.destination / "keep.txt").write_text("other run")
        self.assertEqual(self.run_download(competing_writer), 1)
        self.assertEqual((self.destination / "keep.txt").read_text(), "other run")
        self.assertEqual(list(self.parent.iterdir()), [self.destination])


if __name__ == "__main__":
    unittest.main()
