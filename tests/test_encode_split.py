import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from coral.datasets.encode import verify_fold_window


class EncodeSplitTest(unittest.TestCase):
    def fixture(self, path, training_overlap=False, train_chrom=False):
        split = {"test": ["chr1"], "train": ["chr1"] if train_chrom else ["chr2"], "valid": ["chr8"]}
        peak = "chr1\t1006427\t1006711\t.\t1\t.\t40\t-1\t4\t142\n"
        entries = {"cv_params.fold_0.json": json.dumps(split)}
        for kind in ("peaks", "nonpeaks"):
            for group in ("training", "validation", "test"):
                entries[f"{kind}.{group}set.fold_0.ENCSR865RXA.bed"] = (
                    peak if kind == "peaks" and (group == "test" or group == "training" and training_overlap) else "")
        with tarfile.open(path, "w:gz") as archive:
            for name, value in entries.items():
                value = value.encode()
                member = tarfile.TarInfo(f"./fold_0/{name}")
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_test_window_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.tar.gz"
            checksum = self.fixture(path)
            window = {"genome": "hg38", "chrom": "chr1", "start": 1005512, "end": 1007626}
            result = verify_fold_window(path, window, expected_sha256=checksum)
            self.assertEqual(result["status"], "verified_test_window")
            self.assertEqual(result["overlap_counts"]["peaks_test"], 1)
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_fold_window(path, window)
            with self.assertRaisesRegex(ValueError, "summit"):
                verify_fold_window(path, dict(window, start=1005513, end=1007627), expected_sha256=checksum)

    def test_rejects_chromosome_and_interval_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.tar.gz"
            window = {"genome": "hg38", "chrom": "chr1", "start": 1005512, "end": 1007626}
            for kwargs, error in (({"train_chrom": True}, "exclusively"), ({"training_overlap": True}, "overlaps")):
                checksum = self.fixture(path, **kwargs)
                with self.assertRaisesRegex(ValueError, error):
                    verify_fold_window(path, window, expected_sha256=checksum)


if __name__ == "__main__":
    unittest.main()
