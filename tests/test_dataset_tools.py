import csv
import tempfile
import unittest
from pathlib import Path

from benchmarks.hajj_count_forecast import read_counts
from benchmarks.masjid_frame_inventory import inventory


class DatasetToolTests(unittest.TestCase):
    def test_hajj_reader_rejects_duplicate_video_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text(
                "video,time_s,GT_count\nclip_a,0,10\nclip_a,0,11\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate timestamp"):
                read_counts(path)

    def test_hajj_reader_rejects_negative_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text(
                "video,time_s,GT_count\nclip_a,0,-1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Invalid crowd count"):
                read_counts(path)

    def test_masjid_inventory_skips_face_and_identity_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "RawDataFrames"
            (root / "camera_01").mkdir(parents=True)
            (root / "ProcessedFaceImages").mkdir()
            (root / "camera_01" / "frame_001.png").write_bytes(b"not-decoded")
            (root / "ProcessedFaceImages" / "person_001.png").write_bytes(b"excluded")
            rows = inventory(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["relative_path"], "camera_01/frame_001.png")
            self.assertEqual(rows[0]["size_bytes"], len(b"not-decoded"))


if __name__ == "__main__":
    unittest.main()
