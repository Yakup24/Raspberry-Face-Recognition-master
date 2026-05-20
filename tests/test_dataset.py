from pathlib import Path
import tempfile
import unittest

from raspberry_face_recognition.dataset import (
    DatasetError,
    build_label_map,
    load_label_map,
    next_sample_index,
    normalize_person_name,
    save_label_map,
    validate_training_dataset,
)


class DatasetTests(unittest.TestCase):
    def test_normalize_person_name_keeps_safe_names(self):
        self.assertEqual(normalize_person_name("demo user"), "demo_user")
        self.assertEqual(normalize_person_name("test.user-1"), "test.user-1")

    def test_normalize_person_name_rejects_empty_names(self):
        with self.assertRaises(DatasetError):
            normalize_person_name(" !!! ")

    def test_next_sample_index_ignores_non_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "000001.png").write_text("", encoding="utf-8")
            (root / "000007.jpg").write_text("", encoding="utf-8")
            (root / "notes.txt").write_text("", encoding="utf-8")

            self.assertEqual(next_sample_index(root), 8)

    def test_label_map_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            labels = build_label_map(["demo-user-002", "demo-user-001", "demo-user-001"])
            path = Path(temp_dir) / "labels.json"

            save_label_map(labels, path)

            self.assertEqual(load_label_map(path), {0: "demo-user-001", 1: "demo-user-002"})

    def test_validate_training_dataset_reports_missing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_training_dataset(Path(temp_dir) / "missing")

            self.assertFalse(result.is_valid)
            self.assertIn("does not exist", result.errors[0])

    def test_validate_training_dataset_warns_for_empty_person_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "demo-user-001").mkdir()

            result = validate_training_dataset(root)

            self.assertFalse(result.is_valid)
            self.assertIn("No supported face image", result.errors[0])
            self.assertIn("demo-user-001", result.warnings[0])

    def test_validate_training_dataset_skips_unsupported_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            person = root / "demo-user-001"
            person.mkdir()
            (person / "sample_001.jpg").write_text("placeholder", encoding="utf-8")
            (person / "notes.txt").write_text("ignored", encoding="utf-8")

            result = validate_training_dataset(root)

            self.assertTrue(result.is_valid)
            self.assertEqual(result.people_count, 1)
            self.assertEqual(result.image_count, 1)
            self.assertIn("Unsupported files skipped", result.warnings[0])


if __name__ == "__main__":
    unittest.main()
