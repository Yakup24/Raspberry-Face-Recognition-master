from pathlib import Path
import json
import tempfile
import unittest

from raspberry_face_recognition.audit import build_dataset_audit, format_audit_text
from raspberry_face_recognition.config import load_config
from raspberry_face_recognition.dataset import save_label_map


class AuditTests(unittest.TestCase):
    def test_audit_reports_dataset_counts_without_reading_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            faces = root / "faces"
            ada = faces / "Ada"
            ada.mkdir(parents=True)
            (ada / "000001.png").write_text("not a real image", encoding="utf-8")
            (ada / "notes.txt").write_text("ignored", encoding="utf-8")

            model = root / "model.yml"
            model.write_text("model", encoding="utf-8")
            labels = root / "labels.json"
            save_label_map({0: "Ada"}, labels)

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "faces_dir": "faces",
                        "model_path": "model.yml",
                        "labels_path": "labels.json",
                    }
                ),
                encoding="utf-8",
            )

            audit = build_dataset_audit(load_config(str(config_path)))

            self.assertEqual(audit.people, 1)
            self.assertEqual(audit.images, 1)
            self.assertEqual(audit.images_by_person, {"Ada": 1})
            self.assertTrue(audit.model_exists)
            self.assertTrue(audit.labels_exists)
            self.assertEqual(audit.labels_count, 1)
            self.assertEqual(audit.warnings, [])

    def test_audit_warns_for_missing_runtime_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "faces_dir": "faces",
                        "model_path": "model.yml",
                        "labels_path": "labels.json",
                    }
                ),
                encoding="utf-8",
            )

            audit = build_dataset_audit(load_config(str(config_path)))
            text = format_audit_text(audit)

            self.assertEqual(audit.people, 0)
            self.assertEqual(audit.images, 0)
            self.assertGreaterEqual(len(audit.warnings), 3)
            self.assertIn("warnings:", text)


if __name__ == "__main__":
    unittest.main()
