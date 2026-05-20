from pathlib import Path
import contextlib
import io
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from raspberry_face_recognition import cli


class CliTests(unittest.TestCase):
    def test_parser_accepts_collect_arguments(self):
        parser = cli.build_parser()

        args = parser.parse_args(["--config", "config.yaml", "collect", "--name", "demo-user", "--count", "5"])

        self.assertEqual(args.config, "config.yaml")
        self.assertEqual(args.command, "collect")
        self.assertEqual(args.name, "demo-user")
        self.assertEqual(args.count, 5)

    def test_parser_accepts_train_arguments(self):
        parser = cli.build_parser()

        args = parser.parse_args(["--config", "config.yaml", "train"])

        self.assertEqual(args.command, "train")
        self.assertEqual(args.config, "config.yaml")

    def test_parser_accepts_recognize_arguments(self):
        parser = cli.build_parser()

        args = parser.parse_args(["--config", "config.yaml", "recognize", "--no-window"])

        self.assertEqual(args.command, "recognize")
        self.assertTrue(args.no_window)

    def test_help_command_exits_successfully(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as exit_info:
                cli.main(["--help"])

        self.assertEqual(exit_info.exception.code, 0)
        self.assertIn("Raspberry Pi face recognition toolkit", stdout.getvalue())

    def test_recognize_reports_missing_model_before_opening_camera(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_path": "missing-model.yml",
                        "labels_path": "labels.json",
                        "display": False,
                    }
                ),
                encoding="utf-8",
            )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = cli.main(["--config", str(config_path), "recognize", "--no-window"])

            self.assertEqual(exit_code, 2)
            self.assertIn("Model file not found", stderr.getvalue())

    def test_collect_fails_when_face_sample_cannot_be_written(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "faces_dir": "faces",
                        "sample_count": 1,
                        "display": False,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(config=str(config_path), name="test_user", count=1, no_window=True)
            camera = Mock()
            camera.read.return_value = (True, object())
            fake_cv2 = SimpleNamespace(imwrite=Mock(return_value=False))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch.object(
                cli, "require_cv2", return_value=fake_cv2
            ), patch.object(cli, "create_detector", return_value=object()), patch.object(
                cli, "_open_camera", return_value=camera
            ), patch.object(cli, "detect_faces", return_value=(object(), [(0, 0, 80, 80)])), patch.object(
                cli, "crop_face", return_value=object()
            ):
                with self.assertRaisesRegex(RuntimeError, "Could not write face sample"):
                    cli.command_collect(args)

            camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
