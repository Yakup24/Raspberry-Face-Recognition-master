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

    def test_parser_accepts_enroll_alias(self):
        parser = cli.build_parser()

        args = parser.parse_args(["--config", "config.yaml", "enroll", "--name", "demo-user", "--count", "3"])

        self.assertEqual(args.command, "enroll")
        self.assertEqual(args.name, "demo-user")
        self.assertEqual(args.count, 3)

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

    def test_parser_accepts_autonom_arguments(self):
        parser = cli.build_parser()

        args = parser.parse_args(
            ["--config", "config.yaml", "autonom", "--no-window", "--interval-frames", "5", "--max-frames", "10"]
        )

        self.assertEqual(args.command, "autonom")
        self.assertTrue(args.no_window)
        self.assertEqual(args.interval_frames, 5)
        self.assertEqual(args.max_frames, 10)

    def test_help_command_exits_successfully(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as exit_info:
                cli.main(["--help"])

        self.assertEqual(exit_info.exception.code, 0)
        self.assertIn("PiSight-X Raspberry Pi face embedding and agentic vision toolkit", stdout.getvalue())

    def test_recognize_reports_missing_vectors_before_opening_camera(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "display": False,
                    }
                ),
                encoding="utf-8",
            )

            stderr = io.StringIO()
            fake_db = SimpleNamespace(count=0)
            with contextlib.redirect_stderr(stderr), patch.object(cli, "require_cv2", return_value=SimpleNamespace()), patch.object(
                cli, "get_device", return_value="cpu"
            ), patch.object(cli, "create_deep_detector", return_value=object()), patch.object(
                cli, "create_deep_recognizer", return_value=object()
            ), patch.object(
                cli, "FaceVectorDB", return_value=fake_db
            ), patch.object(
                cli, "_open_camera"
            ) as open_camera:
                exit_code = cli.main(["--config", str(config_path), "recognize", "--no-window"])

            self.assertEqual(exit_code, 2)
            self.assertIn("No enrolled face vectors", stderr.getvalue())
            open_camera.assert_not_called()

    def test_collect_enrolls_vectors_without_writing_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "sample_count": 1,
                        "display": False,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(config=str(config_path), name="test_user", count=1, no_window=True)
            camera = Mock()
            camera.read.return_value = (True, object())
            fake_db = Mock()
            fake_cv2 = SimpleNamespace(imwrite=Mock(return_value=False))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), patch.object(
                cli, "require_cv2", return_value=fake_cv2
            ), patch.object(cli, "get_device", return_value="cpu"), patch.object(
                cli, "create_deep_detector", return_value=object()
            ), patch.object(cli, "create_deep_recognizer", return_value=object()), patch.object(
                cli, "FaceVectorDB", return_value=fake_db
            ), patch.object(
                cli, "_open_camera", return_value=camera
            ), patch.object(
                cli, "detect_and_embed", return_value=([], [[0.0] * 512])
            ):
                exit_code = cli.command_collect(args)

            self.assertEqual(exit_code, 0)
            fake_db.add_face.assert_called_once()
            fake_cv2.imwrite.assert_not_called()
            camera.release.assert_called_once()

    def test_train_is_compatibility_noop(self):
        args = SimpleNamespace(config="missing-config.json")
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            exit_code = cli.command_train(args)

        self.assertEqual(exit_code, 0)
        self.assertIn("not required", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
