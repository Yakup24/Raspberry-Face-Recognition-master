from pathlib import Path
import json
import tempfile
import unittest

from raspberry_face_recognition.config import load_config, validate_config


class ConfigTests(unittest.TestCase):
    def test_load_config_resolves_paths_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "data_dir": "runtime",
                        "faces_dir": "runtime/faces",
                        "model_path": "runtime/model.yml",
                        "labels_path": "runtime/labels.json",
                        "sample_count": 12,
                        "display": False,
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(str(config_path))

            self.assertEqual(config.data_dir, (root / "runtime").resolve())
            self.assertEqual(config.faces_dir, (root / "runtime/faces").resolve())
            self.assertEqual(config.model_path, (root / "runtime/model.yml").resolve())
            self.assertEqual(config.labels_path, (root / "runtime/labels.json").resolve())
            self.assertEqual(config.sample_count, 12)
            self.assertFalse(config.display)

    def test_load_yaml_config_with_nested_sections(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "camera:",
                        "  source: 0",
                        "  width: 640",
                        "  height: 480",
                        "  fps: 15",
                        "paths:",
                        "  dataset_dir: runtime/faces",
                        "  model_path: runtime/model.yml",
                        "  labels_path: runtime/labels.json",
                        "recognition:",
                        "  confidence_threshold: 65",
                        "  unknown_label: demo-unknown",
                        "runtime:",
                        "  debug: true",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(str(config_path))

            self.assertEqual(config.camera_source, 0)
            self.assertEqual(config.camera_width, 640)
            self.assertEqual(config.camera_height, 480)
            self.assertEqual(config.camera_fps, 15)
            self.assertEqual(config.faces_dir, (root / "runtime/faces").resolve())
            self.assertEqual(config.vector_index_path, (root / "data/embeddings/faiss.index").resolve())
            self.assertEqual(config.confidence_threshold, 65)
            self.assertEqual(config.unknown_label, "demo-unknown")
            self.assertTrue(config.debug)

    def test_missing_config_uses_defaults(self):
        config = load_config("missing-config.json")

        self.assertEqual(config.camera_index, 0)
        self.assertEqual(config.face_size, (160, 160))
        self.assertEqual(config.sample_count, 10)
        self.assertEqual(config.embedding_dim, 512)
        self.assertEqual(config.agent_model, "gpt-4.1-mini")
        self.assertEqual(config.agent_interval_frames, 30)
        self.assertEqual(config.omni_device_id, "node-001")
        self.assertFalse(config.omni_swarm_enabled)

    def test_missing_camera_source_in_camera_section_is_invalid(self):
        result = validate_config({"camera": {"width": 640}})

        self.assertFalse(result.is_valid)
        self.assertIn("camera.source", result.errors[0])

    def test_confidence_threshold_validation(self):
        result = validate_config({"recognition": {"confidence_threshold": -1}})

        self.assertFalse(result.is_valid)
        self.assertIn("confidence_threshold", result.errors[0])

    def test_vector_paths_parse_from_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "paths": {
                            "vector_index_path": "runtime/faiss.index",
                            "vector_labels_path": "runtime/labels.json",
                        },
                        "embedding": {"dim": 512},
                        "recognition": {"confidence_threshold": 0.8},
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(str(config_path))

            self.assertEqual(config.vector_index_path, (root / "runtime/faiss.index").resolve())
            self.assertEqual(config.vector_labels_path, (root / "runtime/labels.json").resolve())
            self.assertEqual(config.embedding_dim, 512)
            self.assertEqual(config.confidence_threshold, 0.8)

    def test_agent_config_parses_from_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "agent:",
                        "  model: gpt-4.1-mini",
                        "  base_url: http://localhost:11434/v1",
                        "  interval_frames: 12",
                        "  max_tokens: 200",
                        "  action_mode: disabled",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(str(config_path))

            self.assertEqual(config.agent_model, "gpt-4.1-mini")
            self.assertEqual(config.agent_base_url, "http://localhost:11434/v1")
            self.assertEqual(config.agent_interval_frames, 12)
            self.assertEqual(config.agent_max_tokens, 200)
            self.assertEqual(config.agent_action_mode, "disabled")

    def test_invalid_agent_action_mode_is_rejected(self):
        result = validate_config({"agent": {"action_mode": "gpio"}})

        self.assertFalse(result.is_valid)
        self.assertIn("agent.action_mode", result.errors[0])

    def test_omni_config_parses_from_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "omni:",
                        "  device_id: camera-alpha",
                        "  swarm_enabled: true",
                        "  swarm_host: mqtt.local",
                        "  swarm_port: 1884",
                        "  swarm_topic: pisight/omni/test",
                        "  swarm_dry_run: false",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(str(config_path))

            self.assertEqual(config.omni_device_id, "camera-alpha")
            self.assertTrue(config.omni_swarm_enabled)
            self.assertEqual(config.omni_swarm_host, "mqtt.local")
            self.assertEqual(config.omni_swarm_port, 1884)
            self.assertEqual(config.omni_swarm_topic, "pisight/omni/test")
            self.assertFalse(config.omni_swarm_dry_run)

    def test_invalid_omni_swarm_port_is_rejected(self):
        result = validate_config({"omni": {"swarm_port": 0}})

        self.assertFalse(result.is_valid)
        self.assertIn("omni.swarm_port", result.errors[0])


if __name__ == "__main__":
    unittest.main()
