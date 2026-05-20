from pathlib import Path

from raspberry_face_recognition.dataset import save_label_map
from raspberry_face_recognition.model import load_model


class FakeRecognizer:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.loaded_path = None

    def read(self, path):
        if self.should_fail:
            raise RuntimeError("read failed")
        self.loaded_path = path


def test_missing_model_file_returns_clear_error(tmp_path):
    missing_model = tmp_path / "missing_model.yml"

    result = load_model(str(missing_model))

    assert result.success is False
    assert "model" in result.error.lower()


def test_missing_labels_file_returns_clear_error(tmp_path):
    model_path = tmp_path / "model.yml"
    model_path.write_text("placeholder", encoding="utf-8")

    result = load_model(model_path, tmp_path / "labels.json")

    assert result.success is False
    assert "labels" in result.error.lower()


def test_model_load_failure_is_captured(tmp_path):
    model_path = tmp_path / "model.yml"
    model_path.write_text("placeholder", encoding="utf-8")

    result = load_model(model_path, recognizer_factory=lambda: FakeRecognizer(should_fail=True))

    assert result.success is False
    assert "could not load model" in result.error.lower()


def test_model_and_labels_load_successfully_with_fake_recognizer(tmp_path):
    model_path = tmp_path / "model.yml"
    labels_path = tmp_path / "labels.json"
    model_path.write_text("placeholder", encoding="utf-8")
    save_label_map({0: "demo-user-001"}, labels_path)

    result = load_model(model_path, labels_path, recognizer_factory=FakeRecognizer)

    assert result.success is True
    assert result.labels == {0: "demo-user-001"}
    assert result.model.loaded_path == str(Path(model_path))
