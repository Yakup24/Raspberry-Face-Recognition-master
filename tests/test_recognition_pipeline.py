from unittest.mock import Mock, patch

from raspberry_face_recognition.config import AppConfig
from raspberry_face_recognition.recognition import (
    RecognitionDetection,
    format_detection,
    run_recognition_step,
)


def test_empty_detection_result_does_not_crash(fake_frame):
    config = AppConfig()
    fake_detector = Mock()
    fake_recognizer = Mock()

    with patch("raspberry_face_recognition.recognition.detect_faces", return_value=(object(), [])):
        result = run_recognition_step(fake_frame, fake_detector, fake_recognizer, config, {})

    assert result.faces_detected == 0
    assert result.detections == []
    fake_recognizer.predict.assert_not_called()


def test_unknown_threshold_behavior(fake_frame):
    config = AppConfig(confidence_threshold=70, unknown_label="unknown")
    fake_detector = Mock()
    fake_recognizer = Mock()
    fake_recognizer.predict.return_value = (0, 91.4)

    with patch("raspberry_face_recognition.recognition.detect_faces", return_value=(object(), [(1, 2, 30, 40)])):
        with patch("raspberry_face_recognition.recognition.crop_face", return_value=object()):
            result = run_recognition_step(
                fake_frame,
                fake_detector,
                fake_recognizer,
                config,
                {0: "demo-user-001"},
            )

    assert result.faces_detected == 1
    assert result.detections[0].label == "unknown"
    assert result.detections[0].confidence == 91.4


def test_result_formatter_outputs_expected_text():
    detection = RecognitionDetection(label="demo-user-001", confidence=64.2, box=(1, 2, 3, 4), label_id=0)

    assert format_detection(detection) == "[DETECTED] label=demo-user-001 confidence=64.2"
