from types import SimpleNamespace

import pytest

from raspberry_face_recognition.cli import _open_camera
from raspberry_face_recognition.config import AppConfig


class FakeCamera:
    def __init__(self, opened=True):
        self.opened = opened
        self.settings = []

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.settings.append((prop, value))


def test_camera_unavailable_returns_clear_error():
    fake_cv2 = SimpleNamespace(VideoCapture=lambda source: FakeCamera(opened=False))

    with pytest.raises(RuntimeError, match="Could not open camera/video source"):
        _open_camera(fake_cv2, 99)


def test_camera_settings_are_applied_to_fake_capture():
    camera = FakeCamera(opened=True)
    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda source: camera,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
    )
    config = AppConfig(camera_width=640, camera_height=480, camera_fps=15)

    opened = _open_camera(fake_cv2, 0, config)

    assert opened is camera
    assert camera.settings == [(3, 640), (4, 480), (5, 15)]
