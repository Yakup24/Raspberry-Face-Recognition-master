"""Vision backends for PiSight.

OpenCV remains the camera and drawing layer. The PiSight-Omni path uses
facenet-pytorch for face detection/embedding and stores only vector embeddings.
Legacy Haar/LBPH helpers are kept for compatibility with older modules and
tests, but the CLI now enrolls and recognizes through embeddings by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install the deep runtime with: "
            'python -m pip install -e ".[deep]"'
        ) from exc
    return torch


def require_facenet_pytorch() -> tuple[Any, Any]:
    try:
        from facenet_pytorch import InceptionResnetV1, MTCNN  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "facenet-pytorch is not installed. Install the deep runtime with: "
            'python -m pip install -e ".[deep]"'
        ) from exc
    return MTCNN, InceptionResnetV1


def require_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is not installed. On Raspberry Pi, run: sudo apt install python3-opencv"
        ) from exc
    return cv2


def require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("NumPy is not installed. Run: python3 -m pip install -r requirements.txt") from exc
    return np


def get_device() -> Any:
    """Select CUDA, Apple MPS or CPU for the embedding runtime."""
    torch = require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_deep_detector(device: Any) -> Any:
    """Create an MTCNN detector for face alignment and crops."""
    MTCNN, _ = require_facenet_pytorch()
    return MTCNN(keep_all=True, device=device)


def create_deep_recognizer(device: Any) -> Any:
    """Create a FaceNet-style 512-dimensional embedding model."""
    _, InceptionResnetV1 = require_facenet_pytorch()
    return InceptionResnetV1(pretrained="vggface2").eval().to(device)


def detect_and_embed(frame: Any, detector: Any, recognizer: Any, device: Any) -> tuple[Any, Any]:
    """Detect faces and return boxes plus normalized embeddings.

    The frame is converted in memory from BGR to RGB. No face crop is written to
    disk by this function.
    """
    cv2 = require_cv2()
    np = require_numpy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, _ = detector.detect(rgb_frame)
    faces = detector(rgb_frame)
    if faces is None or boxes is None:
        return [], np.empty((0, 512), dtype=np.float32)

    embeddings = recognizer(faces.to(device)).detach().cpu().numpy().astype(np.float32)
    return boxes, embeddings


def default_cascade_path(cv2: Any) -> Path:
    return Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


def create_detector(config: Any) -> Any:
    cv2 = require_cv2()
    cascade_path = Path(config.cascade_path).expanduser() if config.cascade_path else default_cascade_path(cv2)
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    return detector


def create_recognizer() -> Any:
    cv2 = require_cv2()
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face recognizer support is missing. Install the contrib build or python3-opencv."
        )
    return cv2.face.LBPHFaceRecognizer_create()


def detect_faces(frame: Any, detector: Any, config: Any) -> tuple[Any, Any]:
    cv2 = require_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=config.scale_factor,
        minNeighbors=config.min_neighbors,
        minSize=(config.min_face_size, config.min_face_size),
    )
    return gray, faces


def crop_face(gray_frame: Any, face: tuple[int, int, int, int], size: tuple[int, int]) -> Any:
    cv2 = require_cv2()
    x, y, width, height = face
    crop = gray_frame[y : y + height, x : x + width]
    return cv2.resize(crop, size)
