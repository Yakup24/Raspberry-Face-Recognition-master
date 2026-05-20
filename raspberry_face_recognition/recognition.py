"""Recognition pipeline helpers that are easy to test without a real camera."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple

from .vision import crop_face, detect_faces

FaceBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class RecognitionDetection:
    label: str
    confidence: float
    box: FaceBox
    label_id: int


@dataclass(frozen=True)
class RecognitionStepResult:
    faces_detected: int
    detections: list[RecognitionDetection] = field(default_factory=list)


def resolve_label(
    label_id: int,
    confidence: float,
    labels: Dict[int, str],
    confidence_threshold: float,
    unknown_label: str = "unknown",
) -> str:
    """Map a recognizer prediction to a label using the LBPH confidence threshold."""
    if confidence > confidence_threshold:
        return unknown_label
    return labels.get(int(label_id), unknown_label)


def format_detection(detection: RecognitionDetection) -> str:
    return f"[DETECTED] label={detection.label} confidence={detection.confidence:.1f}"


def run_recognition_step(
    frame: Any,
    detector: Any,
    recognizer: Any,
    config: Any,
    labels: Dict[int, str],
) -> RecognitionStepResult:
    """Run face detection and recognition for one frame."""
    gray, faces = detect_faces(frame, detector, config)
    detections: list[RecognitionDetection] = []

    for face in _iter_faces(faces):
        crop = crop_face(gray, face, config.face_size)
        label_id, confidence = recognizer.predict(crop)
        confidence_value = float(confidence)
        label = resolve_label(
            int(label_id),
            confidence_value,
            labels,
            float(config.confidence_threshold),
            str(config.unknown_label),
        )
        detections.append(
            RecognitionDetection(
                label=label,
                confidence=confidence_value,
                box=(int(face[0]), int(face[1]), int(face[2]), int(face[3])),
                label_id=int(label_id),
            )
        )

    return RecognitionStepResult(faces_detected=len(detections), detections=detections)


def _iter_faces(faces: Iterable[Any]) -> Iterable[FaceBox]:
    for face in faces:
        yield (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
