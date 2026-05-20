"""Model loading helpers for local recognizer artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .dataset import DatasetError, load_label_map
from .vision import create_recognizer


@dataclass(frozen=True)
class ModelLoadResult:
    success: bool
    model: Optional[Any] = None
    labels: Dict[int, str] = field(default_factory=dict)
    error: str = ""


def load_model(
    model_path: Union[str, Path],
    labels_path: Optional[Union[str, Path]] = None,
    recognizer_factory: Optional[Callable[[], Any]] = None,
) -> ModelLoadResult:
    """Load an OpenCV recognizer and optional label map with clear errors."""
    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        return ModelLoadResult(False, error=f"Model file not found: {resolved_model_path}")

    resolved_labels_path = Path(labels_path) if labels_path is not None else None
    if resolved_labels_path is not None and not resolved_labels_path.exists():
        return ModelLoadResult(False, error=f"Labels file not found: {resolved_labels_path}")

    try:
        recognizer = recognizer_factory() if recognizer_factory is not None else create_recognizer()
    except RuntimeError as exc:
        return ModelLoadResult(False, error=str(exc))

    try:
        recognizer.read(str(resolved_model_path))
    except Exception as exc:  # OpenCV raises implementation-specific exceptions.
        return ModelLoadResult(False, error=f"Could not load model file {resolved_model_path}: {exc}")

    labels: Dict[int, str] = {}
    if resolved_labels_path is not None:
        try:
            labels = load_label_map(resolved_labels_path)
        except (DatasetError, ValueError, TypeError) as exc:
            return ModelLoadResult(False, error=f"Could not load labels file {resolved_labels_path}: {exc}")

    return ModelLoadResult(True, model=recognizer, labels=labels)
