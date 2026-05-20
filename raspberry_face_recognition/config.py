"""Configuration loading and validation for the PiSight commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Union


@dataclass(frozen=True)
class AppConfig:
    camera_source: Union[int, str] = 0
    camera_width: Optional[int] = None
    camera_height: Optional[int] = None
    camera_fps: Optional[int] = None
    data_dir: Path = Path("data")
    faces_dir: Path = Path("data/faces")
    model_path: Path = Path("data/model.yml")
    labels_path: Path = Path("data/labels.json")
    log_dir: Path = Path("logs")
    cascade_path: str = ""
    sample_count: int = 40
    face_width: int = 160
    face_height: int = 160
    scale_factor: float = 1.2
    min_neighbors: int = 5
    min_face_size: int = 60
    confidence_threshold: float = 70.0
    unknown_label: str = "unknown"
    draw_bounding_boxes: bool = True
    display: bool = True
    debug: bool = False
    save_unknown_faces: bool = False

    @property
    def face_size(self) -> tuple[int, int]:
        return (self.face_width, self.face_height)

    @property
    def camera_index(self) -> Union[int, str]:
        """Backward-compatible alias for older code and tests."""
        return self.camera_source


@dataclass(frozen=True)
class ConfigValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _read_config_file(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ValueError("YAML configuration requires PyYAML to be installed.") from exc
            data = yaml.safe_load(handle) or {}
        else:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be an object: {path}")
    return data


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{key}' must be an object.")
    return value


def _to_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value.")


def _to_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def _to_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    return _to_int(value, field_name)


def _to_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def _parse_camera_source(value: Any) -> Union[int, str]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        if stripped:
            return stripped
    raise ValueError("camera.source must be a camera index or video source path.")


def validate_config(data: Mapping[str, Any]) -> ConfigValidationResult:
    """Validate raw config data without touching the filesystem or camera."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        camera = _section(data, "camera")
        paths = _section(data, "paths")
        recognition = _section(data, "recognition")
        runtime = _section(data, "runtime")
    except ValueError as exc:
        return ConfigValidationResult(False, [str(exc)], warnings)

    if "camera" in data and "source" not in camera:
        errors.append("camera.source is required when the camera section is provided.")

    camera_source = camera.get("source", data.get("camera_source", data.get("camera_index", AppConfig.camera_source)))
    try:
        _parse_camera_source(camera_source)
    except ValueError as exc:
        errors.append(str(exc))

    numeric_checks = {
        "camera.width": camera.get("width"),
        "camera.height": camera.get("height"),
        "camera.fps": camera.get("fps"),
        "sample_count": data.get("sample_count"),
        "face_width": data.get("face_width"),
        "face_height": data.get("face_height"),
        "min_neighbors": data.get("min_neighbors"),
        "min_face_size": data.get("min_face_size"),
    }
    for field_name, raw_value in numeric_checks.items():
        if raw_value is None:
            continue
        try:
            if _to_int(raw_value, field_name) <= 0:
                errors.append(f"{field_name} must be greater than zero.")
        except ValueError as exc:
            errors.append(str(exc))

    threshold = recognition.get("confidence_threshold", data.get("confidence_threshold"))
    if threshold is not None:
        try:
            if _to_float(threshold, "recognition.confidence_threshold") < 0:
                errors.append("recognition.confidence_threshold must be zero or greater.")
        except ValueError as exc:
            errors.append(str(exc))

    scale_factor = data.get("scale_factor")
    if scale_factor is not None:
        try:
            if _to_float(scale_factor, "scale_factor") <= 1.0:
                errors.append("scale_factor must be greater than 1.0.")
        except ValueError as exc:
            errors.append(str(exc))

    for field_name, raw_value in {
        "runtime.debug": runtime.get("debug"),
        "runtime.save_unknown_faces": runtime.get("save_unknown_faces"),
        "display": data.get("display"),
        "recognition.draw_bounding_boxes": recognition.get("draw_bounding_boxes"),
    }.items():
        if raw_value is None:
            continue
        try:
            _to_bool(raw_value, field_name)
        except ValueError as exc:
            errors.append(str(exc))

    for field_name, raw_value in {
        "paths.dataset_dir": paths.get("dataset_dir"),
        "paths.model_path": paths.get("model_path"),
        "paths.labels_path": paths.get("labels_path"),
        "paths.log_dir": paths.get("log_dir"),
    }.items():
        if raw_value is not None and not str(raw_value).strip():
            errors.append(f"{field_name} must not be empty.")

    if paths.get("log_dir") is not None:
        warnings.append("log_dir is accepted for deployment layouts; current CLI output is stdout/stderr.")

    return ConfigValidationResult(not errors, errors, warnings)


def load_config(path: Optional[str] = None) -> AppConfig:
    """Load JSON or YAML configuration, falling back to safe defaults."""
    config_path = Path(path or "config.json").expanduser()
    base_dir = config_path.parent.resolve() if config_path.exists() else Path.cwd()
    data = _read_config_file(config_path)
    validation = validate_config(data)
    if not validation.is_valid:
        raise ValueError("Invalid configuration: " + "; ".join(validation.errors))

    camera = _section(data, "camera")
    paths = _section(data, "paths")
    recognition = _section(data, "recognition")
    runtime = _section(data, "runtime")

    data_dir_value = paths.get("data_dir", data.get("data_dir", AppConfig.data_dir))
    faces_dir_value = paths.get(
        "faces_dir",
        paths.get("dataset_dir", data.get("faces_dir", AppConfig.faces_dir)),
    )
    model_path_value = paths.get("model_path", data.get("model_path", AppConfig.model_path))
    labels_path_value = paths.get("labels_path", data.get("labels_path", AppConfig.labels_path))
    log_dir_value = paths.get("log_dir", data.get("log_dir", AppConfig.log_dir))

    return AppConfig(
        camera_source=_parse_camera_source(
            camera.get("source", data.get("camera_source", data.get("camera_index", AppConfig.camera_source)))
        ),
        camera_width=_to_optional_int(camera.get("width"), "camera.width"),
        camera_height=_to_optional_int(camera.get("height"), "camera.height"),
        camera_fps=_to_optional_int(camera.get("fps"), "camera.fps"),
        data_dir=_resolve_path(data_dir_value, base_dir),
        faces_dir=_resolve_path(faces_dir_value, base_dir),
        model_path=_resolve_path(model_path_value, base_dir),
        labels_path=_resolve_path(labels_path_value, base_dir),
        log_dir=_resolve_path(log_dir_value, base_dir),
        cascade_path=str(data.get("cascade_path", AppConfig.cascade_path)),
        sample_count=int(data.get("sample_count", AppConfig.sample_count)),
        face_width=int(data.get("face_width", AppConfig.face_width)),
        face_height=int(data.get("face_height", AppConfig.face_height)),
        scale_factor=float(data.get("scale_factor", AppConfig.scale_factor)),
        min_neighbors=int(data.get("min_neighbors", AppConfig.min_neighbors)),
        min_face_size=int(data.get("min_face_size", AppConfig.min_face_size)),
        confidence_threshold=float(
            recognition.get("confidence_threshold", data.get("confidence_threshold", AppConfig.confidence_threshold))
        ),
        unknown_label=str(recognition.get("unknown_label", data.get("unknown_label", AppConfig.unknown_label))),
        draw_bounding_boxes=_to_bool(
            recognition.get("draw_bounding_boxes", data.get("draw_bounding_boxes", AppConfig.draw_bounding_boxes)),
            "recognition.draw_bounding_boxes",
        ),
        display=_to_bool(data.get("display", AppConfig.display), "display"),
        debug=_to_bool(runtime.get("debug", data.get("debug", AppConfig.debug)), "runtime.debug"),
        save_unknown_faces=_to_bool(
            runtime.get("save_unknown_faces", data.get("save_unknown_faces", AppConfig.save_unknown_faces)),
            "runtime.save_unknown_faces",
        ),
    )
