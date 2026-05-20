"""Dataset audit helpers for PiSight.

The audit module intentionally avoids reading image pixels. It only inspects
filesystem metadata and label files so users can validate local dataset hygiene
without exposing biometric samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from .dataset import IMAGE_EXTENSIONS, DatasetError, load_label_map


@dataclass(frozen=True)
class DatasetAudit:
    """Summary of local dataset and model artifacts."""

    faces_dir: str
    model_path: str
    labels_path: str
    people: int
    images: int
    images_by_person: Dict[str, int]
    model_exists: bool
    labels_exists: bool
    labels_count: int
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable audit representation."""
        return asdict(self)


def build_dataset_audit(config: Any) -> DatasetAudit:
    """Build a privacy-preserving summary of the local PiSight dataset."""
    faces_dir = Path(config.faces_dir)
    model_path = Path(config.model_path)
    labels_path = Path(config.labels_path)

    images_by_person: Dict[str, int] = {}
    warnings: List[str] = []

    if faces_dir.exists():
        for person_dir in sorted(path for path in faces_dir.iterdir() if path.is_dir()):
            image_count = sum(
                1
                for image_path in person_dir.iterdir()
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if image_count > 0:
                images_by_person[person_dir.name] = image_count
    else:
        warnings.append(f"Faces directory does not exist: {faces_dir}")

    labels = {}
    if labels_path.exists():
        try:
            labels = load_label_map(labels_path)
        except (DatasetError, ValueError, TypeError) as exc:
            warnings.append(f"Could not read labels file: {exc}")
    else:
        warnings.append(f"Labels file does not exist: {labels_path}")

    if not model_path.exists():
        warnings.append(f"Model file does not exist: {model_path}")

    people_with_images = set(images_by_person)
    people_in_labels = set(labels.values())

    missing_from_labels = sorted(people_with_images - people_in_labels)
    if missing_from_labels:
        warnings.append("People missing from labels: " + ", ".join(missing_from_labels))

    stale_labels = sorted(people_in_labels - people_with_images)
    if stale_labels:
        warnings.append("Labels without image folders: " + ", ".join(stale_labels))

    total_images = sum(images_by_person.values())
    if total_images == 0:
        warnings.append("No face sample images found.")

    return DatasetAudit(
        faces_dir=str(faces_dir),
        model_path=str(model_path),
        labels_path=str(labels_path),
        people=len(images_by_person),
        images=total_images,
        images_by_person=images_by_person,
        model_exists=model_path.exists(),
        labels_exists=labels_path.exists(),
        labels_count=len(labels),
        warnings=warnings,
    )


def format_audit_text(audit: DatasetAudit) -> str:
    """Format an audit result for terminal output."""
    lines = [
        "PiSight dataset audit",
        f"faces_dir   : {audit.faces_dir}",
        f"model_path  : {audit.model_path} ({'exists' if audit.model_exists else 'missing'})",
        f"labels_path : {audit.labels_path} ({'exists' if audit.labels_exists else 'missing'})",
        f"people      : {audit.people}",
        f"images      : {audit.images}",
        f"labels      : {audit.labels_count}",
    ]

    if audit.images_by_person:
        lines.append("")
        lines.append("images by person:")
        for person, count in sorted(audit.images_by_person.items()):
            lines.append(f"- {person}: {count}")

    if audit.warnings:
        lines.append("")
        lines.append("warnings:")
        for warning in audit.warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)
