"""Dataset helpers for local face image storage."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, Tuple

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class DatasetError(RuntimeError):
    """Raised when the local face dataset cannot be used."""


@dataclass(frozen=True)
class DatasetValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    people_count: int = 0
    image_count: int = 0


def normalize_person_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip()).strip("_")
    if not cleaned:
        raise DatasetError("Person name must contain at least one letter or number.")
    return cleaned


def person_directory(faces_dir: Path, name: str) -> Path:
    return faces_dir / normalize_person_name(name)


def next_sample_index(directory: Path) -> int:
    highest = 0
    if not directory.exists():
        return 1
    for path in directory.iterdir():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem.isdigit():
            highest = max(highest, int(path.stem))
    return highest + 1


def iter_person_images(faces_dir: Path) -> Iterator[Tuple[str, Path]]:
    if not faces_dir.exists():
        return
    for person_dir in sorted(path for path in faces_dir.iterdir() if path.is_dir()):
        person = person_dir.name
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield person, image_path


def validate_training_dataset(faces_dir: Path) -> DatasetValidationResult:
    """Validate dataset structure before OpenCV reads image pixels."""
    errors: list[str] = []
    warnings: list[str] = []
    people_count = 0
    image_count = 0

    if not faces_dir.exists():
        return DatasetValidationResult(
            False,
            [f"Dataset directory does not exist: {faces_dir}"],
            warnings,
        )

    if not faces_dir.is_dir():
        return DatasetValidationResult(False, [f"Dataset path is not a directory: {faces_dir}"], warnings)

    person_dirs = sorted(path for path in faces_dir.iterdir() if path.is_dir())
    if not person_dirs:
        return DatasetValidationResult(False, [f"No person folders found in dataset: {faces_dir}"], warnings)

    for person_dir in person_dirs:
        supported_images = [
            path
            for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        unsupported_files = [
            path
            for path in person_dir.iterdir()
            if path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS
        ]
        if unsupported_files:
            warnings.append(f"Unsupported files skipped in {person_dir.name}: {len(unsupported_files)}")
        if not supported_images:
            warnings.append(f"No supported face images found in person folder: {person_dir.name}")
            continue
        people_count += 1
        image_count += len(supported_images)

    if image_count == 0:
        errors.append("No supported face image files found. Run the collect command first.")

    return DatasetValidationResult(not errors, errors, warnings, people_count, image_count)


def build_label_map(people: Iterable[str]) -> Dict[int, str]:
    unique_people = sorted(set(people))
    if not unique_people:
        raise DatasetError("No face images found. Run the collect command first.")
    return {index: person for index, person in enumerate(unique_people)}


def save_label_map(labels: Dict[int, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(index): person for index, person in labels.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_label_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise DatasetError(f"Label map not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {int(index): str(person) for index, person in data.items()}
