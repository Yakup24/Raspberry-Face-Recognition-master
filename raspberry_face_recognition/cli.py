"""Command line interface for Raspberry Pi face recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence, Union

from .audit import build_dataset_audit, format_audit_text
from .config import load_config
from .dataset import (
    DatasetError,
    build_label_map,
    iter_person_images,
    next_sample_index,
    person_directory,
    save_label_map,
    validate_training_dataset,
)
from .model import load_model
from .recognition import format_detection, run_recognition_step
from .vision import create_detector, create_recognizer, crop_face, detect_faces, require_cv2, require_numpy


def _load(args: argparse.Namespace) -> Any:
    return load_config(args.config)


def _open_camera(cv2: Any, camera_source: Union[int, str], config: Optional[Any] = None) -> Any:
    camera = cv2.VideoCapture(camera_source)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera/video source: {camera_source!r}.")
    if config is not None:
        _apply_camera_settings(cv2, camera, config)
    return camera


def _apply_camera_settings(cv2: Any, camera: Any, config: Any) -> None:
    settings = [
        ("camera_width", "CAP_PROP_FRAME_WIDTH"),
        ("camera_height", "CAP_PROP_FRAME_HEIGHT"),
        ("camera_fps", "CAP_PROP_FPS"),
    ]
    for config_attr, cv2_attr in settings:
        value = getattr(config, config_attr, None)
        if value is not None and hasattr(cv2, cv2_attr):
            camera.set(getattr(cv2, cv2_attr), value)


def command_doctor(args: argparse.Namespace) -> int:
    config = _load(args)
    print(f"config: {Path(args.config).resolve()}")
    print(f"camera_source: {config.camera_source}")
    print(f"data: {config.data_dir}")
    print(f"faces: {config.faces_dir}")

    exit_code = 0
    try:
        cv2 = require_cv2()
        print(f"opencv: OK ({cv2.__version__})")
        if hasattr(cv2, "face"):
            print("opencv-face: OK")
        else:
            print("opencv-face: WARN (LBPH recognizer support is missing)")
            exit_code = 1
        create_detector(config)
        print("detector: OK")
    except RuntimeError as exc:
        print(f"opencv: WARN ({exc})")
        exit_code = 1

    return exit_code


def command_audit(args: argparse.Namespace) -> int:
    config = _load(args)
    audit = build_dataset_audit(config)

    if args.json:
        print(json.dumps(audit.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_audit_text(audit))

    return 1 if audit.warnings and args.fail_on_warning else 0


def command_collect(args: argparse.Namespace) -> int:
    config = _load(args)
    cv2 = require_cv2()
    detector = create_detector(config)
    camera = _open_camera(cv2, config.camera_source, config)

    output_dir = person_directory(config.faces_dir, args.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_count = args.count or config.sample_count
    saved = 0
    next_index = next_sample_index(output_dir)
    show_window = config.display and not args.no_window

    print(f"collecting: {output_dir}")
    print("press q to stop" if show_window else "press Ctrl+C to stop")

    try:
        while saved < target_count:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera returned an empty frame.")

            gray, faces = detect_faces(frame, detector, config)
            if len(faces) > 0:
                face = max(faces, key=lambda item: item[2] * item[3])
                crop = crop_face(gray, face, config.face_size)
                image_path = output_dir / f"{next_index:06d}.png"
                if not cv2.imwrite(str(image_path), crop):
                    raise RuntimeError(f"Could not write face sample: {image_path}")
                saved += 1
                next_index += 1
                x, y, width, height = face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (20, 180, 90), 2)
                cv2.putText(
                    frame,
                    f"saved {saved}/{target_count}",
                    (x, max(24, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (20, 180, 90),
                    2,
                )

            if show_window:
                cv2.imshow("collect faces", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        if show_window:
            cv2.destroyAllWindows()

    print(f"saved: {saved}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = _load(args)
    dataset_status = validate_training_dataset(config.faces_dir)
    for warning in dataset_status.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not dataset_status.is_valid:
        raise DatasetError("; ".join(dataset_status.errors))

    cv2 = require_cv2()
    np = require_numpy()
    rows = list(iter_person_images(config.faces_dir))
    labels = build_label_map(person for person, _ in rows)
    label_by_person = {person: index for index, person in labels.items()}

    images = []
    target_labels = []
    for person, image_path in rows:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"skip unreadable image: {image_path}", file=sys.stderr)
            continue
        images.append(cv2.resize(image, config.face_size))
        target_labels.append(label_by_person[person])

    if not images:
        raise DatasetError("No readable face images found.")

    recognizer = create_recognizer()
    recognizer.train(images, np.array(target_labels, dtype=np.int32))
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(config.model_path))
    save_label_map(labels, config.labels_path)

    print(f"model: {config.model_path}")
    print(f"labels: {config.labels_path}")
    print(f"people: {len(labels)}")
    print(f"images: {len(images)}")
    return 0


def command_recognize(args: argparse.Namespace) -> int:
    config = _load(args)
    model_result = load_model(config.model_path, config.labels_path)
    if not model_result.success or model_result.model is None:
        raise DatasetError(f"{model_result.error}. Run the train command first.")
    labels = model_result.labels
    cv2 = require_cv2()
    detector = create_detector(config)
    recognizer = model_result.model
    camera = _open_camera(cv2, config.camera_source, config)
    show_window = config.display and not args.no_window

    print("recognition started")
    print("press q to stop" if show_window else "press Ctrl+C to stop")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera returned an empty frame.")

            step_result = run_recognition_step(frame, detector, recognizer, config, labels)
            for detection in step_result.detections:
                if config.debug or not show_window:
                    print(format_detection(detection))
                if config.draw_bounding_boxes:
                    x, y, width, height = detection.box
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 140, 240), 2)
                    cv2.putText(
                        frame,
                        f"{detection.label} ({detection.confidence:.1f})",
                        (x, max(24, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (30, 140, 240),
                        2,
                    )

            if show_window:
                cv2.imshow("recognize faces", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        if show_window:
            cv2.destroyAllWindows()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raspberry Pi face recognition toolkit")
    parser.add_argument("--config", default="config.json", help="Path to JSON or YAML configuration file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local dependencies and configuration")
    doctor.set_defaults(func=command_doctor)

    audit = subparsers.add_parser("audit", help="Audit local dataset/model metadata without reading image pixels")
    audit.add_argument("--json", action="store_true", help="Print audit result as JSON")
    audit.add_argument("--fail-on-warning", action="store_true", help="Return exit code 1 when warnings are found")
    audit.set_defaults(func=command_audit)

    collect = subparsers.add_parser("collect", help="Collect face samples for one person")
    collect.add_argument("--name", required=True, help="Person name to store under data/faces")
    collect.add_argument("--count", type=int, default=None, help="Number of samples to collect")
    collect.add_argument("--no-window", action="store_true", help="Run without opening a preview window")
    collect.set_defaults(func=command_collect)

    train = subparsers.add_parser("train", help="Train the local LBPH face recognizer")
    train.set_defaults(func=command_train)

    recognize = subparsers.add_parser("recognize", help="Run live face recognition")
    recognize.add_argument("--no-window", action="store_true", help="Run without opening a preview window")
    recognize.set_defaults(func=command_recognize)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (DatasetError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
