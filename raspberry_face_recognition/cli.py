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
)
from .vectordb import FaceVectorDB, require_faiss
from .vision import (
    create_deep_detector,
    create_deep_recognizer,
    detect_and_embed,
    get_device,
    require_cv2,
)


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
    print(f"backend: {config.backend}")
    print(f"vector_index: {config.vector_index_path}")

    exit_code = 0
    try:
        cv2 = require_cv2()
        print(f"opencv: OK ({cv2.__version__})")
        print("opencv-camera: OK")
    except RuntimeError as exc:
        print(f"opencv: WARN ({exc})")
        exit_code = 1
    try:
        device = get_device()
        print(f"torch-device: OK ({device})")
        require_faiss()
        print("faiss: OK")
    except RuntimeError as exc:
        print(f"deep-runtime: WARN ({exc})")
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
    device = get_device()
    detector = create_deep_detector(device)
    recognizer = create_deep_recognizer(device)
    db = FaceVectorDB(config.vector_index_path, config.vector_labels_path, config.embedding_dim)
    camera = _open_camera(cv2, config.camera_source, config)
    target_count = args.count or config.sample_count
    saved = 0
    show_window = config.display and not args.no_window

    print(f"enrolling vectors for: {args.name}")
    print(f"raw face images are not written to disk; index: {config.vector_index_path}")
    print("press q to stop" if show_window else "press Ctrl+C to stop")

    try:
        while saved < target_count:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera returned an empty frame.")

            boxes, embeddings = detect_and_embed(frame, detector, recognizer, device)
            if len(embeddings) == 1:
                db.add_face(embeddings[0], args.name)
                saved += 1
                print(f"vector saved: {saved}/{target_count}")
            elif config.debug and len(embeddings) > 1:
                print("skip frame: multiple faces detected")

            if show_window:
                for box in boxes:
                    x1, y1, x2, y2 = [int(value) for value in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 180, 90), 2)
                cv2.putText(
                    frame,
                    f"vectors {saved}/{target_count}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (20, 180, 90),
                    2,
                )

            if show_window:
                cv2.imshow("PiSight-X enrollment", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        if show_window:
            cv2.destroyAllWindows()

    print(f"vectors saved: {saved}")
    print("enrollment complete: no raw face images were stored.")
    return 0


def command_train(args: argparse.Namespace) -> int:
    _load(args)
    print("train is not required in the PiSight-X embedding pipeline.")
    print("Use `collect --name <demo-user>` to enroll vectors, then `recognize`.")
    return 0


def command_recognize(args: argparse.Namespace) -> int:
    config = _load(args)
    cv2 = require_cv2()
    device = get_device()
    detector = create_deep_detector(device)
    recognizer = create_deep_recognizer(device)
    db = FaceVectorDB(config.vector_index_path, config.vector_labels_path, config.embedding_dim)
    if db.count == 0:
        raise DatasetError("No enrolled face vectors found. Run the collect command first.")
    camera = _open_camera(cv2, config.camera_source, config)
    show_window = config.display and not args.no_window

    print("PiSight-X recognition started")
    print("press q to stop" if show_window else "press Ctrl+C to stop")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera returned an empty frame.")

            boxes, embeddings = detect_and_embed(frame, detector, recognizer, device)
            for index, embedding in enumerate(embeddings):
                result = db.search_face(embedding, threshold=config.confidence_threshold, unknown_label=config.unknown_label)
                if config.debug or not show_window:
                    print(f"[DETECTED] label={result.label} distance={result.distance:.3f}")
                if config.draw_bounding_boxes:
                    x1, y1, x2, y2 = [int(value) for value in boxes[index]]
                    color = (20, 180, 90) if result.matched else (30, 140, 240)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{result.label} ({result.distance:.2f})",
                        (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            if show_window:
                cv2.imshow("PiSight-X", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        if show_window:
            cv2.destroyAllWindows()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PiSight-X Raspberry Pi face embedding toolkit")
    parser.add_argument("--config", default="config.json", help="Path to JSON or YAML configuration file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local dependencies and configuration")
    doctor.set_defaults(func=command_doctor)

    audit = subparsers.add_parser("audit", help="Audit legacy dataset/model metadata without reading image pixels")
    audit.add_argument("--json", action="store_true", help="Print audit result as JSON")
    audit.add_argument("--fail-on-warning", action="store_true", help="Return exit code 1 when warnings are found")
    audit.set_defaults(func=command_audit)

    collect = subparsers.add_parser("collect", help="Enroll face embeddings for one person without storing images")
    collect.add_argument("--name", required=True, help="Label to attach to enrolled embeddings")
    collect.add_argument("--count", type=int, default=None, help="Number of samples to collect")
    collect.add_argument("--no-window", action="store_true", help="Run without opening a preview window")
    collect.set_defaults(func=command_collect)

    enroll = subparsers.add_parser("enroll", help="Alias for collect")
    enroll.add_argument("--name", required=True, help="Label to attach to enrolled embeddings")
    enroll.add_argument("--count", type=int, default=None, help="Number of embeddings to enroll")
    enroll.add_argument("--no-window", action="store_true", help="Run without opening a preview window")
    enroll.set_defaults(func=command_collect)

    train = subparsers.add_parser("train", help="Compatibility command; no offline training is required")
    train.set_defaults(func=command_train)

    recognize = subparsers.add_parser("recognize", help="Run live vector-based face recognition")
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
