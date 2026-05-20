"""Command line interface for Raspberry Pi face recognition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional, Sequence, Union

from .agent import AgentError, AutonomousAgent
from .audit import build_dataset_audit, format_audit_text
from .config import load_config
from .dataset import (
    DatasetError,
)
from .omni_core import OmniContextBuilder, SwarmPublisher
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
                cv2.imshow("PiSight-Omni enrollment", frame)
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
    print("train is not required in the PiSight-Omni embedding pipeline.")
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

    print("PiSight-Omni recognition started")
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
                cv2.imshow("PiSight-Omni", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        if show_window:
            cv2.destroyAllWindows()

    return 0


def _agent_loop(args: argparse.Namespace, *, omni_mode: bool = False) -> int:
    config = _load(args)
    cv2 = require_cv2()
    device = get_device()
    detector = create_deep_detector(device)
    recognizer = create_deep_recognizer(device)
    db = FaceVectorDB(config.vector_index_path, config.vector_labels_path, config.embedding_dim)
    camera = _open_camera(cv2, config.camera_source, config)
    show_window = config.display and not args.no_window
    interval_frames = args.interval_frames or config.agent_interval_frames
    max_frames = args.max_frames
    if interval_frames < 1:
        raise ValueError("interval-frames must be greater than zero.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max-frames must be greater than zero.")
    swarm_enabled = bool(getattr(args, "swarm", False) or config.omni_swarm_enabled)
    swarm_live = bool(getattr(args, "swarm_live", False))
    swarm = SwarmPublisher(
        enabled=swarm_enabled,
        dry_run=False if swarm_live else config.omni_swarm_dry_run,
        broker_host=config.omni_swarm_host,
        broker_port=config.omni_swarm_port,
        topic=config.omni_swarm_topic,
    )
    omni_context = None
    if omni_mode:
        omni_context = OmniContextBuilder(device_id=getattr(args, "device_id", None) or config.omni_device_id)
    agent = AutonomousAgent(
        base_url=config.agent_base_url or None,
        model_name=args.model or config.agent_model,
        max_tokens=config.agent_max_tokens,
        action_mode=config.agent_action_mode,
    )

    mode_label = "PiSight-Omni advisory agent" if omni_mode else "PiSight-Omni autonomous agent"
    print(f"{mode_label} started")
    print("VLM analysis is opt-in and may send frames to the configured model endpoint.")
    print("actions are advisory only; GPIO/locks/network alerts are not executed")
    if omni_mode:
        print(f"omni device_id={omni_context.device_id if omni_context else config.omni_device_id}")
        print(f"swarm telemetry={swarm.status}")

    frame_counter = 0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera returned an empty frame.")

            frame_counter += 1
            boxes, embeddings = detect_and_embed(frame, detector, recognizer, device)
            faiss_results: list[dict[str, Any]] = []

            for index, embedding in enumerate(embeddings):
                result = db.search_face(
                    embedding,
                    threshold=config.confidence_threshold,
                    unknown_label=config.unknown_label,
                )
                box = boxes[index]
                faiss_results.append(
                    {
                        "label": result.label,
                        "distance": result.distance,
                        "matched": result.matched,
                        "box": [float(value) for value in box],
                    }
                )

                if config.draw_bounding_boxes:
                    x1, y1, x2, y2 = [int(value) for value in box]
                    color = (20, 180, 90) if result.matched else (30, 140, 240)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        result.label,
                        (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            if faiss_results and frame_counter % interval_frames == 0:
                telemetry = None
                if omni_context is not None:
                    telemetry = omni_context.build(
                        frame=frame,
                        faiss_results=faiss_results,
                        frame_index=frame_counter,
                        unknown_label=config.unknown_label,
                        swarm_status=swarm.status,
                    )
                    if config.debug or not show_window:
                        print(f"[OMNI][TELEMETRY] {json.dumps(telemetry.to_dict(), sort_keys=True)}")
                try:
                    decision = agent.analyze_scene_and_act(frame, faiss_results)
                    if telemetry is not None and decision.action in {"ALERT", "LOCKDOWN"}:
                        payload = telemetry.to_dict()
                        payload["agent_action"] = decision.action
                        payload["agent_message"] = decision.message
                        swarm.publish_alert(payload)
                except AgentError as exc:
                    print(f"agent warning: {exc}", file=sys.stderr)

            if show_window:
                cv2.imshow("PiSight-Omni Agent View", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames is not None and frame_counter >= max_frames:
                break
    finally:
        camera.release()
        if show_window:
            cv2.destroyAllWindows()

    return 0


def command_autonom(args: argparse.Namespace) -> int:
    return _agent_loop(args, omni_mode=False)


def command_omni(args: argparse.Namespace) -> int:
    return _agent_loop(args, omni_mode=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PiSight-Omni Raspberry Pi face embedding and agentic vision toolkit")
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

    autonom = subparsers.add_parser("autonom", help="Run the advisory Vision-Language autonomous agent")
    autonom.add_argument("--no-window", action="store_true", help="Run without opening a preview window")
    autonom.add_argument("--interval-frames", type=int, default=None, help="Run VLM analysis every N frames")
    autonom.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    autonom.add_argument("--model", default=None, help="Override the configured VLM model name")
    autonom.set_defaults(func=command_autonom)

    omni = subparsers.add_parser("omni", help="Run PiSight-Omni advisory agent mode with local telemetry")
    omni.add_argument("--no-window", action="store_true", help="Run without opening a preview window")
    omni.add_argument("--interval-frames", type=int, default=None, help="Run VLM analysis every N frames")
    omni.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    omni.add_argument("--model", default=None, help="Override the configured VLM model name")
    omni.add_argument("--device-id", default=None, help="Override omni.device_id for telemetry")
    omni.add_argument("--swarm", action="store_true", help="Enable swarm telemetry in dry-run mode unless --swarm-live is set")
    omni.add_argument("--swarm-live", action="store_true", help="Allow MQTT publish to the configured broker; physical actions remain disabled")
    omni.set_defaults(func=command_omni)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (AgentError, DatasetError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
