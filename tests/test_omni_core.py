import numpy as np

from raspberry_face_recognition.omni_core import OmniContextBuilder, RPPGEstimator, SwarmPublisher


def test_rppg_estimator_reports_stable_visual_signal_after_window():
    estimator = RPPGEstimator(min_samples=3, variation_threshold=10.0)
    frame = np.full((8, 8, 3), 42, dtype=np.uint8)

    signal = estimator.update(frame)
    signal = estimator.update(frame)
    signal = estimator.update(frame)

    assert signal.status == "stable_visual_signal"
    assert signal.samples == 3
    assert signal.variance == 0.0


def test_rppg_estimator_flags_variation_without_claiming_biometrics():
    estimator = RPPGEstimator(min_samples=3, variation_threshold=1.0)

    for value in [10, 80, 10]:
        signal = estimator.update(np.full((8, 8, 3), value, dtype=np.uint8))

    assert signal.status == "motion_or_signal_variation"
    assert signal.variance > 1.0


def test_omni_context_builder_summarizes_recognition_results():
    builder = OmniContextBuilder(device_id="camera-alpha", signal_estimator=RPPGEstimator(min_samples=1))
    frame = np.full((20, 20, 3), 20, dtype=np.uint8)
    results = [
        {"label": "demo-user", "matched": True, "box": [1, 1, 10, 10]},
        {"label": "unknown", "matched": False, "box": [12, 12, 18, 18]},
    ]

    telemetry = builder.build(
        frame=frame,
        faiss_results=results,
        frame_index=7,
        unknown_label="unknown",
        swarm_status="dry_run",
    )

    payload = telemetry.to_dict()
    assert payload["device_id"] == "camera-alpha"
    assert payload["faces_detected"] == 2
    assert payload["recognized_labels"] == ["demo-user"]
    assert payload["unknown_count"] == 1
    assert payload["swarm_status"] == "dry_run"


def test_swarm_publisher_dry_run_does_not_require_mqtt(capsys):
    publisher = SwarmPublisher(enabled=True, dry_run=True, topic="pisight/omni/test")

    result = publisher.publish_alert({"agent_action": "ALERT"})

    output = capsys.readouterr().out
    assert result.success is True
    assert result.status == "dry_run"
    assert "pisight/omni/test" in output


def test_swarm_publisher_disabled_is_noop(capsys):
    publisher = SwarmPublisher(enabled=False)

    result = publisher.publish_alert({"agent_action": "LOCKDOWN"})

    output = capsys.readouterr().out
    assert result.status == "disabled"
    assert output == ""
