"""Safe PiSight-Omni context and telemetry helpers.

This module deliberately keeps the "Omni" layer advisory. It can summarize
local recognition context, estimate basic visual signal variation from a face
ROI, and optionally publish dry-run swarm telemetry. It does not execute
generated code, control locks, infer health status, or claim calibrated
biometrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RPPGSignal:
    """Non-clinical visual signal summary for a face crop."""

    status: str
    samples: int
    variance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "samples": self.samples,
            "variance": round(self.variance, 4),
        }


@dataclass(frozen=True)
class OmniTelemetry:
    """Local context sent to logs, dry-run swarm output, or future adapters."""

    device_id: str
    frame_index: int
    faces_detected: int
    recognized_labels: list[str] = field(default_factory=list)
    unknown_count: int = 0
    signal: RPPGSignal = field(default_factory=lambda: RPPGSignal("not_available", 0, 0.0))
    depth_status: str = "not_configured"
    swarm_status: str = "disabled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "frame_index": self.frame_index,
            "faces_detected": self.faces_detected,
            "recognized_labels": list(self.recognized_labels),
            "unknown_count": self.unknown_count,
            "signal": self.signal.to_dict(),
            "depth_status": self.depth_status,
            "swarm_status": self.swarm_status,
        }


@dataclass(frozen=True)
class SwarmPublishResult:
    success: bool
    status: str
    topic: str


class RPPGEstimator:
    """Rolling green-channel signal variation estimator.

    The output is useful as a debugging/telemetry hint only. It is not a heart
    rate monitor, liveness check, stress detector, or medical signal.
    """

    def __init__(self, *, window_size: int = 30, min_samples: int = 8, variation_threshold: float = 15.0) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2.")
        if min_samples < 1:
            raise ValueError("min_samples must be greater than zero.")
        self.window_size = window_size
        self.min_samples = min_samples
        self.variation_threshold = variation_threshold
        self._green_means: list[float] = []

    def update(self, face_crop: Any) -> RPPGSignal:
        if face_crop is None:
            return RPPGSignal("no_face_roi", len(self._green_means), 0.0)

        crop = np.asarray(face_crop)
        if crop.size == 0 or crop.ndim != 3 or crop.shape[2] < 2:
            return RPPGSignal("invalid_face_roi", len(self._green_means), 0.0)

        mean_green = float(np.mean(crop[:, :, 1]))
        self._green_means.append(mean_green)
        if len(self._green_means) > self.window_size:
            self._green_means.pop(0)

        variance = float(np.var(self._green_means))
        if len(self._green_means) < self.min_samples:
            return RPPGSignal("insufficient_signal", len(self._green_means), variance)
        if variance > self.variation_threshold:
            return RPPGSignal("motion_or_signal_variation", len(self._green_means), variance)
        return RPPGSignal("stable_visual_signal", len(self._green_means), variance)


class OmniContextBuilder:
    """Build safe local telemetry from frame and recognition results."""

    def __init__(self, *, device_id: str = "node-001", signal_estimator: Optional[RPPGEstimator] = None) -> None:
        self.device_id = device_id
        self.signal_estimator = signal_estimator or RPPGEstimator()

    def build(
        self,
        *,
        frame: Any,
        faiss_results: Sequence[Mapping[str, Any]],
        frame_index: int,
        unknown_label: str = "unknown",
        swarm_status: str = "disabled",
    ) -> OmniTelemetry:
        recognized_labels = [
            str(result.get("label", unknown_label))
            for result in faiss_results
            if bool(result.get("matched")) and str(result.get("label", unknown_label)) != unknown_label
        ]
        unknown_count = sum(
            1
            for result in faiss_results
            if not bool(result.get("matched")) or str(result.get("label", unknown_label)) == unknown_label
        )
        signal = self.signal_estimator.update(_first_face_crop(frame, faiss_results))
        return OmniTelemetry(
            device_id=self.device_id,
            frame_index=frame_index,
            faces_detected=len(faiss_results),
            recognized_labels=recognized_labels,
            unknown_count=unknown_count,
            signal=signal,
            swarm_status=swarm_status,
        )


class SwarmPublisher:
    """Optional MQTT publisher with dry-run safety by default."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        dry_run: bool = True,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic: str = "pisight/omni/swarm",
        client: Optional[Any] = None,
    ) -> None:
        if broker_port < 1:
            raise ValueError("broker_port must be greater than zero.")
        self.enabled = enabled
        self.dry_run = dry_run
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self._client = client
        self._connected = False

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.dry_run:
            return "dry_run"
        return "live"

    def publish_alert(self, payload: Mapping[str, Any]) -> SwarmPublishResult:
        if not self.enabled:
            return SwarmPublishResult(True, "disabled", self.topic)

        serialized = json.dumps(dict(payload), sort_keys=True)
        if self.dry_run:
            print(f"[OMNI][SWARM][DRY_RUN] topic={self.topic} payload={serialized}")
            return SwarmPublishResult(True, "dry_run", self.topic)

        client = self._client_instance()
        if not self._connected:
            client.connect(self.broker_host, self.broker_port, 60)
            self._connected = True
        client.publish(self.topic, serialized)
        return SwarmPublishResult(True, "published", self.topic)

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import paho.mqtt.client as mqtt  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("MQTT swarm mode requires paho-mqtt. Install with: python -m pip install -e \".[omni]\"") from exc
        self._client = mqtt.Client()
        return self._client


def _first_face_crop(frame: Any, faiss_results: Sequence[Mapping[str, Any]]) -> Any:
    if frame is None or not faiss_results:
        return None
    array = np.asarray(frame)
    if array.ndim < 2:
        return None

    height, width = array.shape[:2]
    for result in faiss_results:
        box = result.get("box")
        if not isinstance(box, Sequence) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [int(float(value)) for value in box]
        except (TypeError, ValueError):
            continue
        left = max(0, min(width, x1))
        right = max(0, min(width, x2))
        top = max(0, min(height, y1))
        bottom = max(0, min(height, y2))
        if right > left and bottom > top:
            return array[top:bottom, left:right]
    return None
