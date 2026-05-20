"""Agentic vision layer for PiSight-X.

The agent is intentionally advisory by default. It can ask a Vision Language
Model for a scene decision, but physical actions are not executed unless a
future, explicitly reviewed action adapter is added.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import os
from typing import Any, Iterable, Mapping, Optional


ALLOWED_ACTIONS = {"IGNORE", "GREET", "ALERT", "LOCKDOWN"}
SAFE_ACTION_MODES = {"dry_run", "disabled"}


@dataclass(frozen=True)
class AgentDecision:
    analysis: str
    action: str
    message: str = ""


class AgentError(RuntimeError):
    """Raised when agent analysis cannot be completed."""


def normalize_decision(payload: Mapping[str, Any]) -> AgentDecision:
    action = str(payload.get("action", "IGNORE")).upper().strip()
    if action not in ALLOWED_ACTIONS:
        action = "IGNORE"
    return AgentDecision(
        analysis=str(payload.get("analysis", "")).strip(),
        action=action,
        message=str(payload.get("message", "")).strip(),
    )


def parse_decision_json(text: str) -> AgentDecision:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AgentError(f"Agent response was not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AgentError("Agent response must be a JSON object.")
    return normalize_decision(payload)


def format_detected_people(faiss_results: Iterable[Mapping[str, Any]]) -> str:
    rows = []
    for result in faiss_results:
        label = str(result.get("label", "unknown"))
        try:
            distance = float(result.get("distance", float("inf")))
            rows.append(f"{label} (distance={distance:.3f})")
        except (TypeError, ValueError):
            rows.append(f"{label} (distance=unknown)")
    return ", ".join(rows) if rows else "no recognized face vectors"


class SafeActionDispatcher:
    """Action boundary that keeps physical actions disabled by default."""

    def __init__(self, mode: str = "dry_run") -> None:
        if mode not in SAFE_ACTION_MODES:
            raise ValueError(f"Unsupported action mode: {mode}. Expected one of {sorted(SAFE_ACTION_MODES)}.")
        self.mode = mode

    def execute(self, decision: AgentDecision) -> AgentDecision:
        if self.mode == "disabled":
            print(f"[AGENT][DISABLED] action={decision.action} analysis={decision.analysis}")
            return decision

        print(f"[AGENT][DRY_RUN] action={decision.action}")
        print(f"[AGENT][ANALYSIS] {decision.analysis}")
        if decision.message:
            print(f"[AGENT][MESSAGE] {decision.message}")
        if decision.action in {"ALERT", "LOCKDOWN"}:
            print("[AGENT][SAFEGUARD] No GPIO, lock, relay, Telegram or network action was executed.")
        return decision


class AutonomousAgent:
    """Vision-language reasoning wrapper.

    The default OpenAI client is loaded lazily so tests and non-agent commands do
    not require the `openai` package or an API key.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4.1-mini",
        max_tokens: int = 300,
        action_mode: str = "dry_run",
        client: Optional[Any] = None,
        cv2_module: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self._client = client
        self._cv2 = cv2_module
        self.dispatcher = SafeActionDispatcher(action_mode)

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AgentError("OpenAI SDK is not installed. Install with: python -m pip install -e \".[agent]\"") from exc

        kwargs: dict[str, str] = {}
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _cv2_module(self) -> Any:
        if self._cv2 is not None:
            return self._cv2
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AgentError("OpenCV is required to encode frames for VLM analysis.") from exc
        self._cv2 = cv2
        return self._cv2

    def _encode_image(self, frame: Any) -> str:
        cv2 = self._cv2_module()
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            raise AgentError("Could not encode frame for VLM analysis.")
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_prompt(self, faiss_results: Iterable[Mapping[str, Any]]) -> str:
        detected_people = format_detected_people(faiss_results)
        return (
            "You are an advisory local security assistant for PiSight-X.\n"
            "Use the image and local FAISS face-vector matches to describe the scene and choose one action.\n"
            f"Local face-vector matches: [{detected_people}]\n\n"
            "Rules:\n"
            "- Do not identify people beyond the provided local labels.\n"
            "- Do not infer sensitive attributes.\n"
            "- Treat the output as advisory; physical actions are disabled by default.\n"
            "- Choose exactly one action: IGNORE, GREET, ALERT, LOCKDOWN.\n"
            "- Use LOCKDOWN only for an immediate visible threat.\n\n"
            "Return only JSON with this schema:\n"
            '{"analysis":"short scene analysis","action":"IGNORE|GREET|ALERT|LOCKDOWN","message":"short message"}'
        )

    def analyze_scene_and_act(self, frame: Any, faiss_results: Iterable[Mapping[str, Any]]) -> AgentDecision:
        data_url = self._encode_image(frame)
        prompt = self._build_prompt(faiss_results)
        client = self._client_instance()

        try:
            response = client.responses.create(
                model=self.model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                max_output_tokens=self.max_tokens,
            )
            output_text = getattr(response, "output_text", "")
            decision = parse_decision_json(output_text)
        except Exception as exc:
            raise AgentError(f"VLM agent analysis failed: {exc}") from exc

        return self.dispatcher.execute(decision)
