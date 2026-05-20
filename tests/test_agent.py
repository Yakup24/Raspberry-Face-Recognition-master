from types import SimpleNamespace

import pytest

from raspberry_face_recognition.agent import (
    AgentError,
    AutonomousAgent,
    SafeActionDispatcher,
    format_detected_people,
    parse_decision_json,
)


class FakeCV2:
    @staticmethod
    def imencode(extension, frame):
        return True, b"fake-jpeg"


class FakeResponses:
    def __init__(self, output_text):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    def __init__(self, output_text):
        self.responses = FakeResponses(output_text)


def test_parse_decision_json_normalizes_unknown_action():
    decision = parse_decision_json('{"analysis":"ok","action":"wave","message":"hello"}')

    assert decision.action == "IGNORE"
    assert decision.analysis == "ok"


def test_parse_decision_json_rejects_invalid_json():
    with pytest.raises(AgentError):
        parse_decision_json("not-json")


def test_format_detected_people_outputs_distances():
    text = format_detected_people([{"label": "demo-user", "distance": 0.42}])

    assert "demo-user" in text
    assert "0.420" in text


def test_safe_action_dispatcher_dry_run_blocks_physical_actions(capsys):
    dispatcher = SafeActionDispatcher("dry_run")
    decision = parse_decision_json('{"analysis":"threat visible","action":"LOCKDOWN","message":"check door"}')

    returned = dispatcher.execute(decision)

    output = capsys.readouterr().out
    assert returned.action == "LOCKDOWN"
    assert "No GPIO" in output


def test_agent_uses_responses_api_and_returns_decision():
    client = FakeClient('{"analysis":"known person present","action":"GREET","message":"hello"}')
    agent = AutonomousAgent(client=client, cv2_module=FakeCV2(), action_mode="disabled")

    decision = agent.analyze_scene_and_act(
        object(),
        [{"label": "demo-user", "distance": 0.2, "box": [1, 2, 3, 4]}],
    )

    assert decision.action == "GREET"
    call = client.responses.calls[0]
    assert call["model"] == "gpt-4.1-mini"
    assert call["input"][0]["content"][1]["type"] == "input_image"
