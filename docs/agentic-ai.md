# Agentic AI

PiSight-X includes an optional advisory agent loop named `autonom`.

## Perception -> Reasoning -> Action

```text
Camera Frame
  -> Face Embeddings
  -> FAISS Match Context
  -> VLM Scene Analysis
  -> JSON Decision
  -> Safe Action Dispatcher
```

## Command

Install agent dependencies:

```sh
python -m pip install -e ".[deep,agent]"
```

```sh
pisight --config config.yaml autonom --no-window --interval-frames 30
```

The agent does not call a VLM on every frame. `interval_frames` controls how often scene reasoning is requested.

## VLM provider

The default client uses the OpenAI Python SDK and the Responses API shape for image inputs. The `agent.base_url` config value can point to an OpenAI-compatible local endpoint such as an internal gateway or local model server.

```yaml
agent:
  model: "gpt-4.1-mini"
  base_url: ""
  interval_frames: 30
  max_tokens: 300
  action_mode: "dry_run"
```

If `base_url` is empty, the SDK default endpoint is used and `OPENAI_API_KEY` should be provided through the environment. Do not commit API keys.

## Decision schema

The VLM is asked to return JSON:

```json
{
  "analysis": "short scene analysis",
  "action": "IGNORE",
  "message": "short message"
}
```

Allowed actions:

- `IGNORE`
- `GREET`
- `ALERT`
- `LOCKDOWN`

## Safety boundary

Actions are advisory by default. The current dispatcher supports only:

- `dry_run`: print the decision and explicitly skip GPIO, locks, relays, Telegram or network actions.
- `disabled`: print that action execution is disabled.

Physical actions require a future reviewed adapter, tests and a deployment-specific risk assessment.

## Privacy boundary

`recognize` stays local. `autonom` can send encoded frames to the configured VLM endpoint. Use a local VLM endpoint if camera frames must not leave the device or local network.
