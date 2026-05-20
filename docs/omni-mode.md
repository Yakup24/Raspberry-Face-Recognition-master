# PiSight-Omni Mode

PiSight-Omni mode is the safe agentic layer on top of local face embeddings and FAISS search. It adds local telemetry, optional VLM scene reasoning and dry-run swarm publishing without turning the project into an uncontrolled automation system.

## 1. Scope

`omni` is for edge AI experimentation:

- local face-vector match context
- advisory VLM scene analysis
- non-clinical visual signal telemetry
- optional dry-run MQTT swarm messages
- explicit safety boundaries around actions

It is not AGI, ASI, a medical system, a security authentication product or a physical access-control system.

## 2. Runtime flow

```text
Camera / Video Source
  -> OpenCV Frame Capture
  -> MTCNN Face Detection
  -> InceptionResnetV1 Embeddings
  -> FAISS Match Context
  -> Omni Telemetry Builder
  -> Optional VLM Scene Reasoning
  -> Safe Action Dispatcher
  -> Optional Dry-Run MQTT Swarm Output
```

## 3. Command

```sh
pisight --config config.yaml omni --no-window --interval-frames 30 --swarm
```

`--swarm` enables the publisher interface, but it remains dry-run unless live MQTT is explicitly requested.

```sh
pisight --config config.yaml omni --no-window --swarm --swarm-live
```

Live MQTT publishing should be used only after reviewing broker security, network privacy and log retention.

## 4. Configuration

```yaml
omni:
  device_id: "node-001"
  swarm_enabled: false
  swarm_host: "localhost"
  swarm_port: 1883
  swarm_topic: "pisight/omni/swarm"
  swarm_dry_run: true
```

## 5. Telemetry

Omni telemetry includes:

- device id
- frame index
- detected face count
- recognized placeholder labels
- unknown count
- visual signal variation status
- swarm publish status

It does not include raw images or embeddings.

## 6. Safety boundaries

The implementation intentionally avoids:

- executing model-generated code
- controlling locks, relays, doors or GPIO pins
- sending Telegram/e-mail alerts from the core loop
- claiming heart-rate, stress, liveness or medical inference
- claiming quantum, BCI, AGI or self-evolving behavior

Future physical action adapters should be separate modules with explicit tests, review, consent model, audit logging and deployment-specific risk assessment.
