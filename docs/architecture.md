# Architecture

PiSight-Omni is a local edge vision toolkit built around live face embeddings, vector search, optional VLM reasoning and safe advisory telemetry.

## 1. High-level architecture

```text
Camera / Video Source
  -> OpenCV VideoCapture
  -> MTCNN Face Detection
  -> In-memory Face Tensor
  -> InceptionResnetV1 Embedding
  -> FAISS Vector Index
  -> Match Result
  -> Optional VLM Scene Reasoning
  -> Advisory Action Decision
  -> Optional Omni Telemetry / Dry-Run Swarm Publish
  -> Console / Optional Preview Window
```

The default flow does not write cropped face images to disk.

## 2. Component responsibilities

- `raspberry_face_recognition.cli`: command orchestration for doctor, collect/enroll, train compatibility, recognize and audit.
- `raspberry_face_recognition.config`: JSON/YAML loading, path resolution and validation.
- `raspberry_face_recognition.vision`: OpenCV imports plus deep detector/recognizer factories.
- `raspberry_face_recognition.vectordb`: FAISS index, label metadata and vector search.
- `raspberry_face_recognition.agent`: optional VLM scene analysis and safe action dispatch.
- `raspberry_face_recognition.omni_core`: local telemetry, non-clinical visual signal summary and optional MQTT dry-run publisher.
- `raspberry_face_recognition.audit`: legacy filesystem metadata checks.
- `raspberry_face_recognition.dataset`, `model`, `recognition`: compatibility helpers for the former Haar/LBPH flow and tests.

## 3. Enrollment flow

```text
Camera frame
  -> RGB conversion
  -> MTCNN detection
  -> single-face guard
  -> 512-dimensional embedding
  -> FAISS add
  -> JSON label metadata update
```

Frames and tensors are transient runtime data. The durable artifacts are the FAISS index and labels JSON.

## 4. Recognition flow

```text
Camera frame
  -> MTCNN detection
  -> embedding extraction
  -> FAISS nearest-neighbor search
  -> distance threshold
  -> label or unknown
```

`recognition.confidence_threshold` is a vector distance threshold, not an accuracy percentage.

## 5. Error handling

Expected failures use clear CLI errors:

- missing OpenCV
- missing PyTorch/facenet-pytorch/FAISS deep runtime
- unavailable camera source
- empty vector index
- invalid config values
- FAISS index/metadata dimension mismatch

## 6. Logging approach

The CLI writes to stdout/stderr. In systemd deployments, journald captures that output. Structured file logging is still roadmap work.

## 7. Agentic AI flow

The `autonom` command combines local vector recognition with optional VLM scene reasoning:

```text
frame + FAISS match context
  -> VLM JSON decision
  -> safe dispatcher
  -> dry-run action log
```

The dispatcher does not execute GPIO, locks, relays, Telegram or network actions. Any future real-world action adapter should be reviewed as a safety-critical integration.

## 8. PiSight-Omni telemetry flow

The `omni` command extends the advisory loop with explicit local context:

```text
frame + FAISS match context
  -> Omni telemetry builder
  -> visual signal summary
  -> VLM JSON decision
  -> safe dispatcher
  -> optional dry-run MQTT telemetry
```

The visual signal summary is not a heart-rate, stress, liveness or health detector. It is only a rolling green-channel variation indicator for debugging and future research hooks.

## 9. Operational constraints

Deep models are heavier than the former Haar/LBPH path. Measure FPS on the actual device and do not publish hardware-independent performance claims.
