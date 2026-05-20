# Privacy

PiSight-Omni minimizes raw image storage, but it does not eliminate biometric risk.

## 1. Local-only processing

Frames are processed locally. The repository does not include upload, cloud sync or remote recognition code.

## 2. Raw face images

The default `collect/enroll` workflow does not write cropped face images to `data/faces/`. Frames are converted to tensors and embeddings during runtime.

## 3. Embeddings and vector index

The FAISS index and label metadata are durable local artifacts:

```text
data/embeddings/faiss.index
data/embeddings/labels.json
```

Embeddings are biometric-derived data. They are not raw photos, but they should still be treated as sensitive and kept out of public repositories.

## 4. Remote upload

PiSight-Omni has no remote upload workflow by default. Any future cloud storage, dashboard sync or remote logging should be reviewed as a privacy-impacting feature.

The `autonom` and `omni` commands are explicit opt-in exceptions: they can send encoded frames to the configured VLM endpoint for scene reasoning. Use a local endpoint through `agent.base_url` when camera frames must remain on the device or local network.

`omni` can also publish local telemetry to an MQTT broker when explicitly enabled. Swarm telemetry should not include real names, private scene descriptions or raw images.

## 5. Personal data in labels and logs

Labels can identify people if real names are used. Public demos should use placeholders such as `demo-user-001`.

## 6. Public repository rules

Do not commit:

- real face images
- private camera recordings
- screenshots from real environments
- FAISS indexes created from real people
- VLM request/response logs that contain private scene context
- MQTT telemetry containing private scene context
- labels that identify real people
- secrets, API keys or environment files

## 7. User responsibility

Users are responsible for notice, consent, retention, deletion and local file protection.

## 8. Security warning

PiSight-Omni is not a standalone security authentication system. Critical access decisions require liveness detection, access policy, encrypted storage, audit logging and formal risk assessment.

The Omni visual signal summary is not a medical, stress, heart-rate or liveness detector. It should be treated as non-clinical debugging telemetry only.
