# Security Policy

## Data Policy

This repository must not include real face samples, personal photos, private camera recordings, FAISS indexes created from real people, access tokens or sensitive personal information.

## Privacy

PiSight-Omni is designed for local processing. The default enrollment flow stores embeddings rather than cropped face images, but embeddings and labels are still biometric-derived sensitive artifacts and should remain on the local device unless the user explicitly exports them.

The optional `autonom` and `omni` commands can send encoded frames to the configured VLM endpoint. Use a local endpoint when privacy requirements do not allow frames to leave the device or local network.

The optional `omni` MQTT publisher is disabled by default and dry-run by default. Do not publish real names, private scene details, raw frames or sensitive labels to shared brokers.

## Responsible Usage

Face recognition systems can create privacy and consent risks. Use this project only in environments where people are informed and authorized.

## Production Warning

This project should not be used as a standalone security authentication system without additional controls such as liveness detection, access control review and formal risk assessment.

Agent actions are advisory by default. The current implementation does not execute GPIO, locks, relays, Telegram messages, generated Python code or other physical actions.

## Reporting a Vulnerability

If you identify a security issue, report it privately to the maintainer. Avoid sharing real face samples, FAISS indexes, labels with real names, logs with personal context or private camera recordings in public issues.
