# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - Unreleased

### Added

- Added PiSight-Omni embedding architecture with PyTorch/facenet-pytorch and FAISS vector search modules.
- Added vector enrollment through the existing `collect` command and new `enroll` alias without storing cropped face images.
- Added FAISS/label metadata config paths and tests for vector database behavior.
- Added optional `autonom` agent loop for VLM scene reasoning with dry-run action dispatch.
- Added agent configuration, docs and tests for JSON decisions and safety boundaries.
- Added `omni` command, local telemetry helpers and dry-run MQTT swarm publisher.
- Added PiSight-Omni documentation covering scope, telemetry and safety boundaries.
- Expanded documentation for architecture, Raspberry Pi setup, camera setup, enrollment, recognition, systemd and privacy model.
- Added YAML example configuration and placeholder examples for service usage, commands, embedding structure and expected errors.
- Added pytest-compatible tests for config validation, vector DB behavior, dataset compatibility, camera unavailable behavior, model loading and recognition pipeline behavior.
- Added a CI workflow that installs dependencies, runs ruff and executes pytest without requiring real camera hardware.
- Added `python -m pisight` compatibility entry point for service examples.
- Added an MIT license file and package license metadata.

### Changed

- Improved README structure for technical review and portfolio presentation.
- Strengthened config loading with JSON/YAML support and validation.
- Replaced the default image-dataset/LBPH train workflow with live embedding enrollment and FAISS search.
- Changed `train` into a compatibility command that explains offline training is no longer required.
- Added OpenAI-compatible VLM dependency extras for opt-in agent mode.
- Replaced boilerplate security policy with project-specific privacy and data handling guidance.

## [0.1.0] - Initial Public Release

### Added

- Initial OpenCV-based face detection/recognition workflow.
- Dataset collection and training concept.
- Local processing approach.
- CLI commands for setup checks, sample collection, model training, recognition and dataset audit.
