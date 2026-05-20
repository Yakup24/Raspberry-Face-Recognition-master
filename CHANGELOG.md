# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - Unreleased

### Added

- Expanded documentation for architecture, Raspberry Pi setup, camera setup, training, recognition, systemd and privacy model.
- Added YAML example configuration and placeholder examples for service usage, commands, dataset structure and expected errors.
- Added pytest-compatible tests for config validation, dataset validation, camera unavailable behavior, model loading and recognition pipeline behavior.
- Added a CI workflow that installs dependencies, runs ruff in report mode and executes pytest without requiring real camera hardware.
- Added `python -m pisight` compatibility entry point for service examples.

### Changed

- Improved README structure for technical review and portfolio presentation.
- Strengthened config loading with JSON/YAML support and validation.
- Improved training and recognition error handling for missing datasets, models and labels.
- Replaced boilerplate security policy with project-specific privacy and data handling guidance.

## [0.1.0] - Initial Public Release

### Added

- Initial OpenCV-based face detection/recognition workflow.
- Dataset collection and training concept.
- Local processing approach.
- CLI commands for setup checks, sample collection, model training, recognition and dataset audit.
