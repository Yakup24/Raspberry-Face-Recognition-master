# PiSight

[![CI](https://github.com/Yakup24/pisight/actions/workflows/ci.yml/badge.svg)](https://github.com/Yakup24/pisight/actions/workflows/ci.yml)

PiSight is an OpenCV-based face detection and recognition toolkit designed to run locally on Raspberry Pi. It provides repeatable workflows for collecting face samples, training a local model, and running real-time recognition with privacy-aware defaults.

## Overview

PiSight turns a Raspberry Pi plus a camera into a local edge computer vision workflow. It is useful for learning, prototyping, and technical demos where camera frames and face samples should stay on the device instead of being uploaded to a remote recognition service.

The project is intentionally more than a single webcam script. It includes a CLI, configuration loading, dataset collection, LBPH model training, live recognition, dataset auditing, tests, and a systemd service template for Linux deployment.

## Problem

Camera-based face recognition projects can become risky or hard to operate when they mix personal data, ad hoc scripts, and manual runtime steps. Common issues include:

- Face samples and labels can expose sensitive personal information.
- Raspberry Pi hardware has limited CPU, storage, and camera bandwidth.
- Dataset collection, training, and recognition often live in separate manual scripts.
- Long-running deployments need a service manager such as systemd.
- Missing camera devices, empty datasets, missing models, and invalid config values need clear errors.

## Solution

PiSight keeps the workflow local and repeatable:

- Camera frames are processed locally with OpenCV.
- Face samples are stored in a local dataset directory.
- Training builds a local OpenCV LBPH recognizer model.
- Recognition loads the local model and label map from disk.
- JSON and YAML configuration files control runtime paths, camera source, and thresholds.
- systemd examples show how to run recognition as a Linux service.
- Audit and tests focus on dataset/model health without requiring a real camera in CI.

## Architecture

```text
Camera / Video Source
  -> Frame Capture
  -> Face Detector
  -> Dataset Collector OR Recognition Pipeline
  -> Local Model
  -> Result Formatter
  -> Console / Log Output
```

The CLI loads configuration first, then routes to one of the operational flows: `collect`, `train`, `recognize`, `audit`, or `doctor`. OpenCV handles frame capture, Haar cascade detection, face cropping, and LBPH recognition. Runtime artifacts remain on the local filesystem.

## Design Philosophy

PiSight is designed around four principles:

1. Local-first processing
   Camera frames, face samples and model files should remain on the device by default.

2. Reproducible workflow
   Dataset collection, training and recognition should be documented as repeatable commands instead of manual experiments.

3. Edge-device awareness
   The project should consider Raspberry Pi limitations such as CPU usage, camera quality, lighting and service stability.

4. Privacy-aware development
   Real face data should not be committed to the repository, and users should understand the privacy implications of face recognition.

## Core Features

- Raspberry Pi camera or USB webcam input through OpenCV `VideoCapture`
- OpenCV Haar cascade face detection
- Local face sample collection with one folder per label
- OpenCV LBPH model training
- Real-time recognition from a camera or configured video source
- CLI commands: `doctor`, `collect`, `train`, `recognize`, and `audit`
- JSON and YAML config file support
- systemd service templates for Linux/Raspberry Pi deployment
- Local-only processing with ignored runtime data under `data/`
- pytest-compatible tests that do not require a real camera

## Tech Stack

- Python 3.9 or newer
- OpenCV, installed from Raspberry Pi OS packages for camera support
- NumPy
- PyYAML for YAML configuration
- pytest for automated tests
- systemd for service deployment
- Raspberry Pi OS or another Linux environment for device deployment

## Hardware Requirements

- Raspberry Pi 4 or newer recommended for smoother camera processing
- Raspberry Pi Camera Module or USB webcam supported by the OS
- MicroSD card or local storage with enough space for datasets and model files
- Python virtual environment
- Stable lighting and a fixed camera angle for better recognition quality

Lower-end Raspberry Pi boards can still run the workflow, but resolution, FPS, lighting, and sample quality matter more on constrained hardware.

## Project Structure

```text
raspberry_face_recognition/  Core package, CLI, config, dataset, model and OpenCV helpers
pisight/                     Compatibility module for python -m pisight
docs/                        Architecture, setup, privacy, testing and deployment documentation
examples/                    Placeholder config, commands, service file and console output
scripts/install_pi.sh        Raspberry Pi setup helper
systemd/pisight.service      Installable systemd service template
tests/                       Camera-independent unit and pipeline tests
config.example.json          Flat JSON config example
config.example.yaml          Nested YAML config example for deployment-style paths
LICENSE                      MIT license
requirements.txt             Python dependencies for local development and CI
pyproject.toml               Package metadata and console scripts
```

## Getting Started

Clone the repository:

```sh
git clone https://github.com/Yakup24/pisight.git
cd pisight
```

Create a virtual environment. On Raspberry Pi OS, use system OpenCV packages:

```sh
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-opencv python3-numpy
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install -e .
cp config.example.yaml config.yaml
```

On a development machine without camera access, install dependencies and run tests:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

Check the runtime environment:

```sh
pisight --config config.yaml doctor
```

Collect local samples for a placeholder label:

```sh
pisight --config config.yaml collect --name demo-user-001 --count 40
```

Train the local model:

```sh
pisight --config config.yaml train
```

Run recognition:

```sh
pisight --config config.yaml recognize
```

## Configuration

PiSight supports the original flat JSON config and the nested YAML format used by the examples. The CLI accepts either file through `--config`.

```yaml
camera:
  source: 0
  width: 640
  height: 480
  fps: 15

paths:
  dataset_dir: "./data/dataset"
  model_path: "./data/models/face_model.yml"
  labels_path: "./data/models/labels.json"
  log_dir: "./logs"

recognition:
  confidence_threshold: 70
  unknown_label: "unknown"
  draw_bounding_boxes: true

runtime:
  debug: false
  save_unknown_faces: false
```

Notes:

- `camera.source` can be a camera index such as `0` or a video file path.
- `paths.dataset_dir` stores local face sample folders.
- `paths.model_path` and `paths.labels_path` point to local training artifacts.
- `recognition.confidence_threshold` controls when LBPH predictions become `unknown`; lower LBPH confidence is usually better.
- `paths.log_dir` is accepted for deployment layouts, while the current CLI writes to stdout/stderr and systemd captures that output through journald.

## Usage

Collect face samples:

```sh
pisight --config config.yaml collect --name demo-user-001 --count 40
```

Train the model:

```sh
pisight --config config.yaml train
```

Recognize from the configured camera:

```sh
pisight --config config.yaml recognize
```

Run headless, which is useful for SSH and systemd:

```sh
pisight --config config.yaml recognize --no-window
```

Test with a video file by setting `camera.source` in the config:

```yaml
camera:
  source: "./examples/demo-video-placeholder.mp4"
```

Enable debug console detections by setting:

```yaml
runtime:
  debug: true
```

Audit local dataset/model metadata without reading image pixels:

```sh
pisight --config config.yaml audit
pisight --config config.yaml audit --json
```

## Running as a Service

PiSight includes service templates in `systemd/pisight.service` and `examples/systemd/pisight.service`. Adjust paths, user, and config location before installation.

```sh
sudo cp systemd/pisight.service /etc/systemd/system/pisight.service
sudo systemctl daemon-reload
sudo systemctl enable pisight
sudo systemctl start pisight
sudo systemctl status pisight
```

Inspect logs:

```sh
journalctl -u pisight -f
```

## Testing

Run the camera-independent test suite:

```sh
python -m pytest -q
```

Current tests cover:

- Config loading and validation
- Dataset path validation and label mapping
- Camera unavailable behavior with fake captures
- Model and labels loading failure paths
- Recognition pipeline behavior with mocked detector/model objects
- CLI argument parsing and help output

CI intentionally does not access a real camera, real face data, or hardware-specific FPS checks.

## Privacy and Safety

- Face samples are stored locally.
- Model files and label maps are stored locally.
- PiSight does not upload camera frames, face samples, labels, or models to a remote service.
- Real face samples, personal photos, private camera recordings, and sensitive files must not be committed.
- Logs can still reveal labels or operational context, so avoid using real names in public demos.
- PiSight should be treated as a demo and edge vision toolkit, not as a standalone security-grade authentication system.

PiSight requires additional security controls, liveness detection, access control review, and formal risk assessment before it is used for critical identity or access decisions.

## Limitations

- Low light can reduce detection and recognition quality.
- Camera quality, angle, and motion blur affect results.
- Dataset quality strongly affects model behavior.
- Raspberry Pi CPU and memory constraints limit FPS and resolution choices.
- LBPH confidence values are not accuracy percentages.
- Face recognition alone should not be used as a security verification system.

## Roadmap

- Web dashboard for local monitoring
- Better model evaluation with holdout datasets
- Docker support
- More camera backend options
- Face tracking metrics
- Encrypted local dataset option
- Performance benchmark script with honest hardware notes

## My Contributions

This project includes the local OpenCV workflow, CLI commands, config loading, dataset helpers, LBPH training flow, recognition flow, privacy-preserving audit command, camera-independent tests, Raspberry Pi setup notes, and systemd deployment examples. The repository avoids committing real biometric data and keeps generated runtime artifacts under ignored local paths.

## License

PiSight is released under the MIT License. See [LICENSE](LICENSE).

The repository includes a local MIT license file so GitHub and downstream users can detect the project license directly. The MIT licensing approach is compatible with the hosted MIT license pattern described by [remy/mit-license](https://github.com/remy/mit-license).
