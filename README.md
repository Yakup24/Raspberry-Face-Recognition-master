# PiSight-X

[![CI](https://github.com/Yakup24/pisight/actions/workflows/ci.yml/badge.svg)](https://github.com/Yakup24/pisight/actions/workflows/ci.yml)

PiSight-X is a Raspberry Pi oriented local face embedding toolkit. It uses OpenCV for camera I/O, facenet-pytorch for MTCNN face detection and 512-dimensional embeddings, and FAISS for local vector search.

The default workflow does not write cropped face images to `data/faces/`. Enrollment converts live camera frames into embeddings and stores a local FAISS index plus JSON label metadata.

## Overview

PiSight-X is designed for edge computer vision experiments where raw camera frames should stay on the device. The project is not a cloud recognition service and is not a security authentication product.

The current pipeline is:

```text
Camera / Video Source
  -> OpenCV Frame Capture
  -> MTCNN Face Detection
  -> InceptionResnetV1 Embeddings
  -> FAISS Vector Index
  -> Local Match Result
  -> Console / Optional Preview Window
```

## Design Philosophy

1. Local-first processing
   Frames are processed on the device and no upload path is included.

2. Image-minimizing enrollment
   The default CLI stores embeddings, not cropped face image datasets.

3. Explicit biometric risk
   Embeddings are still sensitive biometric-derived data. Treat the FAISS index and labels as private runtime artifacts.

4. Edge-device honesty
   Deep models are heavier than Haar/LBPH. Raspberry Pi CPU performance should be measured on the actual device before making FPS claims.

## Architecture Maturity

| Area | Implementation |
| --- | --- |
| Detection | MTCNN through `facenet-pytorch` |
| Embedding | `InceptionResnetV1(pretrained="vggface2")`, 512-dimensional vectors |
| Search | FAISS `IndexFlatL2` local vector index |
| Privacy posture | No raw face crop storage in the default collect/enroll flow |
| Compatibility | Legacy Haar/LBPH helper modules remain for older tests and migration context |
| Operations | CLI, config files, systemd examples, doctor command and CI |
| Testability | Camera-independent tests with mocks/fakes; no real face images in CI |

## Core Features

- Raspberry Pi camera or USB webcam input through OpenCV `VideoCapture`
- Deep face detection and embedding path with MTCNN and InceptionResnetV1
- Real-time vector enrollment through `collect` or `enroll`
- Local FAISS index and JSON label metadata
- Real-time recognition through vector nearest-neighbor search
- `train` compatibility command that explains offline training is no longer required
- JSON and YAML config support
- Dataset/model audit command retained for legacy runtime hygiene checks
- systemd service templates for Linux/Raspberry Pi deployment
- Camera-independent pytest suite and GitHub Actions CI

## Tech Stack

- Python 3.9+
- OpenCV for camera capture and drawing
- NumPy
- PyYAML
- PyTorch
- facenet-pytorch
- FAISS CPU
- pytest and ruff
- systemd for Linux service deployment

## Hardware Notes

- Raspberry Pi 4/5 can run the pipeline, but deep embeddings are CPU-heavy.
- For smoother real-time performance, test NVIDIA Jetson, Google Coral class accelerators, or a machine with CUDA/MPS support.
- Lighting, camera quality, resolution and number of faces directly affect latency and match quality.
- No fixed FPS or accuracy number is claimed by this repository.

## Project Structure

```text
raspberry_face_recognition/  CLI, config, vision, vector DB and compatibility helpers
pisight/                     Compatibility module for python -m pisight
docs/                        Architecture, setup, privacy, testing and deployment notes
examples/                    Placeholder config, commands, service file and console output
scripts/install_pi.sh        Raspberry Pi setup helper
systemd/pisight.service      Installable systemd service template
tests/                       Camera-independent tests
config.example.yaml          YAML config example
config.example.json          JSON config example
requirements.txt             Full deep-runtime dependency list
pyproject.toml               Package metadata, extras and console scripts
```

## Installation

On Raspberry Pi OS, install OpenCV from apt so camera backends work cleanly:

```sh
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-opencv python3-numpy
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[deep]"
cp config.example.yaml config.yaml
```

For development and CI-style checks without the heavy deep runtime:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Configuration

```yaml
camera:
  source: 0
  width: 640
  height: 480
  fps: 15

paths:
  vector_index_path: "./data/embeddings/faiss.index"
  vector_labels_path: "./data/embeddings/labels.json"
  log_dir: "./logs"

recognition:
  backend: "deep"
  confidence_threshold: 0.8
  unknown_label: "unknown"
  draw_bounding_boxes: true

embedding:
  dim: 512

runtime:
  debug: false
  save_unknown_faces: false
```

Notes:

- `camera.source` can be a camera index such as `0` or a video path.
- `paths.vector_index_path` stores the FAISS index.
- `paths.vector_labels_path` stores label metadata in JSON.
- `recognition.confidence_threshold` is a vector distance threshold, not an accuracy percentage.
- `runtime.save_unknown_faces` remains false by default and the current CLI does not save unknown face crops.

## Usage

Check local dependencies:

```sh
pisight --config config.yaml doctor
```

Enroll embeddings for a placeholder label:

```sh
pisight --config config.yaml collect --name demo-user-001 --count 10
```

Equivalent alias:

```sh
pisight --config config.yaml enroll --name demo-user-001 --count 10
```

Run recognition:

```sh
pisight --config config.yaml recognize
```

Run headless over SSH or systemd:

```sh
pisight --config config.yaml recognize --no-window
```

Compatibility train command:

```sh
pisight --config config.yaml train
```

The command does not train an offline model in the embedding pipeline; it explains that enrollment writes vectors directly.

## Privacy and Safety

- Raw face crops are not written in the default `collect/enroll` flow.
- FAISS index files and label metadata are local runtime artifacts.
- Embeddings are biometric-derived data and should not be committed or published.
- PiSight-X does not upload frames, embeddings or labels.
- Do not use real names or real face data in public demos.
- This project is not identity verification, surveillance infrastructure or access-control authentication.

For security-grade use, add liveness detection, consent workflow, encrypted storage, access control, retention policy and formal risk assessment.

## Testing

```sh
python -m pytest -q
python -m ruff check .
```

CI intentionally avoids real camera access, real face images, private videos and hardware-specific FPS assertions.

## Limitations

- Deep face embeddings are heavier than Haar/LBPH on Raspberry Pi CPU.
- FAISS nearest-neighbor distance is not a calibrated identity guarantee.
- Embeddings reduce raw image exposure but remain sensitive.
- Face recognition alone should not be used as a security verification system.
- Lighting, camera quality, pose and occlusion can affect detection and embedding quality.

## Roadmap

- Encrypted local embedding store
- Liveness/anti-spoofing module
- Detector backend selection from config
- Structured event logging
- Benchmark script with hardware metadata
- Local-only dashboard
- Edge accelerator notes for Jetson/Coral-class devices

## License

PiSight-X is released under the MIT License. See [LICENSE](LICENSE).
