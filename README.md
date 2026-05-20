# PiSight

![Python CI](https://github.com/Yakup24/pisight/actions/workflows/python-ci.yml/badge.svg)

PiSight is a privacy-first Raspberry Pi face-recognition toolkit built around OpenCV. The workflow is intentionally small: collect face samples, train a local recognizer, audit local dataset health, then run recognition from a camera stream.

The project is designed for local devices. It does **not** upload camera frames, face samples, labels, or trained models to a remote service.

## What It Does

- Captures labeled face samples from a Raspberry Pi camera or USB camera
- Trains an OpenCV LBPH face recognizer from a local dataset
- Runs live recognition with configurable confidence thresholds
- Audits local dataset/model/label metadata without reading image pixels
- Stores datasets and trained models on disk
- Includes setup notes, tests, privacy notes, and a systemd service template
- Runs unit tests automatically with GitHub Actions

## Repository Layout

```text
raspberry_face_recognition/  PiSight Python package and CLI
config.example.json          Example runtime configuration
docs/                        Setup, usage, architecture, runbook, and privacy notes
scripts/install_pi.sh        Raspberry Pi setup helper
systemd/                     Optional service template
tests/                       Unit tests for non-camera code
```

## Install

On Raspberry Pi OS, prefer system packages for OpenCV:

```sh
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-opencv python3-numpy
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install -e .
cp config.example.json config.json
```

The `--system-site-packages` flag lets the virtual environment use the OpenCV package installed by apt.

## Quick Start

Check the environment:

```sh
pisight --config config.json doctor
```

Collect samples for a person:

```sh
pisight --config config.json collect --name person_name
```

Train the recognizer:

```sh
pisight --config config.json train
```

Audit local dataset/model health:

```sh
pisight --config config.json audit
pisight --config config.json audit --json
```

Run live recognition:

```sh
pisight --config config.json recognize
```

## Data

By default, local runtime files are stored under `data/`:

- `data/faces/` contains captured face crops
- `data/model.yml` contains the trained recognizer
- `data/labels.json` maps numeric IDs to names

These files are ignored by Git because they may contain personal biometric data.

## Privacy and Safety

PiSight is intended for local, consent-based experiments and edge-device demos. Do not collect or process face samples without clear permission from the people involved.

Recommended safety practices:

- Use test participants who understand what the project does
- Keep `data/` out of Git
- Do not upload trained models or labels to public repositories
- Use `pisight audit` to inspect dataset metadata without opening image files
- Delete local datasets when the experiment is finished

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Raspberry Pi setup](docs/SETUP_RASPBERRY_PI.md)
- [Usage guide](docs/USAGE.md)
- [Operational runbook](docs/RUNBOOK.md)
- [Privacy notes](docs/PRIVACY.md)

## Development

Run tests that do not require a camera:

```sh
python -m unittest discover -s tests
```

Run the CLI smoke checks:

```sh
pisight --help
pisight audit --json
```

## Roadmap

- Optional structured event logging
- Recognition session summary export
- Config validation command
- Camera backend abstraction
- Model evaluation command with holdout images
- Optional encrypted local dataset storage
