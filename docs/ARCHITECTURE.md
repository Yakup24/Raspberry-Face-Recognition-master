# Architecture

PiSight is a local-first Raspberry Pi face-recognition toolkit built as a small Python package with a command-line interface.

## High-level flow

```text
CLI command
   ↓
Configuration loader
   ↓
Dataset / camera / model operation
   ↓
OpenCV integration
   ↓
Local filesystem output
```

## Main modules

### `raspberry_face_recognition.cli`

Defines the command-line interface.

Main commands:

- `doctor`: checks local OpenCV and detector availability
- `collect`: captures local face crops for a named person
- `train`: trains the local LBPH recognizer
- `recognize`: runs live recognition from camera stream
- `audit`: summarizes dataset/model/label metadata without reading image pixels

### `raspberry_face_recognition.config`

Loads JSON configuration and resolves local runtime paths.

Important settings:

- `camera_index`
- `data_dir`
- `faces_dir`
- `model_path`
- `labels_path`
- `sample_count`
- `confidence_threshold`
- `display`

### `raspberry_face_recognition.dataset`

Manages local dataset structure.

Responsibilities:

- Normalize person names for safe folder names
- Create person directories
- Find next sample index
- Iterate local face samples
- Save and load label maps

### `raspberry_face_recognition.vision`

Contains OpenCV integration points.

Responsibilities:

- Load OpenCV and NumPy
- Resolve Haar cascade path
- Create face detector
- Create LBPH recognizer
- Detect and crop faces

### `raspberry_face_recognition.audit`

Provides privacy-preserving metadata checks.

The audit module does not read image pixels. It checks:

- People folder count
- Image file count
- Model existence
- Label file existence
- Label-to-folder consistency

## Data layout

Default local runtime layout:

```text
data/
├─ faces/
│  ├─ person_one/
│  │  ├─ 000001.png
│  │  └─ 000002.png
│  └─ person_two/
├─ model.yml
└─ labels.json
```

`data/` is ignored by Git because it may contain biometric data.

## Design principles

1. Local-first processing
2. No remote upload by default
3. Small CLI workflow
4. Clear privacy boundary
5. Testable non-camera logic
6. Raspberry Pi friendly installation

## Extension points

Recommended future extension points:

- Add camera backend abstraction
- Add dataset export/import with encryption
- Add model evaluation command
- Add structured recognition event logs
- Add confidence calibration helper
- Add optional web dashboard for local network use

## Security and privacy notes

Face images, labels and trained models can be sensitive biometric data. They should not be committed, uploaded or shared without consent.

The project intentionally keeps local runtime data under `data/`, and `.gitignore` excludes that folder.
