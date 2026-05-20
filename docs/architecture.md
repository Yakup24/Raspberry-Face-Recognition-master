# Architecture

PiSight is a local-first computer vision toolkit built around a small Python package and CLI. The runtime stays on the device: camera frames, collected samples, labels and model files are stored locally unless the user explicitly moves them.

## 1. High-level architecture

```text
Camera / Video Source
  -> Frame Capture
  -> Face Detector
  -> Dataset Collector OR Recognition Pipeline
  -> Local Model
  -> Result Formatter
  -> Console / Log Output
```

The `pisight` CLI loads configuration first, then dispatches to `doctor`, `collect`, `train`, `recognize`, or `audit`.

## 2. Component responsibilities

- `raspberry_face_recognition.cli`: argument parsing and command orchestration.
- `raspberry_face_recognition.config`: JSON/YAML config loading, path resolution and validation.
- `raspberry_face_recognition.dataset`: dataset folder handling, label mapping and training dataset validation.
- `raspberry_face_recognition.vision`: OpenCV imports, Haar cascade setup, LBPH recognizer creation and face cropping.
- `raspberry_face_recognition.model`: model and label file loading with explicit error results.
- `raspberry_face_recognition.recognition`: single-frame recognition pipeline and detection formatting.
- `raspberry_face_recognition.audit`: privacy-preserving filesystem metadata checks.

## 3. Camera input flow

Camera input uses OpenCV `VideoCapture`. The configured `camera.source` can be an integer camera index or a video file path. Optional width, height and FPS settings are applied when the OpenCV backend exposes the matching properties.

If the source cannot be opened, PiSight raises a clear runtime error instead of continuing with an invalid stream.

## 4. Dataset collection flow

The `collect` command opens the configured camera, detects faces with a Haar cascade, crops the largest detected face and writes local image samples under the configured dataset directory.

```text
data/faces/<label>/000001.png
data/faces/<label>/000002.png
```

Labels are sanitized before folder creation. Collected samples are runtime data and must not be committed.

## 5. Training flow

The `train` command validates the dataset structure before loading images. It skips unsupported file extensions, warns about empty person folders, loads supported images in grayscale, resizes them to the configured face size and trains an OpenCV LBPH recognizer.

Training writes:

- local model file, for example `data/model.yml`
- local label map, for example `data/labels.json`

## 6. Recognition flow

The `recognize` command checks the model and labels first, then opens the camera source. For each frame, PiSight detects faces, crops each face, asks the recognizer for a label/confidence pair and maps high-confidence-distance results to the configured `unknown_label`.

When a preview window is disabled or debug mode is enabled, detections can be printed to stdout.

## 7. Error handling

Expected failure cases are surfaced as CLI errors with exit code `2`:

- invalid config values
- missing dataset directories
- empty training datasets
- missing model or labels files
- unavailable camera/video source
- missing OpenCV or missing `cv2.face` support

## 8. Logging approach

The current CLI writes operational output to stdout and stderr. When running under systemd, journald captures this output. A `log_dir` config value is accepted for deployment layouts, but structured file logging is a roadmap item rather than a completed feature.
