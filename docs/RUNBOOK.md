# Operational Runbook

This runbook describes how to operate PiSight safely on a Raspberry Pi or a Linux development machine.

## 1. Environment check

```sh
pisight --config config.json doctor
```

Use this before collecting or recognizing faces.

Expected checks:

- OpenCV is installed
- OpenCV face recognizer support is available
- Haar cascade detector can be loaded
- Config paths are resolved correctly

## 2. Dataset audit

```sh
pisight --config config.json audit
```

JSON output:

```sh
pisight --config config.json audit --json
```

Fail when warnings exist:

```sh
pisight --config config.json audit --fail-on-warning
```

Audit checks:

- Face sample directory exists
- Image counts by person
- Model file exists
- Label file exists
- Labels match person folders

The audit command does not read image pixels.

## 3. Collect samples

```sh
pisight --config config.json collect --name person_name
```

Headless mode:

```sh
pisight --config config.json collect --name person_name --no-window
```

Recommended collection rules:

- Use clear permission from the participant
- Capture varied angles and lighting
- Keep background simple
- Avoid collecting unnecessary people in frame
- Delete test data when finished

## 4. Train model

```sh
pisight --config config.json train
```

This creates:

```text
data/model.yml
data/labels.json
```

## 5. Run recognition

```sh
pisight --config config.json recognize
```

Headless mode:

```sh
pisight --config config.json recognize --no-window
```

## 6. Common issues

### OpenCV is not installed

Install OS packages on Raspberry Pi OS:

```sh
sudo apt update
sudo apt install -y python3-opencv python3-numpy
```

### LBPH recognizer support is missing

The installed OpenCV build does not include the `cv2.face` module.

Use Raspberry Pi OS package `python3-opencv`, or install an OpenCV contrib build compatible with your device.

### Camera cannot be opened

Check:

- Camera index in `config.json`
- USB camera permissions
- Raspberry Pi camera interface
- Whether another process is using the camera

### No readable face images found

Run collection first:

```sh
pisight --config config.json collect --name person_name
```

Then train again.

## 7. Privacy checklist

Before using PiSight:

- [ ] Participants understand the project
- [ ] Data is stored locally
- [ ] `data/` is not committed
- [ ] No model or label files are uploaded publicly
- [ ] Dataset is deleted when no longer needed

## 8. CI behavior

GitHub Actions runs non-camera checks:

1. Install package
2. Run unit tests
3. Run CLI smoke test
4. Run audit command

Camera-dependent commands are intentionally not executed in CI.
