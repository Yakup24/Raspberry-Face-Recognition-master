# Raspberry Pi Setup

This guide describes a conservative Raspberry Pi OS setup for running PiSight locally with OpenCV.

## 1. Raspberry Pi OS preparation

Use an up-to-date Raspberry Pi OS image and confirm that the camera works at the OS level before debugging PiSight.

```sh
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

Avoid changing kernel, boot or camera stack settings until the basic camera test passes.

## 2. Python installation

PiSight requires Python 3.9 or newer. Check the version:

```sh
python3 --version
```

## 3. Virtual environment setup

On Raspberry Pi OS, OpenCV from apt is usually more reliable than camera-related pip wheels. Create the environment with access to system packages:

```sh
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 4. OpenCV dependency notes

Install OpenCV and NumPy from apt:

```sh
sudo apt install -y python3-opencv python3-numpy
```

Run:

```sh
pisight --config config.yaml doctor
```

The doctor command checks OpenCV import, Haar cascade loading and `cv2.face` availability for LBPH recognition.

## 5. Camera access permissions

For USB webcams, confirm that the device appears:

```sh
ls /dev/video*
```

For Raspberry Pi Camera Module, use the current Raspberry Pi OS camera tools to verify hardware access before running PiSight. If the camera is busy, stop any process that already owns the device.

## 6. Performance recommendations

- Start with 640x480 at 15 FPS.
- Prefer stable lighting and a fixed camera position.
- Use `--no-window` for headless operation.
- Keep datasets small and intentional during early tests.
- Lower resolution if CPU usage is too high.

## 7. Headless usage notes

For SSH or service usage, disable the preview window:

```sh
pisight --config config.yaml recognize --no-window
```

systemd services should generally use `--no-window` because no desktop display is available.

## 8. Troubleshooting

- `OpenCV is not installed`: install `python3-opencv` and recreate the venv with `--system-site-packages`.
- `OpenCV face recognizer support is missing`: confirm the installed OpenCV build exposes `cv2.face`.
- `Could not open camera/video source`: check `camera.source`, camera permissions and whether another process is using the camera.
- `Model file not found`: run `pisight --config config.yaml train` after collecting samples.
- `No supported face image files found`: check the dataset directory and file extensions.
