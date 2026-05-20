# Camera Setup

PiSight reads frames through OpenCV `VideoCapture`, so camera behavior depends on the operating system and OpenCV backend.

## 1. USB webcam usage

Most USB webcams appear as `/dev/video0`, `/dev/video1`, and so on. Use `camera.source: 0` for the first camera.

```yaml
camera:
  source: 0
  width: 640
  height: 480
  fps: 15
```

## 2. Raspberry Pi Camera Module usage

Confirm the camera works with Raspberry Pi OS tools before using PiSight. Once the OS exposes the camera to OpenCV, use the matching camera index in `camera.source`.

## 3. Camera index selection

Try indexes conservatively:

```yaml
camera:
  source: 0
```

If another device owns index `0`, try `1`. Avoid guessing many indexes in a loop on production devices.

## 4. Camera test command

Run:

```sh
pisight --config config.yaml doctor
```

Then test a camera-dependent command:

```sh
pisight --config config.yaml collect --name demo-user-001 --count 5
```

## 5. Camera not found checks

If PiSight reports that it cannot open the source:

- confirm the camera is connected
- check `/dev/video*`
- verify `camera.source`
- close other camera applications
- confirm the service user has permission to access the camera
- test without systemd first

## 6. Lighting and angle

Use stable front lighting, avoid strong backlight, and keep the camera angle close to the expected recognition angle. Enrollment frames should include modest variation, but not uncontrolled blur or extreme lighting.

## 7. FPS and resolution

Start with 640x480 at 15 FPS on Raspberry Pi. Increase only after recognition is stable. Higher resolution can improve face detail but costs CPU, memory bandwidth and latency.
