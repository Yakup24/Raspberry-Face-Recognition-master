# Recognition Flow

Recognition loads a local model and runs face detection plus LBPH prediction on frames from the configured source.

## 1. Camera frame reading

The `recognize` command opens `camera.source` with OpenCV `VideoCapture`. The source can be a camera index or a video file path.

## 2. Face detection

Each frame is converted to grayscale and passed to the configured Haar cascade detector. If no faces are detected, the step returns an empty result and continues.

## 3. Model loading

PiSight checks that the model and labels files exist before opening the camera. This avoids starting a camera session when recognition artifacts are missing.

## 4. Recognition and confidence

For each detected face, PiSight crops and resizes the face region, then calls the LBPH recognizer. Lower LBPH confidence values are usually better. If the returned confidence is greater than `recognition.confidence_threshold`, PiSight labels the detection as `unknown`.

## 5. Unknown person behavior

Unknown detections use the configured `recognition.unknown_label`, which defaults to `unknown`.

## 6. Console and log output

The CLI writes to stdout and stderr. In systemd deployments, journald captures this output. With `runtime.debug: true` or `--no-window`, detection lines can be printed to the console.

## 7. Error conditions

Common recognition errors:

- missing model file
- missing labels file
- failed OpenCV model read
- unavailable camera/video source
- missing OpenCV face recognizer support

## 8. Performance considerations

Resolution, FPS, lighting and number of faces in frame all affect Raspberry Pi performance. Start with modest camera settings and measure on the actual device before claiming performance numbers.
