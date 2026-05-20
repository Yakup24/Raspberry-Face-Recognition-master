# Recognition Flow

Recognition uses live embeddings and a local FAISS index.

## 1. Camera frame reading

`recognize` opens `camera.source` with OpenCV `VideoCapture`. The source can be a camera index or a local video path.

## 2. Face detection

Each frame is converted from BGR to RGB and passed to MTCNN from `facenet-pytorch`.

## 3. Embedding extraction

Detected faces are converted to 512-dimensional embeddings with `InceptionResnetV1(pretrained="vggface2")`.

## 4. Vector search

Each embedding is normalized and searched against the local FAISS index. The nearest vector is accepted only when its distance is below `recognition.confidence_threshold`.

## 5. Unknown behavior

If the index is empty or no vector is close enough, PiSight-X returns the configured `unknown_label`.

## 6. Console and preview output

The CLI can draw bounding boxes in a preview window or print detections to stdout in debug/headless mode.

## 7. Error conditions

Common recognition errors:

- missing deep runtime dependency
- empty vector index
- unavailable camera/video source
- invalid FAISS index metadata
- camera returning empty frames

## 8. Performance considerations

Deep embedding models are CPU-heavy on Raspberry Pi. Measure on the actual hardware, with the actual camera resolution and lighting, before documenting performance.
