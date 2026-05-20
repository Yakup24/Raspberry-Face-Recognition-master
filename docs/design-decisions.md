# Design Decisions

## 1. Why Python?

Python keeps the vision pipeline readable and gives access to OpenCV, PyTorch, facenet-pytorch and FAISS without a large build chain.

## 2. Why OpenCV?

OpenCV remains the camera I/O and drawing layer. It is widely available on Raspberry Pi OS and works well with `VideoCapture`.

## 3. Why MTCNN and InceptionResnetV1?

The deep path is more robust than the former Haar/LBPH demo pipeline for alignment and embedding extraction. It also separates identity matching from raw image storage by converting faces into vectors.

## 4. Why FAISS?

FAISS provides local nearest-neighbor vector search. It lets the project store embeddings and search them without a custom linear-search implementation.

## 5. Why local-only processing?

Local processing avoids adding cloud upload risk and keeps the data boundary inspectable. It does not remove biometric risk, so embeddings still need local protection.

## 6. Why no offline train step?

The embedding model is pretrained. Enrollment adds vectors to the local FAISS index, so a separate LBPH-style `train` command is no longer part of the default workflow.

## 7. Alternatives

- Cloud recognition: easier centralized inference, higher privacy and network dependency risk.
- Haar/LBPH: lighter, but less robust and requires image datasets.
- Full desktop application: richer UI, but heavier deployment and less useful for headless Raspberry Pi.
- Microcontroller-based solution: lower power, but not realistic for this deep embedding workflow.

## 8. Trade-offs

- Embeddings avoid raw image storage by default, but they remain sensitive biometric-derived data.
- Deep models improve architecture maturity, but Raspberry Pi CPU FPS can be low.
- FAISS is powerful, but distance thresholds still need local calibration and are not security guarantees.
- systemd improves service stability, but logs and labels still need privacy review.
