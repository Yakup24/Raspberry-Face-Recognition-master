# Design Decisions

## 1. Why Python?

Python keeps the OpenCV workflow readable and accessible on Raspberry Pi. It also supports quick testing and CLI development without a large build chain.

## 2. Why OpenCV?

OpenCV provides camera capture, Haar cascade detection, image preprocessing and LBPH recognition in one mature toolkit. On Raspberry Pi OS, the apt package is usually easier to integrate with camera support.

## 3. Why Raspberry Pi?

Raspberry Pi is affordable, Linux-based and suitable for local edge vision demos. It also forces realistic thinking about CPU, storage, camera quality and service stability.

## 4. Why local-only processing?

Local processing reduces unnecessary exposure of face samples and camera frames. It also makes the project usable without a cloud account or remote API.

## 5. Why systemd service support?

Computer vision demos often need to run after boot or recover from process failure. systemd provides a standard Linux way to manage that lifecycle.

## 6. Why config-driven execution?

Config files make camera source, dataset paths, model paths and thresholds repeatable. This avoids hardcoded paths and manual edits between runs.

## 7. Alternatives

- Cloud-based recognition: easier centralized management, but higher privacy and network dependency risks.
- Full desktop application: richer UI, but heavier deployment and less suitable for headless Raspberry Pi usage.
- Web dashboard: useful future direction, but not required for the core CLI workflow.
- Microcontroller-based solution: lower power, but not realistic for this OpenCV recognition workflow.

## 8. Trade-offs

- Local privacy boundary is clearer, but device loss or weak filesystem permissions still matter.
- Raspberry Pi keeps deployment lightweight, but CPU and camera constraints limit throughput.
- LBPH is simple and local, but dataset quality strongly influences behavior.
- systemd improves service stability, but operational logs still need privacy review.
