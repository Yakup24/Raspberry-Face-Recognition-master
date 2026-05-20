# Testing Strategy

PiSight tests focus on behavior that can be validated reliably without real camera hardware or real face data.

## 1. Test Goals

- Config validation
- Dataset path validation
- Camera unavailable behavior
- Model file loading behavior
- Recognition pipeline stability

## 2. Test Types

- Unit tests for config, dataset and model helpers
- Integration-like tests with mocks for camera and recognition flow
- CLI tests for argument parsing and help behavior
- Negative tests for missing files and invalid input

## 3. What should not be tested in CI?

- Real camera access
- Real face data
- Hardware-specific FPS claims
- Accuracy claims without a documented evaluation dataset

## 4. Mocking Strategy

- Fake camera capture objects with `isOpened`, `read`, `set` and `release`
- Fake detector results for empty and populated face lists
- Fake recognizer objects with deterministic `predict` and `read` behavior
- Temporary model and labels paths created under pytest `tmp_path`

Run:

```sh
python -m pytest -q
```
