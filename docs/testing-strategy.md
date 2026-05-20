# Testing Strategy

PiSight tests focus on behavior that can be validated reliably without real camera hardware or real face data.

## 1. Test Goals

- Config validation
- Vector index and legacy dataset path validation
- Camera unavailable behavior
- Vector DB behavior
- Agent decision parsing and safe action dispatch
- Recognition pipeline compatibility behavior

## 2. Test Types

- Unit tests for config, vector DB, dataset compatibility and model helper behavior
- Integration-like tests with mocks for camera and recognition flow
- CLI tests for argument parsing and help behavior
- Negative tests for missing files and invalid input

## 3. What should not be tested in CI?

- Real camera access
- Real face data
- Hardware-specific FPS claims
- Accuracy or FPS claims without documented local evaluation data

## 4. Mocking Strategy

- Fake camera capture objects with `isOpened`, `read`, `set` and `release`
- Fake detector results for empty and populated face lists
- Fake vector DB objects and deterministic embeddings
- Fake VLM clients for agent tests
- Temporary index, model and labels paths created under pytest `tmp_path`

Run:

```sh
python -m pytest -q
```
