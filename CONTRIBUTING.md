# Contributing

## Local Setup

1. Create a virtual environment.
2. Install dependencies.
3. Run tests before opening a pull request.

```sh
python -m pip install -e .
python -m pip install pytest ruff
python -m pytest -q
```

On Raspberry Pi OS, install OpenCV from apt and create the venv with `--system-site-packages`.

## Branch Naming

- `feature/add-camera-backend`
- `fix/model-loading-error`
- `docs/update-raspberry-pi-setup`
- `test/add-config-validation`

## Commit Style

- `feat: add new recognition option`
- `fix: handle missing camera error`
- `docs: update privacy notes`
- `test: add dataset validation tests`
- `chore: update CI`

## Pull Request Checklist

- Tests pass locally.
- No real face images are included.
- No secrets are committed.
- README/docs updated if behavior changed.
- Privacy impact considered.
- Camera-dependent behavior is guarded or mocked in tests.
