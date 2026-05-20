# systemd Service

systemd lets PiSight run as a managed Linux service after boot or after manual start.

## 1. Service purpose

Use a service when recognition should run continuously on a Raspberry Pi without an interactive shell. Prefer `--no-window` for service mode.

## 2. Service file example

See:

```text
systemd/pisight.service
examples/systemd/pisight.service
```

## 3. WorkingDirectory

`WorkingDirectory` should point to the installed project directory, for example `/opt/pisight`.

## 4. ExecStart

Use the virtual environment Python or console script:

```ini
ExecStart=/opt/pisight/.venv/bin/python -m pisight --config /opt/pisight/config.yaml recognize --no-window
```

## 5. EnvironmentFile

PiSight does not require secrets. If local environment variables are needed, use an optional file:

```ini
EnvironmentFile=-/etc/default/pisight
```

Do not store secrets or personal data in public example files.

## 6. Restart policy

`Restart=on-failure` and `RestartSec=5` are reasonable defaults for a demo service. Repeated failures should still be investigated instead of hidden by aggressive restart loops.

## 7. Enable, start and status

```sh
sudo cp systemd/pisight.service /etc/systemd/system/pisight.service
sudo systemctl daemon-reload
sudo systemctl enable pisight
sudo systemctl start pisight
sudo systemctl status pisight
```

## 8. journalctl logs

```sh
journalctl -u pisight -f
```

Avoid logging real names in public demos.

## 9. Troubleshooting

- Check paths in `WorkingDirectory` and `ExecStart`.
- Confirm the service user can read the config and access the camera.
- Run the same command manually before enabling the service.
- Check model and labels files exist.
- Use `doctor` for dependency checks.
