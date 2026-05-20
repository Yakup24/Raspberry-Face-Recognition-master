# Privacy

PiSight is designed as a local edge vision toolkit. That does not remove privacy risk, but it keeps the default data boundary clear and inspectable.

## 1. Local-only processing approach

PiSight processes frames locally with OpenCV. The repository does not contain code that uploads camera frames, collected samples, labels or model files to a remote service.

## 2. Face sample files

Collected face crops are stored under the configured dataset directory, for example `data/faces/`. These files can contain biometric personal data and should stay out of Git.

## 3. Model files

The trained LBPH model and label map are local runtime artifacts. A model file may still encode information derived from face samples, and a labels file can connect model IDs to people or placeholder labels.

## 4. Remote upload

PiSight has no remote upload workflow by default. If a user later adds cloud storage, dashboard sync or remote logging, that change should be reviewed as a privacy-impacting feature.

## 5. Personal data in logs

Recognition output can include labels such as `demo-user-001` or user-provided names. Avoid real names in public demos and avoid publishing logs that identify people.

## 6. Public repository rules

Do not commit:

- real face samples
- personal photos
- private camera recordings
- trained model files from real people
- label maps that identify real people
- secrets, API keys or environment files

## 7. User responsibility

Users are responsible for consent, notice, retention and deletion policies. Run PiSight only in environments where people are informed and authorized.

## 8. Security verification warning

PiSight should not be used as a standalone security authentication system.

PiSight requires additional security controls, liveness detection, access control review and formal risk assessment before it is used for critical face-recognition-based security verification.
