# Enrollment Flow

PiSight-X replaces offline image-dataset training with live vector enrollment.

## 1. What changed

The previous OpenCV LBPH flow used:

```text
collect images -> train model.yml -> recognize
```

The default PiSight-X flow is now:

```text
collect/enroll embeddings -> FAISS index -> recognize
```

## 2. No raw image dataset by default

`collect` and `enroll` do not write cropped face images. They store only the FAISS index and JSON labels.

## 3. Enrollment command

```sh
pisight --config config.yaml collect --name demo-user-001 --count 10
```

Alias:

```sh
pisight --config config.yaml enroll --name demo-user-001 --count 10
```

## 4. Single-face guard

Enrollment only stores a vector when exactly one face is detected in the frame. This avoids accidentally labeling another person.

## 5. Stored artifacts

```text
data/
  embeddings/
    faiss.index
    labels.json
```

These files are local runtime artifacts and should not be committed when produced from real people.

## 6. Compatibility train command

`pisight train` remains as a compatibility command, but it does not train an offline model in the embedding pipeline. It tells users to enroll vectors directly.

## 7. Quality impact

Vector match quality depends on lighting, pose, camera quality, face alignment and threshold selection. FAISS distance is not an accuracy percentage.
