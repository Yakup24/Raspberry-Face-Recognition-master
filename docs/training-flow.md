# Training Flow

Training converts local face samples into an OpenCV LBPH recognizer model and a label map.

## 1. Dataset

The dataset is a local directory of face crop images grouped by label. Each label represents one demo identity or placeholder user.

## 2. Samples per person

Collect multiple samples per label so the recognizer sees normal variation in position, expression and lighting. More samples do not guarantee better results if they are blurry or inconsistent.

## 3. Dataset folder structure

```text
data/
  faces/
    demo-user-001/
      000001.png
      000002.png
    demo-user-002/
      000001.png
```

Real face samples must not be added to a public repository.

## 4. Sample collection command

```sh
pisight --config config.yaml collect --name demo-user-001 --count 40
```

The command stores cropped face images under the configured dataset directory.

## 5. Training command

```sh
pisight --config config.yaml train
```

Before reading image pixels, PiSight validates that the dataset directory exists, contains person folders and includes supported image files.

## 6. Model output

Training writes:

```text
data/model.yml
data/labels.json
```

The model and labels are local runtime artifacts and should not be committed when based on real data.

## 7. Dataset quality impact

Recognition quality depends on sample clarity, lighting, camera angle, consistent labels and enough variation. LBPH confidence is not an accuracy percentage.

## 8. Invalid or incomplete datasets

Expected failures include missing dataset directories, empty person folders, unsupported file types and unreadable images. PiSight reports these cases through validation errors or warnings before and during training.
