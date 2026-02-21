import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

IMAGE_COL_CANDIDATES = ["image", "image_id", "image_name", "isic_id", "id"]


def _resolve_image_col(columns: List[str]) -> str:
    for name in IMAGE_COL_CANDIDATES:
        if name in columns:
            return name
    raise ValueError("No image id column found. Expected one of: " + ", ".join(IMAGE_COL_CANDIDATES))


def _extract_labels(df: pd.DataFrame, image_col: str) -> Tuple[List[str], List[str]]:
    columns = [col for col in df.columns if col != image_col]
    if "label" in columns:
        labels = df["label"].astype(str).tolist()
        return df[image_col].astype(str).tolist(), labels

    if len(columns) == 1:
        labels = df[columns[0]].astype(str).tolist()
        return df[image_col].astype(str).tolist(), labels

    # Assume one-hot columns.
    one_hot = df[columns].to_numpy()
    max_idx = np.argmax(one_hot, axis=1)
    labels = [columns[idx] for idx in max_idx]
    return df[image_col].astype(str).tolist(), labels


def _load_pairs(images_dir: Path, labels_csv: Path) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(labels_csv)
    image_col = _resolve_image_col(list(df.columns))
    image_ids, labels = _extract_labels(df, image_col)

    paths = []
    filtered_labels = []
    for image_id, label in zip(image_ids, labels):
        candidate = images_dir / f"{image_id}.jpg"
        if not candidate.exists():
            candidate = images_dir / f"{image_id}.png"
        if candidate.exists():
            paths.append(str(candidate))
            filtered_labels.append(label)

    if not paths:
        raise ValueError("No images found that match the CSV identifiers.")

    return paths, filtered_labels


def _build_label_map(labels: List[str]) -> Dict[str, int]:
    unique = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique)}


def _make_dataset(paths: List[str], labels: List[str], label_map: Dict[str, int], img_size: int,
                  batch_size: int, shuffle: bool) -> tf.data.Dataset:
    y = np.array([label_map[label] for label in labels], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 1000), reshuffle_each_iteration=True)

    def _load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (img_size, img_size))
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, depth=len(label_map))
        return image, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--model-dir", default="../model")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    labels_csv = Path(args.labels_csv).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    paths, labels = _load_pairs(images_dir, labels_csv)
    label_map = _build_label_map(labels)

    indices = np.arange(len(paths))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - args.val_split))
    train_idx, val_idx = indices[:split], indices[split:]

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_ds = _make_dataset(train_paths, train_labels, label_map, args.img_size, args.batch_size, True)
    val_ds = _make_dataset(val_paths, val_labels, label_map, args.img_size, args.batch_size, False)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    preprocess = tf.keras.applications.efficientnet.preprocess_input

    def prep_train(x, y):
        x = augmentation(x)
        x = preprocess(x)
        return x, y

    def prep_val(x, y):
        x = preprocess(x)
        return x, y

    train_ds = train_ds.map(prep_train, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep_val, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    base = tf.keras.applications.EfficientNetB0(
        input_shape=(args.img_size, args.img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(label_map), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    if hasattr(tf.keras.optimizers, "legacy"):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    model_path = model_dir / "isic2019_model.h5"
    model.save(model_path)

    labels_path = model_dir / "isic2019_labels.json"
    labels_path.write_text(json.dumps({str(v): k for k, v in label_map.items()}, indent=2))

    info = {
        "dataset": "isic2019",
        "image_size": args.img_size,
        "channels": 3,
        "num_classes": len(label_map),
    }
    (model_dir / "isic2019_info.json").write_text(json.dumps(info, indent=2))

    print(f"Saved ISIC 2019 model to {model_path}")


if __name__ == "__main__":
    main()
