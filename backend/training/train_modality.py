import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", default="../model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=args.val_split,
        subset="training",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="categorical",
        validation_split=args.val_split,
        subset="validation",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    class_names = list(train_ds.class_names)
    num_classes = len(class_names)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
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

    try:
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(args.img_size, args.img_size, 3),
            include_top=False,
            weights="imagenet",
        )
    except Exception:
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(args.img_size, args.img_size, 3),
            include_top=False,
            weights=None,
        )
    base.trainable = False

    inputs = tf.keras.Input(shape=(args.img_size, args.img_size, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    if hasattr(tf.keras.optimizers, "legacy"):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Balance dermoscopy/clinical contribution if dataset is imbalanced.
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=args.val_split,
        subset="training",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    label_counts = np.zeros(num_classes, dtype=np.int64)
    for _, batch_labels in raw_train_ds:
        values, counts = np.unique(batch_labels.numpy(), return_counts=True)
        label_counts[values] += counts
    total = float(label_counts.sum())
    class_weight = {
        idx: total / (num_classes * max(1.0, float(count)))
        for idx, count in enumerate(label_counts)
    }

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "modality_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    model_path = model_dir / "modality_model.h5"
    model.save(model_path)

    labels_path = model_dir / "modality_labels.json"
    labels_path.write_text(json.dumps({str(i): name for i, name in enumerate(class_names)}, indent=2))

    info = {
        "dataset": "modality",
        "image_size": args.img_size,
        "channels": 3,
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_weight": class_weight,
    }
    (model_dir / "modality_info.json").write_text(json.dumps(info, indent=2))

    print(f"Saved modality model to {model_path}")


if __name__ == "__main__":
    main()
