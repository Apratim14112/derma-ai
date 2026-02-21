import argparse
import json
from collections import Counter
from pathlib import Path

import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", default="../model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--acne-classes",
        default="Acne_Vulgaris,Acne_Keloidalis_Nuchae,Pityrosporum_Folliculitis,Rosacea,Perioral_Dermatitis,Pomade_Acne",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    acne_classes = {c.strip() for c in args.acne_classes.split(",") if c.strip()}

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=args.val_split,
        subset="training",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=args.val_split,
        subset="validation",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names
    acne_indices = [class_names.index(c) for c in acne_classes if c in class_names]
    if not acne_indices:
        raise SystemExit("No acne classes found in dataset. Check --acne-classes.")

    acne_idx = tf.constant(acne_indices, dtype=tf.int64)

    def to_binary(x, y):
        label = tf.cast(y, tf.int64)
        # label shape: (batch,). Compare against acne indices.
        matches = tf.equal(tf.expand_dims(label, -1), acne_idx)
        is_acne = tf.reduce_any(matches, axis=-1)
        return x, tf.one_hot(tf.cast(is_acne, tf.int64), depth=2)

    train_ds = train_ds.map(to_binary, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(to_binary, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

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
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    if hasattr(tf.keras.optimizers, "legacy"):
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        fine_tune_optimizer = tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr)
        if hasattr(tf.keras.optimizers, "legacy"):
            fine_tune_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.fine_tune_lr)

        model.compile(
            optimizer=fine_tune_optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs)

    model_path = model_dir / "acne_binary_model.h5"
    model.save(model_path)

    labels_path = model_dir / "acne_binary_labels.json"
    labels_path.write_text(json.dumps({"0": "not_acne", "1": "acne_like"}, indent=2))

    info = {
        "dataset": "sd198_acne_binary",
        "image_size": args.img_size,
        "channels": 3,
        "num_classes": 2,
        "acne_classes": sorted(list(acne_classes)),
    }
    (model_dir / "acne_binary_info.json").write_text(json.dumps(info, indent=2))

    print(f"Saved acne binary model to {model_path}")


if __name__ == "__main__":
    main()
