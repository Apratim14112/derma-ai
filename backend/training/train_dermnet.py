import argparse
import json
from pathlib import Path
from collections import Counter

import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", default="../model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--include-classes", default="")
    parser.add_argument("--model-prefix", default="sd198")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    include_classes = [c.strip() for c in args.include_classes.split(",") if c.strip()]
    available_classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    if include_classes:
        missing = [c for c in include_classes if c not in available_classes]
        if missing:
            print(f"Warning: missing classes skipped: {', '.join(missing)}")
        include_classes = [c for c in include_classes if c in available_classes]
        if not include_classes:
            raise SystemExit("No valid classes found after filtering. Check --include-classes.")
    else:
        include_classes = available_classes

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

    full_class_names = train_ds.class_names
    keep_indices = [full_class_names.index(name) for name in include_classes]
    index_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(keep_indices, dtype=tf.int64),
            values=tf.constant(list(range(len(keep_indices))), dtype=tf.int64),
        ),
        default_value=-1,
    )

    def _filter_and_remap(ds):
        # Unbatch -> filter -> re-batch so filter predicate is scalar.
        ds = ds.unbatch()

        def _map(x, y):
            label = tf.argmax(y, axis=-1, output_type=tf.int64)
            new_label = index_table.lookup(label)
            return x, new_label

        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.filter(lambda _x, y: y >= 0)
        ds = ds.map(
            lambda x, y: (x, tf.one_hot(y, depth=len(keep_indices))),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(args.batch_size)
        return ds

    train_ds = _filter_and_remap(train_ds)
    val_ds = _filter_and_remap(val_ds)

    class_names = include_classes

    # Compute class weights from folder counts to help class imbalance.
    counts = Counter()
    for class_name in class_names:
        class_dir = data_dir / class_name
        if class_dir.exists():
            counts[class_name] = len([p for p in class_dir.iterdir() if p.is_file()])

    total = sum(counts.values())
    class_weight = {}
    for idx, name in enumerate(class_names):
        # Avoid division by zero
        count = max(counts.get(name, 1), 1)
        class_weight[idx] = total / (len(class_names) * count)

    # Data pipeline + MobileNetV2 transfer learning for stronger baseline.
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
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
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

    # Stage 1: train classifier head
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight)

    # Stage 2: fine-tune top layers of the base model (optional but recommended)
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
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, class_weight=class_weight)

    # Temperature scaling calibration on validation set
    val_probs = model.predict(val_ds, verbose=0)
    val_labels = []
    for _, y in val_ds:
        val_labels.append(y.numpy())
    val_labels = tf.concat(val_labels, axis=0).numpy()

    def nll(temp: float) -> float:
        temp = max(temp, 1e-3)
        adjusted = tf.nn.softmax(tf.math.log(val_probs + 1e-8) / temp, axis=1).numpy()
        return -float((val_labels * tf.math.log(adjusted + 1e-8)).numpy().sum() / val_labels.shape[0])

    best_t = 1.0
    best_nll = nll(best_t)
    for t in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        score = nll(t)
        if score < best_nll:
            best_nll = score
            best_t = t

    model_path = model_dir / f"{args.model_prefix}_model.h5"
    model.save(model_path)

    labels_path = model_dir / f"{args.model_prefix}_labels.json"
    labels_path.write_text(json.dumps({str(i): name for i, name in enumerate(class_names)}, indent=2))

    info = {
        "dataset": args.model_prefix,
        "image_size": args.img_size,
        "channels": 3,
        "num_classes": len(class_names),
        "temperature": best_t,
    }
    (model_dir / f"{args.model_prefix}_info.json").write_text(json.dumps(info, indent=2))

    print(f"Saved {args.model_prefix} model to {model_path}")


if __name__ == "__main__":
    main()
