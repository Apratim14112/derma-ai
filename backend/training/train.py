import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf

from model_utils import build_cnn

try:
    from medmnist import INFO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "medmnist is required for training. Install backend/requirements-train.txt"
    ) from exc


def download_dataset(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, target_path)


def load_npz(path: Path):
    with np.load(path) as data:
        return (
            data["train_images"],
            data["train_labels"],
            data["val_images"],
            data["val_labels"],
            data["test_images"],
            data["test_labels"],
        )


def build_model(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    return build_cnn(input_shape, num_classes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dermamnist")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--model-dir", default="../model")
    args = parser.parse_args()

    if args.dataset not in INFO:
        raise SystemExit(f"Unknown dataset '{args.dataset}'. Available: {', '.join(INFO)}")

    info = INFO[args.dataset]
    url = info.get("url")
    if not url:
        raise SystemExit("Dataset URL not found in medmnist INFO.")

    data_dir = Path(args.data_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    dataset_path = data_dir / filename
    download_dataset(url, dataset_path)

    train_images, train_labels, val_images, val_labels, _, _ = load_npz(dataset_path)

    n_channels = int(info.get("n_channels", 3))
    # DermaMNIST uses shape (N, H, W, C); fall back to H if INFO missing.
    if info.get("image_size") is not None:
        image_size = int(info["image_size"])
    else:
        image_size = int(train_images.shape[1])
    num_classes = int(info.get("num_classes", len(info.get("label", {}))))

    def prep(images: np.ndarray) -> np.ndarray:
        images = images.astype(np.float32) / 255.0
        if n_channels == 1 and images.ndim == 3:
            images = np.expand_dims(images, axis=-1)
        return images

    x_train = prep(train_images)
    x_val = prep(val_images)

    y_train = tf.keras.utils.to_categorical(train_labels.squeeze(), num_classes)
    y_val = tf.keras.utils.to_categorical(val_labels.squeeze(), num_classes)

    model = build_model((image_size, image_size, n_channels), num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    model_path = model_dir / "dermamnist_cnn.h5"
    model.save(model_path)

    labels = info.get("label", {})
    labels_path = model_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2))

    model_info = {
        "dataset": args.dataset,
        "image_size": image_size,
        "channels": n_channels,
        "num_classes": num_classes,
    }
    (model_dir / "model_info.json").write_text(json.dumps(model_info, indent=2))

    print(f"Saved model to {model_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
