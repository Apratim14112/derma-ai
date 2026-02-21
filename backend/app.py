import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

from guidance import build_guidance

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "model"))

MODALITY_MODEL_PATH = Path(os.environ.get("MODALITY_MODEL_PATH", MODEL_DIR / "modality_model.h5"))
MODALITY_LABELS_PATH = Path(os.environ.get("MODALITY_LABELS_PATH", MODEL_DIR / "modality_labels.json"))
MODALITY_INFO_PATH = Path(os.environ.get("MODALITY_INFO_PATH", MODEL_DIR / "modality_info.json"))

ISIC_MODEL_PATH = Path(os.environ.get("ISIC_MODEL_PATH", MODEL_DIR / "isic2019_model.h5"))
ISIC_LABELS_PATH = Path(os.environ.get("ISIC_LABELS_PATH", MODEL_DIR / "isic2019_labels.json"))
ISIC_INFO_PATH = Path(os.environ.get("ISIC_INFO_PATH", MODEL_DIR / "isic2019_info.json"))

SD198_MODEL_PATH = Path(os.environ.get("SD198_MODEL_PATH", MODEL_DIR / "sd198_model.h5"))
SD198_LABELS_PATH = Path(os.environ.get("SD198_LABELS_PATH", MODEL_DIR / "sd198_labels.json"))
SD198_INFO_PATH = Path(os.environ.get("SD198_INFO_PATH", MODEL_DIR / "sd198_info.json"))

ACNE_BINARY_MODEL_PATH = Path(os.environ.get("ACNE_BINARY_MODEL_PATH", MODEL_DIR / "acne_binary_model.h5"))
ACNE_BINARY_LABELS_PATH = Path(os.environ.get("ACNE_BINARY_LABELS_PATH", MODEL_DIR / "acne_binary_labels.json"))
ACNE_BINARY_INFO_PATH = Path(os.environ.get("ACNE_BINARY_INFO_PATH", MODEL_DIR / "acne_binary_info.json"))

ACNE_SUBTYPE_MODEL_PATH = Path(os.environ.get("ACNE_SUBTYPE_MODEL_PATH", MODEL_DIR / "acne_subtype_model.h5"))
ACNE_SUBTYPE_LABELS_PATH = Path(os.environ.get("ACNE_SUBTYPE_LABELS_PATH", MODEL_DIR / "acne_subtype_labels.json"))
ACNE_SUBTYPE_INFO_PATH = Path(os.environ.get("ACNE_SUBTYPE_INFO_PATH", MODEL_DIR / "acne_subtype_info.json"))

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": "*"}})

_model_cache = {}


@app.get("/api/health")
def health() -> Any:
    return jsonify(
        status="ok",
        modality_model_loaded=_model_cache.get("modality") is not None,
        isic_model_loaded=_model_cache.get("isic") is not None,
        sd198_model_loaded=_model_cache.get("sd198") is not None,
        acne_binary_loaded=_model_cache.get("acne_binary") is not None,
        acne_subtype_loaded=_model_cache.get("acne_subtype") is not None,
        model_dir=str(MODEL_DIR),
    )


def _load_assets(kind: str, model_path: Path, labels_path: Path, info_path: Path) -> Tuple[tf.keras.Model, Dict[str, str], Dict[str, Any]]:
    if kind not in _model_cache:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}.")
        _model_cache[kind] = tf.keras.models.load_model(model_path)

    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    model_info = json.loads(info_path.read_text()) if info_path.exists() else {"image_size": 256, "channels": 3}

    return _model_cache[kind], labels, model_info


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _prepare_image(image: Image.Image, size: int, channels: int) -> np.ndarray:
    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize((size, size))
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    if channels == 1:
        img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if not temperature or temperature <= 0:
        return probs
    # Lower temperature (<1.0) makes predictions more confident; >1.0 softens.
    logits = np.log(probs + 1e-8) / temperature
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)


def _predict(model: tf.keras.Model, img_array: np.ndarray, temperature: float | None = None) -> np.ndarray:
    preds = model.predict(img_array, verbose=0)
    preds = np.squeeze(preds)
    if preds.ndim == 0:
        preds = np.array([preds])
    if np.any(preds < 0) or not np.isclose(np.sum(preds), 1.0, atol=1e-2):
        preds = tf.nn.softmax(preds).numpy()
    if temperature:
        preds = _apply_temperature(preds, temperature)
    return preds


def _format_topk(labels: Dict[str, str], probs: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
    indices = probs.argsort()[::-1][:k]
    topk = []
    for idx in indices:
        label_text = labels.get(str(idx), f"Class {idx}")
        topk.append(
            {
                "index": int(idx),
                "label": label_text,
                "probability": float(probs[idx]),
            }
        )
    return topk


def _normalize_modality(label: str) -> str:
    text = label.lower()
    if "dermo" in text:
        return "dermoscopy"
    if "clinical" in text:
        return "clinical"
    return label.lower()


@app.post("/api/predict")
def predict() -> Any:
    if "image" not in request.files:
        return jsonify(error="Missing image file field 'image'."), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify(error="No file selected."), 400

    if not _allowed_file(file.filename):
        return jsonify(error="Unsupported file type."), 400

    try:
        image = Image.open(file.stream)
        modality_override = request.form.get("modality")
        if modality_override in {"clinical", "dermoscopy"}:
            modality_label = modality_override
            modality_confidence = 1.0
            modality_top = [{"index": 0, "label": modality_override, "probability": 1.0}]
        else:
            modality_model, modality_labels, modality_info = _load_assets(
                "modality", MODALITY_MODEL_PATH, MODALITY_LABELS_PATH, MODALITY_INFO_PATH
            )
            modality_size = int(modality_info.get("image_size", 256))
            modality_channels = int(modality_info.get("channels", 3))
            modality_array = _prepare_image(image, size=modality_size, channels=modality_channels)
            modality_probs = _predict(modality_model, modality_array)
            modality_top = _format_topk(modality_labels, modality_probs, k=2)
            modality_label = _normalize_modality(modality_top[0]["label"])
            modality_confidence = float(modality_top[0]["probability"])

        acne_top = [{"label": "n/a", "probability": 0.0, "index": 0}]
        acne_conf = 0.0

        dermoscopy_prob = 0.0
        clinical_prob = 0.0
        for item in modality_top:
            normalized = _normalize_modality(item["label"])
            if normalized == "dermoscopy":
                dermoscopy_prob = float(item["probability"])
            elif normalized == "clinical":
                clinical_prob = float(item["probability"])

        # Fast, conservative auto routing:
        # use ISIC only when dermoscopy evidence is very strong;
        # otherwise route to clinical pipeline.
        if modality_override == "clinical":
            use_dermoscopy = False
            modality_label = "clinical"
            modality_confidence = 1.0
        elif modality_override == "dermoscopy":
            use_dermoscopy = True
            modality_label = "dermoscopy"
            modality_confidence = 1.0
        else:
            use_dermoscopy = (
                modality_label == "dermoscopy"
                and modality_confidence >= 0.9
                and dermoscopy_prob >= (clinical_prob + 0.1)
            )
            modality_label = "dermoscopy" if use_dermoscopy else "clinical"

        if use_dermoscopy:
            model, labels, model_info = _load_assets(
                "isic", ISIC_MODEL_PATH, ISIC_LABELS_PATH, ISIC_INFO_PATH
            )
            used_model = "isic2019"
        else:
            # Reflect final routing decision in the displayed modality.
            modality_label = "clinical"
            # Stage 1: acne-like vs not
            acne_model, acne_labels, acne_info = _load_assets(
                "acne_binary", ACNE_BINARY_MODEL_PATH, ACNE_BINARY_LABELS_PATH, ACNE_BINARY_INFO_PATH
            )
            if acne_top[0]["label"] == "n/a":
                acne_size = int(acne_info.get("image_size", 256))
                acne_channels = int(acne_info.get("channels", 3))
                acne_array = _prepare_image(image, size=acne_size, channels=acne_channels)
                acne_probs = _predict(acne_model, acne_array)
                acne_top = _format_topk(acne_labels, acne_probs, k=2)
                acne_conf = float(acne_top[0]["probability"])

            if acne_top[0]["label"] == "acne_like" and acne_conf >= 0.6:
                # Stage 2: acne subtype
                model, labels, model_info = _load_assets(
                    "acne_subtype", ACNE_SUBTYPE_MODEL_PATH, ACNE_SUBTYPE_LABELS_PATH, ACNE_SUBTYPE_INFO_PATH
                )
                used_model = "acne_subtype"
            else:
                model, labels, model_info = _load_assets(
                    "sd198", SD198_MODEL_PATH, SD198_LABELS_PATH, SD198_INFO_PATH
                )
                used_model = "sd198"

        size = int(model_info.get("image_size", 256))
        channels = int(model_info.get("channels", 3))
        img_array = _prepare_image(image, size=size, channels=channels)
        # Apply optional confidence shaping consistently across models.
        base_temperature = model_info.get("temperature")
        temperature = float(base_temperature) if base_temperature is not None else 1.0
        confidence_boost = float(model_info.get("confidence_boost", 1.0) or 1.0)
        temperature = max(temperature * confidence_boost, 0.1)
        probs = _predict(model, img_array, temperature=temperature)
        topk = _format_topk(labels, probs, k=5)
        best = topk[0]
        top5_combined = float(sum(item["probability"] for item in topk[:5]))

        guidance = build_guidance(best["label"], confidence=best["probability"])

        response = {
            "modality": {
                "label": modality_label,
                "confidence": modality_confidence,
                "top2": modality_top,
            },
            "acne_stage": {
                "label": acne_top[0]["label"] if modality_label == "clinical" else "n/a",
                "confidence": acne_conf if modality_label == "clinical" else None,
            },
            "used_model": used_model,
            "prediction": best,
            "top3": topk[:3],
            "top5": topk,
            "top5_combined": top5_combined,
            "guidance": guidance,
        }
        return jsonify(response)
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 500
    except Exception as exc:
        return jsonify(error="Prediction failed.", detail=str(exc)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
