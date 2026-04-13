import json
import os
import sqlite3
import time
import threading
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas
from werkzeug.security import check_password_hash, generate_password_hash
from web3 import Web3
from web3.middleware import geth_poa_middleware

from guidance import build_guidance

# -----------------------------
# ENV / CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# Loads environment variables (commonly from a dotenv file) into os.environ.
# Keep secrets OUT of code; set these in your environment:
# PRIVATE_KEY=...
# WALLET_ADDRESS=...
# INFURA_URL=...
# PINATA_API_KEY=...
# PINATA_SECRET_API_KEY=...
# CONTRACT_ADDRESS=...
# CONTRACT_ABI_JSON=...
# Optional:
# CONTRACT_DEPLOY_BLOCK=0
# MODEL_VERSION=v1.0
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

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

SEPOLIA_CHAIN_ID = 11155111

# -----------------------------
# APP SETUP
# -----------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024

# Allow both /api/* and /verify/* (since verification route is not under /api).
CORS(app, resources={r"/api/.*": {"origins": "*"}, r"/verify/.*": {"origins": "*"}})

_model_cache: Dict[str, tf.keras.Model] = {}
_web3_cache: Dict[str, Any] = {}  # caches web3+contract objects (NOT verification results)
_report_jobs: Dict[int, Dict[str, Any]] = {}
_report_jobs_lock = threading.Lock()

AUTH_TOKEN_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
DB_PATH = Path(os.environ.get("APP_DB_PATH", BASE_DIR / "data" / "app.db"))


# -----------------------------
# SMALL UTILS
# -----------------------------
def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _auth_secret() -> str:
    return os.environ.get("AUTH_SECRET_KEY") or os.environ.get("FLASK_SECRET_KEY") or "dev-insecure-change-me"


def _token_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_auth_secret())


def _db_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _db_connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                report_id INTEGER PRIMARY KEY,
                user_id TEXT,
                timestamp TEXT NOT NULL,
                model_name TEXT,
                model_version TEXT,
                prediction TEXT,
                json_cid TEXT,
                pdf_cid TEXT,
                json_hash TEXT,
                tx_hash TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _issue_access_token(user_id: str, email: str) -> str:
    payload = {"uid": user_id, "email": email}
    return _token_serializer().dumps(payload)


def _parse_access_token(token: str) -> Dict[str, Any]:
    return _token_serializer().loads(token, max_age=AUTH_TOKEN_MAX_AGE_SECONDS)


def _auth_from_request(optional: bool = False) -> Dict[str, Any] | None:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        if optional:
            return None
        raise RuntimeError("Missing bearer token.")

    token = auth_header.split(" ", 1)[1].strip()
    if not token:
        if optional:
            return None
        raise RuntimeError("Missing bearer token.")

    try:
        payload = _parse_access_token(token)
    except SignatureExpired as exc:
        if optional:
            return None
        raise RuntimeError("Session expired. Please login again.") from exc
    except BadSignature as exc:
        if optional:
            return None
        raise RuntimeError("Invalid auth token.") from exc

    uid = payload.get("uid")
    email = payload.get("email")
    if not uid or not email:
        if optional:
            return None
        raise RuntimeError("Invalid auth token payload.")
    return {"id": uid, "email": email}


def _get_user_by_email(email: str) -> sqlite3.Row | None:
    conn = _db_connect()
    try:
        row = conn.execute("SELECT id, email, password_hash FROM users WHERE email = ?", (email,)).fetchone()
        return row
    finally:
        conn.close()


def _create_user(email: str, password: str) -> Dict[str, str]:
    user_id = str(uuid.uuid4())
    now = _utc_iso_now()
    password_hash = generate_password_hash(password)
    conn = _db_connect()
    try:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, email, password_hash, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": user_id, "email": email}


def _upsert_report_record(
    *,
    report_id: int,
    user_id: str | None,
    timestamp_iso: str,
    model_name: str,
    model_version: str,
    prediction: str,
    status: str,
    json_cid: str | None = None,
    pdf_cid: str | None = None,
    json_hash: str | None = None,
    tx_hash: str | None = None,
) -> None:
    now = _utc_iso_now()
    conn = _db_connect()
    try:
        conn.execute(
            """
            INSERT INTO reports (
                report_id, user_id, timestamp, model_name, model_version, prediction,
                json_cid, pdf_cid, json_hash, tx_hash, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(report_id) DO UPDATE SET
                user_id=excluded.user_id,
                timestamp=excluded.timestamp,
                model_name=excluded.model_name,
                model_version=excluded.model_version,
                prediction=excluded.prediction,
                json_cid=excluded.json_cid,
                pdf_cid=excluded.pdf_cid,
                json_hash=excluded.json_hash,
                tx_hash=excluded.tx_hash,
                status=excluded.status,
                updated_at=excluded.updated_at
            """,
            (
                report_id,
                user_id,
                timestamp_iso,
                model_name,
                model_version,
                prediction,
                json_cid,
                pdf_cid,
                json_hash,
                tx_hash,
                status,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _get_reports_for_user(user_id: str) -> List[Dict[str, Any]]:
    conn = _db_connect()
    try:
        rows = conn.execute(
            """
            SELECT report_id, timestamp, model_name, model_version, prediction,
                   json_cid, pdf_cid, json_hash, tx_hash, status
            FROM reports
            WHERE user_id = ?
            ORDER BY timestamp DESC
            """,
            (user_id,),
        ).fetchall()
    finally:
        conn.close()

    reports: List[Dict[str, Any]] = []
    for row in rows:
        reports.append(
            {
                "reportId": row["report_id"],
                "timestamp": row["timestamp"],
                "prediction": row["prediction"],
                "model": {"name": row["model_name"], "version": row["model_version"]},
                "json_cid": row["json_cid"],
                "pdf_cid": row["pdf_cid"],
                "hash": row["json_hash"],
                "tx_hash": row["tx_hash"],
                "persistence_status": row["status"],
            }
        )
    return reports


_init_db()


# -----------------------------
# MODEL LOADING / INFERENCE
# -----------------------------
def _load_assets(
    kind: str, model_path: Path, labels_path: Path, info_path: Path
) -> Tuple[tf.keras.Model, Dict[str, str], Dict[str, Any]]:
    if kind not in _model_cache:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}.")
        _model_cache[kind] = tf.keras.models.load_model(model_path)

    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    model_info = json.loads(info_path.read_text()) if info_path.exists() else {"image_size": 256, "channels": 3}
    return _model_cache[kind], labels, model_info


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
        topk.append({"index": int(idx), "label": label_text, "probability": float(probs[idx])})
    return topk


def _normalize_modality(label: str) -> str:
    text = label.lower()
    if "dermo" in text:
        return "dermoscopy"
    if "clinical" in text:
        return "clinical"
    return label.lower()


def _store_report_job(report_id: int, payload: Dict[str, Any]) -> None:
    with _report_jobs_lock:
        _report_jobs[report_id] = payload


def _update_report_job(report_id: int, **updates: Any) -> None:
    with _report_jobs_lock:
        current = _report_jobs.get(report_id, {})
        current.update(updates)
        _report_jobs[report_id] = current


def _get_report_job(report_id: int) -> Dict[str, Any] | None:
    with _report_jobs_lock:
        job = _report_jobs.get(report_id)
        return dict(job) if job else None


# -----------------------------
# REPORT GENERATION (JSON + PDF)
# -----------------------------
def generate_json_report(
    *,
    report_id: int,
    user_id: str,
    prediction_label: str,
    confidence_prob: float,
    model_name: str,
    model_version: str,
    timestamp_iso: str,
) -> Tuple[Dict[str, Any], bytes]:
    """
    Creates the machine-readable JSON report and returns:
      - report dict (for convenience)
      - canonical JSON bytes (used for both IPFS upload and hashing)

    IMPORTANT: We use canonical JSON bytes so the hash is stable and verifiable.
    """
    report: Dict[str, Any] = {
        "reportId": report_id,
        "user_id": user_id,
        "prediction": prediction_label,
        "confidence": round(float(confidence_prob) * 100.0, 2),  # percent
        "timestamp": timestamp_iso,
        "model": {"name": model_name, "version": model_version},
    }

    # Canonical form: stable ordering + no extra whitespace.
    report_bytes = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return report, report_bytes


def _persist_report_async(
    *,
    report_id: int,
    user_id: str | None,
    timestamp_iso: str,
    image_bytes: bytes,
    report_dict: Dict[str, Any],
    json_bytes: bytes,
    used_model: str,
    model_version: str,
    topk: List[Dict[str, Any]],
) -> None:
    try:
        _update_report_job(report_id, persistence_status="uploading_to_ipfs", persistence_message="processing transaction...")

        json_cid = upload_to_ipfs(content_bytes=json_bytes, filename=f"report-{report_id}.json", content_type="application/json")
        json_hash = Web3.keccak(json_bytes)
        json_hash_hex = Web3.to_hex(json_hash)

        _update_report_job(report_id, persistence_status="storing_on_chain", persistence_message="processing transaction...")

        tx_hash = store_on_blockchain(report_id=report_id, json_cid=json_cid, json_hash=json_hash)

        uploaded_image = Image.open(BytesIO(image_bytes))
        pdf_bytes = generate_pdf(
            report=report_dict,
            json_cid=json_cid,
            pdf_title="Dermalyze - AI Skin Check Report",
            tx_hash=tx_hash,
            json_hash_hex=json_hash_hex,
            uploaded_image=uploaded_image,
            top_matches=topk,
            top5_combined=float(sum(item["probability"] for item in topk[:5])),
        )
        pdf_cid = upload_to_ipfs(content_bytes=pdf_bytes, filename=f"report-{report_id}.pdf", content_type="application/pdf")

        _update_report_job(
            report_id,
            persistence_status="complete",
            persistence_message="",
            json_cid=json_cid,
            pdf_cid=pdf_cid,
            hash=json_hash_hex,
            tx_hash=tx_hash,
        )
        _upsert_report_record(
            report_id=report_id,
            user_id=user_id,
            timestamp_iso=timestamp_iso,
            model_name=used_model,
            model_version=model_version,
            prediction=str(report_dict.get("prediction", "")),
            status="complete",
            json_cid=json_cid,
            pdf_cid=pdf_cid,
            json_hash=json_hash_hex,
            tx_hash=tx_hash,
        )
    except Exception as exc:
        _update_report_job(report_id, persistence_status="error", persistence_message=str(exc))
        _upsert_report_record(
            report_id=report_id,
            user_id=user_id,
            timestamp_iso=timestamp_iso,
            model_name=used_model,
            model_version=model_version,
            prediction=str(report_dict.get("prediction", "")),
            status="error",
        )


def _format_pdf_label(label: str) -> str:
    label_map = {
        "NV": "Melanocytic nevus (benign mole)",
        "MEL": "Melanoma",
        "BKL": "Benign keratosis-like lesion",
        "BCC": "Basal cell carcinoma",
        "AK": "Actinic keratosis",
        "AKIEC": "Actinic keratosis / intraepithelial carcinoma",
        "VASC": "Vascular lesion",
        "DF": "Dermatofibroma",
        "SCC": "Squamous cell carcinoma",
    }
    if label in label_map:
        return f"{label}: {label_map[label]}"
    return label.replace("_", " ")


def generate_pdf(
    *,
    report: Dict[str, Any],
    json_cid: str,
    pdf_title: str,
    tx_hash: str,
    json_hash_hex: str,
    uploaded_image: Image.Image,
    top_matches: List[Dict[str, Any]],
    top5_combined: float,
) -> bytes:
    """
    Minimal server-side PDF generator (beginner-friendly).
    Produces a simple one-page PDF containing the key session metadata.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    left = 72
    right = width - 72

    def _ensure_space(y_pos: float, needed: float) -> float:
        if y_pos - needed < 72:
            c.showPage()
            c.setFont("Helvetica", 11)
            return height - 72
        return y_pos

    def _draw_wrapped(text: str, y_pos: float, font_name: str = "Helvetica", font_size: int = 11, gap: int = 4) -> float:
        c.setFont(font_name, font_size)
        lines = simpleSplit(text, font_name, font_size, right - left)
        y_pos = _ensure_space(y_pos, (font_size + gap) * max(len(lines), 1))
        for line in lines:
            c.drawString(left, y_pos, line)
            y_pos -= (font_size + gap)
        return y_pos

    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, pdf_title)

    y -= 32
    y = _draw_wrapped(f"Report ID: {report.get('reportId')}", y)
    y = _draw_wrapped(f"User ID: {report.get('user_id')}", y)
    y = _draw_wrapped(f"Timestamp (UTC): {report.get('timestamp')}", y)

    # Draw uploaded image between timestamp and prediction details.
    image_for_pdf = uploaded_image.convert("RGB")
    img_w, img_h = image_for_pdf.size
    max_w = right - left
    max_h = 220
    scale = min(max_w / float(img_w), max_h / float(img_h), 1.0)
    draw_w = img_w * scale
    draw_h = img_h * scale
    y = _ensure_space(y, draw_h + 20)
    c.drawImage(ImageReader(image_for_pdf), left, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, mask='auto')
    y -= (draw_h + 16)

    best_label = str(report.get("prediction", "Unknown"))
    best_conf = float(report.get("confidence", 0.0))
    y = _draw_wrapped(f"Prediction: {_format_pdf_label(best_label)}", y)
    y = _draw_wrapped(f"Confidence: {round(best_conf)}%", y)
    y = _draw_wrapped("How to read this", y, font_name="Helvetica-Bold")
    y = _draw_wrapped(
        "Top-5 accuracy is more reliable for many classes. Low confidence means \"inconclusive.\"",
        y,
    )
    y = _draw_wrapped(f"Top-5 combined confidence: {round(top5_combined * 100)}%", y)

    y -= 2
    y = _draw_wrapped("Top matches", y, font_name="Helvetica-Bold")
    for item in top_matches[:3]:
        lbl = _format_pdf_label(str(item.get("label", "Unknown")))
        pct = round(float(item.get("probability", 0.0)) * 100)
        y = _draw_wrapped(lbl, y)
        y = _draw_wrapped(f"{pct}%", y)

    y -= 4
    y = _draw_wrapped(f"Model: {report.get('model', {}).get('name')} ({report.get('model', {}).get('version')})", y)
    y = _draw_wrapped(f"JSON CID: {json_cid}", y)
    y = _draw_wrapped(f"JSON Keccak256: {json_hash_hex}", y)
    y = _draw_wrapped(f"Blockchain Tx: {tx_hash}", y)

    y -= 8
    y = _draw_wrapped(
        "This AI tool is for educational use only and is not medical advice. Skin conditions can look similar. "
        "If you are concerned or symptoms worsen, seek care from a licensed clinician.",
        y,
    )

    c.showPage()
    c.save()

    return buffer.getvalue()


# -----------------------------
# IPFS (PINATA) UPLOAD
# -----------------------------
def upload_to_ipfs(*, content_bytes: bytes, filename: str, content_type: str) -> str:
    """
    Uploads a file to IPFS via Pinata's pinFileToIPFS endpoint.
    Returns the CID (IpfsHash).
    """
    pinata_key = _require_env("PINATA_API_KEY")
    pinata_secret = _require_env("PINATA_SECRET_API_KEY")

    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {"pinata_api_key": pinata_key, "pinata_secret_api_key": pinata_secret}
    files = {"file": (filename, content_bytes, content_type)}
    data = {"pinataMetadata": json.dumps({"name": filename})}

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"Pinata upload failed ({resp.status_code}): {resp.text}")

    payload = resp.json()
    cid = payload.get("IpfsHash")
    if not cid:
        raise RuntimeError(f"Pinata response missing IpfsHash: {payload}")
    return cid


def fetch_json_from_ipfs(*, json_cid: str) -> bytes:
    """
    Fetches JSON bytes from IPFS. We hash these bytes directly during verification.
    """
    # Public gateway; for production you might prefer a dedicated Pinata gateway domain.
    url = f"https://gateway.pinata.cloud/ipfs/{json_cid}"
    resp = requests.get(url, timeout=60)
    if resp.status_code >= 300:
        raise RuntimeError(f"Failed to fetch JSON from IPFS ({resp.status_code}): {resp.text}")
    return resp.content


# -----------------------------
# BLOCKCHAIN (SEPOLIA via INFURA)
# -----------------------------
def _get_web3_and_contract() -> Tuple[Web3, Any]:
    if "w3" in _web3_cache and "contract" in _web3_cache:
        return _web3_cache["w3"], _web3_cache["contract"]

    infura_url = _require_env("INFURA_URL")
    contract_address = _require_env("CONTRACT_ADDRESS")
    contract_abi_json = _require_env("CONTRACT_ABI_JSON")

    w3 = Web3(Web3.HTTPProvider(infura_url))
    # Sepolia historically needed POA middleware in some setups; safe to include.
    w3.middleware_onion.inject(geth_poa_middleware, layer=0)

    if not w3.is_connected():
        raise RuntimeError("Web3 provider not connected. Check INFURA_URL.")

    abi = json.loads(contract_abi_json)
    contract = w3.eth.contract(address=Web3.to_checksum_address(contract_address), abi=abi)

    _web3_cache["w3"] = w3
    _web3_cache["contract"] = contract
    return w3, contract


def store_on_blockchain(*, report_id: int, json_cid: str, json_hash: bytes) -> str:
    """
    Sends a signed transaction from the backend custodial wallet.
    Stores ONLY: reportId, jsonCID, bytes32 hash (in event logs).
    """
    private_key = _require_env("PRIVATE_KEY")
    expected_wallet = Web3.to_checksum_address(_require_env("WALLET_ADDRESS"))

    w3, contract = _get_web3_and_contract()

    acct = w3.eth.account.from_key(private_key)
    if Web3.to_checksum_address(acct.address) != expected_wallet:
        raise RuntimeError("WALLET_ADDRESS does not match PRIVATE_KEY-derived address.")

    nonce = w3.eth.get_transaction_count(acct.address)

    # Estimate gas
    gas_est = contract.functions.storeReport(report_id, json_cid, json_hash).estimate_gas({"from": acct.address})
    gas_limit = int(gas_est * 1.2)  # small safety margin

    tx_params: Dict[str, Any] = {
        "from": acct.address,
        "nonce": nonce,
        "chainId": SEPOLIA_CHAIN_ID,
        "gas": gas_limit,
    }

    # Prefer EIP-1559 fees; fallback to legacy gasPrice if baseFee is unavailable.
    try:
        pending_block = w3.eth.get_block("pending")
        base_fee = pending_block.get("baseFeePerGas")
        if base_fee is not None:
            max_priority = w3.to_wei(2, "gwei")
            max_fee = int(base_fee * 2 + max_priority)
            tx_params["maxPriorityFeePerGas"] = max_priority
            tx_params["maxFeePerGas"] = max_fee
        else:
            tx_params["gasPrice"] = w3.eth.gas_price
    except Exception:
        tx_params["gasPrice"] = w3.eth.gas_price

    tx = contract.functions.storeReport(report_id, json_cid, json_hash).build_transaction(tx_params)
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)

    # Wait for inclusion (optional but helps return a reliable tx hash).
    w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    return tx_hash.hex()


def verify_report(*, json_cid: str) -> str:
    """
    Real-time verification:
      1) Fetch JSON from IPFS
      2) Recompute keccak hash (JSON only)
      3) Query blockchain event logs filtered by indexed fields (reportHash + uploader)
      4) Confirm jsonCID matches an emitted event

    Returns: "verified" or "tampered"
    """
    expected_wallet = Web3.to_checksum_address(_require_env("WALLET_ADDRESS"))
    from_block = int(os.environ.get("CONTRACT_DEPLOY_BLOCK", "0") or "0")

    json_bytes = fetch_json_from_ipfs(json_cid=json_cid)
    recomputed_hash = Web3.keccak(json_bytes)

    w3, contract = _get_web3_and_contract()

    # Filter by indexed fields to avoid scanning every event:
    # - reportHash is indexed
    # - uploader is indexed
    logs = contract.events.ReportStored().get_logs(
        argument_filters={"reportHash": recomputed_hash, "uploader": expected_wallet},
        fromBlock=from_block,
        toBlock="latest",
    )

    for ev in logs:
        args = ev.get("args", {})
        if args.get("jsonCID") == json_cid:
            return "verified"

    return "tampered"


# -----------------------------
# ROUTES
# -----------------------------
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


@app.post("/api/auth/register")
def register_user() -> Any:
    payload = request.get_json(silent=True) or {}
    email_raw = str(payload.get("email", ""))
    password = str(payload.get("password", ""))

    email = _normalize_email(email_raw)
    if not email or "@" not in email:
        return jsonify(error="A valid email is required."), 400
    if len(password) < 8:
        return jsonify(error="Password must be at least 8 characters."), 400

    if _get_user_by_email(email):
        return jsonify(error="Email already registered."), 409

    user = _create_user(email, password)
    token = _issue_access_token(user["id"], user["email"])
    return jsonify(token=token, user={"id": user["id"], "email": user["email"]})


@app.post("/api/auth/login")
def login_user() -> Any:
    payload = request.get_json(silent=True) or {}
    email_raw = str(payload.get("email", ""))
    password = str(payload.get("password", ""))

    email = _normalize_email(email_raw)
    row = _get_user_by_email(email)
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify(error="Invalid credentials."), 401

    token = _issue_access_token(str(row["id"]), str(row["email"]))
    return jsonify(token=token, user={"id": row["id"], "email": row["email"]})


@app.get("/api/auth/me")
def auth_me() -> Any:
    try:
        auth_user = _auth_from_request(optional=False)
        return jsonify(user=auth_user)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 401


@app.get("/api/reports")
def my_reports() -> Any:
    try:
        auth_user = _auth_from_request(optional=False)
        reports = _get_reports_for_user(str(auth_user["id"]))
        return jsonify(reports=reports)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 401


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
        auth_user = _auth_from_request(optional=True)
        user_id = str(auth_user["id"]) if auth_user else None

        image = Image.open(file.stream)
        modality_override = request.form.get("modality")

        # ---- Modality stage (existing behavior) ----
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

        # ---- Route to correct disease model (existing behavior) ----
        if use_dermoscopy:
            model, labels, model_info = _load_assets("isic", ISIC_MODEL_PATH, ISIC_LABELS_PATH, ISIC_INFO_PATH)
            used_model = "isic2019"
        else:
            modality_label = "clinical"
            acne_model, acne_labels, acne_info = _load_assets(
                "acne_binary", ACNE_BINARY_MODEL_PATH, ACNE_BINARY_LABELS_PATH, ACNE_BINARY_INFO_PATH
            )
            acne_size = int(acne_info.get("image_size", 256))
            acne_channels = int(acne_info.get("channels", 3))
            acne_array = _prepare_image(image, size=acne_size, channels=acne_channels)
            acne_probs = _predict(acne_model, acne_array)
            acne_top = _format_topk(acne_labels, acne_probs, k=2)
            acne_conf = float(acne_top[0]["probability"])

            if acne_top[0]["label"] == "acne_like" and acne_conf >= 0.6:
                model, labels, model_info = _load_assets(
                    "acne_subtype", ACNE_SUBTYPE_MODEL_PATH, ACNE_SUBTYPE_LABELS_PATH, ACNE_SUBTYPE_INFO_PATH
                )
                used_model = "acne_subtype"
            else:
                model, labels, model_info = _load_assets("sd198", SD198_MODEL_PATH, SD198_LABELS_PATH, SD198_INFO_PATH)
                used_model = "sd198"

        size = int(model_info.get("image_size", 256))
        channels = int(model_info.get("channels", 3))
        img_array = _prepare_image(image, size=size, channels=channels)

        base_temperature = model_info.get("temperature")
        temperature = float(base_temperature) if base_temperature is not None else 1.0
        confidence_boost = float(model_info.get("confidence_boost", 1.0) or 1.0)
        temperature = max(temperature * confidence_boost, 0.1)

        probs = _predict(model, img_array, temperature=temperature)
        topk = _format_topk(labels, probs, k=5)
        best = topk[0]

        guidance = build_guidance(best["label"], confidence=best["probability"])

        report_id = int(time.time() * 1000)
        timestamp_iso = _utc_iso_now()
        model_version = os.environ.get("MODEL_VERSION", "v1.0")

        report_dict, json_bytes = generate_json_report(
            report_id=report_id,
            user_id=user_id or "anonymous",
            prediction_label=best["label"],
            confidence_prob=float(best["probability"]),
            model_name=used_model,
            model_version=model_version,
            timestamp_iso=timestamp_iso,
        )

        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()

        _store_report_job(
            report_id,
            {
                "report_id": report_id,
                "user_id": user_id,
                "timestamp": timestamp_iso,
                "persistence_status": "processing",
                "persistence_message": "processing transaction...",
                "json_cid": None,
                "pdf_cid": None,
                "hash": None,
                "tx_hash": None,
            },
        )

        _upsert_report_record(
            report_id=report_id,
            user_id=user_id,
            timestamp_iso=timestamp_iso,
            model_name=used_model,
            model_version=model_version,
            prediction=best["label"],
            status="processing",
        )

        worker = threading.Thread(
            target=_persist_report_async,
            kwargs={
                "report_id": report_id,
                "user_id": user_id,
                "timestamp_iso": timestamp_iso,
                "image_bytes": image_bytes,
                "report_dict": report_dict,
                "json_bytes": json_bytes,
                "used_model": used_model,
                "model_version": model_version,
                "topk": topk,
            },
            daemon=True,
        )
        worker.start()

        return jsonify(
            prediction=best["label"],
            model_version=model_version,
            reportId=report_id,
            timestamp=timestamp_iso,
            model={"name": used_model, "version": model_version},
            persistence_status="processing",
            persistence_message="processing transaction...",
            # Keep existing app response fields if your frontend still expects them:
            modality={"label": modality_label, "confidence": modality_confidence, "top2": modality_top},
            acne_stage={"label": acne_top[0]["label"] if modality_label == "clinical" else "n/a",
                        "confidence": acne_conf if modality_label == "clinical" else None},
            used_model=used_model,
            top3=topk[:3],
            top5=topk,
            guidance=guidance,
        )

    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 500
    except Exception as exc:
        return jsonify(error="Prediction failed.", detail=str(exc)), 500


@app.get("/api/report-status/<int:report_id>")
def report_status(report_id: int) -> Any:
    job = _get_report_job(report_id)
    if not job:
        return jsonify(error="Report not found."), 404

    owner_id = job.get("user_id")
    if owner_id:
        try:
            auth_user = _auth_from_request(optional=False)
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 401
        if str(auth_user["id"]) != str(owner_id):
            return jsonify(error="Forbidden."), 403

    return jsonify(job)


@app.get("/verify/<json_cid>")
def verify(json_cid: str) -> Any:
    try:
        status = verify_report(json_cid=json_cid)
        return jsonify(status=status)
    except Exception as exc:
        return jsonify(error="Verification failed.", detail=str(exc)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
