# SkinWise — AI Skin Check (Demo)

**Important:** This app is for educational purposes only and is **not** medical advice. Skin conditions can look similar. If you are concerned about a lesion, consult a licensed clinician.

## What this is
- Flask API (TensorFlow) that first classifies image type (dermoscopy vs clinical)
- ISIC 2019 classifier for dermoscopy images
- SD-198 classifier for clinical images (public dataset)
- React UI for upload + camera capture and guidance

## Quick start (local)

### 1) Prepare datasets (ISIC 2019 + DermNet + Modality)

You must download the datasets yourself and place them in these folders:

```
backend/data/
  isic2019/
    images/               # dermoscopy images (.jpg/.png)
    labels.csv            # columns: image + label (or one-hot columns)
  sd198/
    acne/
      img001.jpg
    eczema/
      img002.jpg
    ... (one folder per class)
  modality/
    dermoscopy/
      img001.jpg
    clinical/
      img002.jpg
```

Notes:
- `labels.csv` for ISIC 2019 can be a simple two-column file: `image,label`.
- If your ISIC CSV is one-hot encoded, the training script will infer the label using the max column.

### 2) Train the models (256x256)

```bash
pip install -r backend/requirements-train.txt

# Modality classifier (dermoscopy vs clinical)
python backend/training/train_modality.py --data-dir backend/data/modality --img-size 256 --epochs 10

# ISIC 2019 dermoscopy classifier
python backend/training/train_isic2019.py --images-dir backend/data/isic2019/images --labels-csv backend/data/isic2019/labels.csv --img-size 256 --epochs 15

# SD-198 clinical classifier
python backend/training/train_dermnet.py --data-dir backend/data/sd198 --img-size 256 --epochs 15
```

### 3) Start the backend
```bash
pip install -r backend/requirements.txt
python backend/app.py
```

### 4) Start the frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Docker (deployable)
```bash
docker compose build
docker compose up
```
- Frontend: `http://localhost:8080`
- Backend: `http://localhost:5000`

## Configuration
- `MODEL_DIR` (backend) defaults to `backend/model`
- `VITE_API_URL` (frontend) defaults to `http://localhost:5000`

## Medical safety
- The model output is **not** a diagnosis.
- OTC suggestions are general and may not be appropriate for your condition.
- Seek urgent care for rapidly changing, bleeding, or painful lesions.

## License
This project is provided as a demo. Ensure you follow the license terms of any datasets you use.
