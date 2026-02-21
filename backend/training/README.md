# Training

This project uses three models:
- **Modality classifier**: dermoscopy vs clinical images
- **ISIC 2019 classifier** (dermoscopy)
- **SD-198 classifier** (clinical)

## Install training deps

```bash
pip install -r backend/requirements-train.txt
```

## Dataset layout

```
backend/data/
  isic2019/
    images/
    labels.csv
  sd198/
    class_a/
      img1.jpg
    class_b/
      img2.jpg
  modality/
    dermoscopy/
      img1.jpg
    clinical/
      img2.jpg
```

`labels.csv` should include `image` and `label` columns, or use one-hot label columns.

## Train (256x256)

```bash
python backend/training/train_modality.py --data-dir backend/data/modality --img-size 256 --epochs 10
python backend/training/train_isic2019.py --images-dir backend/data/isic2019/images --labels-csv backend/data/isic2019/labels.csv --img-size 256 --epochs 15
python backend/training/train_dermnet.py --data-dir backend/data/sd198 --img-size 256 --epochs 15
```

Artifacts are written to `backend/model/`:
- `modality_model.h5`, `modality_labels.json`, `modality_info.json`
- `isic2019_model.h5`, `isic2019_labels.json`, `isic2019_info.json`
- `sd198_model.h5`, `sd198_labels.json`, `sd198_info.json`
