# Wound Segmentation — U-Net
### Bachelor's Thesis: AI-Powered Chronic Wound Monitoring

---

## Quick start

```bash
# 1. Clone / open in Cursor
# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data
#    data/images/  → wound photos  (.jpg or .png)
#    data/masks/   → binary masks  (.png): white=wound, black=background
#    Naming rule:  001.jpg  →  001.png  (same stem)

# 4. Train
python -m src.train

# 5. Evaluate + generate thesis plots
python -m src.evaluate

# 6. Predict on a new image
python -m src.predict --image path/to/photo.jpg

# 7. Run tests
python -m pytest tests/ -v
```

---

## Project structure

```
wound_segmentation/
├── .cursorrules          ← Cursor AI rules (READ THIS FIRST)
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py         ← ALL settings live here
│   ├── dataset.py        ← Data loading + augmentation
│   ├── model.py          ← U-Net with ResNet34 encoder
│   ├── losses.py         ← BCE + Dice combined loss
│   ├── metrics.py        ← All metrics from classification_metrics.pdf
│   ├── train.py          ← Training loop
│   ├── evaluate.py       ← Evaluation + thesis plots
│   └── predict.py        ← Inference on new images
│
├── data/
│   ├── images/           ← Wound photos go here
│   ├── masks/            ← Binary masks go here
│   └── raw/              ← Original downloads (before processing)
│
├── checkpoints/
│   ├── best_model.pth    ← Best IoU checkpoint (auto-saved)
│   └── last_model.pth    ← Latest epoch checkpoint
│
├── results/
│   ├── metrics/
│   │   └── history.csv   ← Per-epoch metrics log
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── pr_curve.png
│       ├── training_history.png
│       └── prediction_samples.png
│
├── notebooks/            ← Jupyter exploratory analysis
└── tests/
    └── test_all.py       ← Unit tests
```

---

## Data format

| Item | Format | Details |
|------|--------|---------|
| Images | `.jpg` or `.png` | RGB, any size (auto-resized to 256×256) |
| Masks  | `.png` | Grayscale: **white (255) = wound**, black (0) = background |
| Pairing | Same filename stem | `001.jpg` → `001.png` |

**Recommended Kaggle datasets to start with:**
- `kaggle.com/datasets/leoscode/wound-segmentation-images` (2760 samples)
- `kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu`
- `kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages`

---

## Metrics (from classification_metrics.pdf)

Every pixel is a binary classification: wound(1) vs background(0).

| Metric | PDF ref | Target | Why it matters |
|--------|---------|--------|----------------|
| **IoU / Jaccard** | — | ≥ 0.75 | Standard segmentation benchmark |
| **Dice / F1** | #11 | ≥ 0.80 | = F1 at pixel level; primary metric |
| **Recall** | #7 | ≥ 0.85 | Missing a wound is worse than false alarm |
| **Precision** | #8 | ≥ 0.78 | How many detected pixels are actually wound |
| **ROC AUC** | #16 | ≥ 0.90 | Overall classifier quality |
| **PR AUC** | #18 | ≥ 0.85 | Better than ROC for class imbalance |
| **MCC** | #14 | ≥ 0.70 | Most balanced single metric; strong for defence |
| **Specificity** | #4 | — | Background detection rate |
| **FPR** | #2 | — | Type I error |
| **FNR** | #3 | — | Type II error — keep this LOW |

---

## Adjusting settings

All settings are in `src/config.py`. Change there, never in other files.

```python
CFG.EPOCHS      = 50        # increase for better results
CFG.BATCH_SIZE  = 8         # reduce if GPU runs out of memory
CFG.IMG_SIZE    = 256       # increase to 512 for higher detail (more VRAM)
CFG.THRESHOLD   = 0.5       # lower to 0.4 to increase Recall (catch more wounds)
```

---

## Using Cursor effectively

Cursor reads `.cursorrules` automatically. When asking Cursor for help:

- **"Add learning rate warmup to train.py"** — it knows the project structure
- **"Why is my IoU low?"** — it knows the target metrics
- **"Add test for the dataset"** — it knows tests go in `tests/test_all.py`
- **"Explain what Dice loss does"** — it knows the architecture context

The `.cursorrules` file is the memory of the project.
