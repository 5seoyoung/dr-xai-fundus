# Fundus-Based Diabetic Retinopathy Classification with Explainability

**An Explainable Framework Using Fundus Image Preprocessing and Grad-CAM++**

> Warning: This code is for research and educational purposes only. Any clinical deployment requires IRB approval and prospective field validation.

## Overview

Fundus image classifiers must achieve accuracy and efficiency while remaining clinically interpretable. This repository provides a reproducible pipeline on APTOS 2019 that combines (i) robust preprocessing (circular masking, CLAHE, green-channel emphasis), (ii) lightweight EfficientNet models, and (iii) Grad-CAM++ visual explanations.

## Research Questions

* Does strengthened preprocessing improve performance and stability on DR screening?
* Can lightweight models (EfficientNet-B0/B3) reach near-clinical screening performance?
* Do Grad-CAM++ visualizations align with pathologic regions used in clinical reading?

## Key Results

* **Validation:** AUROC 0.9800, AUPRC 0.9689
* **Test:** AUROC 0.9855, AUPRC 0.9780
* **Threshold at specificity ≈ 0.95:** `thr = 0.467 → Spec 0.940, Sens 0.933`
* Grad-CAM++ highlights optic disc, vessel branching, and hemorrhage/exudate regions consistent with clinical findings.

---

## Methods

### Data

* **APTOS 2019 Blindness Detection** (3,662 images).
https://www.kaggle.com/competitions/aptos2019-blindness-detection
* Reformulated as **binary**: non-referable (0–1) vs **referable DR (≥2)**.

### Preprocessing

1. Circular masking (retina extraction)
2. CLAHE contrast enhancement
3. Green-channel emphasis
4. Resize to 448×448

### Modeling

* Backbones: **EfficientNet-B0/B3** (ImageNet pretrained)
* Optimizer: **AdamW** (`lr=3e-4`, `weight_decay=1e-4`)
* Scheduler: **Cosine LR**

### Evaluation

* **AUROC, AUPRC**
* **Sensitivity at fixed specificity ≈ 0.95** (validation-optimized threshold)

### Explainability

* **Grad-CAM++** on the last conv block; overlay boards for validation cases.

---

## Figure

> <img width="1920" height="1080" alt="figure1_board" src="https://github.com/user-attachments/assets/c41641b2-cb93-466a-b972-7ba45a16a7c0" />



<p align="center">
</p>
<p align="center"><b>Figure 1.</b> Validation: Original fundus (left of each pair) and Grad-CAM++ overlay (right). 
  
  The model consistently attends to clinically relevant structures—optic disc, vessel bifurcations, and lesion regions (hemorrhages/exudates)—supporting interpretability of predictions.</p>

---

## Repository Structure

```
dr-xai/
├── configs/                 # experiment configs
├── src/
│   ├── dataset.py           # loader / splits / labels
│   ├── preprocess.py        # circular mask, CLAHE, green channel
│   ├── model.py             # EfficientNet wrappers
│   ├── train.py             # training loop (AMP optional)
│   ├── evaluate.py          # ROC/PR, threshold tuning, metrics
│   └── cam.py               # Grad-CAM++ generation and boards
├── scripts/                 # CLI utilities (boards, exports, end-to-end)
├── notebooks/               # Kaggle-friendly reproduction
├── artifacts/
│   ├── figs/                # test_roc.png, test_pr.png, metrics_summary.csv
│   └── cams_resume/         # val_*.png (original vs CAM overlays)
└── README.md
```

## Environment

* Python 3.10+
* PyTorch ≥ 2.1, `timm`, `albumentations`, `opencv-python`, `scikit-learn`, `matplotlib`, `pandas`
* Grad-CAM++: `pytorch-grad-cam`

```bash
pip install torch torchvision torchaudio
pip install timm albumentations opencv-python scikit-learn matplotlib pandas tqdm pytorch-grad-cam
```

## Reproduction

### Train

```bash
python -m src.train \
  --config configs/aptos.yaml \
  --model effnet_b0 \
  --img-size 448 \
  --epochs 3 \
  --batch-size 32
```

### Evaluate

```bash
python -m src.evaluate \
  --ckpt runs/effb0_best.pt \
  --img-size 448 \
  --fix-spec 0.95 \
  --out-dir artifacts/figs
```

### Grad-CAM++ Boards

```bash
python -m src.cam \
  --ckpt runs/effb0_best.pt \
  --split val \
  --n 12 \
  --out-dir artifacts/cams_resume
```

## Figures

* `artifacts/figs/test_roc.png`
* `artifacts/figs/test_pr.png`
* `artifacts/cams_resume/val_*.png` (original vs CAM overlays)

**[Figure 1]** Validation: Original Fundus vs CAM Overlay — Grad-CAM++ shows attention over optic disc, vessel bifurcations, and lesion areas, supporting clinical interpretability.

---

## Limitations and Future Work

* External validation on **EyePACS** and **Messidor-2** is required to assess generalization.
* Extension to **multi-class DR severity**.
* Latency & memory profiling for **mobile/edge** deployment.

## References

* Gulshan V, et al. *JAMA*, 2016. doi:10.1001/jama.2016.17216
* Chattopadhay A, et al. *WACV*, 2018. [https://doi.org/10.48550/arXiv.1710.11063](https://doi.org/10.48550/arXiv.1710.11063)
* Tan M, Le Q. *EfficientNet*, ICML 2019
* **APTOS 2019** Blindness Detection Dataset (Kaggle)

## Keywords

Diabetic Retinopathy, Fundus Imaging, Deep Learning, Explainable AI, Preprocessing, Grad-CAM++

## License & Responsible AI

* Code: MIT (update as appropriate).
* Dataset: follow APTOS 2019 license and terms; data is **not** bundled.
* This project is a decision-support research tool, **not** an autonomous diagnostic system. Deployment requires independent validation, human oversight, conservative thresholds, ongoing monitoring, and rollback procedures.

---
