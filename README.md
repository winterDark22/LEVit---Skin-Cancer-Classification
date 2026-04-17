# LEVit-Skin: Skin Cancer Classification on HAM10000

Reproduction of the LEVit-Skin paper — a hybrid Transformer-CNN model for multi-class skin cancer diagnosis using the HAM10000 dataset.

**Authors:** Sadia Tabassum (1905091) · Md. Azizul Haque Nadim (1905059)  
**Course:** CSE Machine Learning Project · April 2026

---

## Overview

This project reproduces the pipeline from:
> Sakib et al., "LEVit-Skin: A balanced and interpretable transformer-CNN model for multi-class skin cancer diagnosis," IJSRA, vol. 15, no. 01, 2025.

The model classifies dermoscopic images into 7 skin lesion categories using LEViT-256 (pretrained on ImageNet), augmentation-based oversampling to handle class imbalance, and Grad-CAM for interpretability.

---

## Dataset

**HAM10000** — 10,015 dermoscopic images across 7 classes:

| Code | Full Name | Count | Type |
|------|-----------|-------|------|
| nv | Melanocytic nevi | 6,705 | Benign |
| mel | Melanoma | 1,113 | Malignant |
| bkl | Benign keratosis | 1,099 | Benign |
| bcc | Basal cell carcinoma | 514 | Malignant |
| akiec | Actinic keratoses | 327 | Pre-malignant |
| vasc | Vascular lesions | 142 | Benign |
| df | Dermatofibroma | 115 | Benign |

The dataset has a severe class imbalance (58:1 ratio). Splits are done at the lesion-ID level to prevent data leakage.

---

## Methodology

- **Preprocessing:** Resized to 224x224, normalized with dataset-specific mean/std
- **Augmentation:** Random rotation, flips, crop, brightness/contrast, shear, sigmoid intensity correction
- **Oversampling:** Minority classes upsampled to 6,705 samples each (training set: 46,935 images)
- **Model:** Pretrained LEViT-256 from `timm`, classification head replaced for 7 classes
- **Training:**
  - Phase 1 (5 epochs): Frozen backbone, head only — LR 1e-3
  - Phase 2 (15 epochs): Full fine-tuning — LR 1e-4
- **Validation:** 2-fold cross-validation split on lesion IDs

---

## Results

| | Accuracy | F1 | Specificity | MCC | PR-AUC |
|--|--|--|--|--|--|
| Fold 1 | 0.7151 | 0.6946 | 0.9525 | 0.6836 | 0.8414 |
| Fold 2 | 0.7339 | 0.7148 | 0.9557 | 0.7007 | 0.8373 |
| **Mean** | **0.7245** | **0.7047** | **0.9541** | **0.6922** | **0.8394** |
| Paper (10-fold) | — | 0.9611 | 0.9629 | 0.9551 | 0.9662 |

The gap from the paper is primarily due to using 2-fold instead of 10-fold CV (50% vs 90% training data per fold) and limited training epochs. Specificity nearly matches the paper's reported value.

---

## Repository Contents

| File | Description |
|------|-------------|
| `ham10000.ipynb` | Main notebook — full training and evaluation pipeline |
| `confusion_matrix.png` | Aggregated confusion matrix over both folds |
| `learning_curves.png` | Training/validation loss and accuracy curves |
| `gradcam_visualization.png` | Grad-CAM heatmaps for one sample per class |
| `report.pdf` | Full project report |
| `paper.pdf` | Original LEVit-Skin paper |
| `presentation.pdf` | Project presentation slides |

---

## Setup and Run

**1. Clone the repository**
```bash
git clone https://github.com/winterDark22/LEVit---Skin-Cancer-Classification.git
cd LEVit---Skin-Cancer-Classification
```

**2. Install dependencies**
```bash
pip install torch torchvision timm numpy pandas matplotlib scikit-learn opencv-python
```

**3. Download the HAM10000 dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place it in your project directory:
```
project/
  HAM10000_images_part_1/
  HAM10000_images_part_2/
  HAM10000_metadata.csv
```

**4. Run the notebook**
```bash
jupyter notebook ham10000.ipynb
```

Run all cells from top to bottom. The notebook handles preprocessing, oversampling, training (both phases), evaluation, and Grad-CAM visualization automatically.

---

## Requirements

```
torch
timm>=1.0.25
torchvision
numpy
pandas
matplotlib
scikit-learn
opencv-python
```

---

## References

1. Sakib et al., "LEVit-Skin," IJSRA, vol. 15, no. 01, 2025.
2. Tschandl et al., "The HAM10000 dataset," Scientific Data, 2018.
3. Graham, "LeViT: a Vision Transformer in ConvNet's Clothing," ICCV, 2021.
4. Selvaraju et al., "Grad-CAM," ICCV, 2017.
5. Wightman, "PyTorch Image Models (timm)," 2019.
