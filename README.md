# LEVit-Skin: Reproducing and Improving a Balanced and Interpretable Transformer-CNN Model for Multi-Class Skin Cancer Diagnosis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## 📌 Overview

This repository contains the reproduction and improvement of **LEVit-Skin**, a hybrid Transformer-CNN architecture designed for balanced, interpretable, and accurate **multi-class skin cancer diagnosis** using dermoscopic images.

The original paper proposes a lightweight vision transformer model that combines the local feature extraction power of CNNs with the global context awareness of Vision Transformers (ViT), specifically tailored for skin lesion classification.

---

## 📄 Original Paper

> **LEVit-Skin: A balanced and interpretable transformer-CNN model for multi-class skin cancer diagnosis**

---

## 🗂️ Dataset

This project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset.

### Download the dataset:
1. Go to [Kaggle - HAM10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
2. Download and extract into the project folder

### Dataset Classes:
| Label | Disease |
|-------|---------|
| `mel` | Melanoma |
| `nv` | Melanocytic Nevi |
| `bcc` | Basal Cell Carcinoma |
| `akiec` | Actinic Keratoses |
| `bkl` | Benign Keratosis |
| `df` | Dermatofibroma |
| `vasc` | Vascular Lesions |

---

## 🏗️ Project Structure

```
project_ham10000/
│
├── HAM_project.ipynb       # Main notebook (training, evaluation, visualization)
├── results                 # Evaluation results and metrics
├── .gitignore              # Ignoring large files (datasets, model weights)
└── README.md               # Project documentation
```

> ⚠️ **Note:** Dataset files and model weights are not included in this repository due to size constraints. Please download them separately (see Dataset section above).

---

## 🧠 Model Architecture

**LEVit-Skin** is a hybrid model combining:
- **CNN layers** — for local feature extraction from dermoscopic images
- **Vision Transformer (ViT/LeViT)** — for global context and attention
- **Interpretability module** — for generating attention/saliency maps

Key features:
- Handles **class imbalance** in skin lesion datasets
- Produces **interpretable predictions** via attention visualization
- Lightweight and efficient for medical imaging tasks

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

### Main Dependencies:
- Python 3.8+
- PyTorch 2.0+
- torchvision
- timm
- numpy
- pandas
- matplotlib
- scikit-learn
- Pillow
- jupyter

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/winterDark22/LEVit---Skin-Cancer-Classification.git
cd LEVit---Skin-Cancer-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download HAM10000 from Kaggle and place it in the project folder.

### 4. Run the notebook
```bash
jupyter notebook HAM_project.ipynb
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| F1-Score | TBD |
| AUC-ROC | TBD |
| Balanced Accuracy | TBD |

> Results will be updated after full training and evaluation.

---

## 🔍 Improvements Over Original Paper

- [ ] Data augmentation strategies for class imbalance
- [ ] Hyperparameter tuning
- [ ] Additional interpretability methods (GradCAM, SHAP)
- [ ] Comparison with other baseline models

---

## 📁 What's Not Included (Due to Size)

| File/Folder | Reason |
|-------------|--------|
| `archive/` (dataset) | Too large (GBs) — download from Kaggle |
| `best_model.pth` | Large model weights |
| `*.csv` dataset files | Large data files |

---

## 👤 Author

**winterDark22**
- GitHub: [@winterDark22](https://github.com/winterDark22)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Original LEVit-Skin paper authors
- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection) — ISIC Archive
- [timm library](https://github.com/huggingface/pytorch-image-models) for Vision Transformer implementations
