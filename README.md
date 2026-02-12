# 🧬 CoSMPAD Tools

**Comparative Secretory Microbial Preprotein Activity Database Tools**

A complete, production-ready pipeline for feature extraction and prediction of microbial secretory preprotein signal peptides using ESM-2 embeddings and ensemble XGBoost classifiers.

[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://hub.docker.com/r/yourusername/cosmpad_tools)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](https://doi.org/)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
  - [Option 1: Docker (Recommended)](#option-1-docker-recommended)
  - [Option 2: Local Installation](#option-2-local-installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
  - [1. Feature Extraction](#1-feature-extraction)
  - [2. Model Inference](#2-model-inference)
- [Output Format](#output-format)
- [Directory Structure](#directory-structure)
- [Performance](#performance)
- [Citation](#citation)
- [License](#license)

---

## 🔬 Overview

CoSMPAD Tools is a comprehensive bioinformatics pipeline designed for the classification of microbial secretory signal peptides. It transforms protein sequences into interpretable predictions across **six distinct secretion pathway classes** using state-of-the-art deep learning embeddings and ensemble tree-based methods.

**Key Features:**
- 🚀 **Zero-configuration inference** 
- 🧠 **ESM-2 protein language model** 
- 📊 **Physicochemical descriptors** 
- 🎯 **6-class secretion pathway prediction** 

**Supported Prediction Classes:**

| Class | Description |
|-------|------------|
| **Sec/SPI** | Standard secretory pathway|
| **Sec/SPII** | Lipoproteins|
| **Sec/SPIII** | Pilin-like proteins|
| **Tat/SPI** | Twin-arginine translocation (standard)|
| **Tat/SPII** | Twin-arginine translocation (alternative)|
| **TM/Globular** | No signal peptide|

---

## 💻 Installation

### Option 1: Docker (Recommended) 🐳

**📥 Pull the image**
```bash
docker pull umorator/cosmpad_tools:latest
```
🚀 Run interactive Python session

```bash
docker run -it --rm umorator/cosmpad_tools:latest python
```

📁 Work With Your Local Files (Running Your Own Script)

If you want to run your own Python script (e.g., `run_test.py`) using the Docker image, you need to mount your local working folder into the container.

```bash
docker run -it --rm -v \Users\your_username\your_working_folder:/workspace -w /workspace umorator/cosmpad_tools:latest python run_test.py

```
### Option 2: Local Installation (pip)

**📋 Step-by-step setup:**

1️⃣ **Clone the repository**
```bash
git clone https://github.com/Umorator/CoSMPAD_tools.git
cd CoSMPAD_tools
```
Make sure you are in the repo root where pyproject.toml is located.

2️⃣ Create a fresh conda environment with Python 3.11
```
bash
conda create -n CoSMPAD_tools python=3.11 -y
conda activate CoSMPAD_tools
```

3️⃣ Install CPU-only PyTorch
```
bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

4️⃣ Install CoSMPAD in editable mode
```
bash
pip install -e .
```

This installs the package with all dependencies from pyproject.toml.

5️⃣ Verify installation
```
bash
python -c "from cosmpad_predictor import CosmpadPredictor; print('✅ CoSMPAD imported successfully')"
```

## ⚡ Quick Start

### Python API (run_test.py)

```python
from cosmpad_predictor import CosmpadPredictor

# Initialize predictor (loads model)
predictor = CosmpadPredictor()


# ==============================
# 1️⃣ Single Sequence Prediction
# ==============================

single_sequence = "MKPKKIISNKAQISLELALLLGALVVAASIVG"

single_result = predictor.predict_from_sequence([single_sequence])

print("========== Single Prediction ==========")

for _, row in single_result.iterrows():
    print("\nSequence:", row["sequence"])
    print("Prediction:", row["pred_label_name"])

    print("Probabilities:")
    for label, prob in row["pred_proba"].items():
        print(f"  {label}: {prob:.3f}")

    print("Model Confidence:", f"{row['ensemble_confidence']:.3f}")

print()


# ==============================
# 2️⃣ Batch Prediction
# ==============================

sequences = [
    "MKPKKIISNKAQISLELALLLGALVVAASIVG",
    "MPLNVSFTLFIASVLMLVVAKPLGVAQ",
    "MNKIKYLLLSLVGFLVFADPAFAKRE"
]

batch_results = predictor.predict_from_sequence(sequences)

print("========== Batch Predictions ==========")

for i, (_, row) in enumerate(batch_results.iterrows(), 1):
    print(f"\n--- Sequence {i} ---")
    print("Sequence:", row["sequence"])
    print("Prediction:", row["pred_label_name"])

    print("Probabilities:")
    for label, prob in row["pred_proba"].items():
        print(f"  {label}: {prob:.3f}")

    print("Model Confidence:", f"{row['ensemble_confidence']:.3f}")

print()


# ==============================
# 3️⃣ Single Sequence Feature Extraction
# ==============================

features_single = predictor.extract_from_sequence(single_sequence)

print("========== Single Sequence Feature Extraction ==========")
print("Number of features:", len(features_single))
print("First 10 features:")

for k, v in list(features_single.items())[:10]:
    print(f"  {k}: {v}")

print()


# ==============================
# 4️⃣ Batch Feature Extraction
# ==============================

features_batch = [predictor.extract_from_sequence(seq) for seq in sequences]

print("========== Batch Feature Extraction ==========")
print(f"Extracted features for {len(features_batch)} sequences")

for i, features in enumerate(features_batch, 1):
    print(f"\n--- Sequence {i} Feature Preview ---")
    for k, v in list(features.items())[:5]:
        print(f"  {k}: {v}")

```

## 🔧 Pipeline Stages

### 1. Feature Extraction

Each protein sequence is transformed into a **fixed-length numerical representation** through a multi-view feature extraction pipeline:

| Feature Type | Description | Dimension |
|-------------|-------------|-----------|
| **🧬 ESM-2 Embeddings** | Per-token representations from ESM-2 (650M), mean-pooled | 2560 |
| **📊 ProPy** | CTD descriptors, autocorrelation, composition, transition, distribution | 1547 |
| **🧪 Peptide Properties** | Physicochemical properties, hydrophobicity, charge, isoelectric point, etc. | 76 |
| **🔬 BioPython** | Molecular weight, aromaticity, instability index, flexibility, etc. | 9 |

**Total feature dimension:** **4,192** (strictly ordered for reproducibility)

> **⚠️ Important:** Feature ordering exactly matches the configuration used during model training. The `feature_order.pkl` file ensures reproducibility across inference runs.

Predictions are generated using a **robust ensemble approach**:

- **Base classifier:** XGBoost (default hyperparameters)
- **Ensemble strategy:** 3-fold cross-validation models (folds 0, 1, 2)
- **Aggregation:** Soft voting (mean probability across ensemble members)
- **Confidence score:** Combined metric of mean max probability and vote agreement

**Confidence Calculation:**
```
confidence = (mean_max_proba + vote_agreement) / 2
```
where:
- **mean_max_proba**: Average of the highest probability across ensemble models
- **vote_agreement**: Proportion of models agreeing on the final class

**Ensemble Architecture:**
```

                     ┌─ Model Fold 0 ─┐
                     ├─ Model Fold 1 ─┤
Sequence ──► Features ── Model Fold 2 ──► Probabilities ──► Mean ──► Final Prediction
                     └───────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              Mean Max      Vote Agreement
              Probability        │
                    │             │
                    └──────┬──────┘
                           ▼
                  Ensemble Confidence

```                
---

## 📊 Output Format

### Single Sequence Output
```python
{
    'sequence': 'MKKKKTIIALSYIFCLVFADYKDDDDK',
    'pred_label_name': 'Sec/SPI',            # Human-readable class
    'pred_proba': {
        'TM/Globular': 0.01,
        'Sec/SPI': 0.96,                    # ✅ Predicted class
        'Sec/SPII': 0.02,
        'Sec/SPIII': 0.00,
        'Tat/SPI': 0.01,
        'Tat/SPII': 0.00
    },
    'ensemble_confidence': 0.94              # Combined confidence score (0-1)
}
```

## 📁 Directory Structure

```
CoSMPAD_tools/
├── 📦 cosmpad_predictor/           # Main package
│   ├── __init__.py
│   ├── api.py                     # Main predictor class
│   ├── model.py                   # ESM-2 caching & management
│   ├── features.py                # Multi-view feature extraction
│   ├── utils.py                   # FASTA parsing, helpers
│   │
│   ├── 🧠 models/                  # Trained ensemble (included in package)
│   │   ├── model_fold_1.pkl      # XGBoost fold 1
│   │   ├── model_fold_2.pkl      # XGBoost fold 2
│   │   ├── model_fold_3.pkl      # XGBoost fold 3
│   │   ├── feature_order.pkl     # Critical: ensures feature ordering
│   │   ├── label_encoders.pkl    # Class encoding
│   │
│   └── tests/                    # Unit tests
│       ├── test_api.py
│       └── test_features.py
│
├── 🐳 docker/                     # Docker configuration
│   └── Dockerfile                # Multi-stage build with model pre-caching
│
├── pyproject.toml               # Modern Python packaging
├── MANIFEST.in                 # Include model files in package
├── LICENSE                     # MIT License
└── README.md                   # You are here
```

---

## 📈 Performance

**Cross-validation performance (3-fold OOF):**

| Class | Precision | Recall | F1-Score | MCC1 | MCC2 |
|-------|-----------|--------|----------|------|------|
| **Sec/SPI** (SP) | 0.9334 | 0.9438 | 0.9386 | 0.9441 | 0.9296 |
| **Sec/SPII** (LIPO) | 0.9790 | 0.9226 | 0.9500 | 0.9556 | 0.9463 |
| **Sec/SPIII** (PILIN) | 0.9701 | 0.9286 | 0.9489 | 0.9561 | 0.9490 |
| **Tat/SPI** (TAT) | 0.9609 | 0.9425 | 0.9516 | 0.9687 | 0.9508 |
| **Tat/SPII** (TATLIPO) | 0.7586 | 0.6667 | 0.7097 | 0.8162 | 0.7107 |
| **TM/Globular** (NO_SP) | 0.9883 | 0.9932 | 0.9907 | — | 0.9595 |


**📌 Note on MCC metrics:**  
* **MCC1**: Measures discrimination of each SP type against **non-SP sequences** (where applicable).  
* **MCC2**: Measures discrimination of each SP type against **all remaining classes**.  
 
MCC1 is undefined (—) for TM/Globular because it cannot discriminate against itself when comparing to non-SP sequences.

---

## 📚 Citation

If you use CoSMPAD Tools in your research, please cite:

```bibtex
@phdthesis{cosmpad2026,
  title = {CoSMPAD: Comparative Secretory Microbial Preprotein Activity Database},
  author = {Moran-Torres, Rafael},
  year = {2026},
  school = {Humboldt-Universität zu Berlin},
}
```

---

## 📄 License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file for full details.

---

## 🙏 Acknowledgments

- **ESM-2** model from [Meta AI Research](https://github.com/facebookresearch/esm)
- **ProPy** for sequence descriptors
- **BioPython** community
- **People Program (Marie Skłodowska-Curie Actions) of the European Union’s Horizon 2020 Program under REA grant agreement no. 813979 (SECRETERS)**

---

<div align="center">
  
**Made with 🧬 for the computational biology community**

[Report Bug](https://github.com/yourusername/cosmpad-tools/issues) · 
[Request Feature](https://github.com/yourusername/cosmpad-tools/issues) · 
[Star Repository](https://github.com/yourusername/cosmpad-tools)

</div>