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

```bash
# Pull and run in one command
docker run -it --rm \
  umorator/cosmpad_tools:latest \
  python
```

**For working with your local files:**
```bash
docker run -it --rm \
  -v "$(pwd):/workspace" \
  -w /workspace \
  umorator/cosmpad_tools:latest \
  python
```

### Option 2: Local Installation (pip)

```bash
# Create a fresh environment (recommended)
conda create -n cosmpad python=3.11
conda activate cosmpad

# Install the package
pip install git+https://github.com/yourusername/cosmpad-tools.git

# Or for development
git clone https://github.com/yourusername/cosmpad-tools.git
cd cosmpad-tools
pip install -e .
```

**Note:** Local installation will download the 2.5GB ESM-2 model on first use. The model is cached in `~/.cache/cosmpad/` and will not require re-downloading.

---

## ⚡ Quick Start

### Python API (Docker or Local)

```python
from cosmpad_predictor import CosmpadPredictor

# Initialize predictor
# - With Docker: instant (model pre-loaded)
# - With pip: downloads model on first use (~2.5GB, cached forever)
predictor = CosmpadPredictor()

# Single sequence prediction
sequence = "MKKKKTIIALSYIFCLVFADYKDDDDK"
result = predictor.predict_from_sequence(sequence)
print(f"Prediction: {result['pred_label_name']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
sequences = [
    "MKKKKTIIALSYIFCLVFADYKDDDDK",
    "MPLNVSFTLFIASVLMLVVAKPLGVAQ",
    "MNKIKYLLLSLVGFLVFADPAFAKRE"
]
results = predictor.predict_from_sequence(sequences)

for seq, res in zip(sequences, results):
    print(f"\nSequence: {seq[:20]}...")
    print(f"Prediction: {res['pred_label_name']}")
    print(f"Confidence: {res['confidence']:.3f}")
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