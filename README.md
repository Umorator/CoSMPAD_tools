# CoSMPAD Tools

A comprehensive pipeline for feature extraction and prediction of microbial secretory preproteins. This repository supports reproducible inference of secretion pathway classes from protein sequences as part of the work:

**CoSMPAD: Comparative Secretory Microbial Preprotein Activity Database**

## Overview

This pipeline processes SPs sequences for:

- **Feature Extraction**: Transforms sequences into numerical representations 
- **Probability Estimation**: Provides probabilities for all secretion classes

## Quick Start

### Set up environment:
```bash
pip install .
```

### Run predictions from Python:
```bash
from cosmpad_predictor.api import CosmpadPredictor

predictor = CosmpadPredictor()

sequences = [
    "MKKKKTIIALSYIFCLVFADYKDDDDK",
    "MPLNVSFTL..."
]

results = predictor.predict_from_sequence(sequences)

for r in results:
    print(r)
```

### Pipeline Stages

### 1. Feature Extraction
Each sequence is transformed into a fixed-length numerical representation using:

ESM-2 protein language model embeddings

Peptides physicochemical descriptors

BioPython protein properties

ProPy sequence descriptors

Feature ordering strictly matches the configuration used during model training.

### 2. Model Inference
Predictions are generated using:

An ensemble of XGBoost classifiers trained on 3 cross-validation folds

Mean probability aggregation across models is display as well as the confidence score 

#### Supported Prediction Classes: 
Sec/SPI: Standard secretory pathway

Sec/SPII: Lipoproteins

Sec/SPIII: Pilin-like proteins

Tat/SPI: Twin-arginine translocation (standard)

Tat/SPII: Twin-arginine translocation (alternative)

TM/Globular: No signal peptide (transmembrane or globular)

Example Output
```bash
{
  'sequence': 'MKKKKTIIALSYIFCLVFADYKDDDDK',
  'pred_label_name': 'Sec/SPI',
  'pred_proba': {
      'TM/Globular': 0.03,
      'Sec/SPI': 0.96,
      'Sec/SPII': 0.03,
      'Sec/SPIIII': 0.00,
      'Tat/SPI': 0.00,
      'Tat/SPII': 0.00
}
```

### Directory Structure
```
CoSMPAD_tools/
├── cosmpad_predictor/
│   ├── api.py
│   ├── esm_model.py
│   ├── features.py
│   ├── models/
│   │   ├── model_fold_1.pkl
│   │   ├── model_fold_2.pkl
│   │   ├── model_fold_3.pkl
│   │   ├── trained_feature_order.pkl
│   │   └── label_encoders.pkl
│   └── __init__.py
├── pyproject.toml
├── README.md
├── LICENSE
└── MANIFEST.in
```

### Citation
If you use CoSMPAD Tools in your research, please cite:

Rafael Moran-Torres
PhD work in progress
CoSMPAD: Comparative Secretory Microbial Preprotein Activity Database

### License
This project is released under the MIT License.
See the LICENSE file for details.
