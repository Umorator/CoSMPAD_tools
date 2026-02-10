import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="propy.AAIndex")

import joblib
import os
import numpy as np
import pandas as pd
from .esm_model import extract_esm_features
from .features import extract_classical_features, clean_sequence
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# Ensemble paths
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATHS = [
    os.path.join(MODELS_DIR, "model_fold_0.pkl"),
    os.path.join(MODELS_DIR, "model_fold_1.pkl"),
    os.path.join(MODELS_DIR, "model_fold_2.pkl")
]

FEATURE_ORDER_PATH = os.path.join(BASE_DIR, "trained_feature_order.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")

# Load feature order
FEATURE_ORDER = joblib.load(FEATURE_ORDER_PATH)

# ----------------------------
# Feature extraction
# ----------------------------
def extract_from_sequence(sequence: str, feature_order=None) -> dict:
    if feature_order is None:
        feature_order = FEATURE_ORDER

    seq = clean_sequence(sequence)
    if not seq:
        return {}

    features = {}
    features.update(extract_esm_features(seq, "full"))
    features.update(extract_classical_features(seq, "full"))

    missing_features = [f for f in feature_order if f not in features]
    if missing_features:
        raise ValueError(f"Missing required features for COSMPAD prediction.")

    ordered_features = {f: features[f] for f in feature_order}
    return ordered_features

# ----------------------------
# Predictor
# ----------------------------
class CosmpadPredictor:
    def __init__(self):
        self.models = [joblib.load(p) for p in MODEL_PATHS]
        self.feature_order = FEATURE_ORDER

        if os.path.exists(ENCODERS_PATH):
            encoders = joblib.load(ENCODERS_PATH)
            self.label_encoder = encoders.get('label_encoder', None)
            self.kingdom_encoder = encoders.get('kingdom_encoder', None)
            self.feature_order = encoders.get('feature_columns', self.feature_order)
        else:
            self.label_encoder = None
            self.kingdom_encoder = None

        # Map short SP labels → scientific names
        self.sp_name_mapping = {
            'LIPO': 'Sec/SPII',
            'NO_SP': 'TM/Globular',
            'PILIN': 'Sec/SPIII',
            'SP': 'Sec/SPI',
            'TAT': 'Tat/SPI',
            'TATLIPO': 'Tat/SPII'
        }

    def predict_from_sequence(self, sequences: list[str]) -> pd.DataFrame:
        results = []

        for seq in sequences:
            # Extract features
            features = extract_from_sequence(seq, feature_order=self.feature_order)
            X = pd.DataFrame([features], columns=self.feature_order)

            # Ensemble probability
            probas = np.array([model.predict_proba(X)[0] for model in self.models])
            avg_proba = probas.mean(axis=0)

            # Get SP labels from LabelEncoder
            sp_labels = self.label_encoder.classes_ if self.label_encoder else list(range(len(avg_proba)))

            # Map probabilities to scientific names with rounding to 3 decimal places
            proba_dict = {
                self.sp_name_mapping.get(lbl, lbl): float(f"{prob:.3f}")
                for lbl, prob in zip(sp_labels, avg_proba)
            }

            # Predicted SP type (highest probability)
            pred_label_name = max(proba_dict, key=proba_dict.get)

            results.append({
                "sequence": seq,
                "pred_label_name": pred_label_name,
                "pred_proba": proba_dict
            })

        return pd.DataFrame(results)