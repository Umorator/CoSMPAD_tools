import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="propy.AAIndex")

import joblib
import os
import numpy as np
import pandas as pd
from .esm_model import extract_esm_features
from .features import extract_classical_features, clean_sequence


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


class CosmpadPredictor:
    def __init__(self):
        self.models = [joblib.load(p) for p in MODEL_PATHS]
        self.feature_order = joblib.load(FEATURE_ORDER_PATH)

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

    # ----------------------------
    # Feature extraction (moved inside class)
    # ----------------------------
    def extract_from_sequence(self, sequence: str) -> dict:
        seq = clean_sequence(sequence)
        if not seq:
            return {}

        features = {}
        features.update(extract_esm_features(seq, "full"))
        features.update(extract_classical_features(seq, "full"))

        missing_features = [f for f in self.feature_order if f not in features]
        if missing_features:
            raise ValueError("Missing required features for COSMPAD prediction.")

        return {f: features[f] for f in self.feature_order}

    # ----------------------------
    # Prediction
    # ----------------------------
    def predict_from_sequence(self, sequences: list[str]) -> pd.DataFrame:
        results = []

        for seq in sequences:
            # ---- Feature extraction ----
            features = self.extract_from_sequence(seq)
            X = pd.DataFrame([features], columns=self.feature_order)

            # ---- Collect model probabilities ----
            model_probas = np.array([
                model.predict_proba(X)[0]
                for model in self.models
            ])

            # ---- Soft voting (average probabilities) ----
            avg_proba = model_probas.mean(axis=0)

            # ---- Class labels ----
            sp_labels = (
                self.label_encoder.classes_
                if self.label_encoder
                else list(range(len(avg_proba)))
            )

            # ---- Final prediction ----
            pred_index = np.argmax(avg_proba)
            pred_label_short = sp_labels[pred_index]
            pred_label_name = self.sp_name_mapping.get(
                pred_label_short,
                pred_label_short
            )

            # ---- Ensemble confidence ----
            mean_max_proba = np.mean(np.max(model_probas, axis=1))
            vote_agreement = np.mean(
                np.argmax(model_probas, axis=1) == pred_index
            )

            ensemble_confidence = (mean_max_proba + vote_agreement) / 2

            # ---- Probability dictionary (rounded for display only) ----
            proba_dict = {
                self.sp_name_mapping.get(lbl, lbl): round(float(prob), 3)
                for lbl, prob in zip(sp_labels, avg_proba)
            }

            results.append({
                "sequence": seq,
                "pred_label_name": pred_label_name,
                "pred_proba": proba_dict,
                "ensemble_confidence": round(float(ensemble_confidence), 3)
            })

        return pd.DataFrame(results)

