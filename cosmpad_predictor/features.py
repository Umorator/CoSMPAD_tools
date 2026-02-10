# -*- coding: utf-8 -*-
"""
Full COSMPAD feature extraction for a single protein sequence
Includes: ESM-2, Peptides, BioPython, and ProPy descriptors
"""

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import peptides
import propy
from propy.PyPro import GetProDes
import warnings
import torch
import esm

# -----------------------
# Global ESM-2 model variables
# -----------------------
_G_ESM_MODEL = None
_G_ESM_ALPHABET = None
_G_ESM_BATCH_CONVERTER = None
_G_ESM_DEVICE = None

def init_esm_model():
    """Initialize global ESM-2 model for feature extraction"""
    global _G_ESM_MODEL, _G_ESM_ALPHABET, _G_ESM_BATCH_CONVERTER, _G_ESM_DEVICE
    if _G_ESM_MODEL is None:
        try:
            _G_ESM_MODEL, _G_ESM_ALPHABET = esm.pretrained.esm2_t33_650M_UR50D()
            _G_ESM_BATCH_CONVERTER = _G_ESM_ALPHABET.get_batch_converter()
            _G_ESM_MODEL.eval()
            _G_ESM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _G_ESM_MODEL = _G_ESM_MODEL.to(_G_ESM_DEVICE)
        except Exception as e:
            warnings.warn(f"ESM-2 model initialization failed: {e}")

# -----------------------
# Clean sequence
# -----------------------
def clean_sequence(seq: str) -> str:
    if seq is None:
        return ""
    seq = str(seq).strip().replace(" ", "").replace("*", "").upper()
    return seq if seq.isalpha() else ""

# -----------------------
# ESM-2 feature extraction
# -----------------------
def extract_esm_features(sequence: str, seq_name: str) -> dict:
    features = {}
    init_esm_model()
    global _G_ESM_MODEL, _G_ESM_BATCH_CONVERTER, _G_ESM_DEVICE

    if _G_ESM_MODEL is None or len(sequence) < 2:
        return features

    try:
        layer = 24  # same as training
        data = [("protein", sequence)]
        _, _, batch_tokens = _G_ESM_BATCH_CONVERTER(data)
        batch_tokens = batch_tokens.to(_G_ESM_DEVICE)

        with torch.no_grad():
            results = _G_ESM_MODEL(batch_tokens, repr_layers=[layer], return_contacts=False)

        token_representations = results["representations"][layer]
        seq_tokens = token_representations[0, 1 : len(sequence) + 1]

        mean_pool = seq_tokens.mean(dim=0).cpu().numpy()
        max_pool = seq_tokens.max(dim=0).values.cpu().numpy()

        for i, val in enumerate(mean_pool):
            features[f"ESM2_L{layer}_mean_{i}_{seq_name}"] = float(val)
        for i, val in enumerate(max_pool):
            features[f"ESM2_L{layer}_max_{i}_{seq_name}"] = float(val)

    except Exception as e:
        warnings.warn(f"ESM-2 feature extraction failed for {seq_name}: {e}")

    return features

# -----------------------
# Peptides and BioPython features
# -----------------------
def extract_peptides_features(sequence: str, seq_name: str) -> dict:
    features = {}
    if len(sequence) < 2:
        return features
    try:
        pep = peptides.Peptide(sequence)
        descriptor_methods = [
            pep.kidera_factors, pep.z_scales, pep.t_scales, pep.vhse_scales,
            pep.ms_whim_scores, pep.blosum_indices, pep.cruciani_properties,
            pep.fasgai_vectors, pep.pcp_descriptors, pep.physical_descriptors,
            pep.protfp_descriptors, pep.sneath_vectors
        ]
        for method in descriptor_methods:
            try:
                vals = method()
                method_name = method.__name__.replace("_factors","").replace("_descriptors","")\
                    .replace("_vectors","").replace("_properties","").replace("_scales","")\
                    .replace("_scores","").replace("_indices","")
                for i, val in enumerate(vals):
                    features[f"{method_name}_{i+1}_{seq_name}"] = float(val)
            except Exception:
                continue

        # Additional physicochemical properties
        additional_props = {
            "Molecular_Weight": pep.molecular_weight(),
            "BomanInd": pep.boman(),
            "Hydrophobic_Moment": pep.hydrophobic_moment(),
            "Net_Charge": pep.charge(),
            "Aliphatic_Index": pep.aliphatic_index(),
            "Hydrophobicity": pep.hydrophobicity(),
            "Length": len(sequence)
        }
        for k, v in additional_props.items():
            features[f"{k}_{seq_name}"] = float(v) if v is not None else 0.0

    except Exception as e:
        warnings.warn(f"Peptides descriptors failed for {seq_name}: {e}")

    # BioPython features
    try:
        pa = ProteinAnalysis(sequence)
        biopython_features = {
            "Aromaticity": pa.aromaticity(),
            "Instability_Index": pa.instability_index(),
            "Isoelectric_Point": pa.isoelectric_point(),
            "GRAVY": pa.gravy(),
            "Molecular_Weight_BP": pa.molecular_weight(),
        }
        ss = pa.secondary_structure_fraction()
        if len(ss) == 3:
            biopython_features["Helix_Fraction"] = ss[0]
            biopython_features["Turn_Fraction"] = ss[1]
            biopython_features["Sheet_Fraction"] = ss[2]
        flex = pa.flexibility()
        biopython_features["Flexibility"] = np.mean(flex) if flex else 0.0

        for k, v in biopython_features.items():
            features[f"{k}_{seq_name}"] = float(v)
    except Exception as e:
        warnings.warn(f"BioPython descriptors failed for {seq_name}: {e}")

    return features

# -----------------------
# ProPy descriptors
# -----------------------
def extract_propy_features(sequence: str, seq_name: str) -> dict:
    features = {}
    if len(sequence) < 2:
        return features
    try:
        des = GetProDes(sequence)
        dipeptide_composition = propy.AAComposition.CalculateDipeptideComposition(sequence)
        descriptors = {
            **des.GetMoreauBrotoAuto(),
            **des.GetMoranAuto(),
            **des.GetGearyAuto(),
            **dipeptide_composition,
            **des.GetCTD(),
            **des.GetSOCN(),
            **des.GetQSO(),
            **des.GetPAAC(),
            **des.GetAPAAC(),
            **des.GetAAComp()
        }
        features.update({f"{key}_{seq_name}": float(value) for key, value in descriptors.items()})
    except Exception as e:
        warnings.warn(f"ProPy descriptor failed for {seq_name}: {e}")
    return features

# -----------------------
# Unified feature extraction
# -----------------------
def extract_classical_features(sequence: str, seq_name: str) -> dict:
    """Combine Peptides + BioPython + ProPy features"""
    features = {}
    features.update(extract_peptides_features(sequence, seq_name))
    features.update(extract_propy_features(sequence, seq_name))
    return features
