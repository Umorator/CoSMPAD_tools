import warnings
import torch
import esm

_ESM_MODEL = None
_ESM_ALPHABET = None
_ESM_BATCH_CONVERTER = None
_DEVICE = None


def load_esm():
    global _ESM_MODEL, _ESM_ALPHABET, _ESM_BATCH_CONVERTER, _DEVICE

    if _ESM_MODEL is not None:
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ESM_MODEL, _ESM_ALPHABET = esm.pretrained.esm2_t33_650M_UR50D()

    _ESM_BATCH_CONVERTER = _ESM_ALPHABET.get_batch_converter()
    _ESM_MODEL.eval()

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ESM_MODEL = _ESM_MODEL.to(_DEVICE)


def extract_esm_features(sequence: str, sequence_name: str) -> dict:
    load_esm()

    layer = 24
    features = {}

    data = [("protein", sequence)]
    _, _, batch_tokens = _ESM_BATCH_CONVERTER(data)
    batch_tokens = batch_tokens.to(_DEVICE)

    with torch.no_grad():
        results = _ESM_MODEL(batch_tokens, repr_layers=[layer])

    reps = results["representations"][layer][0, 1:len(sequence) + 1]

    mean_pool = reps.mean(dim=0).cpu().numpy()
    max_pool = reps.max(dim=0).values.cpu().numpy()

    for i, v in enumerate(mean_pool):
        features[f"ESM2_L{layer}_mean_{i}_{sequence_name}"] = float(v)

    for i, v in enumerate(max_pool):
        features[f"ESM2_L{layer}_max_{i}_{sequence_name}"] = float(v)

    return features
