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