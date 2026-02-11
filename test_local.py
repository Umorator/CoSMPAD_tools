from cosmpad_predictor import CosmpadPredictor

# Example sequence
sequence = 'MKPKKIISNKAQISLELALLLGALVVAASIVG'

predictor = CosmpadPredictor()
df = predictor.predict_from_sequence([sequence])
print(df)

df.pred_proba.values


# Extract features directly
features = predictor.extract_from_sequence(sequence)

# Inspect
print("Number of features:", len(features))
print("Some example features:")
for k, v in list(features.items())[:10]:  # show first 10 for brevity
    print(f"{k}: {v}")