from cosmpad_predictor import CosmpadPredictor

predictor = CosmpadPredictor()
df = predictor.predict_from_sequence(["MKPKKIISNKAQISLELALLLGALVVAASIVGFYYLKSVTRGTSTAESISKNITLAAKNKALDNIYKVKR", 'MKPKKIISNKAQISLELALLLGALVVAASIVG'])
print(df)

df.pred_proba.values