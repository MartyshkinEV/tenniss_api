import joblib
import pandas as pd

from config import resolve_default_model_artifact_path
from src.db.engine import get_engine

engine = get_engine()

model = joblib.load(resolve_default_model_artifact_path())

query = """
SELECT *
FROM match_features
WHERE (p1_id=208046 AND p2_id=208293)
   OR (p1_id=208293 AND p2_id=208046)
ORDER BY tourney_date DESC
LIMIT 1
"""

df = pd.read_sql(query, engine)

print(df[["p1_name","p2_name","tourney_date"]])

features = [c for c in df.columns if c not in [
    "match_id","p1_name","p2_name","label","tourney_date"
]]

X = df[features].fillna(0)

proba = model.predict_proba(X)[0][1]

print()
print("MODEL PREDICTION")
print("------------------")
print("P(p1 wins) =", round(proba,3))
print("P(p2 wins) =", round(1-proba,3))
