import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

from src.db.engine import get_engine
from config import settings

engine = get_engine()

MODEL_DIR=settings.models_dir
os.makedirs(MODEL_DIR,exist_ok=True)

print("Loading dataset...")

df=pd.read_sql("""
SELECT *
FROM match_features_elo
WHERE label IS NOT NULL
ORDER BY tourney_date
""",engine)

df["tourney_date"]=pd.to_datetime(df["tourney_date"])

train=df[df["tourney_date"]<"2022-01-01"]
valid=df[(df["tourney_date"]>="2022-01-01")&(df["tourney_date"]<"2024-01-01")]
test=df[df["tourney_date"]>="2024-01-01"]

features=[
"surface",
"tourney_level",
"round",
"best_of",

"p1_rank",
"p2_rank",

"p1_rank_points",
"p2_rank_points",

"p1_winrate_last10",
"p2_winrate_last10",

"p1_winrate_surface",
"p2_winrate_surface",

"p1_hold_rate",
"p2_hold_rate",

"p1_break_rate",
"p2_break_rate",

"rank_diff",
"rank_points_diff",

"winrate_last10_diff",
"winrate_surface_diff",

"hold_rate_diff",
"break_rate_diff",

"p1_h2h_wins",
"p2_h2h_wins",

"elo_diff",
"surface_elo_diff"
]

target="label"

X_train=train[features]
y_train=train[target]

X_valid=valid[features]
y_valid=valid[target]

X_test=test[features]
y_test=test[target]

categorical=["surface","tourney_level","round"]
numeric=[c for c in features if c not in categorical]

pre=ColumnTransformer([
("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),numeric),
("cat",Pipeline([
("imp",SimpleImputer(strategy="most_frequent")),
("oh",OneHotEncoder(handle_unknown="ignore"))
]),categorical)
])

model=Pipeline([
("prep",pre),
("clf",LGBMClassifier(
n_estimators=400,
learning_rate=0.05,
num_leaves=31,
subsample=0.8,
colsample_bytree=0.8,
random_state=42
))
])

print("Training LightGBM + ELO...")

model.fit(X_train,y_train)

for name,X,y in[
("VALID",X_valid,y_valid),
("TEST",X_test,y_test)
]:

    proba=model.predict_proba(X)[:,1]
    pred=(proba>=0.5).astype(int)

    print("\n",name)
    print("accuracy",accuracy_score(y,pred))
    print("roc_auc",roc_auc_score(y,proba))
    print("log_loss",log_loss(y,proba))
    print("brier",brier_score_loss(y,proba))

path=f"{MODEL_DIR}/lightgbm_elo.joblib"

joblib.dump(model,path)

print("\nModel saved:",path)
