import os, json, pathlib, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- Config via env (AI Core sets DATA_PATH via artifact mount) ---
DATA_PATH     = os.getenv("DATA_PATH", "/app/data/cpi_logs_500.csv")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "CUSTOM_STATUS")
MODEL_DIR     = os.getenv("MODEL_DIR", "/app/model")
MODEL_PATH    = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "model.pkl"))

# --- IO ---
pathlib.Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA_PATH)

# --- Features/labels ---
feat_candidates = ["ARTIFACT_NAME", "ORIGIN_COMPONENT_NAME", "LOG_LEVEL"]
feat_cols = [c for c in feat_candidates if c in df.columns]
if not feat_cols:
    raise ValueError(f"Missing expected feature columns; looked for any of {feat_candidates}")
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Missing target column '{TARGET_COLUMN}'")

X_dict = df[feat_cols].fillna("").astype(str).to_dict(orient="records")
y = df[TARGET_COLUMN].astype(str)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_dict, y, test_size=0.2, random_state=42, stratify=y
)

# --- Single-file model via Pipeline (DictVectorizer + LR) ---
pipe = Pipeline([
    ("vec", DictVectorizer(sparse=True)),
    ("clf", LogisticRegression(max_iter=500, solver="saga", multi_class="multinomial")),
])
pipe.fit(X_tr, y_tr)

# --- Eval ---
y_pred = pipe.predict(X_te)
acc = float(accuracy_score(y_te, y_pred))
print("Accuracy:", acc)
print(classification_report(y_te, y_pred, zero_division=0))

# --- Save model + metrics ---
joblib.dump(pipe, MODEL_PATH)
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump({"accuracy": acc, "features": feat_cols, "target": TARGET_COLUMN}, f)

print(f"[info] Saved single-file model → {MODEL_PATH}")
