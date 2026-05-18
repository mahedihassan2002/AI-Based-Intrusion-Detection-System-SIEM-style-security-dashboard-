import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score
)
import xgboost as xgb
import joblib

from utils.preprocess import preprocess
from utils.autoencoder import NumpyAutoencoder
from utils.visualise import build_dashboard_data

DATA_PATH   = Path("data/cybersecurity_attacks.csv")
MODELS_DIR  = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ─── 1. Load & Preprocess ────────────────────────────────────────────────────

print("=" * 60)
print("  AI Intrusion Detection System — Training Pipeline")
print("=" * 60)

df_raw = pd.read_csv(DATA_PATH)
print(f"\n[+] Loaded {len(df_raw):,} records, {df_raw.shape[1]} features")

X, y_attack, y_binary, scaler, feature_names, le_map = preprocess(df_raw)
print(f"[+] Features after engineering: {len(feature_names)}")
print(f"[+] Attack distribution:\n{pd.Series(y_attack).value_counts().to_string()}")

# ─── 2. Train / Test Split ───────────────────────────────────────────────────

X_train, X_test, yb_train, yb_test, ya_train, ya_test = train_test_split(
    X, y_binary, y_attack, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"\n[+] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─── 3. Isolation Forest (Unsupervised Anomaly Detection) ────────────────────

print("\n── Isolation Forest ──────────────────────────────────────")
iso = IsolationForest(
    n_estimators=300,
    contamination=0.85,          # ~85 % of training data are attacks
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_train)
iso_raw   = iso.predict(X_test)                     # +1 = normal, -1 = anomaly
iso_pred  = (iso_raw == -1).astype(int)             # 1 = attack, 0 = normal
iso_score = iso.decision_function(X_test)           # higher → more normal

print(classification_report(yb_test, iso_pred, target_names=["Normal", "Attack"]))
iso_auc = roc_auc_score(yb_test, -iso_score)        # negate: lower score → attack
print(f"ROC-AUC: {iso_auc:.4f}")

joblib.dump(iso, MODELS_DIR / "isolation_forest.pkl")
print("[+] Saved models/isolation_forest.pkl")

# ─── 4. XGBoost Classifier (Supervised) ──────────────────────────────────────

print("\n── XGBoost Classifier ────────────────────────────────────")
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(
    X_train, yb_train,
    eval_set=[(X_test, yb_test)],
    verbose=False,
)
xgb_pred  = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print(classification_report(yb_test, xgb_pred, target_names=["Normal", "Attack"]))
xgb_auc = roc_auc_score(yb_test, xgb_proba)
print(f"ROC-AUC: {xgb_auc:.4f}")

joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
print("[+] Saved models/xgboost.pkl")

# ─── 5. Autoencoder (Unsupervised Reconstruction-Error) ──────────────────────

print("\n── Autoencoder (NumPy) ───────────────────────────────────")
# Train only on "normal" samples so the AE learns the normal manifold
X_normal = X_train[yb_train == 0]
ae = NumpyAutoencoder(hidden_dims=[16, 8, 4], epochs=80, lr=0.01, batch_size=64)
ae.fit(X_normal, verbose=True)

# Reconstruction error = anomaly score
ae_errors_test = ae.reconstruction_error(X_test)
threshold      = np.percentile(ae.reconstruction_error(X_normal), 95)
ae_pred        = (ae_errors_test > threshold).astype(int)

print(f"\nThreshold (95th pct on normal): {threshold:.5f}")
print(classification_report(yb_test, ae_pred, target_names=["Normal", "Attack"]))
ae_auc = roc_auc_score(yb_test, ae_errors_test)
print(f"ROC-AUC: {ae_auc:.4f}")

ae.save(MODELS_DIR / "autoencoder.npz")
print("[+] Saved models/autoencoder.npz")

# ─── 6. Ensemble Verdict ─────────────────────────────────────────────────────

print("\n── Ensemble (majority vote) ──────────────────────────────")
ensemble_pred = (xgb_pred + iso_pred + ae_pred >= 2).astype(int)
print(classification_report(yb_test, ensemble_pred, target_names=["Normal", "Attack"]))

# ─── 7. Save artefacts & dashboard data ──────────────────────────────────────

joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(le_map, MODELS_DIR / "label_encoders.pkl")
joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

dashboard = build_dashboard_data(
    df_raw, X_test, yb_test, ya_test,
    xgb_model, iso, ae,
    xgb_pred, iso_pred, ae_pred, ensemble_pred,
    xgb_proba, iso_score, ae_errors_test,
    threshold, feature_names,
)
with open("dashboard_data.json", "w") as f:
    json.dump(dashboard, f, indent=2)
print("\n[+] Saved dashboard_data.json")

print("\n✅ Training complete.  Run  python ids_realtime.py  for live capture.")
print("=" * 60)
