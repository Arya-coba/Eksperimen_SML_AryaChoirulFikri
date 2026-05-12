import os
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "creditcard_preprocessing")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")

EXPERIMENT_NAME = "CreditCard-Fraud-Detection"
TARGET_COL      = "Class"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
print("[1/3] Memuat dataset...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

X_train = train.drop(columns=[TARGET_COL])
y_train = train[TARGET_COL]
X_test  = test.drop(columns=[TARGET_COL])
y_test  = test[TARGET_COL]

print(f"      Train : {X_train.shape} | fraud={int(y_train.sum())}")
print(f"      Test  : {X_test.shape}  | fraud={int(y_test.sum())}")

# ──────────────────────────────────────────────
# TRAINING + LOGGING
# ──────────────────────────────────────────────
print("\n[2/3] Training model...")

# Gunakan autolog bawaan sklearn MLflow
mlflow.sklearn.autolog(
    log_models=False,        # matikan log_model agar tidak error 404
    log_datasets=False,
    log_post_training_metrics=True
)

with mlflow.start_run(run_name="RandomForest-Baseline") as run:

    # Training
    model = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = 10,
        random_state = 42,
        n_jobs       = -1
    )
    model.fit(X_train, y_train)

    # Prediksi
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Hitung metrik
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)

    # Log metrik manual (backup autolog)
    mlflow.log_metric("test_accuracy",  acc)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall",    recall)
    mlflow.log_metric("test_f1_score",  f1)
    mlflow.log_metric("test_roc_auc",   roc_auc)

    # Simpan model sebagai pickle lalu log artifact
    model_path = os.path.join(BASE_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact(model_path, artifact_path="model")

    # Log feature names sebagai artifact
    feat_path = os.path.join(BASE_DIR, "feature_names.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(X_train.columns.tolist()))
    mlflow.log_artifact(feat_path, artifact_path="model")

    print(f"\n[3/3] Hasil evaluasi:")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("=" * 50)
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Fraud'])}")
    print(f"✅ Selesai! Run ID: {run.info.run_id}")
    print(f"🔗 http://127.0.0.1:5000/#/experiments")