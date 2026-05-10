"""
modelling_tuning.py (FAST VERSION)
===================================
Training dengan hyperparameter tuning + manual logging + DagsHub.
Optimasi kecepatan:
  - Subsample data untuk tuning (50K baris)
  - RandomizedSearchCV (lebih cepat dari GridSearchCV)
  - Train final model dengan full data

Estimasi waktu: ~5-8 menit total

Cara menjalankan (dari folder Membangun_model/):
    python modelling_tuning.py
"""

import os
import pickle
import json
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, ConfusionMatrixDisplay
)

# ──────────────────────────────────────────────
# KONFIGURASI
# ──────────────────────────────────────────────
DAGSHUB_USERNAME  = "Arya-coba"
DAGSHUB_REPO_NAME = "Workflow-CI"

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "creditcard_preprocessing")
TRAIN_PATH   = os.path.join(DATA_DIR, "train.csv")
TEST_PATH    = os.path.join(DATA_DIR, "test.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

TARGET_COL      = "Class"
EXPERIMENT_NAME = "CreditCard-Fraud-Detection-Tuning"

# ──────────────────────────────────────────────
# INIT DAGSHUB
# ──────────────────────────────────────────────
print("[INFO] Menghubungkan ke DagsHub...")
dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPO_NAME,
    mlflow=True
)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"[INFO] Tracking URL: https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow\n")


# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
def load_data():
    print("[1/5] Memuat dataset...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test  = test.drop(columns=[TARGET_COL])
    y_test  = test[TARGET_COL]

    print(f"      Train : {X_train.shape} | fraud={int(y_train.sum())}")
    print(f"      Test  : {X_test.shape}  | fraud={int(y_test.sum())}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# 2. HYPERPARAMETER TUNING (CEPAT)
# ──────────────────────────────────────────────
def tune_model(X_train, y_train):
    print("\n[2/5] Hyperparameter tuning (RandomizedSearchCV)...")

    # Subsample 50K baris untuk tuning agar cepat
    SAMPLE_SIZE = 50000
    if len(X_train) > SAMPLE_SIZE:
        idx = np.random.RandomState(42).choice(len(X_train), SAMPLE_SIZE, replace=False)
        X_sample = X_train.iloc[idx]
        y_sample = y_train.iloc[idx]
        print(f"      Subsample {SAMPLE_SIZE} baris untuk tuning...")
    else:
        X_sample, y_sample = X_train, y_train

    param_dist = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [6, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf, param_dist,
        n_iter=10,        # hanya coba 10 kombinasi acak
        cv=3,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X_sample, y_sample)

    print(f"      Best params : {search.best_params_}")
    print(f"      Best CV F1  : {search.best_score_:.4f}")

    # Train ulang model terbaik dengan FULL data
    print(f"      Training ulang dengan full data ({len(X_train)} baris)...")
    best_model = RandomForestClassifier(
        **search.best_params_,
        random_state=42,
        n_jobs=-1
    )
    best_model.fit(X_train, y_train)
    print(f"      ✅ Training selesai!")

    return best_model, search.best_params_


# ──────────────────────────────────────────────
# 3. BUAT ARTEFAK TAMBAHAN
# ──────────────────────────────────────────────
def make_artifacts(y_test, y_pred, y_pred_prob, model, feature_names):
    print("\n[3/5] Membuat artefak tambahan...")

    # Confusion Matrix
    cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Credit Card Fraud")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ROC Curve
    roc_path = os.path.join(ARTIFACT_DIR, "roc_curve.png")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Feature Importance
    fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.png")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importance")
    plt.tight_layout()
    plt.savefig(fi_path, dpi=150)
    plt.close()

    # Classification Report
    cr_path = os.path.join(ARTIFACT_DIR, "classification_report.txt")
    report = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"])
    with open(cr_path, "w") as f:
        f.write("Classification Report — Credit Card Fraud Detection\n")
        f.write("=" * 55 + "\n")
        f.write(report)

    print("      ✅ Semua artefak selesai dibuat")
    return cm_path, roc_path, fi_path, cr_path


# ──────────────────────────────────────────────
# 4. LOG KE DAGSHUB
# ──────────────────────────────────────────────
def log_to_dagshub(X_train, X_test, y_train, y_test, model, best_params):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    cm_path, roc_path, fi_path, cr_path = make_artifacts(
        y_test, y_pred, y_pred_prob, model, list(X_train.columns)
    )

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n[4/5] Logging ke DagsHub MLflow...")
    mlflow.sklearn.autolog(disable=True)

    with mlflow.start_run(run_name="RandomForest-Tuned") as run:
        # Parameter
        mlflow.log_param("model",            "RandomForestClassifier")
        mlflow.log_param("best_params",      json.dumps(best_params))
        mlflow.log_param("train_size",       len(X_train))
        mlflow.log_param("test_size",        len(X_test))
        mlflow.log_param("n_features",       X_train.shape[1])
        mlflow.log_param("tuning_method",    "RandomizedSearchCV")
        mlflow.log_param("tuning_n_iter",    10)
        mlflow.log_param("tuning_cv",        3)

        # Metrik standar
        mlflow.log_metric("accuracy",        acc)
        mlflow.log_metric("precision",       precision)
        mlflow.log_metric("recall",          recall)
        mlflow.log_metric("f1_score",        f1)
        mlflow.log_metric("roc_auc",         roc_auc)

        # Metrik tambahan
        mlflow.log_metric("true_positives",  int(tp))
        mlflow.log_metric("false_positives", int(fp))
        mlflow.log_metric("true_negatives",  int(tn))
        mlflow.log_metric("false_negatives", int(fn))
        mlflow.log_metric("fraud_recall",    float(tp / (tp + fn)) if (tp + fn) > 0 else 0)

        # Model pickle
        model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_path,  artifact_path="model")

        # Artefak plot & report
        mlflow.log_artifact(cm_path,  artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")
        mlflow.log_artifact(fi_path,  artifact_path="plots")
        mlflow.log_artifact(cr_path,  artifact_path="reports")

        print(f"\n[5/5] Hasil evaluasi:")
        print("=" * 55)
        print(f"  Accuracy        : {acc:.4f}")
        print(f"  Precision       : {precision:.4f}")
        print(f"  Recall          : {recall:.4f}")
        print(f"  F1-Score        : {f1:.4f}")
        print(f"  ROC-AUC         : {roc_auc:.4f}")
        print(f"  True Positives  : {tp}")
        print(f"  False Positives : {fp}")
        print(f"  True Negatives  : {tn}")
        print(f"  False Negatives : {fn}")
        print(f"  Fraud Recall    : {tp/(tp+fn):.4f}")
        print("=" * 55)
        print(f"\n✅ Selesai! Run ID: {run.info.run_id}")
        print(f"🔗 https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    best_model, best_params = tune_model(X_train, y_train)
    log_to_dagshub(X_train, X_test, y_train, y_test, best_model, best_params)
