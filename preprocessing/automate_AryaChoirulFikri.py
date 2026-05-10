"""
automate_Siswa.py
=================
Pipeline otomatisasi preprocessing untuk Credit Card Fraud Detection Dataset.

Cara menjalankan (dari folder preprocessing/):
    python automate_Siswa.py

Output yang dihasilkan:
    preprocessing/creditcard_preprocessing/train.csv
    preprocessing/creditcard_preprocessing/test.csv
    preprocessing/creditcard_preprocessing/creditcard_preprocessed.csv

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ──────────────────────────────────────────────────────────────
# KONFIGURASI PATH
# ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_PATH   = os.path.join(BASE_DIR, "..", "creditcard_raw", "creditcard.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "creditcard_preprocessing")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    print(f"\n[1/6] LOAD DATA")
    print(f"      Path  : {path}")
    df = pd.read_csv(path)
    print(f"      Shape : {df.shape}")
    print(f"      Distribusi kelas:")
    print(f"        {df['Class'].value_counts().to_dict()}")
    return df


# ──────────────────────────────────────────────────────────────
# 2. HANDLING MISSING VALUES
# ──────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Mengisi missing values dengan median untuk kolom numerik."""
    print(f"\n[2/6] HANDLING MISSING VALUES")
    total_missing = df.isnull().sum().sum()
    print(f"      Total missing: {total_missing}")

    if total_missing > 0:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        print(f"      Selesai diisi dengan median.")
    else:
        print(f"      Tidak ada missing values ditemukan.")
    return df


# ──────────────────────────────────────────────────────────────
# 3. HANDLING DUPLICATES
# ──────────────────────────────────────────────────────────────
def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    print(f"\n[3/6] HANDLING DUPLICATES")
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"      Dihapus : {removed} baris duplikat")
    print(f"      Shape   : {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan fitur baru:
    - Hour       : jam transaksi dari kolom Time
    - Log_Amount : log(1+Amount) untuk mengurangi skewness
    Kolom Time dan Amount asli di-drop.
    """
    print(f"\n[4/6] FEATURE ENGINEERING")

    df["Hour"]       = (df["Time"] // 3600) % 24
    df["Log_Amount"] = np.log1p(df["Amount"])
    df.drop(columns=["Time", "Amount"], inplace=True)

    print(f"      Ditambahkan : Hour, Log_Amount")
    print(f"      Di-drop     : Time, Amount")
    print(f"      Shape       : {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────
# 5. FEATURE SCALING
# ──────────────────────────────────────────────────────────────
def scale_features(df: pd.DataFrame, target_col: str = "Class") -> pd.DataFrame:
    """
    StandardScaler pada Hour dan Log_Amount.
    V1–V28 tidak di-scale karena sudah hasil PCA.
    """
    print(f"\n[5/6] FEATURE SCALING")
    scale_cols = [c for c in ["Hour", "Log_Amount"] if c in df.columns]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"      StandardScaler diterapkan pada: {scale_cols}")
    return df


# ──────────────────────────────────────────────────────────────
# 6. SPLIT + SMOTE + SAVE
# ──────────────────────────────────────────────────────────────
def split_and_save(
    df: pd.DataFrame,
    target_col: str = "Class",
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Train/test split 80:20, lalu SMOTE pada train set
    untuk menangani class imbalance. Hasil disimpan ke CSV.
    """
    print(f"\n[6/6] SPLIT + SMOTE + SAVE")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train (sebelum SMOTE) : {X_train.shape} | fraud={y_train.sum()}")
    print(f"      Test                  : {X_test.shape}  | fraud={y_test.sum()}")

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"      Train (setelah SMOTE) : {X_train_res.shape} | fraud={y_train_res.sum()}")

    train_df = pd.concat([
        pd.DataFrame(X_train_res, columns=X.columns),
        pd.Series(y_train_res, name=target_col)
    ], axis=1)

    test_df = pd.concat([
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True)
    ], axis=1)

    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")
    full_path  = os.path.join(output_dir, "creditcard_preprocessed.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    df.to_csv(full_path,       index=False)

    print(f"\n      File tersimpan:")
    print(f"        ✅ {train_path}")
    print(f"        ✅ {test_path}")
    print(f"        ✅ {full_path}")


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────
def run_preprocessing(
    raw_path:   str = RAW_PATH,
    output_dir: str = OUTPUT_DIR
) -> pd.DataFrame:
    """Menjalankan seluruh pipeline preprocessing secara berurutan."""
    print("=" * 60)
    print("  PREPROCESSING PIPELINE — Credit Card Fraud Detection")
    print("=" * 60)

    df = load_data(raw_path)
    df = handle_missing_values(df)
    df = handle_duplicates(df)
    df = feature_engineering(df)
    df = scale_features(df)
    split_and_save(df, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("  ✅  PREPROCESSING SELESAI")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_preprocessing()
