# Eksperimen_SML_AryaChoirulFikri
## Credit Card Fraud Detection — MLOps Pipeline

> Submission Proyek Akhir: Membangun Sistem Machine Learning  
> Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📁 Struktur Repository

```
Eksperimen_SML_AryaChoirulFikri/
├── .github/
│   └── workflows/
│       └── preprocessing.yml       ← GitHub Actions auto-preprocessing
├── creditcard_raw/
│   └── creditcard.csv              ← Dataset mentah 
└── preprocessing/
    ├── Eksperimen_Siswa.ipynb      ← Notebook EDA & preprocessing manual
    ├── automate_Siswa.py           ← Script preprocessing otomatis
    └── creditcard_preprocessing/   ← Output (di-generate otomatis)
        ├── train.csv
        ├── test.csv
        └── creditcard_preprocessed.csv
```

---

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### 2. Jalankan preprocessing manual (Jupyter)
Buka dan jalankan `preprocessing/Eksperimen_AryaChoirulFikri.ipynb`

### 3. Jalankan preprocessing otomatis
```bash
cd preprocessing
python automate_AryaChoirulFikri.py
```

---

## ⚙️ GitHub Actions
Workflow otomatis akan berjalan setiap kali ada push ke:
- `creditcard_raw/**`
- `preprocessing/automate_AryaChoirulFikri.py`

Atau bisa di-trigger manual dari tab **Actions** di GitHub.
