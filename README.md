```markdown
# 🛡️ Cryptocurrency Fraud Detection — Naive Bayes (Baseline)

Kelompok 1 — Universitas Pamulang  
Project untuk mendeteksi transaksi kripto **normal (0)** vs **fraud (1)** menggunakan **Gaussian Naive Bayes** sebagai **baseline**.

---

## 🎯 Tujuan

- Membuat sistem sederhana untuk klasifikasi **normal/fraud**.
- Menyediakan **baseline model** (Naive Bayes) sebelum mencoba model lain.
- Menunjukkan bagaimana **machine learning** membantu **keamanan transaksi digital**.

## 🗂️ Dataset

- **Jenis:** _Synthetic dataset_ (dibuat sendiri demi privasi; bukan data real).
- **Ukuran:** ~**5.000 baris** (otomatis dibuat jika `data/transactions.csv` tidak ada).
- **Fitur (X):**
  - `amount` — jumlah transaksi
  - `transaction_freq_24h` — frekuensi transaksi 24 jam
  - `account_age_days` — umur akun (hari)
  - `is_weekend` — 0/1 apakah transaksi terjadi saat weekend
- **Label (y):** `is_fraud` — 0 (normal) / 1 (fraud)

## ⚙️ Teknologi

Python · scikit-learn · pandas · numpy · matplotlib · seaborn · Jupyter

## 📁 Struktur Repo
```

.
├─ Fraud_Detection_Flow_Presentation_clean.ipynb # notebook untuk demo/presentasi
├─ fraud_detection_baseline.py # script .py (jalan via terminal)
├─ requirements.txt # dependensi
├─ data/
│ └─ transactions.csv # opsional; auto-dibuat kalau kosong
├─ outputs_basic/
│ ├─ confusion_matrix.png
│ ├─ roc_curve.png
│ └─ metrics.json # opsional; dibuat dari notebook
└─ README.md

````

---

## 🔧 Setup (macOS/Linux/Windows PowerShell)
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name fraud-nb --display-name "Python (fraud-nb)"
````

## ▶️ Menjalankan

**Opsi A — Notebook**

1. `jupyter lab`
2. Buka `Fraud_Detection_Flow_Presentation_clean.ipynb`
3. Pilih kernel **Python (fraud-nb)** → **Run All Cells**
4. Output gambar tersimpan di `outputs_basic/`

**Opsi B — Script**

```bash
python fraud_detection_baseline.py
```

---

## 🔍 EDA (ringkas)

- Cek ukuran & tipe data (`df.info()`), ringkasan (`df.describe()`), missing & duplikat.
- Distribusi fitur (`histplot/boxplot`), **imbalance** label (`value_counts`).
- Insight EDA menentukan langkah preprocessing (impute/scale/balance).

## ✂️ Split & Validasi

- **Train/Test:** 80% / 20% (stratified).
- **(Opsional) K-Fold:** aktifkan di notebook `use_kfold = True` (default `k = 5`).

## 📈 Evaluasi

- Metrik: **Accuracy, Precision, Recall, F1, ROC-AUC**.
- Visual: **Confusion Matrix** & **ROC Curve** (`outputs_basic/*.png`).

> Catatan: Akurasi bisa tinggi kalau kelas **tidak seimbang**. Fokus tingkatkan **recall fraud**.

## 🚀 Next Step

- **Balancing** (contoh: `imblearn.SMOTE`) atau undersampling.
- **Threshold tuning** (menurunkan ambang prediksi agar lebih peka ke fraud).
- Coba model lain (Logistic Regression, Decision Tree/Random Forest) dan bandingkan **recall** kelas fraud.

## ❓ FAQ Singkat

- **Kenapa crypto?** Relevan & rawan fraud → studi kasus nyata.
- **Kenapa Naive Bayes?** Sederhana, cepat, cocok baseline sebelum model kompleks.
- **Dataset dari mana?** _Synthetic_ dibuat sendiri dengan Python (privasi).
- **EDA itu apa?** Exploratory Data Analysis: pahami kualitas & pola data sebelum modeling.

## 🔁 Reproducibility

Set seed & simpan versi paket:

```python
import sys, numpy as np, pandas as pd, sklearn, matplotlib, seaborn as sns
print("Python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", sns.__version__)
```

## 📝 Lisensi

Untuk kebutuhan akademik/pembelajaran.

````

Kalau sudah ditempatkan, tinggal commit & push:
```bash
git add README.md
git commit -m "Add README"
git push
````
