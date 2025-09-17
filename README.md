# 🛡️ Data Mining — Cryptocurrency Fraud Detection

Kelompok 1 — Universitas Pamulang  
📚 **Mata Kuliah:** Data Mining  
👨‍🏫 **Dosen:** Tri Prasetyo  

---

## 📌 Judul
**Prediksi Transaksi Fraud pada Cryptocurrency Menggunakan Algoritma Naive Bayes**

---

## 👥 Anggota Kelompok
- Achmad Syahril Fauzi (231011450396)  
- Abdul Fakhry (231011450644)  
- Ahmad Imam (231011450458)  

---

## 🎯 Tujuan
- Membuat sistem sederhana untuk klasifikasi **normal/fraud**.  
- Menyusun **baseline model** (Naive Bayes) sebelum mencoba model lain.  
- Menunjukkan bagaimana **machine learning** membantu **keamanan transaksi digital**.  

---

## 📊 Deskripsi Singkat
Proyek ini mendeteksi **transaksi fraud pada cryptocurrency** menggunakan **Gaussian Naive Bayes** untuk klasifikasi biner.  
Dataset yang digunakan adalah **synthetic dataset** (~**5.000 baris**) dengan fitur utama:

- `amount`  
- `transaction_freq_24h`  
- `account_age_days`  
- `is_weekend`  

**Label target:** `is_fraud` (0 = Normal, 1 = Fraud).  

> **Catatan:** Dataset synthetic dibuat sendiri demi privasi; bukan data real transaksi.  

---

## 🗂️ Dataset
- **Jenis:** _Synthetic dataset_ (dibuat dengan Python).  
- **Ukuran:** ~**5.000 baris** (otomatis dibuat jika `data/transactions.csv` belum ada).  
- **Fitur (X):**  
  - `amount` — jumlah transaksi  
  - `transaction_freq_24h` — frekuensi transaksi 24 jam  
  - `account_age_days` — umur akun (hari)  
  - `is_weekend` — 0/1 apakah transaksi terjadi saat weekend  
- **Label (y):** `is_fraud` — 0 (normal) / 1 (fraud)  

---

## 🛠️ Teknologi
Python · scikit-learn · pandas · numpy · matplotlib · seaborn · Jupyter  

---

## 📁 Struktur Repo
```text
.
├── Fraud_Detection_Flow_Presentation_clean.ipynb   # Notebook untuk demo/presentasi
├── fraud_detection_baseline.py                     # Script: generate data → train NB → evaluasi
├── requirements.txt                                # Dependensi (pinned)
├── data/
│   └── transactions.csv                            # Synthetic dataset (~5k baris)
├── outputs_basic/
│   ├── confusion_matrix.png                        # Confusion Matrix (test set)
│   └── roc_curve.png                               # ROC Curve (test set)
└── README.md                                       # Dokumentasi proyek
```

---

## 💻 Prasyarat
- **Python 3.9+** (disarankan 3.10/3.11)  
- OS: macOS / Linux / Windows  
- Git (opsional, untuk clone repo)  

---

## ⚙️ Cara Menjalankan

### 1) Clone Repository
```bash
git clone https://github.com/DevLabsss/crypto-fraud-detection.git
cd crypto-fraud-detection
```

> Tidak punya Git? Klik **Code → Download ZIP**, ekstrak, lalu `cd` ke foldernya.  

### 2) Buat Virtual Environment
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3) Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Jalankan

#### A. Jalankan Script Python
```bash
python fraud_detection_baseline.py
```

**Output:**  
- Metrik tampil di terminal (**Accuracy, Precision, Recall, F1, ROC-AUC**)  
- Gambar tersimpan di `outputs_basic/`:  
  - `confusion_matrix.png`  
  - `roc_curve.png`  

#### B. Jalankan Notebook
```bash
pip install notebook jupyterlab ipykernel
jupyter lab
```

Buka `Fraud_Detection_Flow_Presentation_clean.ipynb` → **Run All Cells**.

---

## 🔍 EDA (ringkas)
- Ukuran/tipe (`df.info()`), ringkasan (`df.describe()`), missing & duplikat.  
- Distribusi fitur (hist/boxplot) dan **imbalance** label (`value_counts`).  
- Insight EDA → tentukan preprocessing (impute/scale/balance).  

---

## ✂️ Split & Validasi
- **Train/Test:** 80% / 20% (stratified).  
- **(Opsional) K-Fold:** aktifkan di notebook (`use_kfold = True`, `k = 5`).  

---

## 📈 Evaluasi
- **Metrik:** Accuracy, Precision, Recall, F1, ROC-AUC.  
- **Visual:** Confusion Matrix & ROC Curve (tersimpan di `outputs_basic/`).  

> Catatan: Akurasi bisa tinggi saat kelas **tidak seimbang** — fokus tingkatkan **recall** kelas fraud.  

---

## 🚀 Next Step
- **Balancing** data (mis. `imblearn.SMOTE`) atau undersampling.  
- **Threshold tuning** (turunkan ambang prediksi agar lebih peka ke fraud).  
- Coba model lain (Logistic Regression, Decision Tree, Random Forest) untuk bandingkan recall kelas fraud.  

---

## ❓ FAQ
- **Kenapa crypto?** Relevan & rawan fraud → studi kasus nyata.  
- **Kenapa Naive Bayes?** Sederhana, cepat, cocok baseline sebelum model kompleks.  
- **Dataset dari mana?** _Synthetic_ dibuat sendiri (privasi).  
- **EDA itu apa?** Exploratory Data Analysis: pahami kualitas & pola data sebelum modeling.  

---

## 🔁 Reproducibility
```python
import sys, numpy as np, pandas as pd, sklearn, matplotlib, seaborn as sns
print("Python:", sys.version.split()[0])
print("numpy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("seaborn:", sns.__version__)
```

---

## 📝 Lisensi
Untuk kebutuhan akademik/pembelajaran.  
