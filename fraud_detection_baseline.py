
# 🛡️ Fraud Detection on Cryptocurrency — Naive Bayes Baseline (Pure .py)
# Kelompok 1 — Universitas Pamulang

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             RocCurveDisplay, roc_auc_score)

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR = Path("outputs_basic"); OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def generate_synthetic(n_rows=5000, fraud_ratio=0.08, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    amount = np.round(np.exp(rng.normal(3, 1.0, n_rows)) * 10, 2)
    tx_freq = rng.poisson(lam=3, size=n_rows)
    account_age = rng.integers(low=1, high=365*3, size=n_rows)
    is_weekend = rng.integers(low=0, high=2, size=n_rows)

    base = np.full(n_rows, fraud_ratio, dtype=float)
    base += (amount > np.percentile(amount, 85)) * 0.06
    base += (account_age < 30) * 0.05
    base += (tx_freq > 6) * 0.04
    base += (is_weekend == 1) * 0.02
    base = np.clip(base, 0.01, 0.85)

    is_fraud = rng.binomial(1, base)

    return pd.DataFrame({
        "amount": amount,
        "transaction_freq_24h": tx_freq,
        "account_age_days": account_age,
        "is_weekend": is_weekend,
        "is_fraud": is_fraud
    })

csv_path = DATA_DIR / "transactions.csv"
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print("Loaded dataset:", csv_path)
else:
    df = generate_synthetic()
    df.to_csv(csv_path, index=False)
    print("Generated synthetic dataset (~5k rows).")

print(df.head())

# EDA singkat
print("\nShape:", df.shape)
print("\nClass balance ratio:")
print(df['is_fraud'].value_counts(normalize=True))

# Split
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("\nTrain:", X_train.shape, " | Test:", X_test.shape)

# Train
gnb = GaussianNB().fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)[:, 1]

# Evaluasi
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)

print("\n--- Evaluation ---")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}\n")
print(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
ax.set_title("Confusion Matrix — GaussianNB")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# ROC Curve
fig, ax = plt.subplots(figsize=(5, 4))
RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
ax.set_title("ROC Curve — GaussianNB")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "roc_curve.png", dpi=150)
plt.close()

print("Saved confusion_matrix.png and roc_curve.png in outputs_basic/")
