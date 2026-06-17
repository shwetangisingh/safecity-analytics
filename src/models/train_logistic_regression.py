import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)

import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, get_classification_features

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/logistic_regression", exist_ok=True)

print("Loading data...")
df = load_data()

X, _, _, feature_names, encoders = get_classification_features(df)

feature_cols_no_weapon = [
    "AREA", "Hour", "Month", "IsWeekend",
    "Premise Category", "TimeBucket", "Severity", "Part 1-2",
    "Reporting Delay (Days)"
]
df_feat = df[feature_cols_no_weapon].copy()
df_feat["IsWeekend"] = df_feat["IsWeekend"].astype(int)
from sklearn.preprocessing import LabelEncoder
for col in ["Premise Category", "TimeBucket", "Severity"]:
    le = LabelEncoder()
    df_feat[col] = le.fit_transform(df_feat[col].astype(str))

X = df_feat.values
feature_names_clean = feature_cols_no_weapon
y = df["Has Weapon"].astype(int).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print("Comparing regularisation C values via 5-fold CV...")
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
cv_aucs = []
for C in C_values:
    lr_tmp = LogisticRegression(C=C, class_weight="balanced",
                                 max_iter=1000, random_state=RANDOM_SEED)
    aucs = cross_val_score(lr_tmp, X_train, y_train, cv=5, scoring="roc_auc")
    cv_aucs.append(aucs.mean())
    print(f"  C={C:6}  AUC={aucs.mean():.4f} ± {aucs.std():.4f}")

best_C = C_values[np.argmax(cv_aucs)]
print(f"\nBest C = {best_C}")

lr = LogisticRegression(C=best_C, class_weight="balanced",
                         max_iter=1000, random_state=RANDOM_SEED)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
ap   = average_precision_score(y_test, y_prob)
print(f"\nTest Accuracy : {acc:.4f}")
print(f"ROC-AUC       : {auc:.4f}")
print(f"Avg Precision : {ap:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Weapon", "Weapon"]))

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Weapon", "Weapon"],
            yticklabels=["No Weapon", "Weapon"], ax=axes[0])
axes[0].set_xlabel("Predicted", fontsize=11)
axes[0].set_ylabel("Actual", fontsize=11)
axes[0].set_title("Confusion Matrix", fontsize=13)

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="#2E86AB", linewidth=2, label=f"AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[1].set_xlabel("False Positive Rate", fontsize=11)
axes[1].set_ylabel("True Positive Rate", fontsize=11)
axes[1].set_title("ROC Curve", fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

coefs = pd.Series(lr.coef_[0], index=feature_names_clean).sort_values()
colors = ["#E84855" if v > 0 else "#2E86AB" for v in coefs]
coefs.plot(kind="barh", color=colors, ax=axes[2])
axes[2].axvline(0, color="black", linewidth=0.8)
axes[2].set_title("Feature Coefficients\n(red=increases weapon risk)", fontsize=12)
axes[2].set_xlabel("Coefficient Value", fontsize=11)
axes[2].grid(True, alpha=0.3, axis="x")

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[3].plot(rec, prec, color="#52B788", linewidth=2, label=f"AP = {ap:.3f}")
axes[3].axhline(y_test.mean(), color="gray", linestyle="--", linewidth=1, label="Baseline")
axes[3].set_xlabel("Recall", fontsize=11)
axes[3].set_ylabel("Precision", fontsize=11)
axes[3].set_title("Precision-Recall Curve", fontsize=13)
axes[3].legend(fontsize=11)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/logistic_regression/lr_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/logistic_regression/lr_results.png")

report_dict = classification_report(
    y_test, y_pred, target_names=["No Weapon", "Weapon"], output_dict=True
)
metrics_df = pd.DataFrame(report_dict).T.round(3)
metrics_df["ROC-AUC"] = ""
metrics_df.loc["weighted avg", "ROC-AUC"] = str(round(auc, 4))
metrics_df.to_csv("outputs/logistic_regression/lr_metrics.csv")
print("Saved: outputs/logistic_regression/lr_metrics.csv")

with open("models/logistic_regression_model.pkl", "wb") as f:
    pickle.dump({"model": lr, "scaler": scaler, "encoders": encoders,
                 "feature_names": feature_names_clean, "best_C": best_C}, f)
print("Saved: models/logistic_regression_model.pkl")
