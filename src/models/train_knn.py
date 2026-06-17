import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, get_classification_features

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/knn", exist_ok=True)

print("Loading data...")
df = load_data()
X, _, y_severity, feature_names, encoders = get_classification_features(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
label_names = encoders["Severity"].classes_

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_severity, test_size=0.2, random_state=RANDOM_SEED, stratify=y_severity
)

print("Tuning k (3-15, odd values)...")
k_values = range(3, 16, 2)
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    score = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy").mean()
    cv_scores.append(score)
    print(f"  k={k:2d}  CV accuracy={score:.4f}")

best_k = list(k_values)[np.argmax(cv_scores)]
print(f"\nBest k = {best_k}  (CV acc = {max(cv_scores):.4f})")

knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean", n_jobs=-1)
t0 = time.time()
knn.fit(X_train, y_train)
print(f"Training time: {time.time()-t0:.1f}s")

y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_names))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_values), cv_scores, marker="o", color="#2E86AB", linewidth=2)
axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
axes[0].set_xlabel("k (number of neighbours)", fontsize=12)
axes[0].set_ylabel("5-Fold CV Accuracy", fontsize=12)
axes[0].set_title("kNN Hyperparameter Tuning", fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names, ax=axes[1])
axes[1].set_xlabel("Predicted", fontsize=12)
axes[1].set_ylabel("Actual", fontsize=12)
axes[1].set_title(f"kNN Confusion Matrix (k={best_k})", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/knn/knn_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/knn/knn_results.png")

report_dict = classification_report(
    y_test, y_pred, target_names=label_names, output_dict=True
)
metrics_df = pd.DataFrame(report_dict).T.round(3)
metrics_df.to_csv("outputs/knn/knn_metrics.csv")
print("Saved: outputs/knn/knn_metrics.csv")

import pickle
with open("models/knn_model.pkl", "wb") as f:
    pickle.dump({"model": knn, "scaler": scaler, "encoders": encoders,
                 "feature_names": feature_names, "best_k": best_k}, f)
print("Saved: models/knn_model.pkl")
