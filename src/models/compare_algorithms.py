import os, pickle, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier

# Local imports
import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, get_classification_features

# ----------------------------- Constants -----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
OUTPUT_DIR = "outputs/comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------- Load Data -----------------------------
print("Loading data...")
df = load_data()
X, y_category, _, feature_names, encoders = get_classification_features(df)
label_names = encoders["Crime Category"].classes_

# Handle rare classes: remove classes with < 2 samples
counts = pd.Series(y_category).value_counts()
rare_classes = counts[counts < 2].index
if len(rare_classes) > 0:
    mask = ~np.isin(y_category, rare_classes)
    X = X[mask]
    y_category = y_category[mask]
    print(f"Removed rare classes: {list(rare_classes)}")

# ----------------------------- Train-Test Split -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_category, test_size=0.2, random_state=RANDOM_SEED, stratify=y_category
)

# ----------------------------- Scalers -----------------------------
scalers = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler()
}

X_train_scaled = {}
X_test_scaled = {}
for key, scaler in scalers.items():
    X_train_scaled[key] = scaler.fit_transform(X_train)
    X_test_scaled[key]  = scaler.transform(X_test)

# ----------------------------- Classifiers -----------------------------
classifiers = {
    "kNN (k=7)": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=RANDOM_SEED),
    "Naive Bayes": ComplementNB(alpha=1.0),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=20,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    ),
}

# Mapping classifier → appropriate scaled data
data_mapping = {
    "kNN (k=7)": ("standard",),
    "Decision Tree": ("raw",),
    "Naive Bayes": ("minmax",),
    "Random Forest": ("raw",)
}

# ----------------------------- Evaluation -----------------------------
results = []
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

for name, clf in classifiers.items():
    scale_key = data_mapping[name][0]
    if scale_key == "standard":
        Xtr, Xte = X_train_scaled["standard"], X_test_scaled["standard"]
    elif scale_key == "minmax":
        Xtr, Xte = X_train_scaled["minmax"], X_test_scaled["minmax"]
    else:  # raw
        Xtr, Xte = X_train, X_test

    # Train
    t0 = time.time()
    clf.fit(Xtr, y_train)
    train_time = time.time() - t0

    # Predict
    t0 = time.time()
    y_pred = clf.predict(Xte)
    pred_time = time.time() - t0

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    cv_score = cross_val_score(clf, Xtr, y_train, cv=cv_strategy, scoring="accuracy").mean()

    results.append({
        "Algorithm": name,
        "Test Accuracy": round(acc, 4),
        "Weighted F1": round(f1, 4),
        "3-Fold CV Acc": round(cv_score, 4),
        "Train Time (s)": round(train_time, 2),
        "Predict Time (s)": round(pred_time, 4),
    })

    print(f"{name:20s}  acc={acc:.4f}  f1={f1:.4f}  cv={cv_score:.4f}  train={train_time:.1f}s")

# ----------------------------- Save Results -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUTPUT_DIR}/algorithm_comparison.csv", index=False)
print(f"\nSaved CSV: {OUTPUT_DIR}/algorithm_comparison.csv")

# ----------------------------- Plotting -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ["Test Accuracy", "Weighted F1", "Train Time (s)"]
colors  = ["#2E86AB", "#52B788", "#E84855"]

for i, (metric, color) in enumerate(zip(metrics, colors)):
    axes[i].bar(results_df["Algorithm"], results_df[metric], color=color, alpha=0.85)
    axes[i].set_title(metric, fontsize=13)
    axes[i].set_ylabel(metric, fontsize=11)
    axes[i].set_xticks(range(len(results_df)))
    axes[i].set_xticklabels(results_df["Algorithm"], rotation=25, ha="right", fontsize=9)
    axes[i].grid(True, alpha=0.3, axis="y")
    for j, v in enumerate(results_df[metric]):
        axes[i].text(j, v + results_df[metric].max() * 0.01, str(v),
                     ha="center", va="bottom", fontsize=9)

plt.suptitle("Algorithm Comparison — Crime Category Classification", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/algorithm_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved PNG: {OUTPUT_DIR}/algorithm_comparison.png")