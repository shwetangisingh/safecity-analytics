import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, get_classification_features

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODEL_DIR = "models"
OUTPUT_DIR = "outputs/random_forest"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")

df = load_data()

X, y_category, _, feature_names, encoders = get_classification_features(df)

class_counts = pd.Series(y_category).value_counts()

valid_classes = class_counts[class_counts >= 5].index

mask = np.isin(y_category, valid_classes)

X = X[mask]
y_category = y_category[mask]

print("Samples after filtering:", len(y_category))
print("Classes remaining:", len(valid_classes))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_category,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_category
)

print("\nRunning RandomizedSearchCV...")

param_dist = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [5, 10, 50],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"]
}

rscv = RandomizedSearchCV(
    RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=-1
    ),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    verbose=1,
    random_state=RANDOM_SEED,
    n_jobs=-1
)

rscv.fit(X_train, y_train)

print("\nBest Parameters:", rscv.best_params_)
print("Best CV Accuracy:", round(rscv.best_score_, 4))

rf = rscv.best_estimator_

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", round(accuracy, 4))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 8))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=np.unique(y_category),
    yticklabels=np.unique(y_category)
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

cm_path = f"{OUTPUT_DIR}/confusion_matrix.png"

plt.savefig(cm_path, dpi=150)
plt.close()

print("Saved:", cm_path)

importances = pd.Series(
    rf.feature_importances_,
    index=feature_names
).sort_values(ascending=True)

plt.figure(figsize=(8, 6))

importances.plot(kind="barh", color="#52B788")

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")

plt.grid(axis="x", alpha=0.3)

fi_path = f"{OUTPUT_DIR}/feature_importance.png"

plt.tight_layout()
plt.savefig(fi_path, dpi=150)
plt.close()

# -----------------------------
# Learning Curve (train vs val)
# -----------------------------
print("Plotting learning curve...")
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    rf, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 6),
    cv=3, scoring="accuracy",
    n_jobs=-1, random_state=RANDOM_SEED
)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Train Accuracy", color="#2E86AB")
plt.plot(train_sizes, val_mean,   marker="o", label="Val Accuracy",   color="#E84855")
plt.xlabel("Training Set Size", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title("Random Forest Learning Curve", fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
lc_path = f"{OUTPUT_DIR}/learning_curve.png"
plt.savefig(lc_path, dpi=150)
plt.close()
print("Saved:", lc_path) 

print("Saved:", fi_path)

print("Plotting accuracy vs number of trees...")

n_trees = [10, 50, 100, 150, 200, 300]

accs = []

for n in n_trees:

    rf_tmp = RandomForestClassifier(
        n_estimators=n,
        max_depth=rscv.best_params_["max_depth"],
        min_samples_split=rscv.best_params_["min_samples_split"],
        max_features=rscv.best_params_["max_features"],
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    rf_tmp.fit(X_train, y_train)

    accs.append(
        accuracy_score(y_test, rf_tmp.predict(X_test))
    )

plt.figure(figsize=(9, 5))

plt.plot(n_trees, accs, marker="o")

plt.xlabel("Number of Trees")
plt.ylabel("Test Accuracy")
plt.title("Random Forest Accuracy vs Trees")

plt.grid(True, alpha=0.3)

acc_plot = f"{OUTPUT_DIR}/accuracy_vs_trees.png"

plt.tight_layout()
plt.savefig(acc_plot, dpi=150)
plt.close()

print("Saved:", acc_plot)

report = classification_report(y_test, y_pred, output_dict=True)

metrics_df = pd.DataFrame(report).T.round(3)

metrics_path = f"{OUTPUT_DIR}/metrics.csv"

metrics_df.to_csv(metrics_path)

print("Saved:", metrics_path)

model_path = f"{MODEL_DIR}/random_forest_model.pkl"

with open(model_path, "wb") as f:

    pickle.dump({
        "model": rf,
        "encoders": encoders,
        "feature_names": feature_names,
        "best_params": rscv.best_params_
    }, f)

print("Saved:", model_path)