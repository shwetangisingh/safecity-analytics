import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, get_classification_features

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODEL_DIR = "models"
OUTPUT_DIR = "outputs/decision_tree"

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
print("Number of classes:", len(valid_classes))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_category,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_category
)

print("\nRunning GridSearch...")

param_grid = {
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [10, 50, 100],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_SEED),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", round(grid.best_score_, 4))

dt = grid.best_estimator_

y_pred = dt.predict(X_test)

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
    cmap="Oranges",
    xticklabels=np.unique(y_category),
    yticklabels=np.unique(y_category)
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

cm_path = f"{OUTPUT_DIR}/confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
plt.close()

print("Saved:", cm_path)

importances = pd.Series(
    dt.feature_importances_,
    index=feature_names
).sort_values(ascending=True)

plt.figure(figsize=(8, 6))

importances.plot(kind="barh", color="#E84855")

plt.title("Feature Importance")
plt.xlabel("Importance Score")

plt.grid(axis="x", alpha=0.3)

fi_path = f"{OUTPUT_DIR}/feature_importance.png"

plt.tight_layout()
plt.savefig(fi_path, dpi=150)
plt.close()

print("Saved:", fi_path)

plt.figure(figsize=(22, 8))

plot_tree(
    dt,
    max_depth=3,
    feature_names=feature_names,
    filled=True,
    fontsize=7
)

plt.title("Decision Tree (Top 3 Levels)")

tree_path = f"{OUTPUT_DIR}/tree_structure.png"

plt.tight_layout()
plt.savefig(tree_path, dpi=120)
plt.close()

print("Saved:", tree_path)


report = classification_report(y_test, y_pred, output_dict=True)

metrics_df = pd.DataFrame(report).T.round(3)

metrics_path = f"{OUTPUT_DIR}/metrics.csv"

metrics_df.to_csv(metrics_path)

print("Saved:", metrics_path)

model_path = f"{MODEL_DIR}/decision_tree_model.pkl"

with open(model_path, "wb") as f:

    pickle.dump({
        "model": dt,
        "encoders": encoders,
        "feature_names": feature_names,
        "best_params": grid.best_params_
    }, f)

print("Saved:", model_path)