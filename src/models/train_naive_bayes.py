import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, get_classification_features

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODEL_DIR = "models"
OUTPUT_DIR = "outputs/naive_bayes"

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

label_names = encoders["Crime Category"].classes_[valid_classes]

print("Number of samples:", len(y_category))
print("Number of classes:", len(label_names))


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_category,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_category
)


print("\nComparing GaussianNB vs ComplementNB (5-fold CV)...")

models = {
    "GaussianNB": GaussianNB(),
    "ComplementNB": ComplementNB(alpha=1.0)
}

best_model = None
best_score = 0

for name, clf in models.items():

    cv_scores = cross_val_score(
        clf,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    mean_score = cv_scores.mean()
    std_score = cv_scores.std()

    print(f"{name}: mean={mean_score:.4f}  std={std_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        best_model = clf


print("\nTraining best model...")

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 8))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=np.unique(y_category),
    yticklabels=np.unique(y_category)
)

plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

print("Saved:", f"{OUTPUT_DIR}/confusion_matrix.png")


report = classification_report(y_test, y_pred, output_dict=True)

metrics_df = pd.DataFrame(report).T.round(3)

metrics_path = f"{OUTPUT_DIR}/nb_metrics.csv"

metrics_df.to_csv(metrics_path)

print("Saved:", metrics_path)

model_path = f"{MODEL_DIR}/naive_bayes_model.pkl"

with open(model_path, "wb") as f:

    pickle.dump({
        "model": best_model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names
    }, f)

print("Saved:", model_path)