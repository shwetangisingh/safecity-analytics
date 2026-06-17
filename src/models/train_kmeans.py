import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, get_geo_features

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/kmeans", exist_ok=True)

print("Loading data...")
df = load_data()
X_geo, df_valid = get_geo_features(df)

X_latlon = X_geo[:, :2]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_latlon)

print("Running elbow method (k=2..12)...")
inertias, sil_scores = [], []
k_range = range(2, 13)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=10000, random_state=RANDOM_SEED)
    sil_scores.append(sil)
    print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

best_k = list(k_range)[np.argmax(sil_scores)]
print(f"\nBest k = {best_k}  (silhouette = {max(sil_scores):.4f})")

km_final = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
cluster_labels = km_final.fit_predict(X_scaled)
df_valid = df_valid.copy()
df_valid["Cluster"] = cluster_labels

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(k_range), inertias, marker="o", color="#2E86AB", linewidth=2)
axes[0].set_xlabel("Number of Clusters (k)", fontsize=12)
axes[0].set_ylabel("Inertia (WCSS)", fontsize=12)
axes[0].set_title("Elbow Method", fontsize=14)
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), sil_scores, marker="s", color="#E84855", linewidth=2)
axes[1].axvline(best_k, color="green", linestyle="--", label=f"Best k={best_k}")
axes[1].set_xlabel("Number of Clusters (k)", fontsize=12)
axes[1].set_ylabel("Silhouette Score", fontsize=12)
axes[1].set_title("Silhouette Scores", fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/kmeans/kmeans_elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/kmeans/kmeans_elbow_silhouette.png")

fig2, ax2 = plt.subplots(figsize=(12, 9))
colors = cm.tab10(np.linspace(0, 1, best_k))

for c in range(best_k):
    mask = df_valid["Cluster"] == c
    ax2.scatter(
        df_valid.loc[mask, "LON"],
        df_valid.loc[mask, "LAT"],
        s=0.5, alpha=0.3, color=colors[c], label=f"Cluster {c}"
    )

centres_orig = scaler.inverse_transform(km_final.cluster_centers_)
ax2.scatter(
    centres_orig[:, 1], centres_orig[:, 0],
    s=200, marker="X", color="black", zorder=5, label="Centroids"
)
ax2.set_xlabel("Longitude", fontsize=12)
ax2.set_ylabel("Latitude", fontsize=12)
ax2.set_title(f"Crime Hotspot Clusters (k={best_k}) — Los Angeles", fontsize=14)
ax2.legend(markerscale=6, loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("outputs/kmeans/kmeans_geo_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/kmeans/kmeans_geo_clusters.png")

cluster_crime = (
    df_valid.groupby(["Cluster", "Crime Category"])
    .size().unstack(fill_value=0)
)
cluster_crime_pct = cluster_crime.div(cluster_crime.sum(axis=1), axis=0) * 100

fig3, ax3 = plt.subplots(figsize=(14, 6))
cluster_crime_pct.plot(kind="bar", stacked=True, ax=ax3, colormap="tab10")
ax3.set_xlabel("Cluster", fontsize=12)
ax3.set_ylabel("% of Crimes", fontsize=12)
ax3.set_title("Crime Category Composition per Cluster", fontsize=14)
ax3.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/kmeans/kmeans_cluster_composition.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/kmeans/kmeans_cluster_composition.png")

sil_final = silhouette_score(X_scaled, cluster_labels, sample_size=10000,
                              random_state=RANDOM_SEED)
metrics = pd.DataFrame({
    "k": [best_k],
    "Inertia": [km_final.inertia_],
    "Silhouette Score": [round(sil_final, 4)],
})
metrics.to_csv("outputs/kmeans/kmeans_metrics.csv", index=False)
print(f"\nFinal silhouette score: {sil_final:.4f}")
print("Saved: outputs/kmeans/kmeans_metrics.csv")

with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump({"model": km_final, "scaler": scaler, "best_k": best_k}, f)
print("Saved: models/kmeans_model.pkl")
