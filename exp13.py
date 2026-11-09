# EXPERIMENT 13: K-Means Clustering on Digits Dataset

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load and scale dataset
X = load_digits().data
X_scaled = StandardScaler().fit_transform(X)

# Apply K-Means for different K values
K_range = range(2, 15)
inertia_values, silhouette_scores = [], []

for k in K_range:
    model = KMeans(n_clusters=k, init='k-means++', random_state=0)
    labels = model.fit_predict(X_scaled)
    inertia_values.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot and save Elbow Method graph
plt.figure()
plt.plot(K_range, inertia_values, marker='o')
plt.title("Elbow Method (Inertia vs K)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.savefig("exp13_elbow_method_inertia.png", bbox_inches="tight")

# Plot and save Silhouette Score graph
plt.figure()
plt.plot(K_range, silhouette_scores, marker='o')
plt.title("Silhouette Score vs K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.savefig("exp13_silhouette_score_vs_k.png", bbox_inches="tight")

# Print values
for k, inertia, score in zip(K_range, inertia_values, silhouette_scores):
    print(f"K={k}, Inertia={inertia:.2f}, Silhouette={score:.4f}")
