import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, DBSCAN
from kmodes.kmodes import KModes
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# === 1. Load Data ===
file_path = "..\..\car+evaluation\car.data"  # Replace with your actual path
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, names=column_names)

X_raw = df.drop(columns=["car_acceptability"])
y_true = df["car_acceptability"]

# Encodings
X_label = X_raw.apply(LabelEncoder().fit_transform)
X_onehot = OneHotEncoder().fit_transform(X_raw).toarray()

results = {}
cluster_labels = {}

# === 2. Evaluation Function ===
def evaluate(name, y_pred, X_data):
    try:
        silhouette = silhouette_score(X_data, y_pred) if len(set(y_pred)) > 1 else None
    except:
        silhouette = None
    result = {
        "K值": len(set(y_pred)) - (1 if -1 in y_pred else 0),
        "Silhouette": silhouette,
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "V-Measure": v_measure_score(y_true, y_pred)
    }
    results[name] = result
    cluster_labels[name] = y_pred

# === 3. Clustering Models with Best Params ===
# KMeans (n_clusters=8, init='k-means++', n_init=5, max_iter=100)
kmeans = KMeans(n_clusters=8, init='k-means++', n_init=5, max_iter=100, random_state=42)
evaluate("KMeans", kmeans.fit_predict(X_label), X_label)

# KModes (n_clusters=3, init='Cao', n_init=5)
kmodes = KModes(n_clusters=3, init='Cao', n_init=5, random_state=42)
evaluate("KModes", kmodes.fit_predict(X_label), X_label)

# Agglomerative (n_clusters=8, linkage='ward', affinity='euclidean')
agg = AgglomerativeClustering(n_clusters=8, linkage='ward', affinity='euclidean')
evaluate("Agglomerative", agg.fit_predict(X_label), X_label)

# MeanShift (bandwidth=2.0, bin_seeding=True, cluster_all=True)
meanshift = MeanShift(bandwidth=2.0, bin_seeding=True, cluster_all=True)
evaluate("MeanShift", meanshift.fit_predict(X_label), X_label)

# DBSCAN (eps=0.3, min_samples=3, metric='hamming')
dbscan = DBSCAN(eps=0.3, min_samples=3, metric='hamming')
evaluate("DBSCAN", dbscan.fit_predict(X_onehot), X_onehot)

# === 4. Save Evaluation ===
df_result = pd.DataFrame(results).T.reset_index().rename(columns={"index": "模型"})
df_result.to_csv("all_best_clustering_results.csv", index=False)
print("\n✅ 聚类模型评价表保存为 all_best_clustering_results.csv")
print(df_result.to_string(index=False))

# === 5. PCA Visualization ===
X_pca = PCA(n_components=2).fit_transform(X_label)
plt.figure(figsize=(15, 10))
for i, (name, labels) in enumerate(cluster_labels.items()):
    plt.subplot(2, 3, i+1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(f"{name} Clustering (k={results[name]['K值']})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
plt.tight_layout()
plt.savefig("best_clustering_models_pca.png", dpi=300)
plt.show()

# === 6. Bar Chart of All Scores ===
metrics = ["Silhouette", "ARI", "NMI", "Homogeneity", "Completeness", "V-Measure"]
df_plot = df_result.set_index("模型")[metrics].copy()

plt.figure(figsize=(14, 8))
df_plot.plot(kind="bar", rot=0)
plt.title("最佳参数下各聚类模型的指标对比")
plt.ylabel("得分")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("best_clustering_metrics_comparison.png", dpi=300)
plt.show()
print("✅ 指标对比柱状图保存为 best_clustering_metrics_comparison.png")
