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

# === 1. 加载数据 ===
file_path = "..\..\car+evaluation\car.data"  # 修改为你本地 car.julei 文件路径
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, names=column_names)

X_raw = df.drop(columns=["car_acceptability"])
y_true = df["car_acceptability"]

# 编码
X_label = X_raw.apply(LabelEncoder().fit_transform)
X_onehot = OneHotEncoder().fit_transform(X_raw).toarray()

# 结果保存容器
results = {}
cluster_labels = {}

# === 2. 聚类模型构建与评估 ===
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

# 1. KMeans
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
evaluate("KMeans", kmeans.fit_predict(X_label), X_label)

# 2. KModes
kmodes = KModes(n_clusters=6, init='Huang', n_init=5, random_state=42)
evaluate("KModes", kmodes.fit_predict(X_label), X_label)

# 3. Agglomerative
agg = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
evaluate("Agglomerative", agg.fit_predict(X_label), X_label)

# 4. MeanShift
meanshift = MeanShift()
evaluate("MeanShift", meanshift.fit_predict(X_label), X_label)

# 5. DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=3, metric='hamming')
evaluate("DBSCAN", dbscan.fit_predict(X_onehot), X_onehot)

# === 3. 保存对比表格 ===
df_result = pd.DataFrame(results).T.reset_index().rename(columns={"index": "模型"})
df_result.to_csv("all_clustering_results.csv", index=False)
print("✅ 聚类结果指标保存为 all_clustering_results.csv")
print(df_result.to_string(index=False))

# === 4. 多模型 PCA 可视化 ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_label)

plt.figure(figsize=(15, 10))
for i, (name, labels) in enumerate(cluster_labels.items()):
    plt.subplot(2, 3, i+1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10)
    plt.title(f"{name} Clustering (k={results[name]['K值']})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
plt.tight_layout()
plt.savefig("clustering_all_models_pca.png", dpi=300)
plt.show()
print("✅ 所有模型的 PCA 聚类图保存为 clustering_all_models_pca.png")

# === 5. 指标柱状图对比 ===
metrics = ["Silhouette", "ARI", "NMI", "Homogeneity", "Completeness", "V-Measure"]
df_plot = df_result.set_index("模型")[metrics].copy()

plt.figure(figsize=(14, 8))
df_plot.plot(kind="bar", rot=0)
plt.title("各聚类模型评价指标对比")
plt.ylabel("Score")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("clustering_metrics_comparison.png", dpi=300)
plt.show()
print("✅ 聚类指标对比图保存为 clustering_metrics_comparison.png")
