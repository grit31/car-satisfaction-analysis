import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = "..\\..\\car+evaluation\\car.data"  # 替换为你的文件路径
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, names=column_names)

# 2. 特征编码（使用 OneHot）
X = df.drop(columns=["car_acceptability"])
y_true = df["car_acceptability"]
X_onehot = OneHotEncoder().fit_transform(X).toarray()

# 3. 构建 DBSCAN 模型（Hamming 距离）
dbscan = DBSCAN(eps=0.3, min_samples=3, metric='hamming')
y_pred = dbscan.fit_predict(X_onehot)
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
print(f"✅ DBSCAN 聚出簇数: {n_clusters}")

# 4. PCA 降维
X_pca = PCA(n_components=2).fit_transform(X_onehot)

# 5. 可视化
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=LabelEncoder().fit_transform(y_true), cmap='tab10', s=10)
axs[0].set_title('True Labels (PCA Projection)')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].grid(True)

axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='tab10', s=10)
axs[1].set_title(f'DBSCAN Clustering (k={n_clusters})')
axs[1].set_xlabel('PC1')
axs[1].set_ylabel('PC2')
axs[1].grid(True)

plt.tight_layout()
plt.savefig("dbscan_hamming_clustering.png", dpi=300)
plt.show()

# 6. 聚类评价函数
def evaluate_clustering(y_true, y_pred):
    try:
        silhouette = silhouette_score(X_onehot, y_pred) if len(set(y_pred)) > 1 else None
    except:
        silhouette = None
    return {
        "Silhouette": silhouette,
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "V-Measure": v_measure_score(y_true, y_pred)
    }

# 7. 构建评价结果
results = evaluate_clustering(y_true, y_pred)
comparison_df = pd.DataFrame([{
    "K值": n_clusters,
    "Silhouette": results["Silhouette"],
    "ARI": results["ARI"],
    "NMI": results["NMI"],
    "Homogeneity": results["Homogeneity"],
    "Completeness": results["Completeness"],
    "V-Measure": results["V-Measure"]
}])

# 8. 保存为 CSV
comparison_df.to_csv("dbscan_hamming_results.csv", index=False)
print("✅ 已保存聚类效果表为 'dbscan_hamming_results.csv'")
print(comparison_df.to_string(index=False))
