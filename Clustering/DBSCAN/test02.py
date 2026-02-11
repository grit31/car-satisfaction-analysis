import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 1. 加载数据
file_path = "..\\..\\car+evaluation\\car.data"
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, names=column_names)

# 2. 自定义语义编码（从 1 开始）
map_buying = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
map_maint = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
map_doors = {'2': 1, '3': 2, '4': 3, '5more': 4}
map_persons = {'2': 1, '4': 2, 'more': 3}
map_lug_boot = {'small': 1, 'med': 2, 'big': 3}
map_safety = {'low': 1, 'med': 2, 'high': 3}
map_acceptability = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}

X = df.drop(columns=["car_acceptability"]).copy()
X["buying"] = X["buying"].map(map_buying)
X["maint"] = X["maint"].map(map_maint)
X["doors"] = X["doors"].map(map_doors)
X["persons"] = X["persons"].map(map_persons)
X["lug_boot"] = X["lug_boot"].map(map_lug_boot)
X["safety"] = X["safety"].map(map_safety)
y_true = df["car_acceptability"].map(map_acceptability)

# 3. 构建 DBSCAN 模型
dbscan = DBSCAN(eps=0.8, min_samples=5, metric='euclidean')
y_pred = dbscan.fit_predict(X)

# 4. 获取聚类数（排除噪声）
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
print(f"✅ DBSCAN 聚出簇数: {n_clusters}")

# 5. PCA 降维
X_pca = PCA(n_components=2).fit_transform(X)

# 6. 可视化对比
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='tab10', s=10)
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
plt.savefig("dbscan_cluster_comparison.png", dpi=300)
plt.show()

# 7. 聚类评估函数
def evaluate_clustering(y_true, y_pred):
    try:
        silhouette = silhouette_score(X, y_pred) if len(set(y_pred)) > 1 else None
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

# 8. 保存聚类评价指标
comparison_df.to_csv("dbscan_comparison_results.csv", index=False)
print("✅ 已保存聚类效果表为 'dbscan_comparison_results.csv'")
print(comparison_df.to_string(index=False))

# 9. 模拟“聚类中心”：使用每类样本均值
df_clusters = X.copy()
df_clusters["cluster"] = y_pred
cluster_centers = df_clusters[df_clusters["cluster"] != -1].groupby("cluster").mean().reset_index(drop=True)

# 10. 打印未解码的聚类“中心”
print("\n📌 DBSCAN 各簇均值特征（模拟中心 - 编码数值）:")
print(cluster_centers.round(2).to_string(index=False))

# 11. 解码函数
def decode_centroids(centroids):
    reverse_maps = {
        'buying': {v: k for k, v in map_buying.items()},
        'maint': {v: k for k, v in map_maint.items()},
        'doors': {v: k for k, v in map_doors.items()},
        'persons': {v: k for k, v in map_persons.items()},
        'lug_boot': {v: k for k, v in map_lug_boot.items()},
        'safety': {v: k for k, v in map_safety.items()}
    }
    decoded = []
    for row in centroids.values:
        decoded_row = []
        for idx, col in enumerate(X.columns):
            val = int(round(row[idx]))
            decoded_val = reverse_maps[col].get(val, "(?)")
            decoded_row.append(decoded_val)
        decoded.append(decoded_row)
    return pd.DataFrame(decoded, columns=X.columns)

decoded_centroids = decode_centroids(cluster_centers)
print("\n📌 DBSCAN 各簇中心（解码后）:")
print(decoded_centroids.to_string(index=False))

# 12. 保存中心对比结果
decoded_centroids["聚类模型"] = f"DBSCAN (k={n_clusters})"
decoded_centroids.to_csv("centroids_dbscan.csv", index=False, encoding='utf-8-sig')
print("✅ 已保存聚类中心对比表为 'centroids_dbscan.csv'")

# 13. 打印每类样本数量（不含噪声）
print("\n📦 每类样本数量（不含噪声）：")
cluster_counts = df_clusters[df_clusters["cluster"] != -1]["cluster"].value_counts().sort_index()
for i, count in cluster_counts.items():
    print(f"Cluster {i}: {count} 个样本")

# 14. 可视化样本分布柱状图
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(cluster_counts.index, cluster_counts.values, color='mediumseagreen')
ax.set_title(f"DBSCAN 聚类样本分布（k={n_clusters}）")
ax.set_xlabel("簇编号")
ax.set_ylabel("样本数量")
ax.set_xticks(cluster_counts.index)
for i, v in enumerate(cluster_counts.values):
    ax.text(cluster_counts.index[i], v + 5, str(v), ha='center')
plt.tight_layout()
plt.savefig("dbscan_cluster_distribution.png", dpi=300)
plt.show()

print("✅ 已保存聚类样本分布柱状图为 'dbscan_cluster_distribution.png'")
