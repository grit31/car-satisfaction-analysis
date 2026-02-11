import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools
# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
# 1. 加载数据
file_path = "..\\..\\car+evaluation\\car.data"  # 替换为你的文件路径
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, names=column_names)

# 2. 自定义映射（从1开始的语义编码）
map_buying = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
map_maint = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
map_doors = {'2': 1, '3': 2, '4': 3, '5more': 4}
map_persons = {'2': 1, '4': 2, 'more': 3}
map_lug_boot = {'small': 1, 'med': 2, 'big': 3}
map_safety = {'low': 1, 'med': 2, 'high': 3}
map_acceptability = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}

# 应用编码
X = df.drop(columns=["car_acceptability"]).copy()
X["buying"] = X["buying"].map(map_buying)
X["maint"] = X["maint"].map(map_maint)
X["doors"] = X["doors"].map(map_doors)
X["persons"] = X["persons"].map(map_persons)
X["lug_boot"] = X["lug_boot"].map(map_lug_boot)
X["safety"] = X["safety"].map(map_safety)

# 标签（用于评估）
y_true = df["car_acceptability"].map(map_acceptability)

# 3. 网格搜索最佳参数组合
param_grid = {
    "n_clusters": range(2, 11),
    "init": ['k-means++', 'random'],
    "n_init": [5, 10, 20],
    "max_iter": [100, 300]
}

best_score = -1
best_params = None

for params in itertools.product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    kmeans_model = KMeans(random_state=42, **param_dict)
    labels = kmeans_model.fit_predict(X)
    score = silhouette_score(X, labels)
    if score > best_score:
        best_score = score
        best_params = param_dict

print("✅ 最佳参数组合:", best_params)
print("✅ 最佳Silhouette得分:", best_score)

# 4. 构建 k=4 模型
kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred_4 = kmeans_4.fit_predict(X)

# 5. 使用最优参数构建最佳模型
kmeans_best = KMeans(random_state=42, **best_params)
y_pred_best = kmeans_best.fit_predict(X)
best_k = best_params['n_clusters']

# 6. PCA 降维
X_pca = PCA(n_components=2).fit_transform(X)

# 7. 可视化对比
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='tab10', s=10)
axs[0].set_title('True Labels (PCA Projection)')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].grid(True)

axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_4, cmap='tab10', s=10)
axs[1].set_title('K-Means Clusters (k=4)')
axs[1].set_xlabel('PC1')
axs[1].set_ylabel('PC2')
axs[1].grid(True)

axs[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_best, cmap='tab10', s=10)
axs[2].set_title(f'K-Means Clusters (Best k={best_k})')
axs[2].set_xlabel('PC1')
axs[2].set_ylabel('PC2')
axs[2].grid(True)

plt.tight_layout()
plt.savefig("kmeans_cluster_comparison.png", dpi=300)
plt.show()

# 8. 聚类评价函数
def evaluate_clustering(y_true, y_pred):
    return {
        "Silhouette": silhouette_score(X, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "V-Measure": v_measure_score(y_true, y_pred)
    }

# 9. 构建评价结果
results_4 = evaluate_clustering(y_true, y_pred_4)
results_best = evaluate_clustering(y_true, y_pred_best)

comparison_df = pd.DataFrame({
    "K值": [4, best_k],
    "Silhouette": [results_4["Silhouette"], results_best["Silhouette"]],
    "ARI": [results_4["ARI"], results_best["ARI"]],
    "NMI": [results_4["NMI"], results_best["NMI"]],
    "Homogeneity": [results_4["Homogeneity"], results_best["Homogeneity"]],
    "Completeness": [results_4["Completeness"], results_best["Completeness"]],
    "V-Measure": [results_4["V-Measure"], results_best["V-Measure"]]
})

# 10. 保存为 CSV
comparison_df.to_csv("kmeans_comparison_results.csv", index=False)
print("✅ 已保存聚类效果对比表为 'kmeans_comparison_results.csv'")
print(comparison_df.to_string(index=False))

# 11. 打印聚类中心（未解码）
print("\n📌 K=4 的聚类中心（编码数值）:")
print(pd.DataFrame(kmeans_4.cluster_centers_, columns=X.columns).round(2).to_string(index=False))

print(f"\n📌 最佳K={best_k} 的聚类中心（编码数值）:")
print(pd.DataFrame(kmeans_best.cluster_centers_, columns=X.columns).round(2).to_string(index=False))

# 12. 打印聚类中心（解码后）
def decode_centroids(centroids):
    decoded = []
    reverse_maps = {
        'buying': {v: k for k, v in map_buying.items()},
        'maint': {v: k for k, v in map_maint.items()},
        'doors': {v: k for k, v in map_doors.items()},
        'persons': {v: k for k, v in map_persons.items()},
        'lug_boot': {v: k for k, v in map_lug_boot.items()},
        'safety': {v: k for k, v in map_safety.items()}
    }
    for row in centroids:
        decoded_row = []
        for idx, col in enumerate(X.columns):
            val = int(round(row[idx]))
            decoded_val = reverse_maps[col].get(val, f"(?)")
            decoded_row.append(decoded_val)
        decoded.append(decoded_row)
    return pd.DataFrame(decoded, columns=X.columns)

centroids_k4_decoded = decode_centroids(kmeans_4.cluster_centers_)
centroids_best_decoded = decode_centroids(kmeans_best.cluster_centers_)

print("\n📌 K=4 的聚类中心（解码后）:")
print(centroids_k4_decoded.to_string(index=False))

print(f"\n📌 最佳K={best_k} 的聚类中心（解码后）:")
print(centroids_best_decoded.to_string(index=False))

# 合并显示
print("\n📊 聚类中心对比分析（解码后合并显示）:")
centroids_k4_decoded["聚类模型"] = "K=4"
centroids_best_decoded["聚类模型"] = f"K={best_k}"
combined_centroids = pd.concat([centroids_k4_decoded, centroids_best_decoded], ignore_index=True)
print(combined_centroids.to_string(index=False))

# 13. 保存聚类中心为 CSV
combined_centroids.to_csv("centroids_comparison.csv", index=False, encoding='utf-8-sig')
print("✅ 已保存聚类中心对比表为 'centroids_comparison.csv'")

# 14. 打印每类样本数量统计
print("\n📦 K=4 聚类模型的每类样本数量：")
cluster_counts_k4 = pd.Series(y_pred_4).value_counts().sort_index()
for i, count in cluster_counts_k4.items():
    print(f"Cluster {i}: {count} 个样本")

print(f"\n📦 最佳K={best_k} 聚类模型的每类样本数量：")
cluster_counts_best = pd.Series(y_pred_best).value_counts().sort_index()
for i, count in cluster_counts_best.items():
    print(f"Cluster {i}: {count} 个样本")

# 15. 可视化聚类样本数量分布
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 左图 - K=4
axs[0].bar(cluster_counts_k4.index, cluster_counts_k4.values, color='skyblue')
axs[0].set_title("K=4 聚类样本分布")
axs[0].set_xlabel("簇编号")
axs[0].set_ylabel("样本数量")
axs[0].set_xticks(cluster_counts_k4.index)
for i, v in enumerate(cluster_counts_k4.values):
    axs[0].text(i, v + 5, str(v), ha='center')

# 右图 - Best K
axs[1].bar(cluster_counts_best.index, cluster_counts_best.values, color='salmon')
axs[1].set_title(f"K={best_k} 聚类样本分布")
axs[1].set_xlabel("簇编号")
axs[1].set_ylabel("样本数量")
axs[1].set_xticks(cluster_counts_best.index)
for i, v in enumerate(cluster_counts_best.values):
    axs[1].text(i, v + 5, str(v), ha='center')

plt.tight_layout()
plt.savefig("kmeans_cluster_sample_distribution.png", dpi=300)
plt.show()

print("✅ 已保存聚类样本分布柱状图为 'cluster_sample_distribution.png'")
