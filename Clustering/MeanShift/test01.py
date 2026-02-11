import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools

# 1. 加载数据
file_path = "..\\..\\car+evaluation\\car.data"  # 替换为你的文件路径
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, names=column_names)

# 2. 特征编码
X = df.drop(columns=["car_acceptability"])
y_true = df["car_acceptability"]
X_encoded = X.apply(LabelEncoder().fit_transform)

# 3. 网格搜索最佳 MeanShift 参数组合
bandwidths = [0.5, 1.0, 1.5, 2.0]
bin_seedings = [True, False]
cluster_alls = [True, False]

param_grid = {
    "bandwidth": bandwidths,
    "bin_seeding": bin_seedings,
    "cluster_all": cluster_alls
}

best_score = -1
best_params = None

for params in itertools.product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    try:
        model = MeanShift(**param_dict)
        labels = model.fit_predict(X_encoded)
        if len(set(labels)) < 2:
            continue  # 忽略只有一个簇的情况
        score = silhouette_score(X_encoded, labels)
        if score > best_score:
            best_score = score
            best_params = param_dict
    except Exception:
        continue  # 忽略出错组合

print("✅ 最佳参数组合:", best_params)
print("✅ 最佳Silhouette得分:", best_score)

# 4. 构建“参考 k=4”模型 —— 使用默认 MeanShift
default_bandwidth = estimate_bandwidth(X_encoded, quantile=0.2, n_samples=500)
meanshift_4 = MeanShift(bandwidth=default_bandwidth, bin_seeding=True)
y_pred_4 = meanshift_4.fit_predict(X_encoded)

# 5. 使用最佳参数构建模型
meanshift_best = MeanShift(**best_params)
y_pred_best = meanshift_best.fit_predict(X_encoded)
best_k = len(np.unique(y_pred_best))  # 获取聚类数

# 6. PCA 降维
X_pca = PCA(n_components=2).fit_transform(X_encoded)

# 7. 可视化对比
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].scatter(X_pca[:, 0], X_pca[:, 1], c=LabelEncoder().fit_transform(y_true), cmap='tab10', s=10)
axs[0].set_title('True Labels (PCA Projection)')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].grid(True)

axs[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_4, cmap='tab10', s=10)
axs[1].set_title('MeanShift Default')
axs[1].set_xlabel('PC1')
axs[1].set_ylabel('PC2')
axs[1].grid(True)

axs[2].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_best, cmap='tab10', s=10)
axs[2].set_title(f'MeanShift Best (k={best_k})')
axs[2].set_xlabel('PC1')
axs[2].set_ylabel('PC2')
axs[2].grid(True)

plt.tight_layout()
plt.savefig("meanshift_cluster_comparison.png", dpi=300)
plt.show()

# 8. 聚类评价函数
def evaluate_clustering(y_true, y_pred):
    return {
        "Silhouette": silhouette_score(X_encoded, y_pred) if len(set(y_pred)) > 1 else None,
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
    "K值": [len(np.unique(y_pred_4)), best_k],
    "Silhouette": [results_4["Silhouette"], results_best["Silhouette"]],
    "ARI": [results_4["ARI"], results_best["ARI"]],
    "NMI": [results_4["NMI"], results_best["NMI"]],
    "Homogeneity": [results_4["Homogeneity"], results_best["Homogeneity"]],
    "Completeness": [results_4["Completeness"], results_best["Completeness"]],
    "V-Measure": [results_4["V-Measure"], results_best["V-Measure"]]
})

# 10. 保存为CSV文件
comparison_df.to_csv("meanshift_comparison_results.csv", index=False)
print("✅ 已保存聚类效果对比表为 'meanshift_comparison_results.csv'")
print(comparison_df.to_string(index=False))
