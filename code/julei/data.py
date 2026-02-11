import pandas as pd
from sklearn.datasets import make_blobs

# === 参数设置 ===
n_samples = 500          # 样本总数
n_features = 2           # 每个样本的特征数
n_clusters = 4           # 聚类簇的个数
cluster_std = 1.2        # 每个簇的标准差（控制聚类的紧凑程度）
random_state = 42        # 保证生成数据一致性

# === 生成聚类数据 ===
X, y = make_blobs(n_samples=n_samples,
                  n_features=n_features,
                  centers=n_clusters,
                  cluster_std=cluster_std,
                  random_state=random_state)

# === 构造 DataFrame 并保存为 CSV ===
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['label'] = y  # 可选：真实簇标签用于可视化或验证

# 保存为 CSV 文件
csv_path = 'synthetic_clustering_data.csv'
df.to_csv(csv_path, index=False)

print(f"✅ 聚类数据已生成并保存为 {csv_path}")
