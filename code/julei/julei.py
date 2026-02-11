import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# === 步骤1：加载数据 ===
csv_path = 'synthetic_clustering_data.csv'  # 请根据实际路径修改
df = pd.read_csv(csv_path)

# 选择特征列用于聚类
X = df[['feature1', 'feature2']]

# === 步骤2：K-Means 聚类 ===
n_clusters = 4  # 根据之前生成数据时的中心数设置
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)  # 添加聚类结果列

# 聚类中心坐标
centers = kmeans.cluster_centers_

# === 步骤3：可视化聚类结果 ===
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['feature1'], df['feature2'], c=df['cluster'], cmap='viridis', s=50, alpha=0.7, label='Sample Points')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Cluster Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering Result')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
