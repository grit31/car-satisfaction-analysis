import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

# === 步骤1：加载数据 ===
csv_path = 'synthetic_classification_data.csv'  # 文件路径
df = pd.read_csv(csv_path)

X = df.drop(columns=['label'])
y = df['label']

# === 步骤2：拆分训练集与测试集 ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# === 步骤3：训练 KNN 分类器 ===
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# === 步骤4：评估模型性能 ===
print("📋 Classification Report:\n", classification_report(y_test, y_pred))
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 步骤5：PCA降维并可视化分类结果 ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 预测整个数据集标签用于可视化
full_pred = knn.predict(X)

# === 可视化图像 ===
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=full_pred, cmap='Set1', s=40, alpha=0.7)
plt.title('KNN Classification Result (PCA-reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.colorbar(scatter, ticks=range(len(y.unique())), label='Predicted Class')
plt.tight_layout()
plt.show()
