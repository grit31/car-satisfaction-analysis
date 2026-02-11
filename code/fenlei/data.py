import pandas as pd
from sklearn.datasets import make_classification

# === 参数设置 ===
n_samples = 500           # 样本总数
n_features = 6            # 总特征数（全部为数值型）
n_informative = 4         # 有效特征数
n_redundant = 1           # 冗余特征数（由其他特征线性组合而来）
n_classes = 3             # 分类类别数（可改为2表示二分类）
random_state = 42         # 保证生成数据一致性

# === 生成数据 ===
X, y = make_classification(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           n_redundant=n_redundant,
                           n_classes=n_classes,
                           random_state=random_state)

# === 构造 DataFrame ===
feature_names = [f'feature{i+1}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y  # 添加目标标签列

# === 保存为 CSV 文件 ===
csv_path = 'synthetic_classification_data.csv'
df.to_csv(csv_path, index=False)

print(f"✅ 分类数据已生成并保存为 {csv_path}")
