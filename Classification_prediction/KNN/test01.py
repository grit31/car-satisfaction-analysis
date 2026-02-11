import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === 1. 读取数据 ===
file_path = "../../car+evaluation/car.data"  # 路径可根据你的实际调整
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, header=None, names=column_names)

# === 2. 特征和目标变量 ===
X = df.drop("car_acceptability", axis=1)
y = df["car_acceptability"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === 3. OneHot 编码 ===
categorical_features = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# === 4. KNN 管道 ===
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

# === 5. 划分训练集/测试集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === 6. 扩展的网格搜索参数 ===
param_grid = {
    'classifier__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'classifier__p': [1, 2, 3]  # 仅在 minkowski 距离下生效，但 GridSearchCV 会自动忽略不兼容组合
}

# === 7. 执行 GridSearchCV ===
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='precision_macro',
    n_jobs=-1,
    verbose=2  # 加上这个你可以看到搜索进度
)
grid_search.fit(X_train, y_train)

# === 8. 使用最佳模型评估 ===
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')

# === 9. 输出结果 ===
print(f"Best Params: {grid_search.best_params_}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
