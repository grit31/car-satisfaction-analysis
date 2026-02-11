import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === 1. 读取数据 ===
file_path = "..\..\car+evaluation\car.data"
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, header=None, names=column_names)

# === 2. 特征和目标变量 ===
X = df.drop("car_acceptability", axis=1)
y = df["car_acceptability"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# === 3. OneHot 编码器（忽略未知类别） ===
categorical_features = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# === 4. 管道模型 ===
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'))
])

# === 5. 划分训练集/测试集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === 6. 网格搜索参数 ===
param_grid = {
    'classifier__C': [0.1, 1.0, 10.0, 100.0],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs']
}

# === 7. 执行 GridSearchCV ===
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='precision_macro', n_jobs=-1)
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
