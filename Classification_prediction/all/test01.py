import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy import stats

# === 1. 数据读取与准备 ===
file_path = "../../car+evaluation/car.data"  # 本地或上传路径
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, header=None, names=column_names)

X = df.drop("car_acceptability", axis=1)
y = df["car_acceptability"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_labels = label_encoder.classes_
categorical_features = X.columns.tolist()

# === 2. 统一预处理器 ===
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# === 3. 定义模型及最优参数 ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', C=100.0),
    "KNN": KNeighborsClassifier(n_neighbors=9, weights='uniform', metric='euclidean', p=1),
    "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=15, min_samples_split=2, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.1, fit_prior=True),
    "Random Forest": RandomForestClassifier(
        criterion='entropy', max_depth=None, max_features=None, min_samples_leaf=1,
        min_samples_split=2, n_estimators=50, random_state=42),
    "SVM": SVC(C=10.0, gamma='scale', kernel='rbf')
}

# === 4. 多次实验（10次） ===
n_trials = 10
metrics_result = {name: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for name in models}

for i in range(n_trials):
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=i)

    for name, clf in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics_result[name]['acc'].append(accuracy_score(y_test, y_pred) * 100)
        metrics_result[name]['prec'].append(precision_score(y_test, y_pred, average='macro') * 100)
        metrics_result[name]['rec'].append(recall_score(y_test, y_pred, average='macro') * 100)
        metrics_result[name]['f1'].append(f1_score(y_test, y_pred, average='macro') * 100)

# === 5. 计算均值与置信区间 ===
def mean_ci(data):
    mean = np.mean(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
    return mean, (mean - ci[0], ci[1] - mean)

summary = {'Model': [], 'Metric': [], 'Mean': [], 'CI Lower': [], 'CI Upper': []}
for model, scores in metrics_result.items():
    for metric_name, values in scores.items():
        mean, (err_low, err_high) = mean_ci(values)
        summary['Model'].append(model)
        summary['Metric'].append(metric_name)
        summary['Mean'].append(mean)
        summary['CI Lower'].append(err_low)
        summary['CI Upper'].append(err_high)

summary_df = pd.DataFrame(summary)

# === 6. 绘制误差条图 ===
metrics = ['acc', 'prec', 'rec', 'f1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, metric in enumerate(metrics):
    df = summary_df[summary_df['Metric'] == metric]
    axs[i].barh(df['Model'], df['Mean'], xerr=[df['CI Lower'], df['CI Upper']], capsize=6)
    axs[i].set_title(f"{metric_labels[i]} (95% CI)")
    axs[i].set_xlim(60, 100)
    axs[i].grid(True)

plt.tight_layout()
plt.savefig("model_comparison_ci_final.png")  # 输出为本地文件
plt.show()
