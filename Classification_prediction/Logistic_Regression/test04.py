import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy import stats

# === 1. 数据读取与准备 ===
file_path = "..\\..\\car+evaluation\\car.data"
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, header=None, names=column_names)

# === 2. 自定义映射（从1开始的语义编码） ===
map_buying = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
map_maint = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
map_doors = {'2': 1, '3': 2, '4': 3, '5more': 4}
map_persons = {'2': 1, '4': 2, 'more': 3}
map_lug_boot = {'small': 1, 'med': 2, 'big': 3}
map_safety = {'low': 1, 'med': 2, 'high': 3}
map_acceptability = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}
class_labels = ['unacc', 'acc', 'good', 'vgood']

# === 3. 应用映射 ===
X = df.drop(columns=["car_acceptability"]).copy()
X["buying"] = X["buying"].map(map_buying)
X["maint"] = X["maint"].map(map_maint)
X["doors"] = X["doors"].map(map_doors)
X["persons"] = X["persons"].map(map_persons)
X["lug_boot"] = X["lug_boot"].map(map_lug_boot)
X["safety"] = X["safety"].map(map_safety)

y = df["car_acceptability"].map(map_acceptability)

# === 4. 定义模型与存储容器 ===
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', penalty='l2', C=100.0),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5)
}

n_trials = 10
results = {model_name: {"acc": [], "prec": [], "rec": [], "f1": []} for model_name in models}

# === 5. 多次训练和评估 ===
for i in range(n_trials):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    for model_name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        results[model_name]["acc"].append(accuracy_score(y_test, y_pred) * 100)
        results[model_name]["prec"].append(precision_score(y_test, y_pred, average='macro') * 100)
        results[model_name]["rec"].append(recall_score(y_test, y_pred, average='macro') * 100)
        results[model_name]["f1"].append(f1_score(y_test, y_pred, average='macro') * 100)

    # 最后一次打印报告和混淆矩阵
    if i == n_trials - 1:
        print("=== Classification Report (Last Trial: Logistic Regression) ===")
        print(classification_report(y_test, y_pred, target_names=class_labels))

        print("=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.show()

# === 6. 计算均值和置信区间 ===
def mean_ci(data):
    mean = np.mean(data)
    ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=stats.sem(data))
    lower = mean - ci[0]
    upper = ci[1] - mean
    return mean, (lower, upper)

metrics = ['acc', 'prec', 'rec', 'f1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# === 7. 绘图对比 ===
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()

for idx, metric in enumerate(metrics):
    ax = axs[idx]
    means = []
    errors = [[], []]
    labels = []

    for model_name in models:
        mean, (lower_err, upper_err) = mean_ci(results[model_name][metric])
        means.append(mean)
        errors[0].append(lower_err)
        errors[1].append(upper_err)
        labels.append(model_name)

    ax.barh(labels, means, xerr=errors, capsize=5)
    ax.set_title(f"{metric_names[idx]} (95% CI)")
    ax.set_xlim(60, 100)
    ax.grid(True)

plt.tight_layout()
plt.show()

# === 8. 打印评估汇总 ===
for model_name in models:
    print(f"\n== {model_name} Evaluation ==")
    for metric in metrics:
        mean, (lower_err, upper_err) = mean_ci(results[model_name][metric])
        print(f"{metric.upper():<6}: {mean:.2f}% ± {max(lower_err, upper_err):.2f}%")
