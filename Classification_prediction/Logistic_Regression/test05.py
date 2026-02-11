import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from scipy import stats

# === 1. 数据读取与准备 ===
file_path = "..\\..\\car+evaluation\\car.data"
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, header=None, names=column_names)

# 检查缺失值
missing_summary = df.isnull().sum()

# 输出每列缺失值数量
print("缺失值检查结果：")
print(missing_summary)

# 查看各类别变量的取值情况
for col in ["doors", "persons"]:
    print(f"{col} 字段的唯一取值为：{df[col].unique()}")



# === 2. 自定义映射（从1开始的语义编码） ===
map_buying = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
map_maint = {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}
map_doors = {'2': 1, '3': 2, '4': 3, '5more': 4}
map_persons = {'2': 1, '4': 2, 'more': 3}
map_lug_boot = {'small': 1, 'med': 2, 'big': 3}
map_safety = {'low': 1, 'med': 2, 'high': 3}
map_acceptability = {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}
class_labels = ['unacc', 'acc', 'good', 'vgood']

# 应用编码
X = df.drop("car_acceptability", axis=1).copy()
X["buying"] = X["buying"].map(map_buying)
X["maint"] = X["maint"].map(map_maint)
X["doors"] = X["doors"].map(map_doors)
X["persons"] = X["persons"].map(map_persons)
X["lug_boot"] = X["lug_boot"].map(map_lug_boot)
X["safety"] = X["safety"].map(map_safety)
y = df["car_acceptability"].map(map_acceptability)

# === 3. 模型与参数（保持原样）===
model = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    penalty='l2',
    C=100.0
)

# === 4. 多次实验 ===
n_trials = 10
accuracies, precisions, recalls, f1s = [], [], [], []

for i in range(n_trials):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred) * 100)
    precisions.append(precision_score(y_test, y_pred, average='macro') * 100)
    recalls.append(recall_score(y_test, y_pred, average='macro') * 100)
    f1s.append(f1_score(y_test, y_pred, average='macro') * 100)

    if i == n_trials - 1:
        print("=== Classification Report (Last Fold) ===")
        print(classification_report(y_test, y_pred, target_names=class_labels))

        print("=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

# === 5. 均值与置信区间 ===
def mean_ci(data, confidence=0.95):
    mean = np.mean(data)
    ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=stats.sem(data))
    return mean, ci

acc_mean, acc_ci = mean_ci(accuracies)
prec_mean, prec_ci = mean_ci(precisions)
recall_mean, recall_ci = mean_ci(recalls)
f1_mean, f1_ci = mean_ci(f1s)

# === 6. 误差条图（保持原图结构）===
models = ['Logistic Regression']
def plot_with_ci(ax, mean, ci, label, xlim):
    err = [[mean - ci[0]], [ci[1] - mean]]
    ax.errorbar([mean], models, xerr=err, fmt='o', capsize=5)
    ax.set_title(label + " (95% CI)")
    ax.set_xlabel(label)
    ax.set_xlim(xlim)
    ax.grid(True)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plot_with_ci(axs[0, 0], acc_mean, acc_ci, "Accuracy", (86, 100))
plot_with_ci(axs[0, 1], prec_mean, prec_ci, "Precision", (70, 100))
plot_with_ci(axs[1, 0], recall_mean, recall_ci, "Recall", (70, 100))
plot_with_ci(axs[1, 1], f1_mean, f1_ci, "F1 Score", (70, 100))
plt.tight_layout()
plt.show()

# === 7. 打印所有指标结果 ===
print(f"[Accuracy]  Mean: {acc_mean:.2f}%, 95% CI: ({acc_ci[0]:.2f}%, {acc_ci[1]:.2f}%)")
print(f"[Precision] Mean: {prec_mean:.2f}%, 95% CI: ({prec_ci[0]:.2f}%, {prec_ci[1]:.2f}%)")
print(f"[Recall]    Mean: {recall_mean:.2f}%, 95% CI: ({recall_ci[0]:.2f}%, {recall_ci[1]:.2f}%)")
print(f"[F1 Score]  Mean: {f1_mean:.2f}%, 95% CI: ({f1_ci[0]:.2f}%, {f1_ci[1]:.2f}%)")
