import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats

# === 1. 数据读取与准备 ===
file_path = "../../car+evaluation/car.data"  # 确保路径正确
column_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "car_acceptability"]
df = pd.read_csv(file_path, header=None, names=column_names)

X = df.drop("car_acceptability", axis=1)
y = df["car_acceptability"]
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_labels = label_encoder.classes_

categorical_features = X.columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# === 使用决策树 + 最优参数 ===
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        criterion='gini',
        max_depth=15,
        min_samples_split=2,
        random_state=42
    ))
])

# === 2. 多次实验 ===
n_trials = 10
accuracies, precisions, recalls, f1s = [], [], [], []

for i in range(n_trials):
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=i)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred) * 100)
    precisions.append(precision_score(y_test, y_pred, average='macro') * 100)
    recalls.append(recall_score(y_test, y_pred, average='macro') * 100)
    f1s.append(f1_score(y_test, y_pred, average='macro') * 100)

    # 打印一次报告（最后一次）
    if i == n_trials - 1:
        print("=== Classification Report (Last Fold) ===")
        print(classification_report(y_test, y_pred, target_names=class_labels))

        print("=== Confusion Matrix ===")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Decision Tree Confusion Matrix")
        plt.savefig('Decision Tree Confusion Matrix.png')
        plt.show()

# === 3. 均值与置信区间 ===
def mean_ci(data, confidence=0.95):
    mean = np.mean(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=stats.sem(data))
    return mean, ci

acc_mean, acc_ci = mean_ci(accuracies)
prec_mean, prec_ci = mean_ci(precisions)
recall_mean, recall_ci = mean_ci(recalls)
f1_mean, f1_ci = mean_ci(f1s)

# === 4. 误差条图 ===
models = ['Decision Tree']
def plot_with_ci(ax, mean, ci, label, xlim):
    err = [[mean - ci[0]], [ci[1] - mean]]
    ax.errorbar([mean], models, xerr=err, fmt='o', capsize=5)
    ax.set_title(label + " (95% CI)")
    ax.set_xlabel(label)
    ax.set_xlim(xlim)
    ax.grid(True)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plot_with_ci(axs[0,0], acc_mean, acc_ci, "Accuracy", (86, 100))
plot_with_ci(axs[0,1], prec_mean, prec_ci, "Precision", (70, 100))
plot_with_ci(axs[1,0], recall_mean, recall_ci, "Recall", (70, 100))
plot_with_ci(axs[1,1], f1_mean, f1_ci, "F1 Score", (70, 100))
plt.tight_layout()
plt.savefig('Decision Tree Error bar graph.png')
plt.show()

# === 5. 打印所有指标结果 ===
print(f"[Accuracy]  Mean: {acc_mean:.2f}%, 95% CI: ({acc_ci[0]:.2f}%, {acc_ci[1]:.2f}%)")
print(f"[Precision] Mean: {prec_mean:.2f}%, 95% CI: ({prec_ci[0]:.2f}%, {prec_ci[1]:.2f}%)")
print(f"[Recall]    Mean: {recall_mean:.2f}%, 95% CI: ({recall_ci[0]:.2f}%, {recall_ci[1]:.2f}%)")
print(f"[F1 Score]  Mean: {f1_mean:.2f}%, 95% CI: ({f1_ci[0]:.2f}%, {f1_ci[1]:.2f}%)")
