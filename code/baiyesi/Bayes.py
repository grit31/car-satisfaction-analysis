import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']

# ----------------------------
# 1. 构造模拟文本分类数据
# ----------------------------

# 定义词汇表
spam_words = ["中奖", "免费", "抽奖", "点击", "限时", "促销", "赚钱", "投资", "理财", "广告"]
ham_words = ["会议", "报告", "项目", "进展", "同事", "客户", "请查收", "文件", "安排", "计划"]

def generate_email(is_spam, n_words=20):
    words = []
    if is_spam:
        # 垃圾邮件高频词+少量正常词
        words += random.choices(spam_words, k=random.randint(10, 16))
        words += random.choices(ham_words, k=n_words - len(words))
    else:
        # 正常邮件高频词+少量垃圾词
        words += random.choices(ham_words, k=random.randint(10, 16))
        words += random.choices(spam_words, k=n_words - len(words))
    random.shuffle(words)
    return " ".join(words)

# 生成数据集
n_samples = 500
data = []
for i in range(n_samples):
    label = 1 if i < n_samples // 2 else 0  # 1=垃圾邮件, 0=正常邮件
    text = generate_email(label)
    data.append([text, label])

df = pd.DataFrame(data, columns=['text', 'label'])
print("数据集预览：")
print(df.head())

# ----------------------------
# 2. 划分训练集和测试集
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)

# ----------------------------
# 3. 文本向量化（词袋模型）
# ----------------------------

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# 4. 朴素贝叶斯分类
# ----------------------------

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# ----------------------------
# 5. 结果评估与可视化
# ----------------------------

print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=["正常邮件", "垃圾邮件"]))
print("准确率：", accuracy_score(y_test, y_pred))

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["正常邮件", "垃圾邮件"],
            yticklabels=["正常邮件", "垃圾邮件"])
plt.xlabel("预测")
plt.ylabel("真实")
plt.title("朴素贝叶斯文本分类混淆矩阵")
plt.savefig("朴素贝叶斯文本分类混淆矩阵.png")
plt.show()
