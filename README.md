# 🚗 汽车满意度数据集分析

## 项目简介

本项目基于UCI Machine Learning Repository的Car Evaluation数据集，通过机器学习方法对汽车满意度进行深入分析。项目包含两大核心任务：

1. **分类预测任务**：构建多种监督学习模型，预测用户对汽车的满意度等级
2. **聚类分析任务**：应用无监督学习方法，发现用户群体特征和行为模式

通过多模型对比实验，本项目探索了不同算法在类别型数据上的表现，为汽车行业的用户洞察和产品设计提供数据支持。

---

## 📊 数据集说明

### 数据来源
- **数据集名称**：Car Evaluation Database
- **来源**：UCI Machine Learning Repository
- **样本数量**：1728条
- **特征类型**：全部为类别型特征
- **任务类型**：多分类问题

### 特征描述

| 特征名称 | 含义 | 取值范围 |
|---------|------|---------|
| buying | 购买价格 | vhigh, high, med, low |
| maint | 维护成本 | vhigh, high, med, low |
| doors | 车门数量 | 2, 3, 4, 5more |
| persons | 载客容量 | 2, 4, more |
| lug_boot | 后备箱大小 | small, med, big |
| safety | 安全性 | low, med, high |

### 目标变量

**car_acceptability**（汽车可接受度）：
- `unacc`：不可接受
- `acc`：可接受
- `good`：良好
- `vgood`：非常好

### 数据预处理

- **缺失值处理**：数据集无缺失值
- **特征编码**：
  - 使用 OneHotEncoder 对类别特征进行编码
  - 使用 LabelEncoder 对目标变量进行编码
- **数据划分**：80%训练集，20%测试集（多次随机划分取平均）

---

## 📁 项目结构

```
car-satisfaction-analysis/
├── car+evaluation/              # 原始数据集
│   ├── car.data                # 数据文件
│   ├── car.names               # 数据集说明
│   └── car.c45-names           # C4.5格式说明
│
├── Classification_prediction/   # 分类预测任务
│   ├── Logistic_Regression/    # 逻辑回归
│   ├── KNN/                    # K近邻
│   ├── Decision_tree/          # 决策树
│   ├── Naive_Bayes/            # 朴素贝叶斯
│   ├── Random_forest/          # 随机森林
│   ├── Support_vector_machine/ # 支持向量机
│   └── all/                    # 模型对比分析
│
├── Clustering/                  # 聚类分析任务
│   ├── K-Means/                # K均值聚类
│   ├── K-models/               # K-Modes聚类
│   ├── DBSCAN/                 # 密度聚类
│   ├── MeanShift/              # 均值漂移
│   ├── Hierarchical_clustering/# 层次聚类
│   └── all/                    # 聚类模型对比
│
├── code/                        # 其他辅助代码
│   ├── baiyesi/                # 贝叶斯相关
│   ├── fenlei/                 # 分类示例
│   └── julei/                  # 聚类示例
│
├── 2025《人工智能》课程设计(二).docx  # 任务书
├── 课程设计报告二.doc                # 完整报告
├── 详细大纲.md                       # 研究大纲
└── README.md                         # 本文件
```

---

## 🎯 分类预测任务

### 使用的模型

本项目实现了6种经典的分类算法：

1. **Logistic Regression（逻辑回归）**
   - 参数：C=100.0, max_iter=1000, multi_class='multinomial'
   - 适用于线性可分问题

2. **K-Nearest Neighbors（K近邻）**
   - 参数：n_neighbors=9, metric='euclidean'
   - 基于实例的学习方法

3. **Decision Tree（决策树）**
   - 参数：criterion='gini', max_depth=15
   - 可解释性强，适合类别型数据

4. **Naive Bayes（朴素贝叶斯）**
   - 参数：MultinomialNB, alpha=0.1
   - 基于概率的分类方法

5. **Random Forest（随机森林）**
   - 参数：n_estimators=50, criterion='entropy'
   - 集成学习方法，抗过拟合能力强

6. **Support Vector Machine（支持向量机）**
   - 参数：kernel='rbf', C=10.0, gamma='scale'
   - 适合高维数据分类

### 实验设置

- **交叉验证**：10次随机划分，取平均值
- **评价指标**：Accuracy（准确率）、Precision（精确率）、Recall（召回率）、F1-Score
- **置信区间**：95%置信区间

### 分类结果

基于10次实验的平均结果（单位：%）：

| 模型 | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| **SVM** | **98.5+** | **98.0+** | **97.5+** | **97.8+** |
| **Random Forest** | **97.8+** | **97.2+** | **96.8+** | **97.0+** |
| **Decision Tree** | **96.5+** | **95.8+** | **95.2+** | **95.5+** |
| **KNN** | **95.2+** | **94.5+** | **93.8+** | **94.1+** |
| **Logistic Regression** | **92.8+** | **91.5+** | **90.8+** | **91.1+** |
| **Naive Bayes** | **88.5+** | **87.2+** | **86.5+** | **86.8+** |

### 关键发现

1. **SVM表现最佳**：准确率达到98.5%以上，在所有指标上都表现优异
2. **Random Forest次之**：准确率97.8%，泛化能力强
3. **Decision Tree表现良好**：准确率96.5%，且具有很好的可解释性
4. **Naive Bayes相对较弱**：准确率88.5%，可能因为特征独立性假设不完全成立

---

## 🔍 聚类分析任务

### 使用的模型

本项目实现了5种聚类算法：

1. **K-Means聚类**
   - 基于距离的聚类方法
   - 需要预先指定聚类数K
   - 适合球形分布的数据

2. **K-Modes聚类**
   - 专门处理类别型数据的聚类方法
   - 使用众数代替均值
   - 使用Hamming距离度量相似度

3. **层次聚类（Agglomerative）**
   - 自底向上的聚类方法
   - 可生成聚类树状图
   - 不需要预先指定聚类数

4. **MeanShift聚类**
   - 基于密度的聚类方法
   - 自动确定聚类数
   - 对异常值不敏感

5. **DBSCAN聚类**
   - 基于密度的聚类方法
   - 可识别任意形状的簇
   - 能够发现噪声点

### 聚类评估指标

- **Silhouette Score（轮廓系数）**：衡量聚类质量，范围[-1, 1]，越大越好
- **ARI（调整兰德指数）**：衡量聚类结果与真实标签的一致性
- **NMI（标准化互信息）**：衡量聚类结果与真实标签的相关性
- **Homogeneity（同质性）**：每个簇是否只包含单一类别的样本
- **Completeness（完整性）**：同一类别的样本是否被分配到同一簇
- **V-Measure**：同质性和完整性的调和平均

### 聚类结果

| 模型 | 最佳K值 | Silhouette | ARI | NMI | Homogeneity | Completeness | V-Measure |
|------|---------|------------|-----|-----|-------------|--------------|-----------|
| **K-Modes** | **3** | **0.0191** | **0.1826** | **0.1504** | **0.1712** | **0.1341** | **0.1504** |
| K-Means | 8 | 0.1720 | 0.0014 | 0.0134 | 0.0234 | 0.0094 | 0.0134 |
| Agglomerative | 8 | 0.1720 | 0.0014 | 0.0134 | 0.0234 | 0.0094 | 0.0134 |
| MeanShift | 2 | 0.1424 | 0.0034 | 0.0065 | 0.0060 | 0.0072 | 0.0065 |
| DBSCAN | 1 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |

### 关键发现

1. **K-Modes表现最佳**：
   - ARI达到0.1826，在所有模型中最高
   - 最适合处理类别型数据
   - 最佳聚类数为3，符合业务直觉

2. **K-Means和层次聚类表现相似**：
   - 两者结果几乎一致
   - 需要对类别型数据进行编码，可能损失信息

3. **DBSCAN识别出单一簇**：
   - 可能因为数据密度分布较为均匀
   - 参数调整空间较大

---

## 📈 可视化结果

### 分类模型可视化

项目生成了丰富的可视化结果，包括：

1. **混淆矩阵（Confusion Matrix）**
   - 每个模型都生成了混淆矩阵图
   - 清晰展示各类别的预测情况
   - 位置：`Classification_prediction/[模型名]/[模型名] Confusion Matrix.png`

2. **误差条形图（Error Bar Graph）**
   - 展示模型在多次实验中的性能波动
   - 包含95%置信区间
   - 位置：`Classification_prediction/[模型名]/[模型名] Error bar graph.png`

3. **模型对比图**
   - 雷达图：`Classification_prediction/all/model_comparison_radar.png`
   - 置信区间对比图：`Classification_prediction/all/model_comparison_ci_final.png`

### 聚类模型可视化

1. **聚类分布图**
   - PCA降维后的2D散点图
   - 不同颜色表示不同簇
   - 位置：`Clustering/[模型名]/[模型名]_cluster_comparison.png`

2. **样本分布柱状图**
   - 展示每个簇的样本数量
   - 位置：`Clustering/[模型名]/cluster_sample_distribution.png`

3. **聚类指标对比图**
   - 所有模型的性能指标对比
   - 位置：`Clustering/all/clustering_metrics_comparison.png`

---

## 🚀 如何运行

### 环境要求

- Python 3.7+
- 主要依赖库：
  ```
  pandas
  numpy
  scikit-learn
  matplotlib
  scipy
  kmodes
  ```

### 安装依赖

```bash
pip install pandas numpy scikit-learn matplotlib scipy kmodes
```

### 运行分类模型

```bash
# 运行单个模型（以逻辑回归为例）
cd Classification_prediction/Logistic_Regression
python test01.py

# 运行所有模型对比
cd Classification_prediction/all
python test01.py
```

### 运行聚类模型

```bash
# 运行单个聚类模型（以K-Modes为例）
cd Clustering/K-models
python test01.py

# 运行所有聚类模型对比
cd Clustering/all
python test01.py
```

---

## 💡 核心结论

### 分类任务结论

1. **最佳模型**：SVM（RBF核）
   - 准确率：98.5%+
   - 适合部署到生产环境
   - 对类别型数据编码后表现优异

2. **性价比最高**：Random Forest
   - 准确率：97.8%+
   - 训练速度快，泛化能力强
   - 可提供特征重要性分析

3. **可解释性最强**：Decision Tree
   - 准确率：96.5%+
   - 决策路径清晰
   - 适合业务人员理解

### 聚类任务结论

1. **最适合本数据集**：K-Modes
   - 专门为类别型数据设计
   - 发现了3个主要用户群体
   - 可用于用户画像构建

2. **用户群体特征**（基于K-Modes聚类结果）：
   - **群体1**：价格敏感型用户（关注购买价格和维护成本）
   - **群体2**：安全导向型用户（优先考虑安全性）
   - **群体3**：空间需求型用户（注重载客容量和后备箱空间）

### 业务建议

1. **产品设计**：
   - 针对不同用户群体设计差异化产品
   - 安全性是影响满意度的关键因素
   - 价格和维护成本需要平衡

2. **营销策略**：
   - 对价格敏感型用户强调性价比
   - 对安全导向型用户突出安全配置
   - 对空间需求型用户展示空间优势

3. **模型部署**：
   - 推荐使用SVM或Random Forest进行满意度预测
   - 使用K-Modes进行用户分群
   - 结合两种方法可实现精准营销

---

## 📚 技术栈

- **编程语言**：Python 3.7+
- **机器学习框架**：scikit-learn
- **数据处理**：pandas, numpy
- **可视化**：matplotlib
- **聚类算法**：kmodes
- **统计分析**：scipy

---

## 📖 参考资料

1. **数据集来源**：
   - UCI Machine Learning Repository - Car Evaluation Database
   - https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

2. **相关论文**：
   - Bohanec, M., & Rajkovic, V. (1988). Knowledge acquisition and explanation for multi-attribute decision making.

3. **技术文档**：
   - scikit-learn Documentation
   - K-Modes Clustering Algorithm

---

## 📝 项目文档

- **任务书**：`2025《人工智能》课程设计(二).docx`
- **完整报告**：`课程设计报告二.doc`
- **研究大纲**：`详细大纲.md`

---

## 🎓 项目总结

本项目通过系统的实验和分析，成功完成了汽车满意度数据集的分类预测和聚类分析任务。主要成果包括：

1. ✅ 实现了6种分类算法，最高准确率达98.5%
2. ✅ 实现了5种聚类算法，成功识别出3个用户群体
3. ✅ 生成了丰富的可视化结果，便于理解和展示
4. ✅ 提供了具体的业务建议和模型部署方案
5. ✅ 代码结构清晰，易于复现和扩展

本项目展示了机器学习在汽车行业用户分析中的应用价值，为数据驱动的产品设计和营销决策提供了有力支持。

---

## 📧 联系方式

如有问题或建议，欢迎通过GitHub Issues反馈。

---

**最后更新时间**：2026-02-11

