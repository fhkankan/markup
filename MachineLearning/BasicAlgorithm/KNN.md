# KNN

## 概述

优点

```
简单，易于理解，易于实现
```

缺点

```
如果训练集有m个样本，n个特征，则预测每一个新的数据，需要O(m*n),计算量大，内存开销大
必须指定k值，k值选择不当则分类精度不能保证

优化：使用树结构，KD-Tree,Ball-Tree
```

## 原理

如果一个样本在数据集中，有k个最相近的样本，而k个样本大多数属于某一个类别，那么这个样本也属于该类别

欧拉距离
$$
\sqrt{(x^{(a)}-x^{(b)})^2+(y^{(a)}-y^{(b)})^2}
$$

$$
\sqrt{(x^{(a)}-x^{(b)})^2+(y^{(a)}-y^{(b)})^2+(z^{(a)}-z^{(b)})^2}
$$

$$
\sqrt{(X_1^{(a)}-X_1^{(b)})^2+(X_2^{(a)}-X2^{(b)})^2+...+(X_n^{(a)}-X_n^{(b)})^2}
$$

$$
\sqrt{\sum_{i=1}^{n}(X_i^{(a)}-X_i^{(b)})^2}
$$

曼哈顿距离
$$
\sum_{i=1}^n{|X_i^{(a)}-X_i^{(b)}|}
$$
明可夫斯距离
$$
(\sum_{i=1}^n{|X_i^{(a)}-X_i^{(b)}|^p})^{\frac{1}{p}}
$$


## 实现

自定义类

```python
import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class KNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None, \
                "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
                "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k
```

使用

```python 
knn_clf = KNNClassifier(k=6)
knn_clf.fit(x_train, y_train)
y_predict = knn_clf.predict(X_predict)[0]
```

## sklearn

[KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)

用于分类

```python
from sklearn.neighbors import KNeighborsClassifier

# KNeighborsClassifiler训练模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
# 使用测试集的特征值，预测测试集的特征值对应的目标值
y_predict = knn.predict(x_test)
print(y_predict)
# 测试模型在测试集上的准确性
score = knn.score(x_test, y_test)
print(score)
```

示例

```python
# 1.导入所需的包
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 2.加载数据文件
data = pd.read_csv("./FBlocation/train.csv")
print(len(data))
# 3.缩小数据范围
data = data.query("x > 1 & x < 1.25 & y >3 &y < 3.25")
print(len(data))
# 4.时间特征抽取
# 将时间戳转换为日期
time_value = pd.to_datetime(data["time"], unit="s")
# 将时间转换为DatetimeIndex
date_time_index = pd.DatetimeIndex(time_value)
data["hour"] = date_time_index.hour
data["month"] = date_time_index.month
data["dayofweek"] = date_time_index.dayofweek
# 5.删除掉入住率比较低的样本
# 分组聚合 以place_id分组，count计数，小于3，筛选掉
place_count = data.groupby("place_id").aggregate(np.count_nonzero)
# print(place_count)
#            row_id      x      y  accuracy  time  hour  month  dayofweek
# place_id
# 1009781224     219  219.0  219.0       219   219   216    219        200
# 所有入住次数大于3的结果，数据并不是原始数据，而只是一个统计数据
result = place_count[place_count["row_id"] > 3].reset_index()
# 从原始数据中选择place_id在result中的样本
data = data[data["place_id"].isin(result["place_id"])]
# 6.特征选择
# 特征值
x = data.drop(["row_id", "time", "place_id"], axis=1)
# 目标值
y = data["place_id"]
# 7.分割数据集
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 8.对数据集进标准化
ss = StandardScaler()
# 对特征值进行标准化
x_train = ss.fit_transform(x_train)
# 对测试集的特征值标准化
x_test = ss.transform(x_test)  # 按照原来训练集的平均值做标准化，统一数据转换标准
# 9.KNeighborsClassifiler训练模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
# 10.准确率
# 使用测试集的特征值，预测测试集的特征值对应的目标值place_id
y_predict = knn.predict(x_test)
print(y_predict)
# 测试模型在测试集上的准确性
score = knn.score(x_test, y_test)
print(score)
```

[KNeighborsRegressor](KNeighborsRegressor)

用于回归

```python
from sklearn.neighbors import KNeighborsRegressor

# 创建实例对象
knn_reg = KNeighborsRegressor()  
knn_reg.fit(x_train, y_train)
# 使用测试集的特征值，预测测试集的特征值对应的目标值
y_predict = knn_reg.predict(x_test)
print(y_predict)
# 测试模型在测试集上的准确性
score = knn_reg.score(x_test, y_test)
print(score)
```

