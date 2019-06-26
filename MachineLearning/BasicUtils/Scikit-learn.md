# Scikit-learn

## 安装使用

安装

```python
pip install scikit-learn
```

引入

```python
import sklearn
```

内容

```
- 分类、聚类、回归
- 特征工程
- 模型选择、调优
```

## 数据集

[API]()

- 加载

```python
# 加载获取流行数据集
from sklearn import datasets


# 获取小规模数据集，数据包含在datasets中
datasets.load_*()		

# 获取大规模数据集，需要从网络上下载
# 参数1:data_home表示数据集下载的目录，默认～/scikit_learn_data/
# 参数2:subset表示'all'，'train'或'test'，可选，选择要加载的数据集.
datasets.fetch_*(data_home=None, subset='all')

# 清除目录下的数据
datasets.clear_data_home(data_home=None)
```

常用数据集

| name                                              | desc             |
| ------------------------------------------------- | ---------------- |
| `load_iris()`                                     | 鸢尾花数据集     |
| `load_boston()`                                   | 波士顿房价数据集 |
| `fetch_20newsgroups(data_home=None,subset='all')` | 20类新闻数据集   |

- 属性

```python
# 数据集属性
DESCR			 			数据集描述
feature_names	 	特征名
data			 			特征值数据数组，是[n_samples*n_features]的二维numpy.ndarry数组
target_names	 	标签名，回归数据集没有
target					目标值数组
```

- 分离

```python
sklearn.model_selection.train_test_split(arrays, *options)

# 参数
参数1				 	x数据集的特征值
参数2					y数据集的特征值
test_size			测试集的大小，一般为float
random_state	随机数种子，不同的种子会造成不同的随机采样结果。相同的种子采样结果相同
# 返回
训练集特征值、测试集特征值、训练集标签、测试集标签
```

示例

```python
from sklearn.model_selection import train_test_split

# 特征值
x = np.arange(0,10).reshape([5,2])
# 目标值
y = range(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

print("训练集的特征值")
print(x_train)
print("测试集的特征值")
print(x_test)
print("训练集的目标值")
print(y_train)
print("测试集的目标值")
print(y_test)
```

## 特征工程

### 归一化

- 最值归一化

[API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)

```python
from sklearn.preprocessing import MinMaxScaler

# numpy array格式的数据[n_samples,n_features]
data = ...
# 实例化,每个特征缩放到给定范围(默认[0,1])
mm = MinMaxScaler(feature_range=(0,1)…)
# 归一化，返回转换后的形状相同的array
result = mm.fit_transform(data)  
# 原始数据中每列特征的最小最大值
print(mm.data_min_)
print(mm.data_max_)
```

- 均值方差归一化

[API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)

```python
from sklearn.preprocessing import StandardScaler

data = ...
# 处理之后每列的数据都聚集在均值0，方差1附近
ss = StandardScaler()
# 输入：ndarray，输出:转换后形状相同的array
result = ss.fit_transform(data)
# 原始数据中每列特征的平均值
print(ss.mean_)
```

### 缺失值

[API]()

```python
import numpy as np
from sklearn.preprocessing import Imputer

data = [[1, 2],
        [np.nan, 3],
        [7, 6]]

# 完成缺失值插补
imputer = Imputer(missing_values="NaN", strategy="mean", axis=1)
# 输入numpy array格式的数据，返回转换后形状相同的array
result = imputer.fit_transform(data)
print(result.shape)
```

## 算法评价

### 分类

准确度

```python
from sklearn.metrics import accuracy_score
```

混淆矩阵

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_predict)
```

精准率

```python
from sklearn.metrics import precision_score

precision_score(y_test, y_predict)
```

召回率

```python
from sklearn.metics import recall_score

recall_score(y_test, y_predict)
```

F1 Score

```python
from sklearn.metics import f1_score

f1_score(y_test, y_predict)
```

precision-recall曲线

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)
```

ROC曲线

```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fprs, tprs, thresholds = roc_curve(Y_test, decision_scores)
roc_auc_score(y_test, decision_scores)
```

### 回归

均方误差

```python
from sklearn.metrics import mean_squared_error
```

根均方误差

```python
sqrt(mean_squared_error())
```

平均绝对误差

```python
from sklearn.metrics import mean_absolute_error
```

R方

```python
from sklearn.metrics import r2_score
```

### 聚类

轮廓系数

```python
from sklearn.metrics import silhouette_score
```



## 网格搜索

[API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)

- 交叉验证

将拿到的数据，分为训练和测试集。将数据分成5份，其中一份作为验证集。然后经过5次(组)的测试，每次都更换不同的验证集。即得到5组模型的结果，取平均值作为模型精度的估计。又称5折交叉验证。

- 网格搜索

使用网格搜索确定最优的参数。这种参数，称之为超参数，K近邻算法中的K值。

在网格搜索中每组超参数都采用交叉验证来进行评估。

```python
# 对估计器的指定参数值进行详尽搜索
sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)

# 输入
estimator：估计器对象
param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
cv：指定几折交叉验证
# 方法
fit：输入训练数据
score：准确率
# 属性
best_score_:最好结果
best_estimator_：最好的参数模型
cv_results_:交叉验证的结果
```

示例

```python
param_grid = [
		{
				'weights': ['uniform'],
      	'n_neighbors': [i for i in range(1, 11)]
		},
  	{
      	'weights': ['distance'],
      	'n_neightbors': [i for i in range(1, 11)],
      	'p':[i for i in range(1, 6)]
    }
]

knn_clf = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(knn_clf, param_grid)
grid_search.fit(X_train, y_train)
grid_search.best_estimator_  # 最佳算法
grid_search.best_score_best_params_  

knn_clf = grid_search.best_estimator_
knn_clf.predict(X_test)

grid_search.best_score_  # 采用算法在交叉验证基础上判断准确性
grid_search.best_estimator_.score(X_test, y_test)  # 采用算法进行判断准确性
```

## 常用算法

分类

```python
# k-近邻算法
sklearn.neighbors	
# 朴素贝叶斯
sklearn.naive_bayes	
# 逻辑回归
sklearn.linear_model.LogisticRegression	
```

回归

```python
# 线性回归
sklearn.linear_model.LinearRegression
# 岭回归
sklearn.linear_model.Ridge
# Lasso回归
sklearn.linear_model.Lasso
```

[KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)





