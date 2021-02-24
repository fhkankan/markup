# KNN

## 概述

优点

```
简单，易于理解，易于实现
```

缺点

```
1. 如果训练集有m个样本，n个特征，则预测每一个新的数据，需要O(m*n),计算量大，内存开销大
2. 必须指定k值，k值选择不当则分类精度不能保证

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

### 最近邻

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

X = np.array([[1, 1], [1, 3], [2, 2], [2.5, 5], [3, 1],
              [4, 2], [2, 3.5], [3, 3], [3.5, 4]])

# 寻找最近邻的数量
num_neighbors = 3

# 输入数据点
input_point = [[2.6, 1.7]]

# 画出数据点
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, color='k')

# 建立最近邻
knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)
distances, indices = knn.kneighbors(input_point)

# Print the 'k' nearest neighbors
print("\nk nearest neighbors")
for rank, index in enumerate(indices[0][:num_neighbors]):
    print(str(rank + 1) + " -->", X[index])

# Plot the nearest neighbors
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, color='k')
plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1],
            marker='o', s=150, color='k', facecolors='none')
plt.scatter(input_point[0][0], input_point[0][1],
            marker='x', s=150, color='k', facecolors='none')

plt.show()

```

### KNN分类

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

### KNN回归

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

amplitude = 10
num_points = 100
X = amplitude * np.random.rand(num_points, 1) - 0.5 * amplitude

# 计算目标并添加噪声
y = np.sinc(X).ravel()
y += 0.2 * (0.5 - np.random.rand(y.size))

# 画出输入数据图形
plt.figure()
plt.scatter(X, y, s=40, c='k', facecolors='none')
plt.title('Input data')

# 用输入数据的10倍的密度创建一维网格
x_values = np.linspace(-0.5*amplitude, 0.5*amplitude, 10*num_points)[:, np.newaxis]

# 定义最近邻的个数
n_neighbors = 8

# 定义并训练回归器 
knn_regressor = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
y_values = knn_regressor.fit(X, y).predict(x_values)

# 将输入输出数据交叠，查看回归器效果
plt.figure()
plt.scatter(X, y, s=40, c='k', facecolors='none', label='input data')
plt.plot(x_values, y_values, c='k', linestyle='--', label='predicted values')
plt.xlim(X.min() - 1, X.max() + 1)
plt.ylim(y.min() - 0.2, y.max() + 0.2)
plt.axis('tight')
plt.legend()
plt.title('K Nearest Neighbors Regressor')

plt.show()

```

## tensorflow

### 最近邻

```python
# k-Nearest Neighbor
#----------------------------------
#
# This function illustrates how to use
# k-nearest neighbors in tensorflow
#
# We will use the 1970s Boston housing dataset
# which is available through the UCI
# ML data repository.
#
# Data:
#----------x-values-----------
# CRIM   : per capita crime rate by town
# ZN     : prop. of res. land zones
# INDUS  : prop. of non-retail business acres
# CHAS   : Charles river dummy variable
# NOX    : nitrix oxides concentration / 10 M
# RM     : Avg. # of rooms per building
# AGE    : prop. of buildings built prior to 1940
# DIS    : Weighted distances to employment centers
# RAD    : Index of radian highway access
# TAX    : Full tax rate value per $10k
# PTRATIO: Pupil/Teacher ratio by town
# B      : 1000*(Bk-0.63)^2, Bk=prop. of blacks
# LSTAT  : % lower status of pop
#------------y-value-----------
# MEDV   : Median Value of homes in $1,000's

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Load the data
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_features = len(cols_used)
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
# 分离数据集为特征依赖的数据集和特征无关的数据集，将预测最后一个变量MEDV，该值为恶一组放假中的平均值
y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

## Min-Max Scaling
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

# Split the data into train and test sets
np.random.seed(13)  #make results reproducible
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare k-value and batch size
k = 4
batch_size=len(x_vals_test)

# Placeholders
x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare distance metric
# L1
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)

# L2
#distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=1))

# Predict: Get min distance index (Nearest neighbor)
# prediction = tf.arg_min(distance, 0)
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)  # top_k()函数以张量的方式返回最大值的值和索引
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1),1)
x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
x_val_weights = tf.expand_dims(tf.div(top_k_xvals,x_sums_repeated), 1)

top_k_yvals = tf.gather(y_target_train, top_k_indices)
prediction = tf.squeeze(tf.matmul(x_val_weights,top_k_yvals), axis=[1])

# Calculate MSE
mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

# Calculate how many loops over training data
num_loops = int(np.ceil(len(x_vals_test)/batch_size))

for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size,len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                         y_target_train: y_vals_train, y_target_test: y_batch})
    batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                         y_target_train: y_vals_train, y_target_test: y_batch})

    print('Batch #' + str(i+1) + ' MSE: ' + str(np.round(batch_mse,3)))

# Plot prediction and actual distribution
bins = np.linspace(5, 50, 45)

plt.hist(predictions, bins, alpha=0.5, label='Prediction')
plt.hist(y_batch, bins, alpha=0.5, label='Actual')
plt.title('Histogram of Predicted and Actual Values')
plt.xlabel('Med Home Value in $1,000s')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()
```

### 文本距离

使用tensorflow的文本距离度量-字符串间的编辑距离(Levenshtein距离)。Levenshtein距离是指由一个字符串转换成另一个字符串所需的最少编辑操作次数。允许的编辑操作包括插入一个字符、删除一个字符和将一个字符替换成另一个字符。

使用tensorflow的内建函数`edit_distance()`求解Levenshtein距离。

```python
# Text Distances
#----------------------------------
#
# This function illustrates how to use the Levenstein distance (edit distance) in TensorFlow.

import tensorflow as tf

sess = tf.Session()

#----------------------------------
# First compute the edit distance between 'bear' and 'beers'
hypothesis = list('bear')  # 创建列表，然后将列表映射为一个三维稀疏矩阵
truth = list('beers')
h1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3]],
                     hypothesis,
                     [1,1,1])  # SparseTensor函数需指定字符索引、矩阵形状和张量中的非零值

t1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3],[0,0,4]],
                     truth,
                     [1,1,1])

# edit_distance()仅接受稀疏张量
print(sess.run(tf.edit_distance(h1, t1, normalize=False)))  # False表示计算总的编辑距离，True表示计算归一化编辑距离（通过编辑距离除以第二个单词的长度进行归一化）

#----------------------------------
# Compute the edit distance between ('bear','beer') and 'beers':
hypothesis2 = list('bearbeer')
truth2 = list('beersbeers')  # 为便于比较，重复beers
h2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,1,0], [0,1,1], [0,1,2], [0,1,3]],
                     hypothesis2,
                     [1,2,4])

t2 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,1,0], [0,1,1], [0,1,2], [0,1,3], [0,1,4]],
                     truth2,
                     [1,2,5])

print(sess.run(tf.edit_distance(h2, t2, normalize=True)))

#----------------------------------
# Now compute distance between four words and 'beers' more efficiently with sparse tensors:
hypothesis_words = ['bear','bar','tensor','flow']
truth_word = ['beers']

num_h_words = len(hypothesis_words)
h_indices = [[xi, 0, yi] for xi,x in enumerate(hypothesis_words) for yi,y in enumerate(x)]
h_chars = list(''.join(hypothesis_words))

h3 = tf.SparseTensor(h_indices, h_chars, [num_h_words,1,1])

truth_word_vec = truth_word*num_h_words
t_indices = [[xi, 0, yi] for xi,x in enumerate(truth_word_vec) for yi,y in enumerate(x)]
t_chars = list(''.join(truth_word_vec))

t3 = tf.SparseTensor(t_indices, t_chars, [num_h_words,1,1])

print(sess.run(tf.edit_distance(h3, t3, normalize=True)))

# 使用占位符来计算来那个哥哥单词列表间的编辑距离
def create_sparse_vec(word_list):
    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi, x in enumerate(word_list) for yi, y in enumerate(x)]
    chars = list(''.join(word_list))
    res = tf.compat.v1.SparseTensorValue(indices, chars, [num_words, 1, 1])
    return res


hyp_string_sparse = create_sparse_vec(hypothesis_words)
truth_string_parse = create_sparse_vec(truth_word*len(hypothesis_words))

hyp_input = tf.sparse_placeholder(dtype=tf.string)
truth_input = tf.sparse_placeholder(dtype=tf.string)

edit_distances = tf.edit_distance(hyp_input, truth_input, normalize=True)
feed_dict = {hyp_input: hyp_string_sparse, truth_input: truth_string_parse}
res = sess.run(edit_distances, feed_dict=feed_dict)
print(res)
```

### 混合距离

加权距离函数的关键是使用加权权重矩阵。包含矩阵操作的距离函数表达式如下
$$
D(x,y) = \sqrt{(x-y)^TA(x-y)}
$$
其中，$A$ 是对角权重矩阵，用来对每个特征的距离度量进行调整。

```python
# Mixed Distance Functions for  k-Nearest Neighbor
#----------------------------------
#
# This function shows how to use different distance
# metrics on different features for kNN.
#
# Data:
#----------x-values-----------
# CRIM   : per capita crime rate by town
# ZN     : prop. of res. land zones
# INDUS  : prop. of non-retail business acres
# CHAS   : Charles river dummy variable
# NOX    : nitrix oxides concentration / 10 M
# RM     : Avg. # of rooms per building
# AGE    : prop. of buildings built prior to 1940
# DIS    : Weighted distances to employment centers
# RAD    : Index of radian highway access
# TAX    : Full tax rate value per $10k
# PTRATIO: Pupil/Teacher ratio by town
# B      : 1000*(Bk-0.63)^2, Bk=prop. of blacks
# LSTAT  : % lower status of pop
#------------y-value-----------
# MEDV   : Median Value of homes in $1,000's


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Load the data
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_features = len(cols_used)
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]

y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

## Min-Max Scaling
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

## Create distance metric weight matrix weighted by standard deviation
# 创建对角权重矩阵，该举着嗯提供归一化的距离度量，其值为特征的标准差
weight_diagonal = x_vals.std(0)
weight_matrix = tf.cast(tf.diag(weight_diagonal), dtype=tf.float32)

# Split the data into train and test sets
np.random.seed(13)   # reproducible results
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# Declare k-value and batch size
k = 4
batch_size=len(x_vals_test)

# Placeholders
x_data_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare weighted distance metric
# Weighted L2 = sqrt((x-y)^T * A * (x-y))
subtraction_term =  tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))
first_product = tf.matmul(subtraction_term, tf.tile(tf.expand_dims(weight_matrix,0), [batch_size,1,1]))  # tf.tile()为权重矩阵指定batch_size维度扩展
second_product = tf.matmul(first_product, tf.transpose(subtraction_term, perm=[0,2,1]))
distance = tf.sqrt(tf.matrix_diag_part(second_product))

# Predict: Get min distance index (Nearest neighbor)
# 计算完每个测试数据点的距离，需返回k-NN法的前k个最近邻域.
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1),1)
x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
x_val_weights = tf.expand_dims(tf.div(top_k_xvals,x_sums_repeated), 1)

top_k_yvals = tf.gather(y_target_train, top_k_indices)
prediction = tf.squeeze(tf.matmul(x_val_weights,top_k_yvals), axis=[1])  # 将前k个最近邻域的距离进行加权平均做预测

# Calculate MSE
mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_target_test))), batch_size)

# Calculate how many loops over training data
num_loops = int(np.ceil(len(x_vals_test)/batch_size))

for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size,len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                         y_target_train: y_vals_train, y_target_test: y_batch})
    batch_mse = sess.run(mse, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                         y_target_train: y_vals_train, y_target_test: y_batch})

    print('Batch #' + str(i+1) + ' MSE: ' + str(np.round(batch_mse,3)))

# Plot prediction and actual distribution
bins = np.linspace(5, 50, 45)

plt.hist(predictions, bins, alpha=0.5, label='Prediction')
plt.hist(y_batch, bins, alpha=0.5, label='Actual')
plt.title('Histogram of Predicted and Actual Values')
plt.xlabel('Med Home Value in $1,000s')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()
```

### 地址匹配

地址匹配是一种记录匹配，其匹配的地址涉及多个数据集。在地址匹配中，地址中有许多打印错误，不同的城市或者不同的邮政编码，但是指向同一个地址。使用最近邻域算法综合地址信息的数值部分和字符部分可以帮助鉴定实际相同的地址。

```python
# Address Matching with k-Nearest Neighbors
#----------------------------------
#
# This function illustrates a way to perform
# address matching between two data sets.
#
# For each test address, we will return the
# closest reference address to it.
#
# We will consider two distance functions:
# 1) Edit distance for street number/name and
# 2) Euclidian distance (L2) for the zip codes

import random
import string
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# First we generate the data sets we will need
# n = Size of created data sets
n = 10
street_names = ['abbey', 'baker', 'canal', 'donner', 'elm']
street_types = ['rd', 'st', 'ln', 'pass', 'ave']

random.seed(31)  # make results reproducible
rand_zips = [random.randint(65000, 65999) for i in range(5)]


# Function to randomly create one typo in a string w/ a probability
def create_typo(s, prob=0.75):
    if random.uniform(0, 1) < prob:
        rand_ind = random.choice(range(len(s)))
        s_list = list(s)
        s_list[rand_ind] = random.choice(string.ascii_lowercase)
        s = ''.join(s_list)
    return s

# Generate the reference dataset
numbers = [random.randint(1, 9999) for _ in range(n)]
streets = [random.choice(street_names) for _ in range(n)]
street_suffs = [random.choice(street_types) for _ in range(n)]
zips = [random.choice(rand_zips) for _ in range(n)]
full_streets = [str(x) + ' ' + y + ' ' + z for x, y, z in zip(numbers, streets, street_suffs)]
reference_data = [list(x) for x in zip(full_streets,zips)]

# Generate test dataset with some typos
typo_streets = [create_typo(x) for x in streets]
typo_full_streets = [str(x) + ' ' + y + ' ' + z for x, y, z in zip(numbers, typo_streets, street_suffs)]
test_data = [list(x) for x in zip(typo_full_streets, zips)]

# Now we can perform address matching
# Create graph
sess = tf.Session()

# Placeholders
test_address = tf.sparse_placeholder(dtype=tf.string)
test_zip = tf.placeholder(shape=[None, 1], dtype=tf.float32)
ref_address = tf.sparse_placeholder(dtype=tf.string)
ref_zip = tf.placeholder(shape=[None, n], dtype=tf.float32)

# Declare Zip code distance for a test zip and reference set
zip_dist = tf.square(tf.subtract(ref_zip, test_zip))  # 邮政编码的距离

# Declare Edit distance for address
address_dist = tf.edit_distance(test_address, ref_address, normalize=True)  # 地址字符串的编辑距离

# Create similarity scores
# 把邮政编码距离和地址距离转换成相似度
zip_max = tf.gather(tf.squeeze(zip_dist), tf.argmax(zip_dist, 1))
zip_min = tf.gather(tf.squeeze(zip_dist), tf.argmin(zip_dist, 1))
zip_sim = tf.div(tf.subtract(zip_max, zip_dist), tf.subtract(zip_max, zip_min))
address_sim = tf.subtract(1., address_dist)

# Combine distance functions
# 进行加权平均
address_weight = 0.5
zip_weight = 1. - address_weight
weighted_sim = tf.add(tf.transpose(tf.multiply(address_weight, address_sim)), tf.multiply(zip_weight, zip_sim))

# Predict: Get max similarity entry
top_match_index = tf.argmax(weighted_sim, 1)


# Function to Create a character-sparse tensor from strings
# 将地址字符串转换成稀疏向量
def sparse_from_word_vec(word_vec):
    num_words = len(word_vec)
    indices = [[xi, 0, yi] for xi,x in enumerate(word_vec) for yi,y in enumerate(x)]
    chars = list(''.join(word_vec))
    return tf.SparseTensorValue(indices, chars, [num_words,1,1])

# Loop through test indices
# 分离参考集中的地址和邮政编码，在遍历迭代训练中为恶占位符赋值
reference_addresses = [x[0] for x in reference_data]
reference_zips = np.array([[x[1] for x in reference_data]])

# Create sparse address reference set
sparse_ref_set = sparse_from_word_vec(reference_addresses)

for i in range(n):
    test_address_entry = test_data[i][0]
    test_zip_entry = [[test_data[i][1]]]
    
    # Create sparse address vectors
    test_address_repeated = [test_address_entry] * n
    sparse_test_set = sparse_from_word_vec(test_address_repeated)
    
    feeddict = {test_address: sparse_test_set,
                test_zip: test_zip_entry,
                ref_address: sparse_ref_set,
                ref_zip: reference_zips}
    best_match = sess.run(top_match_index, feed_dict=feeddict)
    best_street = reference_addresses[best_match[0]]
    [best_zip] = reference_zips[0][best_match]
    [[test_zip_]] = test_zip_entry
    print('Address: ' + str(test_address_entry) + ', ' + str(test_zip_))
    print('Match  : ' + str(best_street) + ', ' + str(best_zip))

```

### 图像识别

```python
# MNIST Digit Prediction with k-Nearest Neighbors
#-----------------------------------------------
#
# This script will load the MNIST data, and split
# it into test/train and perform prediction with
# nearest neighbors
#
# For each test integer, we will return the
# closest image/integer.
#
# Integer images are represented as 28x8 matrices
# of floating point numbers

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Load the data，指定one-hot编码
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Random sample
# 由于数据集较大，对数据集抽样成小数据集进行训练测试
np.random.seed(13)  # set seed for reproducibility
train_size = 1000
test_size = 102
rand_train_indices = np.random.choice(len(mnist.train.images), train_size, replace=False)
rand_test_indices = np.random.choice(len(mnist.test.images), test_size, replace=False)
x_vals_train = mnist.train.images[rand_train_indices]
x_vals_test = mnist.test.images[rand_test_indices]
y_vals_train = mnist.train.labels[rand_train_indices]
y_vals_test = mnist.test.labels[rand_test_indices]

# Declare k-value and batch size
k = 4
batch_size=6

# Placeholders
x_data_train = tf.placeholder(shape=[None, 784], dtype=tf.float32)
x_data_test = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 10], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# Declare distance metric
# L1
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), axis=2)

# L2
#distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=1))

# Predict: Get min distance index (Nearest neighbor)
# 找到最接近的topk图片和预测模型。在数据集的one-hot编码索引上进行预测模型计算，然后统计发生的数量，最终预测为数量最多的类别
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
prediction_indices = tf.gather(y_target_train, top_k_indices)
# Predict the mode category
count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
prediction = tf.argmax(count_of_predictions, axis=1)

# Calculate how many loops over training data
num_loops = int(np.ceil(len(x_vals_test)/batch_size))

test_output = []
actual_vals = []
for i in range(num_loops):
    min_index = i*batch_size
    max_index = min((i+1)*batch_size,len(x_vals_train))
    x_batch = x_vals_test[min_index:max_index]
    y_batch = y_vals_test[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                         y_target_train: y_vals_train, y_target_test: y_batch})
    test_output.extend(predictions)
    actual_vals.extend(np.argmax(y_batch, axis=1))

# 计算准确度
accuracy = sum([1./test_size for i in range(test_size) if test_output[i]==actual_vals[i]])
print('Accuracy on test set: ' + str(accuracy))

# Plot the last batch results:
actuals = np.argmax(y_batch, axis=1)

Nrows = 2
Ncols = 3
for i in range(len(actuals)):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(np.reshape(x_batch[i], [28,28]), cmap='Greys_r')
    plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
                               fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    
plt.show()

```



