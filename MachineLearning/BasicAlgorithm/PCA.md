# 主成分分析

## 原理

一个非监督的机器学习算法

主要用于数据的降维

通过降维，可以发现更便于人类理解的特征

其他应用：可视化、去噪
$$
Var(x) = \frac{1}{m}\sum_{i=1}^m(x_i-\bar{x})^2
$$
找到一个轴，使样本空间的所有点映射到这个轴后，方差最大

将样例的均值归0，demean，则方差公式简化为
$$
Var(x) = \frac{1}{m}\sum_{i=1}^mx_i^2
$$

- 目标函数

对于主成分分析：

> 对所有的样本进行demean处理，得到$X^{(i)}=(X_1^{(i)}, X_2^{(i)})$

> 求一个轴的方向$w = (w_1, w_2)$使得所有的样本，映射到w后，得到$X_{project}^{(i)}=(X_{pr1}^{(i)}, X_{pr2}^{(i)})$，有$Var(X_{project}) = \frac{1}{m}\sum_{i=1}^m{(X_{project}^{(i)}-\bar{X}_{project})^2}$最大，即$Var(X_{project}) = \frac{1}{m}\sum_{i=1}^m{\Arrowvert{X_{project}^{(i)}-\bar{X}_{project}\Arrowvert}^2}$最大值，最终即$Var(X_{project}) = \frac{1}{m}\sum_{i=1}^m{\Arrowvert{X_{project}^{(i)}\Arrowvert}^2}$最大

由于
$$
X^{(i)}\cdot{w} = \Arrowvert{X^{(i)}}\Arrowvert\cdot\Arrowvert{w}\Arrowvert\cdot\cos\theta
$$

$$
\Arrowvert{w}\Arrowvert= 1
$$

故
$$
X^{(i)}\cdot{w} = \Arrowvert{X^{(i)}}\Arrowvert\cdot\cos\theta=\Arrowvert{X_{project}^{(i)}}\Arrowvert
$$
则
$$
Var(X_{project}) = \frac{1}{m}\sum_{i=1}^m{\Arrowvert{X_{project}^{(i)}\Arrowvert}^2} = \frac{1}{m}\sum_{i=1}^m{({X^{(i)}\cdot{w})}^2}
$$
对于多维
$$
Var(X_{project}) =  \frac{1}{m}\sum_{i=1}^m{({X_1^{(i)}\cdot{w_1} + X_2^{(i)}\cdot{w_2} +\ldots + X_n^{(i)}\cdot{w_n})}^2}= \frac{1}{m}\sum_{i=1}^{m}(\sum_{j=1}^m{X_j^{(i)}w_j})^2
$$
求其最大值

- 求解

求$w$使得$f(X) = \frac{1}{m}\sum_{i=1}^m{({X_1^{(i)}\cdot{w_1} + X_2^{(i)}\cdot{w_2} +\ldots + X_n^{(i)}\cdot{w_n})}^2}$最大

> 方法一：数学求解



> 方法二：梯度上升法

$$
\nabla{f} = \left( \begin{array}{ccc} {\partial{f}}/{\partial{w_1}} \\ {\partial{f}}/{\partial{w_2}} \\ \ldots\\ {\partial{f}}/{\partial{w_n}} \end{array} \right)= \frac{2}{m}\left( \begin{array}{ccc} {\sum_{i=1}^{m}{(X_1^{(i)}w_1 + \ldots+X_n^{(i)}w_n )X_1^{(i)}}} \\ {\sum_{i=1}^{m}{(X_2^{(i)}w_2 + \ldots+X_n^{(i)}w_n )X_2^{(i)}}} \\ \ldots\\ {\sum_{i=1}^{m}{(X_1^{(i)}w_1 + \ldots+X_n^{(i)}w_n )X_n^{(i)}}} \end{array} \right)
$$

简化
$$
\nabla{f} = \frac{2}{m}\left( \begin{array}{ccc} {\sum_{i=1}^{m}{(X^{(i)}w )X_1^{(i)}}} \\ {\sum_{i=1}^{m}{(X^{(i)}w )X_2^{(i)}}} \\ \ldots\\ {\sum_{i=1}^{m}{(X^{(i)}w )X_n^{(i)}}} \end{array} \right)
$$
向量化
$$
\frac{2}{m}\cdot(X^{(1)}w,X^{(2)}w,\ldots, X^{(m)}w )\cdot\left( \begin{array}{ccc}  X_1^{(1)} & X_2^{(1)} & \ldots & X_n^{(1)} \\ X_1^{(2)} & X_2^{(2)} & \ldots & X_n^{(2)} \\ \cdots &&& \cdots \\ X_1^{(m)} & X_2^{(m)} &\ldots & X_n^{(m)}\end{array} \right)
$$
则
$$
\frac{2}{m}\cdot(Xw)^T\cdot{X}
$$
转换行列
$$
\nabla{f} = \frac{2}{m}\left( \begin{array}{ccc} {\sum_{i=1}^{m}{(X^{(i)}w )X_1^{(i)}}} \\ {\sum_{i=1}^{m}{(X^{(i)}w )X_2^{(i)}}} \\ \ldots\\ {\sum_{i=1}^{m}{(X^{(i)}w )X_n^{(i)}}} \end{array} \right) = \frac{2}{m}\cdot{X^T}\cdot{(Xw)}
$$

- 下一主成分

数据进行改变，将数据在第一个主成分上的分量去掉

由于
$$
X^{(i)}\cdot{w} = \Arrowvert{X^{(i)}}\Arrowvert\cdot\cos\theta=\Arrowvert{X_{project}^{(i)}}\Arrowvert
$$

$$
X_{project}^{(i)} =\Arrowvert{X_{project}^{(i)}}\Arrowvert\cdot{w}
$$

则
$$
X^{'(i)} = X^{(i)}-X_{project}^{(i)}
$$
对新的$X^{'(i)}$求第一主成分，则是$X^{(i)}$的第二主成分，依次类推

## 降维

$$
X = \left( \begin{array}{ccc}  X_1^{(1)} & X_2^{(1)} & \ldots & X_n^{(1)} \\ X_1^{(2)} & X_2^{(2)} & \ldots & X_n^{(2)} \\ \cdots &&& \cdots \\ X_1^{(m)} & X_2^{(m)} &\ldots & X_n^{(m)}\end{array} \right)
$$

$$
W_k = \left( \begin{array}{ccc}  W_1^{(1)} & W_2^{(1)} & \ldots & W_n^{(1)} \\ W_1^{(2)} & W_2^{(2)} & \ldots & W_n^{(2)} \\ \cdots &&& \cdots \\ W_1^{(k)} & W_2^{(k)} &\ldots & W_n^{(k)}\end{array} \right)
$$



由$X$的`m*n`矩阵乘以`n*k`的$W_k$矩阵(前`k`个主成分)，形成`m*k`的矩阵$X_k$
$$
X\cdot{W_k^T} = X_k
$$
反向恢复，虽然丢失信息(与X不同了)，但是还是可以由`m*k`转换为`m*n`维矩阵
$$
X_k\cdot{W_k} = X_m
$$

## 实现

- 第一主成分

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:, 0] = np.random.unifor(0., 100., size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

def demean(X):
  	"""均值归零"""
  	return X - np.mean(X, axis=0)

def f(w, X):
  	"""目标函数"""
  	return np.sum((X.dot(w)**2)) / len(X)

def df_math(w, X):
  	"""目标函数梯度"""
  	return X.T.dot(X.dot(w)) * 2. / len(X)
  
def df_debug(w, X, epsilon=0.0001):
  	"""近似求导，通用"""
  	res = np.empty(len(w))
    for i in range(len(w)):
      	w_1 = w.copy()
        w_1[i] += epsilon
        w_2 = w.copy()
        w_2[i] -= epsilon
        res[i] = (f(w_1, X) - f(w_2, X)) / (2 * epsilon)
    return res
  
def direction(w):
  	"""w为单位向量"""
  	return w / np.linalg.norm(w)
  
def gradient_ascent(df, X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
		"""批量梯度下降法"""
    w = direction(initial_w)
    cur_iter = 0
    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)  # 注意：每次求一个单位方向
        if (abs(f(w, X) - f(last_w, X)) < epsilon):
            break
        cur_iter += 1
    return w
 

initial_w  = np.random.random(X.shape[1])  # 注意：不能用0向量开始
eta = 0.001
X_demean = demean(X)
# 注意：不能使用StandarScaler标准化数据
# debug测试
w = gradient_ascent(df_debug, X_demean, initial_w, eta)
print(w)
w = gradient_ascent(df_math, X_demean, initial_w, eta)
print(w)

# 绘图
plt.scatter(X_demean[:, 0], X_demean[:,1]})
plt.plot([0, w[0]*30], [0, w[1]*30], color='r')  # 由于w比较小，为可视化，扩大30倍
plt.show()
```

- 前n个主成分

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.empty((100, 2))
X[:, 0] = np.random.unifor(0., 100., size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

def demean(X):
  	return X - np.mean(X, axis=0)

def f(w, X):
  	return np.sum((X.dot(w)**2)) / len(X)

def df(w, X):
  	return X.T.dot(X.dot(w)) * 2. / len(X)
  
def direction(w):
  	"""w为单位向量"""
  	return w / np.linalg.norm(w)
  
def first_compent(X, initial_w, eta, n_iters=1e4, epsilon=1e-8):
		"""批量梯度下降法"""
    w = direction(initial_w)
    cur_iter = 0
    while cur_iter < n_iters:
        gradient = df(w, X)
        last_w = w
        w = w + eta * gradient
        w = direction(w)  # 注意：每次求一个单位方向
        if (abs(f(w, X) - f(last_w, X)) < epsilon):
            break
        cur_iter += 1
    return w
 

initial_w  = np.random.random(X.shape[1])  # 注意：不能用0向量开始
eta = 0.001
X = demean(X)
# 求第一主成分的轴
w = first_compent(X, initial_w, eta)
print(w)

# X2 = np.empty(X.shape)
# for i in range(len(X)):  # 循环求解
#   	X2[i] = X[i] - X[i].dot(w) * w
    
X2 = X - X.dot(w).reshape(-1, 1) * w  # 向量化

# 绘图
plt.scatter(X2[:, 0], X2[:, 1])
plt.show()

# 求第二主成分的轴
w2 = first_compent(X2, initial_w, eta)
print(w2)


# 前n项主成分求解方法
def first_n_components(n, X, eta=0.01, n_iters=1e4, epsilon=1e-8):
  	X_pca = X.copy()
    X_pca = demaean(X_pca)
    res = []
    for i in range(n):
      	initial_w = np.random.random(X_pca.shape[1])
        w = first_component(X_pca, initial_w, eta)
        res.append(w)
        X_pca = X_pca -X_pca.dot(w).reshape(-1, 1) * w
    return res
  
res = first_n_components(2, X)
print(res)
```

- 类创建

```python
import numpy as np


class PCA:

    def __init__(self, n_components):
        """初始化PCA"""
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components  # n维主成分 
        self.components_ = None

    def fit(self, X, eta=0.01, n_iters=1e4):
        """获得数据集X的前n个主成分"""
        assert self.n_components <= X.shape[1], \
            "n_components must not be greater than the feature number of X"

        def demean(X):
          	"""均值归零"""
            return X - np.mean(X, axis=0)

        def f(w, X):
          	"""目标函数"""
            return np.sum((X.dot(w) ** 2)) / len(X)

        def df(w, X):
          	"""目标函数梯度"""
            return X.T.dot(X.dot(w)) * 2. / len(X)

        def direction(w):
          	"""单位向量方向"""
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, epsilon=1e-8):
						"""梯度上升法求第一主成分"""
            w = direction(initial_w)
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = df(w, X)
                last_w = w
                w = w + eta * gradient
                w = direction(w)
                if (abs(f(w, X) - f(last_w, X)) < epsilon):
                    break

                cur_iter += 1

            return w

        X_pca = demean(X)
        self.components_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            initial_w = np.random.random(X_pca.shape[1])
            w = first_component(X_pca, initial_w, eta, n_iters) 
            self.components_[i,:] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        """将给定的X，映射到各个主成分分量中"""
        assert X.shape[1] == self.components_.shape[1]

        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """将给定的X，反向映射回原来的特征空间"""
        assert X.shape[1] == self.components_.shape[0]

        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components

```

使用

```python
import numpy as np
import matplotlib.pyplot as plt
from playML.PCA import PCA

X = np.empty((100, 2))
X[:, 0] = np.random.unifor(0., 100., size=100)
X[:, 1] = 0.75 * X[:, 0] + 3. + np.random.normal(0, 10., size=100)

pca = PCA(n_components=2)  # 2维的数据
pca.fit(X)
print(pca.components_)

pca = PCA(n_componets=1)
pca.fit(X)

# 降维
X_reduction = pca.transform(X)
print(X_reduction.shape)

# 恢复
X_restore = pca.inverse_transform(X_reduction)
print(X_restore.shape)

# 绘图
plt.scatter(X[:, 0], X[:, 1], color='b', alpha=0.5)
plt.scatter(X_restore[:, 0], X_restore[:, 1], color='r', alpha=0.5)
plt.show()
```

## sklearn

不是使用梯度上升法，使用了数学解法

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(X)
print(pca.components_)

# 降维
X_reduction = pca.transform(X)
print(X_reduction.shape)

# 恢复
X_restore = pca.inverse_transform(X_reduction)
print(X_restore.shape)
```

示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)
print(X_train.shpe)

# 全维度数据集
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_score = knn_clf.score(X_test, y_test)
print(knn_score)

# 降维处理，维度过少，准确率低，但是可视化
pca = PCA(n_components=2)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
knn_score = knn_clf.score(X_test_reduction, y_test)
print(knn_score)

for i in range(10):
  	plt.scatter(X_reduction[y==i, 0], X_reduction[y==i, 1], alpha=0.8)
plt.show()

# 保证准确率的降维
pca = PCA(0.95)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
knn_score = knn_clf.score(X_test_reduction, y_test)
print(knn_score)
```

## 合适的维度

对于降低到合适的维度才能在效率和准确度上保持平衡的方法

```
1. 网格搜索，将维数信息作为超参数
2. 使用pca.explained_variance_ratio_
```

示例

```python
# 方法一：查看，选取n_components
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
pca.explained_variance_ratio_  # 查看各个维度所能代表信息的比率
# 可视化
plt.plot([i for i in range(X_train.shape[1])],
        [np.sum(pca.explained_variance_ratio_[: i+1]) for i in range(X_train.shape[1])]
        )
plt.show()

# 方法二：输入准确度，自动计算n_components
pca = PCA(0.95)
pca.fit(X_train)
pca.n_components_
```

## MNIST

手写数字识别

```python
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']
print(X.shape)
X_train = np.array(X[:60000], dtype=float)
print(X_train.shape)
y_train = np.array(X[:60000], dtype=float)
X_test = np.array(X[60000:], dtype=float)
y_test = np.array(X[60000:], dtype=float)

# 由于数据在同一量纲维度下，故不需要归一化
# 全特征knn训练
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_score = knn_clf.score(X_test, y_test)
print(knn_score)

# 降维，提高效率，降噪，可能提高了准确率
pca = PCA(0.9)
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
print(X_train_reduction.shape)
X_test_reduction = pca.transform(X_test)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_reduction, y_train)
knn_score = knn_clf.score(X_test_reduction, y_test)
print(knn_score)
```

## 降噪

降低维度，丢失了信息，信息中也包含了噪声，故保留下来的信息，丢失需要的信息的同时也去除了部分噪声

手写识别示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

digits = datasets.load_digits()
X = digits.data
y = digits.target

noisy_digits = X + np.random.normal(0, 4, size=X.shape)

# 可视化噪音数据
example_digits = noisy_digits[y==0, :][: 10]
for num in range(1, 10):
  	X_num = noisy_digits[y==num, :][:10]
    example_digits = np.vstack([example_digits, X_num])
print(example_digits.shape)

def plot_digits(data):
  	fig, axes = plt.subplots(
      10,10, figsize=(10, 10),
      subplot_kw={'xticks':[],'yticks':[]},
      gridspec_kw=dict(hspace=0.1,wspace=0.1)
      )
    for i, ax in enumerate(axes.flat):
      	ax.imshow(
        	data[i].reshape(8, 8),
          cmap='binary',
          interpolation='nearest',
          clim=(0, 16)
        )
    plt.show()
    
plot_digits(example_digits)

# PCA降维降噪
pca = PCA(0.5)
pca.fit(noisy_digits)
components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)
```

## 特征脸

$$
X = \left( \begin{array}{ccc}  X_1^{(1)} & X_2^{(1)} & \ldots & X_n^{(1)} \\ X_1^{(2)} & X_2^{(2)} & \ldots & X_n^{(2)} \\ \cdots &&& \cdots \\ X_1^{(m)} & X_2^{(m)} &\ldots & X_n^{(m)}\end{array} \right)
$$

$$
W_k = \left( \begin{array}{ccc}  W_1^{(1)} & W_2^{(1)} & \ldots & W_n^{(1)} \\ W_1^{(2)} & W_2^{(2)} & \ldots & W_n^{(2)} \\ \cdots &&& \cdots \\ W_1^{(k)} & W_2^{(k)} &\ldots & W_n^{(k)}\end{array} \right)
$$



由$X$的`m*n`矩阵乘以`n*k`的$W_k$矩阵(前`k`个主成分)，形成`m*k`的矩阵$X_k$
$$
X\cdot{W_k^T} = X_k
$$
其中$W_k$的每一行都可以视为$X$的n维特征转换后的重要度依次降低的k维特征

示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

faces = datasets.fetch_lfw_people()
print(faces.keys())
print(faces.shape)
print(faces.images.shape)

random_indexes = np.random.permutation(len(faces.data))
X = faces.data[random_indexes]
example_faces = X[:36, :]
print(example_faces.shape)

def plot_faces(faces):
  	ig, axes = plt.subplots(
      6,6, figsize=(10, 10),
      subplot_kw={'xticks':[],'yticks':[]},
      gridspec_kw=dict(hspace=0.1,wspace=0.1)
      )
    for i, ax in enumerate(axes.flat):
      	ax.imshow(
        	data[i].reshape(62, 47),
          cmap='bone'
        )
    plt.show()
    
plot_faces(example_faces)


# 特征脸
pca = PCA(svd_solver='randomized')
pca.fit(X)
print(pca.components_.shape)
plot_faces(pca.components_[:36, :])


# 人脸识别数据库
faces2 = datasets.fetch_lfw_people(min_faces_per_person=60)
print(faces2.data.shape)
print(faces2.target_names)
print(len(faces2.target_names))
```





