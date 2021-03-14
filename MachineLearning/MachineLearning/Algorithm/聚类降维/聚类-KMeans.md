[TOC]
# 聚类

人们往往根据事物之间的距离远近或相似程度来判定类别。个体与个体之间的距离越近，其相似性可能也越大，是同类的可能性也越大，聚在一起形成类别的可能性也就越大。因此有了聚类分析的基本原则。

聚类的算法主要有：

- 距离切分方法

    根据特征之间的距离进行聚类划分。如K-means

- 密度方法

    通过定义每个子集的最小成员数量和成员之间距离实现划分。如DBSCAN

- 模型方法

    不完全将样本认定为属于某个子集，而是指出样本属于各子集的可能性。如概率模型的高斯混合模型，神经网络模型

- 层次方法

    不同于其他聚类将总体划分成彼此地位平等的多个子集，层次方法最终将数据集划分成有父子关系的树形结构。这样就可以在聚类的同时考虑各子类之间的亲缘关系。如BIRCH

## 基本原则

聚类过程所依据的距离主要与明氏距离、马氏距离等几大类。

设样本数据可以用如下矩阵形式表示
$$
X= \left( \begin{array}{}
x_{11} & x_{12} &  \cdots & x_{1p} \\
x_{21} & x_{22} &  \cdots & x_{2p} \\
\vdots \\
x_{n1} & x_{n2} &  \cdots & x_{np} \\
\end{array} \right),记为X=\{x_{ij}\}_{n\times p}
$$
设 $d_{ij}$ 表示第 $i$ 个样本与第 $j$ 个样本之间的距离。如果 $d_{ij}$满足以下4个条件，则称其为**距离**

- $d_{ij} \geq 0$，对于一切 $i,j$；
- $d_{ij} = 0$，等价于 $i=j$；
- $d_{ij} = d_{ji}$，对于一切 $i,j$；
- $d_{ij} \leq d_{ik}+d_{kj}$，对于一切 $i,j,k$；

第1个条件表明聚类分析中的距离是非负的；第2个条件表明个体自身与自身的距离为0；第3个条件表明距离的对等性，即A和B之间的距离与B和A之间的距离是一致的；第4个条件表明两点之间直线距离是最小的。

**明氏距离**是最常用的距离度量方法之一，其计算公式为
$$
d_{ij}(q) = (\sum_{k=1}^{p}{|x_{ik}-x_{jk}|^q})^{1/q}
$$
有如下几种典型情况

- 当 $q=1$时，$d_{ij}(1) = \sum_{k=1}^{p}{|x_{ik}-x_{jk}|}$ 称为**绝对距离**
- 当 $q=2$时，$d_{ij}(2) = (\sum_{k=1}^{p}{|x_{ik}-x_{jk}|^2})^{1/2}$称为**欧氏距离**
-  当$q=1$时，$d_{ij}(\infty) = \max_{1\leq k \leq p}{|x_{ik}-x_{jk}|}$称为**车比雪夫距离**

但是明氏距离的大小与个体指标的观测单位有关，没有考虑指标之间的相关性。为克服此缺点，可以考虑马氏距离进行改造。**马氏距离** 是由协方差矩阵计算出来的相对距离，具体计算公式如下
$$
d_{ij} = (X_i-X_j)^{'}\Sigma^{-1}(X_i-X_j)
$$
其中，$\Sigma$ 是多维随机变量的协方差矩阵。

除了最短距离原则进行分类之外，还可以采用相关系数、相似系数、匹配系数等指标来衡量个体之间的相似性，以此为依据进行分类。

在分类过程中，为了便于分析，有如下3个重要原则：

- 同质性原则：同一类中个体之间有较大的相似性
- 互斥性原则：不同类中的个体差异很大
- 完备性原则：每个个体在同一次分类过程中，能且只能分在一个类别中

实际应用中，以最短距离原则进行系统聚类比较常用。

## K-Means

是无监督学习中的一种算法

聚类：相似的东西分到一组

难点：如何评估、如何调参

优点

```
简单、快读、适合常规数据集
```

缺点

```
k值难确定
时间复杂度为O(tnmk),t表示迭代次数，n表示数据点个数，m表示特征数，k表示簇数。数据量大时效率下降，且容易陷入局部最优解
很难发现任意形状的簇
初始值选择对结果影响大
```

### 原理

核心思想：对数据集 $D={x^1, x^2, \cdots,x^m}$ ，考虑所有可能的k个簇集合，希望能找到一个簇集合 ${C_1,C_2,\cdots,C_k}$，使得每一个点到其对应簇的中心的距离的平方和最小
$$
min\sum_{i=1}^K\sum_{x^j\in{C_i}}{(c_i, x^j)^2}
$$

其中，$c_i=\frac{1}{|C_i|}\sum_{x^j \in C_i}{x^j}$ 表示簇$C_i$ 的中心

但要找到满足最小化条件的簇非常困难，K-means则采用了贪心的策略，通过迭代的方式找到近似解。

算法首先随机挑选k个样本点作为初始簇，对剩余的样本点，根据其道所有簇中心的距离，将点分配到最近的簇中，然后更新每一个簇的中心位置，不断重复这个过程，直到所有的样本点不再变化为止。

- 算法流程

```
输入：包含m个数据点{x^1,x^2,\cdots,x^m}的数据集D
输出：k个簇{C_1,C_2,\cdots,C_k}
1.若样本集的个数m \le k, 则将每一个点单独作为一个簇，程序退出
2.任意从数据集D中选取k个点，作为初始的k个簇
3.将剩余的数据点，分别计算到k个簇的距离，把数据点分配到其最近的簇中
4.重新计算每个簇的中心的位置
5.重复执行下面的操作，直到没有点需要被调整
	5.1 计算每一个点到簇的距离，并将数据点分配到与其最近的簇中
	5.2 重新计算每个簇的中心位置
6.输出k个簇{C_1,C_2,\cdots,C_k}
```

### 实现

```python
# 自定义实现kmeans
def find_clusters(X, n_clusters, rseed=2):
    # 1.随机选择簇中心点
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # 2a.基于最近的中心指定标签
        labels = pairwise_distances_argmin(X, centers)
        # 2b.根据点的平均值找到新的中心
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # 2c.确认收敛
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


centers, labels = find_clusters(X, 4)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# plt.show()
```

### sklearn

#### 实现

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans, SpectralClustering

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


plt.scatter(X[:, 0], X[:, 1], s=50)

# 使用kmeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()

# 期望最大化
# 期望最大化(E_M)应用于数据科学的很多场景，K-means是该算法的一个简单应用
# 期望最大化的方法步骤：1.猜测一些簇中心点，2.重复直至收敛
# 期望步骤(E-step):将点分配至离其最近的簇中心点；最大化步骤(M-step):将簇中心点设置为所有点坐标的平均值

# 自定义实现kmeans
def find_clusters(X, n_clusters, rseed=2):
    # 1.随机选择簇中心点
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # 2a.基于最近的中心指定标签
        labels = pairwise_distances_argmin(X, centers)
        # 2b.根据点的平均值找到新的中心
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # 2c.确认收敛
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


centers, labels = find_clusters(X, 4)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

# 注意事项
# 1.可能不是全局最优解
centes, labels = find_clusters(X, 4, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
# 2.簇数量需事先定好
labels = KMeans(6, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

# 3.kmeans算法只能确定线性聚类边界
# 边界很复杂时，算法失效
X, y = make_moons(200, noise=0.05, random_state=0)
labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
# 核变换，将数据投影到更高维度，效果较好
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
```

#### 边界评价

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

blob_centers = np.array(
    [[0.2, 2.3],
     [-1.5, 2.3],
     [-2.8, 1.8],
     [-2.8, 2.8],
     [-2.8, 1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


plt.figure(figsize=(8, 4))
plot_clusters(X)
save_fig("blobs_diagram")
plt.show()

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

print(y_pred)
print(y_pred is kmeans.labels_)
# 聚类中心
print(kmeans.cluster_centers_)
# 注意，“KMeans”实例保留了它所训练的实例的标签。有些令人困惑的是，在这个上下文中，实例的_label_是被分配给实例的集群的索引
print(kmeans.labels_)
# 预测
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
res = kmeans.predict(X_new)
print(res)


# 决策边界
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
save_fig("voronoi_diagram")
plt.show()
"""
不错！边缘附近的一些实例可能被分配到了错误的集群中，但总体来看，它看起来相当不错。
"""

# hard和soft聚类
# 与其为每个实例任意选择最近的集群(称为_hard clustering_)，不如测量每个实例到所有5个中心的距离。这就是“transform()”方法的作用:
res = kmeans.transform(X_new)
print(res)
# 您可以验证这确实是每个实例和每个质心之间的欧几里得距离：
res = np.linalg.norm(np.tile(X_new, (1, k)).reshape(-1, k, 2) - kmeans.cluster_centers_, axis=2)
print(res)

# 惯性
# 为了选择最佳模型，我们需要一种评估K均值模型性能的方法。不幸的是，聚类是一个无监督的任务，所以我们没有目标。但至少我们可以测量每个实例和它的质心之间的距离。这是“惯性”度量背后的理念：
print(kmeans.inertia_)  # 惯性是每个训练实例与其最近质心之间平方距离之和：
X_dist = kmeans.transform(X)
res = np.sum(X_dist[np.arange(len(X_dist)), kmeans.labels_]**2)
print(res)
print(kmeans.score(X))  # 惯性的负数

# 不同迭代次数下的kmeans
kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,
                     algorithm="full", max_iter=1, random_state=1)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,
                     algorithm="full", max_iter=2, random_state=1)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,
                     algorithm="full", max_iter=3, random_state=1)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)
plt.figure(figsize=(10, 8))

plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)
plt.title("Update the centroids (initially randomly)", fontsize=14)

plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
plt.title("Label the instances", fontsize=14)

plt.subplot(323)
plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
plot_centroids(kmeans_iter2.cluster_centers_)

plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)

plt.subplot(325)
plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
plot_centroids(kmeans_iter3.cluster_centers_)

plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)

save_fig("kmeans_algorithm_diagram")
plt.show()
```

#### 可变性

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

blob_centers = np.array(
    [[0.2, 2.3],
     [-1.5, 2.3],
     [-2.8, 1.8],
     [-2.8, 2.8],
     [-2.8, 1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


plt.figure(figsize=(8, 4))
plot_clusters(X)
save_fig("blobs_diagram")
plt.show()

# 决策边界
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


# kmeans可变性
# 在原来的K-Means算法中，质心只是随机初始化的，算法只是简单地运行一次迭代来逐步地改进质心，如上所述。
# 但是，这种方法的一个主要问题是，如果多次运行K-Means（或使用不同的随机种子），它可能会收敛到非常不同的解决方案，如下所示：
#
def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)

kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1,
                         algorithm="full", random_state=11)
kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1,
                         algorithm="full", random_state=19)

plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,
                          "Solution 1", "Solution 2 (with a different random init)")

save_fig("kmeans_variability_diagram")
plt.show()

# 多次初始化
# 因此，解决可变性问题的一种方法是简单地用不同的随机初始化多次运行K-Means算法，并选择使惯性最小的解决方案。
print(kmeans_rnd_init1.inertia_)  # 具有较高的惯性，效果较差
print(kmeans_rnd_init2.inertia_)
# 当您设置“n_init”超参数时，Scikit Learn会运行原始算法“n_init”次，并选择使惯性最小化的解决方案。默认情况下，Scikit Learn设置为“n_init=10”。
kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10,
                              algorithm="full", random_state=11)
kmeans_rnd_10_inits.fit(X)

plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans_rnd_10_inits, X)
plt.show()

```

#### K-Means++

与其完全随机初始化质心，不如使用[2006年论文](https://goo.gl/eNUPw6)中提出的以下算法初始化质心：

- 从数据集中随机选取一个质心$c_1$。
- 取一个新的中心 $c_i$，选择一个实例 $\mathbf{x}_i$，概率为：$D(\mathbf{x}_i)^2$/$\sum\limits_{j=1}^{m}{D(\mathbf{x}_j)}^2$，其中$D(\mathbf{x}_i)$是实例 $\mathbf{x}_i$ 与已选择的最近质心之间的距离。这种概率分布确保距离已选择的质心较远的实例更有可能被选为质心。
- 重复上一步，直到选择了所有 $k$ 质心。

K-Means++算法的其余部分只是普通的K-Means。通过这种初始化，K-Means算法收敛到次优解的可能性要小得多，因此可以大大减少`n_init`。大多数时候，这在很大程度上弥补了初始化过程的额外复杂性。

sklearn默认的KMeans中就是采用的此算法

```python
KMeans(init='k-means++')
```

#### 加速聚类

通过避免许多不必要的距离计算，K-Means算法可以大大加快速度：这是通过利用三角形不等式（给定三个点A、B和C，距离AC始终是AC≤AB+BC）和跟踪实例和质心之间距离的上下限（见下文）来实现的。参见[2003年论文](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf)。

````python
# 要使用Elkan的K-Means变体，只需设置`algorithm=“Elkan”`。
# 请注意，它不支持稀疏数据，因此在默认情况下，Scikit Learn对密集数据使用“elkan”，对稀疏数据使用“full”（常规K-Means算法）。
%timeit -n 50 KMeans(algorithm="elkan").fit(X)
%timeit -n 50 KMeans(algorithm="full").fit(X)
````

#### 小批量

Scikit Learn还实现了一个支持小批量的K-Means算法的变体（参见[本文](http://www.tukmets.pdf))

```python
import time
from timeit import timeit

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

blob_centers = np.array(
    [[0.2, 2.3],
     [-1.5, 2.3],
     [-2.8, 1.8],
     [-2.8, 2.8],
     [-2.8, 1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


plt.figure(figsize=(8, 4))
plot_clusters(X)
save_fig("blobs_diagram")
plt.show()

# 方法一：MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)
print(minibatch_kmeans.inertia_)

# 方法二：memmap
filename = "my_mnist.data"
m, n = 50000, 28*28
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)
minibatch_kmeans.fit(X_mm)

# 如果你的数据太大，你不能使用memmap，事情就会变得更复杂。让我们开始写一个函数来加载下一批(在现实生活中，你会从磁盘加载数据)
def load_next_batch(batch_size):
    return X[np.random.choice(len(X), batch_size, replace=False)]
# 现在我们可以训练模型，一次只喂一批。我们还需要实现多次初始化，并使模型保持最小的惯性
np.random.seed(42)

k = 5
n_init = 10
n_iterations = 100
batch_size = 100
init_size = 500  # more data for K-Means++ initialization
evaluate_on_last_n_iters = 10

best_kmeans = None

for init in range(n_init):
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)
    X_init = load_next_batch(init_size)
    minibatch_kmeans.partial_fit(X_init)

    minibatch_kmeans.sum_inertia_ = 0
    for iteration in range(n_iterations):
        X_batch = load_next_batch(batch_size)
        minibatch_kmeans.partial_fit(X_batch)
        if iteration >= n_iterations - evaluate_on_last_n_iters:
            minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_

    if (best_kmeans is None or
        minibatch_kmeans.sum_inertia_ < best_kmeans.sum_inertia_):
        best_kmeans = minibatch_kmeans

res = best_kmeans.score(X)
print(res)

time1 = time.process_time()
KMeans(n_clusters=5).fit(X)
time2 = time.process_time()
print(time2-time1)
time3 = time.process_time()
MiniBatchKMeans(n_clusters=5).fit(X)
time4 = time.process_time()
print(time4-time3)
"""
MiniBatchKMeans快多了！然而，它的性能通常较低（更高的惯性），并且随着k的增加而不断下降。
"""
# 小批量K均值和常规K均值之间的惯性比和训练时间比
times = np.empty((100, 2))
inertias = np.empty((100, 2))
for k in range(1, 101):
    kmeans = KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    print("\r{}/{}".format(k, 100), end="")
    times[k-1, 0] = timeit("kmeans.fit(X)", number=10, globals=globals())
    times[k-1, 1]  = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
    inertias[k-1, 0] = kmeans.inertia_
    inertias[k-1, 1] = minibatch_kmeans.inertia_

plt.figure(figsize=(10,4))

plt.subplot(121)
plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
#plt.ylabel("Inertia", fontsize=14)
plt.title("Inertia", fontsize=14)
plt.legend(fontsize=14)
plt.axis([1, 100, 0, 100])

plt.subplot(122)
plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
#plt.ylabel("Training time (seconds)", fontsize=14)
plt.title("Training time (seconds)", fontsize=14)
plt.axis([1, 100, 0, 6])
#plt.legend(fontsize=14)

save_fig("minibatch_kmeans_vs_kmeans")
plt.show()
```

#### 选择k值

有两种方法来判断k值是否合适：

- 肘部法：通过惯性与k的曲线，在肘部位置即较合适的k值
- 剪影得分：较高的分值表示较好的k值

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


blob_centers = np.array(
    [[0.2, 2.3],
     [-1.5, 2.3],
     [-2.8, 1.8],
     [-2.8, 2.8],
     [-2.8, 1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


plt.figure(figsize=(8, 4))
plot_clusters(X)
save_fig("blobs_diagram")
plt.show()

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

print(y_pred)
print(y_pred is kmeans.labels_)
# 聚类中心
print(kmeans.cluster_centers_)
# 注意，“KMeans”实例保留了它所训练的实例的标签。有些令人困惑的是，在这个上下文中，实例的_label_是被分配给实例的集群的索引
print(kmeans.labels_)
# 预测
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
res = kmeans.predict(X_new)
print(res)


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


kmeans_k3 = KMeans(n_clusters=3, random_state=42)
kmeans_k8 = KMeans(n_clusters=8, random_state=42)

plot_clusterer_comparison(kmeans_k3, kmeans_k8, X, "$k=3$", "$k=8$")
save_fig("bad_n_clusters_diagram")
plt.show()

print(kmeans_k3.inertia_, kmeans_k8.inertia_)
"""
相对于5来说，3和8不大合适，虽然8的惯性更低
我们不能简单地把惯性最小化的k值取出来，因为随着我们增加k，它会不断降低。
实际上，簇越多，每个实例就越接近其最近的质心，因此惯性也就越低。但是，我们可以将惯性绘制为k的函数，并分析结果曲线
"""
# 方法一：肘部法
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(4, inertias[3]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
             )
plt.axis([1, 8.5, 0, 1300])
save_fig("inertia_vs_k_diagram")
plt.show()
"""
正如所看到的，在k=4处有一个拐弯，这意味着少于这个值的集群将是糟糕的，而更多的集群将不会有太大帮助，并且可能会将集群削减一半。
所以k=4是个不错的选择。当然，在这个例子中，它并不完美，因为它意味着左下角的两个blob将被视为一个单独的集群，但是它仍然是一个非常好的集群。
"""
plot_decision_boundaries(kmeans_per_k[4 - 1], X)
plt.show()

# 方法二：剪影得分
# 另一种方法是查看“剪影得分”，即所有实例中的平均“剪影系数”。
# 一个实例的轮廓系数等于(b-a)/\max(a，b)，其中a是到同一个集群中其他实例的平均距离（它是“平均簇内距离”，而b是“平均最近簇距离”，
# 即到下一个最近的集群实例的平均距离（定义为最小化$b$，排除实例自己的集群）。另一个簇的系数从1到1的边界是错误的，而另一个簇的系数可能在1和1之间变化。
res = silhouette_score(X, kmeans.labels_)
print(res)  # 0.655517642572828
silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]
plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.55, 0.7])
save_fig("silhouette_score_vs_k_diagram")
plt.show()
"""
正如看到的，这种可视化比上一个要丰富得多：特别是，虽然它确认了$k=4$是一个非常好的选择，但它也强调了这样一个事实，即$k=5$也相当不错。
"""
# 当您绘制每个实例的轮廓系数时，会给出一个更具信息量的可视化效果，并按它们所分配的集群和系数的值进行排序。这被称为“轮廓图”
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.figure(figsize=(11, 9))

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)

    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")

    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)

save_fig("silhouette_analysis_diagram")
plt.show()

```

#### 最优类的数量

- 优化惯性

假设合适数量的聚类会产生小的惯性。然而，当累的数目等于样本数时，该值为最小值(0.0)。因此，不能寻找最小值，而应该找一个可以平衡惯性与类的数目的值

- 轮廓分数

轮廓分数基于最大内部凝聚和最大类分离原理。想找到使得数据集细分为彼此互相分离的密集块的剧烈数目。以这种方式，每个类将包含非常相似的元素，选择属于不同类的两个元素，它们的距离应该大于类内元素的最大距离

- Calinski-Harabasz指标

该指标基于类内密集且分类合适的概念。

- 类的不稳定性

基于类的不稳定性的概念，直观来说，如果对相同数据集受到干扰后的样本进行聚类能产生非常相似的结果，那么这种聚类方法是稳定的。

```python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabaz_score, \
    homogeneity_score, completeness_score, adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_distances


# For reproducibility
np.random.seed(1000)


nb_samples = 1000


if __name__ == '__main__':
    # Create dataset
    X, _ = make_blobs(n_samples=nb_samples, n_features=2, centers=3, cluster_std=1.5, random_state=1000)

    # Show the dataset
    fig, ax = plt.subplots(1, 1, figsize=(30, 25))

    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax.scatter(X[:, 0], X[:, 1], marker='o', color='b')

    plt.show()

    # Analyze the inertia
    nb_clusters = [2, 3, 5, 6, 7, 8, 9, 10]

    inertias = []

    for n in nb_clusters:
        km = KMeans(n_clusters=n)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(nb_clusters, inertias)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    ax.grid()

    plt.show()

    # Analyze the silhouette scores
    avg_silhouettes = []

    for n in nb_clusters:
        km = KMeans(n_clusters=n)
        Y = km.fit_predict(X)
        avg_silhouettes.append(silhouette_score(X, Y))

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(nb_clusters, avg_silhouettes)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Average Silhouette score')
    ax.grid()

    plt.show()

    # Draw the silhouette plots
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    nb_clusters = [2, 3, 4, 8]
    mapping = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, n in enumerate(nb_clusters):
        km = KMeans(n_clusters=n)
        Y = km.fit_predict(X)

        silhouette_values = silhouette_samples(X, Y)

        ax[mapping[i]].set_xticks([-0.15, 0.0, 0.25, 0.5, 0.75, 1.0])
        ax[mapping[i]].set_yticks([])
        ax[mapping[i]].set_title('%d clusters' % n)
        ax[mapping[i]].set_xlim([-0.15, 1])
        ax[mapping[i]].grid()
        y_lower = 20

        for t in range(n):
            ct_values = silhouette_values[Y == t]
            ct_values.sort()

            y_upper = y_lower + ct_values.shape[0]

            color = cm.Accent(float(t) / n)
            ax[mapping[i]].fill_betweenx(np.arange(y_lower, y_upper), 0,
                                         ct_values, facecolor=color, edgecolor=color)

            y_lower = y_upper + 20

    # Analyze the Calinski-Harabasz scores
    ch_scores = []

    km = KMeans(n_clusters=n)
    Y = km.fit_predict(X)

    for n in nb_clusters:
        km = KMeans(n_clusters=n)
        Y = km.fit_predict(X)
        ch_scores.append(calinski_harabaz_score(X, Y))

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(nb_clusters, ch_scores)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Calinski-Harabasz scores')
    ax.grid()

    plt.show()

    # Analyze the cluster instability
    nb_noisy_datasets = 10

    X_noise = []

    for _ in range(nb_noisy_datasets):
        Xn = np.ndarray(shape=(1000, 2))

        for i, x in enumerate(X):
            if np.random.uniform(0, 1) < 0.25:
                Xn[i] = X[i] + np.random.uniform(-2.0, 2.0)
            else:
                Xn[i] = X[i]

        X_noise.append(Xn)

    instabilities = []

    for n in nb_clusters:
        Yn = []

        for Xn in X_noise:
            km = KMeans(n_clusters=n)
            Yn.append(km.fit_predict(Xn))

        distances = []

        for i in range(len(Yn) - 1):
            for j in range(i, len(Yn)):
                d = pairwise_distances(Yn[i].reshape(-1, 1), Yn[j].reshape(-1, 1), 'hamming')
                distances.append(d[0, 0])

        instability = (2.0 * np.sum(distances)) / float(nb_noisy_datasets ** 2)
        instabilities.append(instability)

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(nb_clusters, instabilities)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Cluster instability')
    ax.grid()

    plt.show()

    # Analyze the homegeneity, completeness, and Adjusted Rand score
    km = KMeans(n_clusters=3)
    Yp = km.fit_predict(X)

    print('Homegeneity: %.3f' % homogeneity_score(Y, Yp))
    print('Completeness: %.3f' % completeness_score(Y, Yp))
    print('Adjusted Rand score: %.3f' % adjusted_rand_score(Y, Yp))

```

#### 限制

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)


plt.figure(figsize=(8, 4))
plot_clusters(X)
save_fig("blobs_diagram")
plt.show()


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                 cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)

kmeans_good = KMeans(n_clusters=3, init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]), n_init=1, random_state=42)
kmeans_bad = KMeans(n_clusters=3, random_state=42)
kmeans_good.fit(X)
kmeans_bad.fit(X)

plt.figure(figsize=(10, 3.2))

plt.subplot(121)
plot_decision_boundaries(kmeans_good, X)
plt.title("Inertia = {:.1f}".format(kmeans_good.inertia_), fontsize=14)

plt.subplot(122)
plot_decision_boundaries(kmeans_bad, X, show_ylabels=False)
plt.title("Inertia = {:.1f}".format(kmeans_bad.inertia_), fontsize=14)

save_fig("bad_kmeans_diagram")
plt.show()
```

#### 图片压缩

示例1

```python
from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('./images/ladybug.png')
io.imshow(image)
io.show()

rows = image.shape[0]
cols = image.shape[1]
# 对图片进行住那还成数组，以便于运行k-means聚类算法
image = image.reshape(image.shape[0]*image.shape[1], 3)
# 对输入数据进行聚类
kmeans = KMeans(n_clusters=128, n_init=10, max_iter=200)
kmeans.fit(image)

cluster = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
labels = np.asarray(kmeans.labels_, dtype=np.uint8)
labels = labels.reshape(rows, cols)

print(cluster.shape)
np.save('codebook_test.npy', cluster)
io.imsave('compressed_test.jpg', labels)

image = io.imread('compressed_test.jpg')
io.imshow(image)
io.show()
```

示例2

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from sklearn.cluster import MiniBatchKMeans

china = load_sample_image("china.jpg")
print(china.shape)  # (427, 640, 3)
# 图片存储在一个三维数组(height, width, RGB)中，以0～255的整数表示红/蓝/绿
# ax = plt.axes(xticks=[], yticks=[])
# ax.imshow(china)
# plt.show()
# 将这组像素转换成三维颜色空间中的一群数据点，先将数据变形为[n_samples * n_features]，然后缩放颜色至取值为0～1
data = china / 255.0  # 转换为0～1区间值
data = data.reshape(427 * 640, 3)
print(data.shape)  # (273280, 3)


# 在颜色空间中对像素进行可视化, 使用包含前10000个像素的子集
def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    # 随机选择一个子集
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(title, size=20)


# plot_pixels(data, title='Input color space: 16 million possible colors')
# 对像素空间(特征矩阵)使用kmeans聚类，将1600万中颜色(255*255*255)缩减至16种
# 由于数据集比较大，使用MiniBatchKMeans算法进行计算
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
# plot_pixels(data, colors=new_colors, title='Reduced color space: 16 colors')
# 用计算结果对原始像素重新着色，即每个像素被指定为距离其最近的簇中心点的颜色
china_recolored = new_colors.reshape(china.shape)
fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image', size=16)
# 虽然丢失了某些细节，但是图像总体还是非常易于辨识的，实现了近100万的压缩比
plt.show()
```

#### 图像分割

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

from matplotlib.image import imread
image = imread(os.path.join("images","ladybug.png"))
print(image.shape)  # (533, 800, 3)

X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)

segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')

save_fig('image_segmentation_diagram', tight_layout=False)
plt.show()
```

#### 预处理

```python
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_digits, y_digits = load_digits(return_X_y=True)
# 训练集测试集划分
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
# 逻辑回归
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)
log_reg.fit(X_train, y_train)
# 评分
res = log_reg.score(X_test, y_test)
print(res)  # 0.97
"""
好吧，这是我们的基准：96.7%的准确率。
让我们来看看使用K均值作为预处理步骤是否可以做得更好。
我们将创建一个管道，首先将训练集聚类成50个簇，然后用它们到50个簇的距离替换图像，然后应用逻辑回归模型
"""
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)),
])
pipeline.fit(X_train, y_train)
res = pipeline.score(X_test, y_test)
print(res)  # 0.98
print(1 - (1 - 0.98) / (1 - 0.97))
"""
那怎么样？我们几乎把错误率除以2倍！但是我们完全随意地选择了集群的数量$k$，我们肯定可以做得更好。
由于K-Means只是分类管道中的一个预处理步骤，因此为$K$找到一个好的值要比以前简单得多：
不需要执行轮廓分析或最小化惯性，因此$K$的最佳值就是产生最佳分类性能的值。
"""
# 交叉验证取最优参数
param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)
print(grid_clf.best_params_)
print(grid_clf.score(X_test, y_test))

```

#### 半监督学习

聚类的另一个用例是在半监督学习中，当我们有大量未标记的实例而很少有标记的实例时。

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


X_digits, y_digits = load_digits(return_X_y=True)
# 训练集测试集划分
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
# 逻辑回归
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)
log_reg.fit(X_train, y_train)
# 评分
res = log_reg.score(X_test, y_test)
print(res)  # 0.97

n_labeled = 50
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
res = log_reg.score(X_test, y_test)
print(res)  # 0.83
"""
当然比早些时候少了很多。让我们看看怎样才能做得更好。
首先，让我们将训练集分为50个簇，然后对于每个簇，让我们找到最接近质心的图像。我们将这些图像称为代表图像
"""
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')

save_fig("representative_images_diagram", tight_layout=False)
plt.show()

y_representative_digits = np.array([
    4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
    5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
    6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
    4, 2, 9, 4, 7, 6, 2, 3, 1, 1])

# 现在我们有一个只有50个标记实例的数据集，但是它们不是完全随机的实例，而是其集群的代表性图像。让我们看看性能是否更好：
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
res = log_reg.score(X_test, y_test)
print(res)
"""
我们从82.7%的准确率跃升到92.4%，尽管我们仍然只在50个实例上训练模型。由于给实例添加标签通常是昂贵和痛苦的，特别是当它必须由专家手动完成时，
最好让它们标记具有代表性的实例，而不仅仅是随机实例。但也许我们可以更进一步：如果我们将标签传播到同一个集群中的所有其他实例呢？
"""
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)
log_reg.fit(X_train, y_train_propagated)
res = log_reg.score(X_test, y_test)
print(res)
"""
我们得到了一点小小的精度提升。总比没有好，但是我们可能应该只将标签传播到最接近质心的实例，因为通过传播到整个集群，我们当然包含了一些异常值。
我们只将标签传播到最接近质心的第20个百分位：
"""
percentile_closest = 20

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)

res = log_reg.score(X_test, y_test)
print(res)
"""
不错！只有50个标记实例（平均每个类只有5个示例！）我们得到了94.2%的性能，这与逻辑回归在全标记数字集上的性能（96.7%）非常接近。
这是因为传播的标签实际上相当不错：它们的准确率非常接近99%：
"""
res = np.mean(y_train_partially_propagated == y_train[partially_propagated])
print(res)
"""
现在，您可以重复“主动学习”：
1手动标记分类器最不确定的实例，如果可能的话，通过在不同的集群中选择它们。
2用这些附加标签训练一个新模型。
"""
```

### tensorflow

将iris数据集聚类成三类

```python
# -*- coding: utf-8 -*-
# K-means with TensorFlow
#----------------------------------
#
# This script shows how to do k-means with TensorFlow

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import datasets
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA  # 为后续将四维的结果转换为二维数据可视化
from sklearn.preprocessing import scale
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

iris = datasets.load_iris()

num_pts = len(iris.data)
num_feats = len(iris.data[0])

# Set k-means parameters
# There are 3 types of iris flowers, see if we can predict them
k = 3
generations = 25

data_points = tf.Variable(iris.data)
cluster_labels = tf.Variable(tf.zeros([num_pts], dtype=tf.int64))

# Randomly choose starting points
rand_starts = np.array([iris.data[np.random.choice(len(iris.data))] for _ in range(k)])

centroids = tf.Variable(rand_starts)

# In order to calculate the distance between every data point and every centroid, we
#  repeat the centroids into a (num_points) by k matrix.
centroid_matrix = tf.reshape(tf.tile(centroids, [num_pts, 1]), [num_pts, k, num_feats])
# Then we reshape the data points into k (3) repeats
point_matrix = tf.reshape(tf.tile(data_points, [1, k]), [num_pts, k, num_feats])
distances = tf.reduce_sum(tf.square(point_matrix - centroid_matrix), axis=2)

# Find the group it belongs to with tf.argmin()
centroid_group = tf.argmin(distances, 1)


# Find the group average
def data_group_avg(group_ids, data):
    # Sum each group
    sum_total = tf.unsorted_segment_sum(data, group_ids, 3)
    # Count each group
    num_total = tf.unsorted_segment_sum(tf.ones_like(data), group_ids, 3)
    # Calculate average
    avg_by_group = sum_total/num_total
    return avg_by_group


means = data_group_avg(centroid_group, data_points)

update = tf.group(centroids.assign(means), cluster_labels.assign(centroid_group))

init = tf.global_variables_initializer()

sess.run(init)

for i in range(generations):
    print('Calculating gen {}, out of {}.'.format(i, generations))
    _, centroid_group_count = sess.run([update, centroid_group])
    group_count = []
    for ix in range(k):
        group_count.append(np.sum(centroid_group_count==ix))
    print('Group counts: {}'.format(group_count))
    

[centers, assignments] = sess.run([centroids, cluster_labels])


# Find which group assignments correspond to which group labels
# First, need a most common element function
def most_common(my_list):
    return max(set(my_list), key=my_list.count)


label0 = most_common(list(assignments[0:50]))
label1 = most_common(list(assignments[50:100]))
label2 = most_common(list(assignments[100:150]))

group0_count = np.sum(assignments[0:50] == label0)
group1_count = np.sum(assignments[50:100] == label1)
group2_count = np.sum(assignments[100:150] == label2)

accuracy = (group0_count + group1_count + group2_count)/150.

print('Accuracy: {:.2}'.format(accuracy))

# Also plot the output
# First use PCA to transform the 4-dimensional data into 2-dimensions
pca_model = PCA(n_components=2)
reduced_data = pca_model.fit_transform(iris.data)
# Transform centers
reduced_centers = pca_model.transform(centers)

# Step size of mesh for plotting
h = .02

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Get k-means classifications for the grid points
xx_pt = list(xx.ravel())
yy_pt = list(yy.ravel())
xy_pts = np.array([[x, y] for x, y in zip(xx_pt, yy_pt)])
mytree = cKDTree(reduced_centers)
dist, indexes = mytree.query(xy_pts)

# Put the result into a color plot
indexes = indexes.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(indexes, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

# Plot each of the true iris data groups
symbols = ['o', '^', 'D']
label_name = ['Setosa', 'Versicolour', 'Virginica']
for i in range(3):
    temp_group = reduced_data[(i*50):(50)*(i+1)]
    plt.plot(temp_group[:, 0], temp_group[:, 1], symbols[i], markersize=10, label=label_name[i])
# Plot the centroids as a white X
plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
plt.title('K-means clustering on Iris Dataset Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='lower right')
plt.show()

```




