# 高斯混合模型

Gaussian Mixture Model

高斯混合模型可被看作是kmeans思想的一个扩展，也是一种非常强大的聚类评估工具。

## kmeans的缺陷

只要给定简单且分离行好的数据，kmeans就可以找到合适的聚类结果

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1]  # 交换列是为了方便画图
# 用kmeans标签画出数据
kmeans = KMeans(4, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.show()

```

通过观察，某些点的归属簇比其他店的归属簇更加明确。如中间的两个簇有区域重合，对于重合部分的点被分配到哪个簇并不很有信心。kmeans模型本身也没有度量簇的分配概率或不确定性的方法（虽然可用数据重抽样方法bootstrap来估计不确定性）。

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1]  # 交换列是为了方便画图


# 聚类可视化
def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)
    # 画出输入数据
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    # 画出kmeans模型的表示
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))


# kmeans = KMeans(n_clusters=4, random_state=0)
# plot_kmeans(kmeans, X)
# plt.show()

# kmeans要求这些簇的模型必须是圆形，对数据进行转换，则分配混乱
rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))
kmeans = KMeans(n_clusters=4, random_state=0)
plot_kmeans(kmeans, X_stretched)
plt.show()
# 观察发现，这些变形的簇并不是圆形，因此袭拟合效果糟糕。
```

kmeans的两个缺点：类的形状缺少灵活性、缺少簇分配的概率，使得它对许多数据集（尤其低纬数据集）的拟合效果不尽如人意。

高斯混合模型解决了这两个不足：通过比较每个点与所有簇中心点的距离来度量簇分配的不确定性，而不仅仅是关注最近的簇；将簇的边界由圆形放宽至椭圆形，从而得到非圆形的簇。

## API

```python
from sklearn.mixture import GaussianMixture

# 初始化参数
n_components
# 聚类分组个数，必须提供
convariance_type
# 协方差矩阵类型，可以是spherical,diag, full,tied
tol
# EM算法廋脸阈值
max_iter
# EM算法最大迭代次数
n_init
# 训练开始前选择p(k),\mu_k,\Sigma_k参数的次数，对数据匹配最好的一次将杯用于开始EM迭代
init_Params
# 用什么方法初始化p(k),\mu_k,\Sigma_k参数，可选K-means或random
weights_init/means_init/precisions_init
# 可以传入调用者自定义的厨师p(k),\mu_k,\Sigma_k，其中\Sigma_k以精度矩阵的形式给出

# 模型属性
weights_							# 每个聚类分组的比重
means_								# 每个高斯分布的均值点
covariances_					# 每个高斯分布的协方差矩阵，矩阵的形状取决于协方差矩阵类型
precisions_						# 精度矩阵
precisions_cholesky_	# 精度矩阵的cholesky_分解
converged_						# EM算法最终是否收敛(如果由于达到max_iter而停止迭代，则不算收敛)
n_inter_							# 迭代次数
lower_bound_					# 最终在训练数据上达到的似然度水平，以对数方式表达
```

## 一般化EM

一个高斯混合模型试图找到多维高斯概率分布的混合题，从而获得任意数据集最好的模型。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
X = X[:, ::-1]  # 交换列是为了方便画图

rng = np.random.RandomState(13)
X_stretched = np.dot(X, rng.randn(2, 2))


# GMM
# gmm = GaussianMixture(n_components=4).fit(X)
# labels = gmm.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
# plt.show()
# GMM中有一个隐含的概率模型，可能找到簇分配的概率结果
# probs = gmm.predict_proba(X)  # 返回[n_samples, n_clusters]的矩阵，给出任意点属于某个簇的概率
# print(probs[:5].round(3))
"""
[[0.    0.469 0.    0.531]
 [0.    0.    1.    0.   ]
 [0.    0.    1.    0.   ]
 [0.    0.    0.    1.   ]
 [0.    0.    1.    0.   ]]
"""


# 不确定性可视化，用每个点的大小体现预测的不确定性，使其成正比
# size = 50 * probs.max(1) ** 2  # 平方强调差异
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap='viridis')
# plt.show()


# 高斯混合模型本质上和kmeans模型列斯，使用了期望最大化
# 1.选择初始簇的中心位置和形状，2.重复直至收敛
# a.期望步骤(E-step):为每个点找到对应每个簇的概率作为权重；
# b.最大化步骤(M-step):更新每个簇的位置，将其标准化，并且基于所有数据点的权重来确定形状
# 每个簇的记过并不与硬边缘的空间有关，而是通过高斯平滑模型实现。
# 正如kmeans中期望最大化，这个算法有时并不是全局最优解，实际中需要使用多个随机初始解

# 可视化GMM簇为何和形状
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """用给定的位置和协方差画一个椭圆"""
    ax = ax or plt.gca()
    # 将协方差转换成主轴
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        widht, height = 2 * np.sqrt(covariance)
    # 画出椭圆
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


gmm = GaussianMixture(n_components=4, random_state=42)
# plot_gmm(gmm, X)
# plt.show()

# 拟合扩展过的数据集，允许使用全协方差来处理非常扁平的椭圆形
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
# 参数covariance_type控制了每个簇的形状自由度
# 默认diag:簇在每个维度的吃醋都可以单独设置，椭圆边界的主轴和坐标轴平行
# spherical:通过约束簇的形状，让所有维度相等。得到结果与kmeans聚类的特征类似
# full:允许每个簇在任意方向上用椭圆建模
plot_gmm(gmm, X_stretched)
plt.show()

```

## 用作密度估计

虽然GMM通常被归类为聚类算法，但它本质上是一个**密度估计**算法；也就是说，从技术的角度考虑，一个GMM拟合的结果并不是一个聚类模型，而是描述数据分布的生成概率模型。

GMM是一种非常方便的建模方法，可以为数据估计出任意维度的随机分布

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture


# 可视化GMM簇为何和形状
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """用给定的位置和协方差画一个椭圆"""
    ax = ax or plt.gca()
    # 将协方差转换成主轴
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        widht, height = 2 * np.sqrt(covariance)
    # 画出椭圆
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


Xmoon, ymoon = make_moons(200, noise=0.05, random_state=0)
# plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
# plt.show()

# 拟合2个成分
# gmm2 = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
# plot_gmm(gmm2, Xmoon)
# plt.show()

# 拟合多个成分而忽视簇标签
gmm16 = GaussianMixture(n_components=16, covariance_type='full', random_state=0)
plot_gmm(gmm16, Xmoon)
# plt.show()

# 这里不是为了找到数据的分割的簇，而是为了对输入数据的总体分布建模
# 这就是分布函数的生成模型-GMM可以生成新的、与输入数据类似的随机分布函数。
# 拟合原始数据获得的16个成分生成的400个新数据点
Xnew = gmm16.sample(400)
plt.scatter(Xnew[0][:, 0], Xnew[0][:, 1])
plt.show()

```

- 需要多少成分

作为一种生成模型，GMM提供了一种确定数据集最优成分数量的方法。由于生成模型本身就是数据集的概率分布，因此可以利用该模型来评估数据的似然估计，并利用交叉检验防止过拟合。还有一些纠正过拟合的方法，如赤池信息量准则（AIC）、贝叶斯信息准则（BIC）调整模型的似然估计。

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture

Xmoon, ymoon = make_moons(200, noise=0.05, random_state=0)
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(Xmoon) for n in n_components]

# AIC，BIC调整模型的似然估计
plt.plot(n_components, [m.bic(Xmoon) for m in models], label='BIC')
plt.plot(n_components, [m.aic(Xmoon) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')
plt.show()
# 类的最优数量出现在AIC/BIC曲线最小值的位置，最终结果取决于使用哪一种近似

```

注意：成分数量的选择度量的是GMM作为一个**密度评估器**的性能，而不是作为一个**聚类算法**的性能。建议把GMM当作一个密度评估器，仅在简单数据集中才将它作为聚类算法使用。

## 用作聚类

示例1

```python
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


# Set random seed for reproducibility
np.random.seed(1000)


# Total number of samples
nb_samples = 800


if __name__ == '__main__':
    # Create the dataset
    X, Y = make_blobs(n_samples=nb_samples, n_features=2, centers=3, cluster_std=2.2, random_state=1000)

    # Show the original dataset
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], c='r', s=20, marker='p', label='Class 0')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], c='g', s=20, marker='d', label='Class 1')
    ax.scatter(X[Y == 2, 0], X[Y == 2, 1], c='b', s=20, marker='s', label='Class 2')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()
    ax.grid()

    plt.show()

    # Create a fit a Gaussian Mixture model
    gm = GaussianMixture(n_components=3, max_iter=1000, random_state=1000)
    gm.fit(X)

    # Print means, covariances, and weights
    print('Means:\n')
    print(gm.means_)

    print('\nCovariances:\n')
    print(gm.covariances_)

    print('\nWeights:\n')
    print(gm.weights_)

    # Show the clustered dataset with the final Gaussian distributions
    fig, ax = plt.subplots(figsize=(15, 8))

    c = gm.covariances_
    m = gm.means_

    g1 = Ellipse(xy=m[0], width=4 * np.sqrt(c[0][0, 0]), height=4 * np.sqrt(c[0][1, 1]), fill=False, linestyle='dashed',
                 linewidth=2)
    g1_1 = Ellipse(xy=m[0], width=3 * np.sqrt(c[0][0, 0]), height=3 * np.sqrt(c[0][1, 1]), fill=False,
                   linestyle='dashed', linewidth=3)
    g1_2 = Ellipse(xy=m[0], width=1.5 * np.sqrt(c[0][0, 0]), height=1.5 * np.sqrt(c[0][1, 1]), fill=False,
                   linestyle='dashed', linewidth=4)

    g2 = Ellipse(xy=m[1], width=4 * np.sqrt(c[1][0, 0]), height=4 * np.sqrt(c[1][1, 1]), fill=False, linestyle='dashed',
                 linewidth=2)
    g2_1 = Ellipse(xy=m[1], width=3 * np.sqrt(c[1][0, 0]), height=3 * np.sqrt(c[1][1, 1]), fill=False,
                   linestyle='dashed', linewidth=3)
    g2_2 = Ellipse(xy=m[1], width=1.5 * np.sqrt(c[1][0, 0]), height=1.5 * np.sqrt(c[1][1, 1]), fill=False,
                   linestyle='dashed', linewidth=4)

    g3 = Ellipse(xy=m[2], width=4 * np.sqrt(c[2][0, 0]), height=4 * np.sqrt(c[2][1, 1]), fill=False, linestyle='dashed',
                 linewidth=2)
    g3_1 = Ellipse(xy=m[2], width=3 * np.sqrt(c[2][0, 0]), height=3 * np.sqrt(c[2][1, 1]), fill=False,
                   linestyle='dashed', linewidth=3)
    g3_2 = Ellipse(xy=m[2], width=1.5 * np.sqrt(c[2][0, 0]), height=1.5 * np.sqrt(c[2][1, 1]), fill=False,
                   linestyle='dashed', linewidth=4)

    ax.add_artist(g1)
    ax.add_artist(g1_1)
    ax.add_artist(g1_2)
    ax.add_artist(g2)
    ax.add_artist(g2_1)
    ax.add_artist(g2_2)
    ax.add_artist(g3)
    ax.add_artist(g3_1)
    ax.add_artist(g3_2)

    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], c='r', s=20, marker='p', label='Class 0')
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], c='g', s=20, marker='d', label='Class 1')
    ax.scatter(X[Y == 2, 0], X[Y == 2, 1], c='b', s=20, marker='s', label='Class 2')
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    ax.legend()
    ax.grid()

    plt.show()

    # Compute AICs and BICs
    nb_components = [2, 3, 4, 5, 6, 7, 8]

    aics = []
    bics = []

    for n in nb_components:
        gm = GaussianMixture(n_components=n, max_iter=1000, random_state=1000)
        gm.fit(X)
        aics.append(gm.aic(X))
        bics.append(gm.bic(X))

    fig, ax = plt.subplots(2, 1, figsize=(15, 8))

    ax[0].plot(nb_components, aics)
    ax[0].set_ylabel('AIC')
    ax[0].grid()

    ax[1].plot(nb_components, bics)
    ax[1].set_xlabel('Number of components')
    ax[1].set_ylabel('BIC')
    ax[1].grid()

    plt.show()
```

示例2

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm

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

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

print(gm.weights_, gm.means_, gm.covariances_)
print(gm.converged_)  # 是否收敛
print(gm.n_iter_)  # 迭代次数

# 使用该模型预测每个实例属于哪个集群（硬集群）或它来自每个集群的概率。为此，只需使用predict或predict_proba
print(gm.predict(X))
print(gm.predict_proba(X))

# 这是一个生成模型，因此您可以从中采样新实例（并获取它们的标签）
# 请注意，它们是从每个簇中按顺序采样的
X_new, y_new = gm.sample(6)
print(X_new, y_new)

# 您也可以使用score_samples方法估计任意位置的概率密度函数（PDF）的日志
res = gm.score_samples(X)
print(res)

# 让我们检查PDF在整个空间中是否集成为1。我们只需在集群周围取一个大的正方形，然后将其切割成一个由小正方形组成的网格，
# 然后计算每个小正方形中生成实例的近似概率（通过将小正方形一角的PDF乘以正方形的面积），最后求出所有这些概率的总和。结果非常接近1
resolution = 100
grid = np.arange(-10, 10, 1 / resolution)
xx, yy = np.meshgrid(grid, grid)
X_full = np.vstack([xx.ravel(), yy.ravel()]).T

pdf = np.exp(gm.score_samples(X_full))
pdf_probas = pdf * (1 / resolution) ** 2
res = pdf_probas.sum()
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


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

plt.figure(figsize=(8, 4))
plot_gaussian_mixture(gm, X)
save_fig("gaussian_mixtures_diagram")
plt.show()

# 通过设置“协方差”超参数，可以对算法查找的协方差矩阵施加约束：
# full（默认）：没有约束，所有簇可以采用任何大小的任何椭球形状。
# tied：所有簇必须具有相同的形状，可以是任何椭球体（也就是说，它们共享相同的协方差矩阵）。
# spherical：所有的簇都必须是球形的，但是它们可以有不同的直径（即不同的方差）。
# diag：簇可以呈现任何大小的椭球形状，但是椭球的轴必须与轴平行（即协方差矩阵必须是对角的）。

gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
gm_full.fit(X)
gm_tied.fit(X)
gm_spherical.fit(X)
gm_diag.fit(X)

def compare_gaussian_mixtures(gm1, gm2, X):
    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(gm1, X)
    plt.title('covariance_type="{}"'.format(gm1.covariance_type), fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(gm2, X, show_ylabels=False)
    plt.title('covariance_type="{}"'.format(gm2.covariance_type), fontsize=14)

compare_gaussian_mixtures(gm_tied, gm_spherical, X)
save_fig("covariance_type_diagram")
plt.show()

compare_gaussian_mixtures(gm_full, gm_diag, X)
plt.tight_layout()
plt.show()
```

## 异常检测

高斯混合可用于异常检测：位于低密度区域的实例可被视为异常。必须定义要使用的密度阈值。

例如，在一家试图检测缺陷产品的制造公司中，缺陷产品的比率通常是众所周知的。假设它等于4%，则可以将“密度阈值”设置为使4%的实例位于该阈值密度以下的区域

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm

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

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)

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


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

# 异常检测
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]

plt.figure(figsize=(8, 4))

plot_gaussian_mixture(gm, X)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
plt.ylim(top=5.1)

save_fig("mixture_anomaly_detection_diagram")
plt.show()
```

## 模型选择

我们不能使用惯性或轮廓分数，因为它们都假设簇是球形的。相反，我们可以尝试找到最小化理论信息准则的模型，例如贝叶斯信息准则（BIC）或Akaike信息准则（AIC）：
$$
{BIC}={\log(m)p-2\log({\hat L})} \\
{AIC}=2p-2\log(\hat L)
$$
其中，$ m$是实例数，$p$ 是模型学习的参数数， $\hat L$ 是模型似然函数的最大值。$\mathbf{X}$ 优化了模型的条件参数。

BIC和AIC都会惩罚需要学习更多参数的模型（例如，更多的集群），并奖励与数据匹配良好的模型（即，给观测数据提供高可能性的模型）。

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm

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

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)


res_bic = gm.bic(X)
res_aic = gm.aic(X)
print(res_bic, res_aic)
"""
8189.662685850679 8102.437405735641
"""

n_clusters = 3
n_dims = 2
n_params_for_weights = n_clusters - 1
n_params_for_means = n_clusters * n_dims
n_params_for_covariance = n_clusters * n_dims * (n_dims + 1) // 2
n_params = n_params_for_weights + n_params_for_means + n_params_for_covariance
max_log_likelihood = gm.score(X) * len(X)  # log(L^)
bic = np.log(len(X)) * n_params - 2 * max_log_likelihood
aic = 2 * n_params - 2 * max_log_likelihood
print(bic, aic, n_params)
"""
8189.662685850679 8102.437405735641 17
"""

# 在不同k下测量
gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X)
             for k in range(1, 11)]
bics = [model.bic(X) for model in gms_per_k]
aics = [model.aic(X) for model in gms_per_k]

plt.figure(figsize=(8, 3))
plt.plot(range(1, 11), bics, "bo-", label="BIC")
plt.plot(range(1, 11), aics, "go--", label="AIC")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Information Criterion", fontsize=14)
plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
plt.annotate('Minimum',
             xy=(3, bics[2]),
             xytext=(0.35, 0.6),
             textcoords='figure fraction',
             fontsize=14,
             arrowprops=dict(facecolor='black', shrink=0.1)
             )
plt.legend()
save_fig("aic_bic_vs_k_diagram")
plt.show()

# 让我们为“簇数”和“协方差”超参数搜索值的最佳组合
min_bic = np.infty

for k in range(1, 11):
    for covariance_type in ("full", "tied", "spherical", "diag"):
        bic = GaussianMixture(n_components=k, n_init=10,
                              covariance_type=covariance_type,
                              random_state=42).fit(X).bic(X)
        if bic < min_bic:
            min_bic = bic
            best_k = k
            best_covariance_type = covariance_type

print(best_k, best_covariance_type)
"""
3 full
"""
```

## 部分贝叶斯高斯混合

与其手动搜索最佳集群数量，还可以使用“BayesianGaussianMixture”类，该类能够为不必要的簇赋予等于（或接近）零的权重。只需将组件数量设置为您认为大于最佳集群数量的值，算法将自动消除不必要的集群。

```python
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.datasets import make_blobs, make_moons
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

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

bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)


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


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


# 算法自动检测到只需要3个组件：
res = np.round(bgm.weights_, 2)
print(res)

plt.figure(figsize=(8, 5))
plot_gaussian_mixture(bgm, X)
plt.show()

bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                  weight_concentration_prior=0.01, random_state=42)
bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                   weight_concentration_prior=10000, random_state=42)
nn = 73
bgm_low.fit(X[:nn])
bgm_high.fit(X[:nn])

res_low = np.round(bgm_low.weights_, 2)
res_high = np.round(bgm_high.weights_, 2)
print(res_low, res_high)

plt.figure(figsize=(9, 4))

plt.subplot(121)
plot_gaussian_mixture(bgm_low, X[:nn])
plt.title("weight_concentration_prior = 0.01", fontsize=14)

plt.subplot(122)
plot_gaussian_mixture(bgm_high, X[:nn], show_ylabels=False)
plt.title("weight_concentration_prior = 10000", fontsize=14)

save_fig("mixture_concentration_prior_diagram")
plt.show()
"""
注意：虽然有4个质心，但在右图中只看到3个区域，这不是一个缺陷。
右上角的簇的权重远大于右下角的簇的权重，因此该区域中任何给定点属于右上角簇的概率都大于它属于右下角簇的概率。
"""

X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X_moons)


plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_data(X_moons)
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.subplot(122)
plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)

save_fig("moons_vs_bgm_diagram")
plt.show()
```

## 似然函数

```python
from scipy.stats import norm
from matplotlib.patches import Polygon
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

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


xx = np.linspace(-6, 4, 101)
ss = np.linspace(1, 2, 101)
XX, SS = np.meshgrid(xx, ss)
ZZ = 2 * norm.pdf(XX - 1.0, 0, SS) + norm.pdf(XX + 4.0, 0, SS)
ZZ = ZZ / ZZ.sum(axis=1) / (xx[1] - xx[0])

plt.figure(figsize=(8, 4.5))

x_idx = 85
s_idx = 30

plt.subplot(221)
plt.contourf(XX, SS, ZZ, cmap="GnBu")
plt.plot([-6, 4], [ss[s_idx], ss[s_idx]], "k-", linewidth=2)
plt.plot([xx[x_idx], xx[x_idx]], [1, 2], "b-", linewidth=2)
plt.xlabel(r"$x$")
plt.ylabel(r"$\theta$", fontsize=14, rotation=0)
plt.title(r"Model $f(x; \theta)$", fontsize=14)

plt.subplot(222)
plt.plot(ss, ZZ[:, x_idx], "b-")
max_idx = np.argmax(ZZ[:, x_idx])
max_val = np.max(ZZ[:, x_idx])
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [0, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.text(1.01, max_val + 0.005, r"$\hat{L}$", fontsize=14)
plt.text(ss[max_idx] + 0.01, 0.055, r"$\hat{\theta}$", fontsize=14)
plt.text(ss[max_idx] + 0.01, max_val - 0.012, r"$Max$", fontsize=12)
plt.axis([1, 2, 0.05, 0.15])
plt.xlabel(r"$\theta$", fontsize=14)
plt.grid(True)
plt.text(1.99, 0.135, r"$=f(x=2.5; \theta)$", fontsize=14, ha="right")
plt.title(r"Likelihood function $\mathcal{L}(\theta|x=2.5)$", fontsize=14)

plt.subplot(223)
plt.plot(xx, ZZ[s_idx], "k-")
plt.axis([-6, 4, 0, 0.25])
plt.xlabel(r"$x$", fontsize=14)
plt.grid(True)
plt.title(r"PDF $f(x; \theta=1.3)$", fontsize=14)
verts = [(xx[41], 0)] + list(zip(xx[41:81], ZZ[s_idx, 41:81])) + [(xx[80], 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
plt.gca().add_patch(poly)

plt.subplot(224)
plt.plot(ss, np.log(ZZ[:, x_idx]), "b-")
max_idx = np.argmax(np.log(ZZ[:, x_idx]))
max_val = np.max(np.log(ZZ[:, x_idx]))
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [-5, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.axis([1, 2, -2.4, -2])
plt.xlabel(r"$\theta$", fontsize=14)
plt.text(ss[max_idx] + 0.01, max_val - 0.05, r"$Max$", fontsize=12)
plt.text(ss[max_idx] + 0.01, -2.39, r"$\hat{\theta}$", fontsize=14)
plt.text(1.01, max_val + 0.02, r"$\log \, \hat{L}$", fontsize=14)
plt.grid(True)
plt.title(r"$\log \, \mathcal{L}(\theta|x=2.5)$", fontsize=14)

save_fig("likelihood_function_diagram")
plt.show()

```



