# 高斯混合模型

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



