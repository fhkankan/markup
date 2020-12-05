# 核密度估计

密度评估器是一种利用D维数据生成D维概率密度分布估计的算法。GMM算法用不同高斯分布的加权汇总来表示概率分布估计。**核密度估计**（`kernel density estimation,KDE`）算法将高斯混合理念扩展到了逻辑极限。它通过对每个点生成高斯分布的混合成分，获得本质上是无参数的密度评估器。

## KDE由来

密度评估器是一种寻找数据集生成概率扽不模型的的算法。一维数据的密度估计-直方图就是一个简单的密度评估器。直方图将数据分成若干区间，统计落入每个区间内的点的数量，然后用直观的方式将结果可视化。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# 创建两组服从正态分布的数据
def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x


x = make_data(1000)

# 正态分布直方图
# hist = plt.hist(x, bins=30, density=True)
# plt.show()
# 在区间不变的条件下，这个标准化(计算概率密度)只是简单第改变了y轴的比例，相对高度依然与频次直方图一直。
# 标准化是为了让直方图的总面积等于1
# desity, bins, patches = hist
# widths = bins[1:] - bins[:-1]
# res = (desity * widths).sum()
# print(res)  # 1.0

# 使用直方图作为密度评估器时，区间大小和位置的选择不同，产生的统计特征也不同
x = make_data(20)
# bins = np.linspace(-5, 10, 10)
# fig, ax = plt.subplots(1,  2, figsize=(12, 4), sharex=True, sharey=True,
#                        subplot_kw={'xlim': (-4, 9), 'ylim': (-0.02, 0.3)})
#
# fig.subplots_adjust(wspace=0.05)
# for i, offset in enumerate([0.0, 0.6]):
#     ax[i].hist(x, bins=bins + offset, density=True)
#     ax[i].plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
# plt.show()
# 左侧为双峰分布，右侧为单峰分布且带有长尾，描述同样的数据却显示不同的分布

# 将直方图看成是一堆方块，把每个方块堆在数据集每个数据点锁在的区间内
# fig, ax = plt.subplots()
# bins = np.arange(-3, 8)
# ax.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
# for count, edge in zip(*np.histogram(x, bins)):
#     for i in range(count):
#         ax.add_patch(plt.Rectangle((edge, i), 1, 1, alpha=0.5))
# ax.set_xlim(-4, 8)
# ax.set_ylim(-0.2, 8)
# plt.show()
# 前面的两种区间之所以造成问题，原因在于方块堆叠的高度通常并不能反映区间附近数据点的实际密度，
# 而是反映了区间如何与数据点对齐。区间内数据点和方块对不齐将可能导致前面那样的问题

# 不采用方块和区间对其的形式，采用方块与相应的数据点对齐，
# 会导致方块对不齐，但是可以将它们在x轴上每个数据点位置的贡献求和来找到结果
# x_d = np.linspace(-4, 8, 2000)
# density = sum(abs(xi - x_d) < 0.5 for xi in x)
# plt.fill_between(x_d, density, alpha=0.5)
# plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
# plt.axis([-4, 8, -0.2, 8])
# plt.show()
# 虽然有些杂乱，但是可以更全面地反映初数据的真实特征

# 使用平滑函数取代每个位置上的方块，如使用高斯函数。使用标准正态分布 曲线代替每个点的方块
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)
plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)
plt.axis([-4, 8, -0.2, 5])
plt.show()
# 这幅平滑图像是由每个点锁在位置的高斯分布构成的，这样可以更准确地表现数据分布的形状，
# 并且拟合方差更小(进行不同的抽样时，数据的改变更小)

```

## 实际应用

核密度估计的自由参数时**核类型(kernel)**参数，它可以指定每个点核密度分布的形状。而**核带宽(kernel bandwidth)** 参数控制每个点的核的大小。在实际应用中，有 很多核可用于核密度估计，特别是sklearn的KDE实现支持六个核（在Scipy和StatsModels中也有相应的版本）。由于KDE计算量比较大，sklearn底层使用了基于树的算法，可以使用绝对容错（atol）和相对容错（rtol）参数来平衡计算时间与准确性。可以用标准交叉检验确定自身自由参数核带宽。

- API

```python
from sklearn.neighbors import KernelDensity
```

示例

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


# 创建两组服从正态分布的数据
def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x


x = make_data(20)
x_d = np.linspace(-4, 8, 1000)

# 初始化并拟合KDE模型
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])

# score_sample返回概率密度的对数值
logprob = kde.score_samples(x_d[:, None])
plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)
plt.show()

```

- 通过交叉检验选择带宽

在KDE中，带宽的选择不仅对找到合适的密度估计非常重要，也是在密度估计中控制偏差-方差平和的关键：带宽过窄将导致估计呈现高方差（过拟合），而且每个点的出现或缺失都会引起很大的不同；带宽过宽将导致估计呈现高偏差（欠拟合），而且带宽较大的核还会破坏数据结构 。

```python
from sklearn.model_selection import GridSearchCV, LeaveOneOut

# 使用交叉检验确定带宽
bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=LeaveOneOut())
grid.fit(x[:, None])
res_p = grid.best_params_
print(res_p)
# {'bandwidth': 1.1233240329780276}
```

## 球形空间的KDE

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.datasets import fetch_species_distributions
from sklearn.datasets._species_distributions import construct_grids
from sklearn.neighbors import KernelDensity

data = fetch_species_distributions()

# 获取物种ID和位置矩阵/数组
latlon = np.vstack([data.train['dd lat'], data.train['dd long']]).T
species = np.array([d.decode('ascii').startswith('micro') for d in data.train['species']], dtype='int')

xgrid, ygrid = construct_grids(data)

# 用Basemap画出海岸线
# m = Basemap(projection='cyl', resolution='c', llcrnrlat=ygrid.min(), urcrnrlat=ygrid.max(),
#             llcrnrlon=xgrid.min(), urcrnrlon=xgrid.max())
# m.drawmapboundary(fill_color='#DDEEFF')
# m.fillcontinents(color='#FFEEDD')
# m.drawcoastlines(color='gray', zorder=2)
# m.drawcountries(color='gray', zorder=2)

# 画出位置
# m.scatter(latlon[:, 1], latlon[:, 0], zorder=3, c=species, cmap='rainbow', latlon=True)
# plt.show()
# 没有很好显示物种密度的信息，应为两个物种的数据点分布有相互重叠。

# 用核密度估计来显示物种分布信息：在地图中平滑地显示密度
# 由于地图坐标系统位于一个球面，而不是一个平面上，因此可以使用haversine度量距离正确表示球面上的距离
# 准备画轮廓图的数据点
X, Y = np.meshgrid(xgrid[::5], ygrid[::5][::-1])
land_reference = data.coverages[6][::5, ::5]
land_mask = (land_reference > -9999).ravel()
xy = np.vstack([Y.ravel(), X.ravel()]).T
xy = np.radians(xy[land_mask])
# 创建两幅并排的图
fig, ax = plt.subplots(1, 2)
fig.subplots_adjust(left=0.05, right=0.95, wspace=0.05)
species_names = ['Bradypus Variegatus', 'Microryzomys Minutus']
cmaps = ['Purples', 'Reds']

for i, axi in enumerate(ax):
    axi.set_title(species_names[i])

    # 用Basemap画出海岸线
    m = Basemap(projection='cyl', llcrnrlat=Y.min(), urcrnrlat=Y.max(),
                llcrnrlon=X.min(), urcrnrlon=X.max(), resolution='c', ax=axi)
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    m.drawcountries()

    # 构建一个球形的分布核密度估计
    kde = KernelDensity(bandwidth=0.03, metric='haversine')
    kde.fit(np.radians(latlon[species == i]))

    # 值计算大陆的值：-9999表示是海洋
    Z = np.full(land_mask.shape[0], -9999.0)
    Z[land_mask] = np.exp(kde.score_samples(xy))
    Z = Z.reshape(X.shape)

    # 画出密度的轮廓
    levels = np.linspace(0, Z.max(), 25)
    axi.contourf(X, Y, Z, levels=levels, cmap=cmaps[i])

plt.show()
# 与简单散点图相比，可以更清楚地展示两个物种的观察分布

```

## 不是很朴素的贝叶斯

在朴素贝叶斯分类方法中，为每一个类创建了一个简单的生成模型，并用这些模型构建了一个快速的分类器。在朴素贝叶斯分类中，生成模型是与坐标轴平行的高斯分布。如果用KDE核密度估计算法，就可以去掉朴素的成分，使用更成熟的生成模型描述每一个类。虽然还是贝叶斯分类，但是不再朴素。

一般分类器的生成算法

```
1.通过标签分割训练数据。
2.为每个集合拟合一个KDE来获得数据的生成模型，这样就可以用任意x观察值和y标签计算出似然估计值P(x|y)。
3.根据训练集中每一类的样本数量，计算每一类的先验概率P(y)。
4.对于一个未知的点x，每一类的后验概率是P(x|y)->P(x|y)P(y)，而后验概率最大的类就是分配给该点的标签。
```

示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB


# 1.创建自定义评估器
class KDEClassifier(BaseEstimator, ClassifierMixin):  # 继承BaseEstimator类包含各种标准功能，支持适当的混合类(mixin)
    """基于KDE的贝叶斯生成分类

    Parameters
    ----------
    bandwidth : float
        每个类中的核带宽
    kernel : str
        核函数的名称，传递给KernelDensity
    """

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        # 在sklearn中除了将参数值传递给self之外，不做任何操作
        # 所有参数都是显式的，不能使用*args和**kwargs,防止交叉检验异常
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        # 训练数据
        # 首先在训练集中找到所有类(对标签去重)，为每一类训练一个KernelDensity模型
        # 然后，根据输入样本的数量计算类的先验概率，最后用fit()函数返回self
        # 所有拟合结果都存在self.logpriors_中
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self

    def predict_proba(self, X):
        # 概率分类器，返回每个类概率的数组，形状为[n_samples, n_classes]，
        # 数组中的[i,j]表示样本i属于j类的后验概率，用似然估计先乘以类先验概率，再进行归一化
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        # 根据概率，返回概率最大的类
        return self.classes_[np.argmax(self.predict_proba(X), 1)]


# 2.使用自定义评估器
digits = load_digits()

bandwidths = 10 ** np.linspace(0, 2, 100)
grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})
grid.fit(digits.data, digits.target)
# scores = [val.mean_validation_score for val in grid.grid_scores_] # 2.0已停用
scores = [val for val in grid.cv_results_['mean_test_score']]
# 画出交叉曲线图
plt.semilogx(bandwidths, scores)
plt.xlabel('bandwidth')
plt.ylabel('accuracy')
plt.title('KDE Model Performance')
print(grid.best_params_)
print('accuracy =', grid.best_score_)
"""
{'bandwidth': 6.135907273413174}
accuracy = 0.9677298050139276
"""
plt.show()

# 朴素贝叶斯效果
res = cross_val_score(GaussianNB(), digits.data, digits.target).mean()
print(res)  # 0.8069281956050759

# 结果差异的原因：在自定义的评估器中，不仅得到了每个未知样本的一个带概率的分类结果，而且还得到了一个可以对比的数据点分布全模型。若需要的话，这个分类器还可以提供一个直观的可视化观察窗口，而SVM和随机森林却难以实现这个功能。
# 这个KDE分类模型进一步优化的方向
# 1.允许每一个类的带宽各不相同
# 2.不用预测值优化带宽，而是基于训练数据中每一饿类生成模型的似然估计值优化宽带(使用KernelDensity的值，而不使用预测的准确值)
# 还可以使用高斯混合模型代替KDE来构建一个自己的类似的贝叶斯分类器
```



