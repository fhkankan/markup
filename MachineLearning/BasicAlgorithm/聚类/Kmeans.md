# KMeans

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
复杂度与样本呈线性关系
很难发现任意形状的簇
初始值选择对结果影响大
```

## 原理

要得到的簇的个数，需要指定k值

质心：均值，即向量各维取平均即可

距离的度量：欧几里得距离和余弦相似度(先标准化)

优化目标
$$
min\sum_{i=1}^K\sum_{x\in{C_i}}{(c_i, x)^2}
$$

## 实现

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

## sklearn

### API

```python
from sklearn.cluster import KMeans, SpectralClustering
```

### 示例

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans, SpectralClustering

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


# plt.scatter(X[:, 0], X[:, 1], s=50)

# 使用kmeans
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)

# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


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
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

# 注意事项
# 1.可能不是全局最优解
centes, labels = find_clusters(X, 4, rseed=0)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# 2.簇数量需事先定好
labels = KMeans(6, random_state=0).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# 3.kmeans算法只能确定线性聚类边界
# 边界很复杂时，算法失效
X, y = make_moons(200, noise=0.05, random_state=0)
labels = KMeans(2, random_state=0).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# 核变换，将数据投影到更高维度，效果较好
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# 4.数据量大时，速度较慢
plt.show()

```

图片压缩1

```python
from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('test.jpg')
io.imshow(image)
io.show()

rows = image.shape[0]
cols = image.shape[1]

image = image.reshape(image.shape[0]*image.shape[1], 3)
kmean = KMeans(n_cluster=128, n_init=10, max_iter=200)
kmean.fit(image)

cluster = np.asarray(kmeans.cluster_cnters_, dtype=np.unit8)
labels = np.asarray(kmeans.labels_, dtype=np.unint8)
labels = labels.reshape(rows. cols)

print(cluster.shape)
np.save('codebook_test.npy', cluster)
io.imsave('compressed_test.jpg', labels)

image = io.read('compressed_test.jpg')
io.imshow(image)
io.show()
```

图片压缩2

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

