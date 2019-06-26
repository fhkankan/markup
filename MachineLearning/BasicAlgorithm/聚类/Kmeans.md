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

## sklearn

图片压缩

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

