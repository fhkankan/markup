# DBSCAN

Density Based Spatial Clustering of Applications with Noise

核心对象：若某个点的密度达到算法设定的阈值则其为核心点(即r邻域内点的数量小于minPts)

$\epsilon$邻域的距离阈值：设定的半径r

直接密度可达：若某点p在点q的r邻域内，且q是核心点则p-q直接密度可达

密度可达：若有一个点的序列q0,q1,…qk，对任意qi~qi-1是直接密度可达的，则称从q0到qk密度可达，这实际上是直接密度可达的传播

密度相连：从某核心点

边界点：

噪声点：

工作流程

```
标记所有对象为unvisited
Do
随机选择一个unvisited对象p
标记p为visited
if p
```

参数选择

```

```

优缺点

```
# 优点
不需要指定簇个数
可以发现任意形状的簇
擅长找到离群点(检测任务)
两个参数就enough

# 缺点
高维数据有些困难(可以降维)
参数难以选择(参数对结果的影响非常大)
sklearn中效率很慢(数据消减策略)
```

## sklearn

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

db = DBSCAN(eps=10, min_samples=2).fix(X)
db_scaled = DBSCAN(eps=10, min_samples=2).fix(X_scaled)

lables = db.labels_
lables_scaled = db_scaled.labels_

score = silhouette_score(X, labels)
score_scaled = silhouette_score(X, labels_scaled)
print(score, score_scaled)
```

