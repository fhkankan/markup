# Scikit-learn

## 安装

```python
pip install scikit-learn
```

## 数据集导入

```python
from sklearn import datasets

# 加载数据集
# 加载并返回鸢尾花数据集
datasets.load_iris()
# 加载并返回波士顿放假数据集
datasets.load_boston()
# 加载并返回20类新闻数据集
sklearn.datasets.fetch_20newsgroups(data_home=None,subset='all')
# 参数
data_home: 表示数据集下载的目录,默认是 ~/scikit_learn_data/
subset:  'all'，'train'或'test'，可选，选择要加载的数据集.
      训练集的“训练”，测试集的“测试”，两者的“全部”

# 清除目录下的数据
datasets.clear_data_home(data_home=None)

# 属性
DESCR			---> 数据集描述
feature_names	---> 特征名
data			---> 特征值数据数组，是[n_samples*n_features]的二维numpy.ndarry数组
target_names	---> 标签名，回归数据集没有
target			---> 目标值数组
```

## 训练测试分离

## 数据归一化

- 最值归一化(Normalization)

将说有数据归一化到0～1的分布中
$$
x_{scale} =\frac{x-x_{min}}{x_{max}-x_{min}}
$$


适用于有明显边界的情况， 不适合有极端值的情况

- 均值方差归一化(Standardization)

将所有数据归一化到均值为0，方差为1的分布中
$$
x_{scale} = \frac{x-x_{mean}}{S}
$$


适用与没有明显边界，有可能存在极端值



## 算法评价

- 准确度

```

```

- 交叉验证

```

```



## 网格搜索







## 常用算法

```python
# knn
from 
```

