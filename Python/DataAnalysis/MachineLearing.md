# 概念

- 机器学习

是从数据中自动分析获得规律(模型)，并利用规律对未知数据进行预测

- 开发流程

数据 ---> 特征工程(数据处理) ---> 模型 ---> 模型评估 ---> 是否合格 ---> 应用

- 算法与学习

分类算法：用来解决分类问题

​	常见算法有：k-近邻算法、朴素贝叶斯、决策树与随机森林、逻辑回归、SVC、神经网络等

回归算法：

​	常见算法有：线型回归、岭回归、Lasson回归、SVR



监督学习：输入数据有特征有目标值

​	算法：分类算法和回归算法

无监督学习：输入数据有特征无目标值

​	算法：聚类算法k-Means

# 特征工程

## 数据

离散型数据：由于记录不同类别个体的数目所获得的数据，又称计数数据，不能再细分，也不能提高精度

连续型数据：变量可以在某个范围内取任一数，即变量的取值可以是连续的，这类数据通常是非整数，含有小数部分

- 可用数据集

Kaggle网址：<https://www.kaggle.com/datasets>

UCI数据集网址：<http://archive.ics.uci.edu/ml/>

scikit-learn网址：[http://scikit-learn.org/stable/datasets/index.html#datasets](http://scikit-learn.org/stable/datasets/index.html)

## scikit-learn

- 安装

```python
# 创建一个基于python3的虚环境
mkvirtualenv -p /usr/bin/python3.6 目录
# 在ubuntu的虚环境中运行
pip install Numpy
pip install scipy
pip install matplotlib
pip install Scikit-learn
# 通过命令查看
import sklearn
```

- 数据集导入

```python
# 加载并返回鸢尾花数据集
sklearn.datasets.load_iris()
# 加载并返回波士顿放假数据集
sklearn.datasets.load_boston()
# 属性
DESCR			---> 数据集描述
feature_names	---> 特征名
data			---> 特征值数据数组，是[n_samples*n_features]的二维numpy.ndarry数组
target_names	---> 标签名，回归数据集没有
target			---> 目标值数组
```

## 特征抽取

- 英文文本

```python
from sklearn.feature_extraction.text import CountVectorizer

data = ["life is short, i like python", "lisfe is too long, i disliake python"]

# 特征抽取，抽取词频矩阵
cv = CountVectorizer()
# fit提取特征名
name = cv.fit(data)
# transform根据提取出来的特征词，统计个数
result = cv.transform(data)
# fit_transform = fit + transform
# data是文本或包含文本字符串的可迭代对象，返回词频矩阵
# result = cv.fit_transform(data)
# 返回单词列表
print(cv.get_feature_names())
# 稀疏矩阵
print(result)
# sparse矩阵转换为array数组
print(result.toarray())
```

- 字典抽取

```python
from sklearn.feature_extraction import DictVectorizer

data = [{'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}]
# 字典数据特征抽取，默认稀疏矩阵
dv = DictVectorizer(sparse=False)
# 提取特征名及词频
# 输入是字典或者包含字典的迭代器，返回值是sparse矩阵
result = dv.fit_transform(data)
# dv.fit(data)
# result = dv.transform(data)
# 输入是array数组后者sparse矩阵，返回值是转换之前数据格式
origin = dv.inverse_transform(result)
# 返回了表的名称
print(dv.get_feature_names())
print(result)
print(origin)
```

- one-hot

```python
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("./one_hot_test.csv")
# print(df)

# 1.DictVecotrizer
# df.gender = df.gender.map({0: "female", 1: "male", })
# print(type(df))
# print(df["gender"])  # Series对象
# print(df[["gender"]]) # DataFrame对象
# print(list(df[["gender"]].T.to_dict().values()))
# data = list(df[["gender"]].T.to_dict().values())
# dv = DictVectorizer(sparse=False)
# result = dv.fit_transform(data)
# print(dv.get_feature_names())
# print(result)

# 2.OneHotEncoder
encoder = OneHotEncoder(sparse=False)
result = encoder.fit_transform(df[["gender"]])
print(result)

# 3.pd.get_dummies转换之后不需要在做合并
data = pd.get_dummies(data=df, columns=["gender"])
print(data)
```

- 中文文本

```python
import jieba
from sklearn.feature_extraction.text import CountVectorizer

data = "生活很短，我喜欢python, 生活太久了，我不喜欢python"

# 分词,返回值是generator
cut_ge = jieba.cut(data)
# 方法一：生成器转列表
# content = []
# for word in cut_ge:
#     content.append(word)
# data = [" ".join(content)]
# 方法二，join(可迭代)
data = " ".join(cut_ge)
cv = CountVectorizer()
result = cv.fit_transform(data)
print(cv.get_feature_names())
print(result)
print(result.toarray())
```

- TF-IDF

```python
import jieba 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cut_words():
    s1 = "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。"
    s2 = "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。"
    s3 = "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"   
    s1_ge = jieba.cut(s1)
    s2_ge = jieba.cut(s2)
    s3_ge = jieba.cut(s3)
    return " ".join(s1_ge), " ".join(s2_ge), " ".join(s3_ge)

words1, words2, words3 = cut_words()
# 使用TFIDF特征抽取
tfidf = TfidfVectorizer(stop_words=["一种", "每个"])
# 输入：文本或包含文本字符创的可迭代对象，返回值：saprse矩阵
result = tfidf.fit_transform([words1, words2, words3])
# 返回值：单词列表
print(tfidf.get_feature_names())
print(result.toarray())
# 输入：array数组或sparse矩阵，返回值：转换之前的数据格式
print(tfidf.inverse_transform(result))
```

## 特征处理

- 归一化

```python
from sklearn.preprocessing import MinMaxScaler

data = [[90, 2, 10, 40],
        [60, 4, 15, 45],
        [75, 3, 13, 46]]

# 将每个特征值缩放到给定范围，默认[0,1]
mm = MinMaxScaler()
# 输入ndarray，返回转换后的形状相同的array
result = mm.fit_transform(data)
# print(mm.fit(data))
# print(mm.transform(data))
print(mm.data_min_)
print(mm.data_max_)
```

- 标准值化

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
# 处理之后每列的数据都聚集在均值0，方差1附近
ss = StandardScaler()
# 输入：ndarray，输出:转换后形状相同的array
result = ss.fit_transform(iris.data)
print(result)
```

- 缺失值

```python
import numpy as np
from sklearn.preprocessing import Imputer

data = [[1, 2],
        [np.nan, 3],
        [7, 6]]

# 完成缺失值插补
imputer = Imputer(missing_values="NaN", strategy="mean", axis=1)
# 输入numpy array格式的数据，返回转换后形状相同的array
result = imputer.fit_transform(data)
print(result)
print(result.shape
```

- 特征选择

主要方法：

Filter:VarianceThreshold

Embedded:正则化、决策树

Wrapper

```python
from sklearn.feature_selection import VarianceThreshold

data = [[0, 2, 0, 3],
        [0, 1, 4, 3],
        [0, 1, 1, 3]]
        
# 删除所有低方差特征，默认0.0
vt = VarianceThreshold()
# 输入值：numpy array格式数据
# 返回值：训练集差异低于threshold的特征将被删除
result = vt.fit_transform(data)
print(result)
print(result.shape)
```

- 降维

分类：

主成成分分析(principal
component analysis,PCA)

因子分析(Factor Analysis)

独立成分分析(Independent Component Analysis，ICA)

```python
from sklearn.decomposition import PCA

data = [[2, 8, 4, 5],
        [6, 3, 0, 8],
        [5, 4, 9, 1]]

# 将数据分解为较低维数空间
pca = PCA(n_components=3)
# 输入：numpy array格式的数据，返回值：转换后制定维度的array
result = pca.fit_transform(data)
# pca降维之后，新的数据具体意义就丧失了，主要信息保留
print(result)
```
**特征选择/降维**

相同点：

特征选择和降维都是降低数据维度

不同点：

特征选择筛选掉的特征不会对模型的训练产生任何影响

降维做了数据的映射，保留主要成分，所有的特征对模型训练有影响

