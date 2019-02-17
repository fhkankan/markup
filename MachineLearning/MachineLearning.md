# 机器学习

是从数据中自动分析获得规律(模型)，并利用规律对未知数据进行预测

## 开发流程

数据 ---> 特征工程(数据处理) ---> 模型 ---> 模型评估 ---> 是否合格 ---> 应用

## 学习分类

- 监督学习：输入数据有特征有目标值

```
算法：分类算法和回归算法
分类：目标是类别
回归：目标是连续数字的值

分类算法：用来解决分类问题
	常见算法有：k-近邻算法、朴素贝叶斯、决策树与随机森林、逻辑回归、SVC、神经网络等

回归算法：
	常见算法有：线型回归、岭回归、Lasson回归、SVR
```

- 无监督学习：输入数据有特征无目标值

```
算法：聚类算法k-Mean
特征提取
特征压缩
```

- 半监督学习：输入数据有特征部分有目标值

```
无监督 ---> 监督
```

- 增强学习

```
根据周围环境情况，采取行动，根据采取行动的结果，学习行动方式
```

## 其他分类

批量学习与在线学习

```
批量学习：
简单，需要定时批量学习，缺点运算量大，变化快

在线学习：
及时反映新的环境变化，需要加强对数据的监控
适用于数据量巨大，完全无法批量学习的环境
```

参数学习与非参数学习

```
参数学习：
需要对模型进行统计假设，一旦学到了参数，就不再需要原有的数据集

非参数学习：
不对模型进行过多假设，不代表没有参数
```

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
# 加载并返回20类新闻数据集
sklearn.datasets.fetch_20newsgroups(data_home=None,subset='all')
data_home:表示数据集下载的目录,默认是 ~/scikit_learn_data/
subset: 'all'，'train'或'test'，可选，选择要加载的数据集.
      训练集的“训练”，测试集的“测试”，两者的“全部”
datasets.clear_data_home(data_home=None)
	清除目录下的数据

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
# 对文本数据进行特征值化
# 类
sklearn.feature_extraction.text.CountVectorizer

# 实例化
CountVectorizer()

CountVectorizer.fit_transform(X)       
X:文本或者包含文本字符串的可迭代对象
返回词频矩阵

CountVectorizer.get_feature_names()
返回值:单词列表
```

实现

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
# 对字典数据进行特征值化

# 类
sklearn.feature_extraction.DictVectorizer
# 实例化
DictVectorizer(sparse=True,…)

DictVectorizer.fit_transform(X)       
X:字典或者包含字典的迭代器
返回值：返回sparse矩阵

DictVectorizer.inverse_transform(X)
X:array数组或者sparse矩阵
返回值:转换之前数据格式

DictVectorizer.get_feature_names()
返回类别名称

DictVectorizer.transform(X)
按照原先的标准转换
```

实现

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

```
3种方式：
1. DictVectorizer(注意，数字不会进行转换)
2. OneHotEncoder(Numpy)
3. pd.get_dummies(Pandas)
```

实现

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

TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。

```python
# 类
sklearn.feature_extraction.text.TfidfVectorizer
# 返回词的权重矩阵
TfidfVectorizer(stop_words=None,…)

TfidfVectorizer.fit_transform(X,y)       
X:文本或者包含文本字符串的可迭代对象
返回值：返回sparse矩阵

TfidfVectorizer.inverse_transform(X)
X:array数组或者sparse矩阵
返回值:转换之前数据格式

TfidfVectorizer.get_feature_names()
返回值:单词列表
```

实现

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
# 类
sklearn.preprocessing.MinMaxScaler
# 实例化
MinMaxScaler(feature_range=(0,1)…)
每个特征缩放到给定范围(默认[0,1])

MinMaxScaler.fit_transform(X)       
X:numpy array格式的数据[n_samples,n_features]
返回值：转换后的形状相同的array
```

实现

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
# 类
scikit-learn.preprocessing.StandardScaler
# 实例化
StandardScaler(…)
处理之后每列来说所有数据都聚集在均值0附近方差为1

StandardScaler.fit_transform(X,y)       
X:numpy array格式的数据[n_samples,n_features]
返回值：转换后的形状相同的array

StandardScaler.mean_
原始数据中每列特征的平均值
```

实现

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
# 类
sklearn.preprocessing.Imputer
# 实例化
Imputer(missing_values='NaN', strategy='mean', axis=0)
完成缺失值插补

Imputer.fit_transform(X,y)       
X:numpy array格式的数据[n_samples,n_features]
返回值：转换后的形状相同的array
```

实现

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

```
Filter:VarianceThreshold

Embedded:正则化、决策树

Wrapper

```

函数

```
# 类
sklearn.feature_selection.VarianceThreshold
# 实例化
VarianceThreshold(threshold = 0.0)
删除所有低方差特征

Variance.fit_transform(X,y)       
X:numpy array格式的数据[n_samples,n_features]
返回值：训练集差异低于threshold的特征将被删除。
默认值是保留所有非零方差特征，即删除所有样本
中具有相同值的特征。
```

实现

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

```
主成成分分析(principalcomponent analysis,PCA)

因子分析(Factor Analysis)

独立成分分析(Independent Component Analysis，ICA)
```

本质：PCA是一种分析、简化数据集的技术。在PCA中，数据从原来的坐标系转换到新的坐标系。

目的：是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。

函数

```python
# 类
sklearn.decomposition.PCA
# 实例化
PCA(n_components=None)
将数据分解为较低维数空间

PCA.fit_transform(X)       
X:numpy array格式的数据[n_samples,n_features]
返回值：转换后指定维度的array
```

实现

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

# 模型

## 数据集的划分

数据集：

训练数据集、测试数据集

```python
sklearn.model_selection.train_test_split(*arrays, **options)

# 输入
x			数据集的特征值
y			数据集的标签值
test_size 	 测试集的比例，默认0.25
random_state 随机数种子，不同的种子会造成不同的随机采样结果。相同的种子采样结果相同

# 返回
训练集特征值，测试集特征值，训练集标签值，测试标签值(默认随机取)
```

实现

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 特征值
x = np.arange(0,10).reshape([5,2])
print(x)
# 目标值
y = range(5)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y)
print("训练集的特征值")
print(x_train)
print("测试集的特征值")
print(x_test)
print("训练集的目标值")
print(y_train)
print("测试集的目标值")
print(y_train)
```

## 估计器

分类估计器

```python
# k-近邻算法
sklearn.neighbors	
# 朴素贝叶斯
sklearn.naive_bayes	
# 逻辑回归
sklearn.linear_model.LogisticRegression	
```

回归估计器

```python
# 线性回归
sklearn.linear_model.LinearRegression
# 岭回归
sklearn.linear_model.Ridge
# Lasso回归
sklearn.linear_model.Lasso
```

# 分类

## k-近邻算法

如果一个样本在数据集中，有k个最相思的样本，而k个样本大多数属于某一个类别，那么这个样本也属于该类别

```python
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')

# 输入
n_neighbors:int,可选，默认=5
```

优缺点

```
优点：
简单，易于理解，易于实现
缺点：
懒惰算法，对测试样本分类时的计算量大，内存开销大
必须指定k值，k值选择不当则分类精度不能保证
```

实现

```python
# 1.导入所需的包
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 2.加载数据文件
data = pd.read_csv("./FBlocation/train.csv")
print(len(data))
# 3.缩小数据范围
data = data.query("x > 1 & x < 1.25 & y >3 &y < 3.25")
print(len(data))
# 4.时间特征抽取
# 将时间戳转换为日期
time_value = pd.to_datetime(data["time"], unit="s")
# 将时间转换为DatetimeIndex
date_time_index = pd.DatetimeIndex(time_value)
data["hour"] = date_time_index.hour
data["month"] = date_time_index.month
data["dayofweek"] = date_time_index.dayofweek
# 5.删除掉入住率比较低的样本
# 分组聚合 以place_id分组，count计数，小于3，筛选掉
place_count = data.groupby("place_id").aggregate(np.count_nonzero)
# print(place_count)
#            row_id      x      y  accuracy  time  hour  month  dayofweek
# place_id
# 1009781224     219  219.0  219.0       219   219   216    219        200
# 所有入住次数大于3的结果，数据并不是原始数据，而只是一个统计数据
result = place_count[place_count["row_id"] > 3].reset_index()
# 从原始数据中选择place_id在result中的样本
data = data[data["place_id"].isin(result["place_id"])]
# 6.特征选择
# 特征值
x = data.drop(["row_id", "time", "place_id"], axis=1)
# 目标值
y = data["place_id"]
# 7.分割数据集
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 8.对数据集进标准化
ss = StandardScaler()
# 对特征值进行标准化
x_train = ss.fit_transform(x_train)
# 对测试集的特征值标准化
x_test = ss.transform(x_test)  # 按照原来训练集的平均值做标准化，统一数据转换标准
# 9.KNeighborsClassifiler训练模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
# 10.准确率
# 使用测试集的特征值，预测测试集的特征值对应的目标值place_id
y_predict = knn.predict(x_test)
print(y_predict)
# 测试模型在测试集上的准确性
score = knn.score(x_test, y_test)
print(score)
```

### 交叉验证

将拿到的数据，分为训练和测试集。将数据分成5份，其中一份作为验证集。然后经过5次(组)的测试，每次都更换不同的验证集。即得到5组模型的结果，取平均值作为模型精度的估计。又称5折交叉验证。

### 网格搜索

使用网格搜索确定最优的参数。这种参数，称之为超参数，K近邻算法中的K值。

在网格搜索中每组超参数都采用交叉验证来进行评估。

```python
# 对估计器的指定参数值进行详尽搜索
sklearn.model_selection.GridSearchCV(estimator, param_grid=None,cv=None)

# 输入
estimator：估计器对象
param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
cv：指定几折交叉验证
# 方法
fit：输入训练数据
score：准确率
# 属性
best_score_:最好结果
best_estimator_：最好的参数模型
cv_results_:交叉验证的结果
```

实现

```python
# 1.导入所需要的包
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 2.载入数据
data = pd.read_csv('./FBlocation/train.csv')
print(len(data))
# 3.缩小数据范围
data = data.query("x > 1 & x < 1.25 & y >3 &y < 3.25")
print(len(data))
# 4.时间特征抽取
# 将时间戳转换为日期
time_value = pd.to_datetime(data["time"], unit="s")
# 将时间转换为DatetimeIndex
date_time_index = pd.DatetimeIndex(time_value)
data["hour"] = date_time_index.hour
data["month"] = date_time_index.month
data["dayofweek"] = date_time_index.dayofweek
# 5.删除掉入住率比较低的样本
# 分组聚合 以place_id分组，count计数，小于3，筛选掉
place_count = data.groupby("place_id").aggregate(np.count_nonzero)
# print(place_count)
#            row_id      x      y  accuracy  time  hour  month  dayofweek
# place_id
# 1009781224     219  219.0  219.0       219   219   216    219        200
# 所有入住次数大于3的结果，数据并不是原始数据，而只是一个统计数据
result = place_count[place_count["row_id"] > 3].reset_index()
# 从原始数据中选择place_id在result中的样本
data = data[data["place_id"].isin(result["place_id"])]
# 6.特征选择
# 特征值
x = data.drop(["row_id", "time", "place_id"], axis=1)
# 目标值
y = data["place_id"]
# 7.分割数据集
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 8.对数据集进标准化
ss = StandardScaler()
# 对特征值进行标准化
x_train = ss.fit_transform(x_train)
# 对测试集的特征值标准化
x_test = ss.transform(x_test)  # 按照原来训练集的平均值做标准化，统一数据转换标准
# 9.KNeighborsClassifiler训练模型
knn = KNeighborsClassifier(n_neighbors=3)
# 网格搜索与交叉验证
params = {"n_neighbors": [1, 3, 5]}
gscv = GridSearchCV(estimator=knn, param_grid=params, cv=2)
gscv.fit(x_train, y_train)
print(gscv.best_params_)
print(gscv.best_estimator_)
print(gscv.best_score_)
print(gscv.cv_results_)
```

## 朴素贝叶斯

$$
p(B|A)=\frac{p(A|B)p(B)}{p(A)}
$$

当事件(特征)相互独立时，贝叶斯准则转变为朴素贝叶斯，朴素贝叶斯是贝叶斯准则中的一种特殊情况。

```python
# 朴素贝叶斯分类
sklearn.naive_bayes.MultinomialNB(alpha = 1.0)

# 输入
alpha：拉普拉斯平滑系数
```

- 拉普拉斯平滑

```python
由于样本数较少，会出现p(A|B)的概率为0，防止此情况出现，使用拉普莱斯平滑

拉普拉斯平滑系数ɑ, 默认为1
p=Ni/N	---> p=(Ni+a)/(N+am)
m为训练文档中特征词个数，Ni为xi在分类ci下出现的次数，N为分类ci下词频总数。
```

优缺点

```
优点：
朴素贝叶斯模型发源于古典数学理论，有稳定的分类效率。
对缺失数据不太敏感，算法也比较简单，常用于文本分类。
分类准确度高，速度快

缺点：
需要知道先验概率P(F1,F2,…|C)，因此在某些时候会由于假设的先验模型的原因导致预测效果不佳
```

实现

```python
# 1.导入需要的包
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# 2.载入数据
news = fetch_20newsgroups(subset="all")
# 3.特征选取
# 特征值,文章内容
x = news.data
# 目标值，文章的类别
y = news.target
print(len(y))
# 4.分割训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
# 5.TF-IDF生成文章特征词
# 特征抽取
cv = TfidfVectorizer()
x_train = cv.fit_transform(x_train)  # 词频矩阵
x_test = cv.transform(x_test)  # 按照训练集抽取特征词统计词频
# 6.朴素贝叶斯estimator流程进行预估
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb.predict(x_test)
score = mnb.score(x_test, y_test)
print(score)
```

## 决策树

### 信息论基础

信息是用来消除随机不确定性的东西

信息量：一个事件的信息量就是这个事件发生概率的负对数，单位bit
$$
-logP(x)
$$

一个事情发生的概率越小，信息量越大。

信息熵：一个事件有很多结果，那么所有结果携带信息量的期望就是信息熵
$$
H(x)=-\sum_{i=1}P(xi)logP(xi)
$$
条件熵：在某一个条件下，随机变量的不确定度
$$
H(Y|X) = \sum_{i}P(xi)H(Y|X=xi)
$$
信息增益：信息增益 = 信息熵-条件熵

代表了在一个条件下，信息复杂度(不确定性)减少的程度

构造决策树常用算法

```
ID3算法
信息增益 最大的准则：若属性信息增益越大，该属性优先判断

C4.5算法
信息增益比 最大的准则：若属性信息增益比越大，该属性优先判断

CART 算法
基尼(gini)系数   最小的准则：若属性基尼系数越小，该属性优先判断

```

### 决策树

决策树（decision tree）是一个树结构。其每个非叶节点表示一个特征属性上的测试，每个分支代表这个特征属性在某个值域上的输出，而每个叶节点存放一个类别。 

```python
# 决策树分类器
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, max_depth=None,random_state=None)

# 输入
criterion:默认是’gini’系数，信息增益’entropy’
max_depth:树的深度大小
random_state:随机数种子
```

可视化

```
1、sklearn.tree.export_graphviz() 该函数能够导出DOT格式
tree.export_graphviz(estimator,out_file='tree.dot’,feature_names=[‘’,’’])


2、工具:(能够将dot文件转换为pdf、png)
安装graphviz
ubuntu:sudo apt-get install graphviz           Mac:brew install graphviz

3、运行命令
然后我们运行这个命令
$ dot -Tpng tree.dot -o tree.png
```

优缺点

```
优点：
简单的理解和解释，树木可视化。
需要很少的数据准备，其他技术通常需要数据归一化。
效果一般比K近邻，朴素贝叶斯要好，企业和比赛中使用较多。

缺点：
容易过拟合：也就是该决策树对训练数据可以得到很低的错误率，
但是运用到测试数据上却得到非常高的错误率。
决策树可能不稳定，因为数据的小变化可能会导致完全不同的树
被生成

改进：
减枝cart算法（基尼系数最小准则）
随机森林
```

实现

```python
# 1.导入合适的包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import classification_report

# 2.加载数据
data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
# 3.数据处理
# 填补缺失值age
data["age"].fillna(data["age"].mean(), inplace=True)
# 雷彪数据进行One-Hot编码
data = pd.get_dummies(data, columns=["pclass", "sex"]) 
print(data.head(2))
# 4.特征选择和数据集分割
# 特征值
x = data[["age", "pclass_1st", "pclass_2nd", "pclass_3rd", "sex_female", "sex_male"]]
# 目标值
y = data["survived"]
# 数据集分割
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 5.决策树估计器流程
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=4)
dtc.fit(x_train, y_train)
# 输出图形
export_graphviz(dtc, out_file="./tree.dot")
# 5.预测
predict = dtc.predict(x_test)
# 6.准确率
score = dtc.score(x_test, y_test)
print(score)
```


## 随机森林

在机器学习中，随机森林是一个包含多个决策树的分类器，并且其输出的类别是由个别树输出的类别的众数而定。

树的建立

```
1. 如果训练集大小为N，对于每棵树而言，随机且有放回地从训练集中的抽取N个训练样本（这种采样方式称为bootstrap sample方法），作为该树的训练集；

2.从M个特征中随机选取m个特征子集（m<<M）
```

函数

```python
# 随机森林分类器
class sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None,bootstrap=True, random_state=None)

# 输入
n_estimators：integer，optional（default = 10） 森林里的树木数量
criteria：string，可选（default =“gini”）分割特征的测量方法
max_depth：integer或None，可选（默认=无）树的最大深度 
bootstrap：boolean，optional（default = True）是否在构建树时使用放回抽样 
```

优缺点

```
优点
能够解决单个决策树不稳定的情况
能够处理具有高维特征的输入样本，而且不需要降维（使用的是特征子集）
对于缺省值问题也能够获得很好得结果（投票）

缺点
随机森林已经被证明在某些噪音较大的分类或回归问题上会过拟。
```

实现

```python
# 1.导入合适的包
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# 2.加载数据
data = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
# 3.数据处理
# 填补缺失值age
data["age"].fillna(data["age"].mean(), inplace=True)
# 雷彪数据进行One-Hot编码
data = pd.get_dummies(data, columns=["pclass", "sex"]) 
print(data.head(2))
# 4.特征选择和数据集分割
# 特征值
x = data[["age", "pclass_1st", "pclass_2nd", "pclass_3rd", "sex_female", "sex_male"]]
# 目标值
y = data["survived"]
# 数据集分割
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 5.随机森林估计器流程
rfc = RandomForestClassifier(n_estimators=5, criterion="entropy", max_depth=4)
rfc.fit(x_train, y_train)
# 5.预测
predict = rfc.predict(x_test)
# 6.准确率
score = rfc.score(x_test, y_test)
print(score)
```

## 支持向量机

### 基本原理

解决线性可分问题:

支持向量：训练数据集的样本点中与分离超平面距离最近的样本点

解决非线性可分问题：

通过核函数将数据映射到高维空间再使用平面进行分割。

### 支持向量机

优缺点

```
优点：
适合小数量样本数据，可以解决高维问题，理论基础比较完善，对于学数学的来说它的理论很美。

缺点： 
一旦数据量上去了，那么计算机的内存什么的资源就支持不了，这时候LR等算法就比SVM 要好。（借助二次规划求解支持向量）
```

函数

```python
sklearn.scm.SVC()
```

实现

```python
# 1.导入合适的包
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 2.载入数据
data = pd.read_csv(r'./water/moment.csv', encoding="gbk")
print(data.head(2))
# 3.特征和数据集划分
# 特征值
x = data.drop(["类别", "序号"], axis=1)
# 目标值
y = data["类别"]
# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y)
# # 4.训练模型
# svc = SVC()
# svc.fit(x_train*30, y_train)
# 4.读取模型
with open("./svc.model", "rb") as f:
    svc = pickle.load(f)
# 5.准确率
score = svc.score(x_test*30, y_test)
print(score)
# # 6.保存模型
# with open("./svc.model", "wb") as f:
#     svc = pickle.dump(svc, f)
```

## 分类评估

准确率

即预测结果正确的百分比，`estimator.score()`

精确率

表示的是预测为正的样本中有多少是真正的正样本。那么预测为正就有两种可能了，一种就是把正类预测为正类(TP)，另一种就是把负类预测为正类(FP)，也就是
$$
P  = \frac{TP}{TP+FP}
$$
召回率

表示的是样本中的正例有多少被预测正确了。那也有两种可能，一种是把原来的正类预测成正类(TP)，另一种就是把原来的正类预测为负类(FN)。
$$
R = \frac{TP}{TP+FN}
$$
F1-score

反应了模型的稳健性
$$
F1=\frac{2TP}{2TP+FN+FP}=\frac{2*Precision*Recall}{Precision+Recall}
$$

函数

```python
sklearn.metrics.confusion_matrix(y_true, y_pred,)

# 输入
y_true：真实目标值
y_pred：估计器预测目标值


sklearn.metrics.classification_report(y_true, y_pred, target_names=None)

# 输入
y_true：真实目标值
y_pred：估计器预测目标值
target_names：目标类别名称
# 返回
每个类别精确率与召回率
```

## 模型保存和加载

```python
# 保存模型
import pickle
with open("./svm.model", "wb") as f:
    pickle.dump(svc, f)

# 加载模型
with open("./svm.model", "rb") as f:
    svc = pickle.load(f)
```

# 回归

## 线性回归

找到一条直线或曲线尽力你和数据分布和走势

