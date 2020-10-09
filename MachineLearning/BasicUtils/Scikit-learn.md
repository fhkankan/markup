# Scikit-Learn

[官网](https://scikit-learn.org/stable/index.html#)

基于NumPy，SciPy和matplotlib构建的开源简单高效的预测数据分析工具

Scikit-Learn API遵循以下设计原则

```
1.统一性：所有对象使用共同接口连接一组方法和统一的文档
2.内省：所有参数值都是公共属性
3.限制对象层级：只有算法可以用python表示，数据集都是用标准数据类型(Numpy数组，Pandas的DataFrame，SciPy稀疏矩阵)表示，参数名称用标准的Python字符串
4.函数组合：很多机器学习任务由一串基本算法实现，尽力支持
5.明智的默认值：当模型需要用户设置参数时，预先定义适当的默认值
```

## 安装引入

安装

```python
pip install scikit-learn
```

引入

```python
import sklearn
```

对象

```
估算器：能够根据数据集对某些参数进行估算的任意对象都可称为估算器，估算由fit()执行
转换器：可以转换数据集的估算器也叫转换器。转换由transform()执行，有时直接使用fit_transform()
预测器：可以基于一个给定的数据集进行预测的估算器也叫预测器。预测由predict()执行
```

## 数据集

```
from sklearn import datasets
```

- 内置数据集

加载

```python
# 获取小规模数据集，数据包含在datasets中
datasets.load_*()		

# 获取大规模数据集，需要从网络上下载
# 参数1:data_home表示数据集下载的目录，默认～/scikit_learn_data/
# 参数2:subset表示'all'，'train'或'test'，可选，选择要加载的数据集.
datasets.fetch_*(data_home=None, subset='all')

# 清除目录下的数据
datasets.clear_data_home(data_home=None)
```

常用数据集

```python
load_iris()		# 鸢尾花
load_boston()	# 波士顿房价
fetch_20newsgroups(data_home=None,subset='all')  # 20类新闻数据集
```

数据集对象属性

```python
print(data.keys())  # 打印数据字典对象的keys

DESCR			 数据集描述
feature_names	 特征名
data			 特征值数据数组，是[n_samples*n_features]的二维numpy.ndarry数组
target_names	 标签名，回归数据集没有
target			 目标值数组
```

- 创建数据集

```
datasets.make_*()
```

常用方法

```python
X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=2, cluster_std=1.5)
# 参数
# n_samples 样本数
# n_features 特征数
# centers	 中心数
# random_state	随机种子
# cluster_std	方差
```

## 特征工程

### 特征提取

#### 分类特征

- LabelEncoder

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

dict_data = {'lang': ['Eng', 'Chi', 'Spa']}
data = pd.DataFrame(dict_data)
encoder = LabelEncoder()
data_encode = encoder.fit_transform(dict_data['lang'])
print(data_encode)  # [1 0 2]
print(encoder.classes_)  # ['Chi' 'Eng' 'Spa']
```

- OneHotEncoder
```python
from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder()
# 整数转换为onehot
data_one_hot = one_hot.fit_transform(data_encode.reshape(-1, 1))
print(data_one_hot.toarray())
"""
[[0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
"""
```
- LabelBinarizer
```python
from sklearn.preprocessing import label_binarize

encoder2 = LabelBinarizer()
# 文本转换为整数，整数转换为onehot
data_encoder2 = encoder2.fit_transform(dict_data['lang'])
print(data_encoder2)
"""
[[0 1 0]
 [1 0 0]
 [0 0 1]]
"""
```

- DictVectorizer

```python
from sklearn.feature_extraction import DictVectorizer

data = [{'city': '北京', 'temperature': 100},
        {'city': '上海', 'temperature': 60},
        {'city': '深圳', 'temperature': 30}]

# 字典数据特征抽取，默认稀疏矩阵
vec = DictVectorizer(sparse=False)
# 输入是字典或者包含字典的迭代器，返回值是sparse矩阵
result = vec.fit_transform(data)
# 输入是array数组后者sparse矩阵，返回值是转换之前数据格式
origin = vec.inverse_transform(result)

print(vec.get_feature_names())  # 返回了表的名称
# ['city=上海', 'city=北京', 'city=深圳', 'temperature']
print(result)
# [[  0.   1.   0. 100.]
#  [  1.   0.   0.  60.]
#  [  0.   0.   1.  30.]]
print(origin)
# [{'city=北京': 1.0, 'temperature': 100.0},
#  {'city=上海': 1.0, 'temperature': 60.0}, 
#  {'city=深圳': 1.0, 'temperature': 30.0}]

vec = DictVectorizer(sparse=True)  # 采用稀疏矩阵
res = vec.fit_transform(data)
print(res)
#   (0, 1)	1.0
#   (0, 3)	100.0
#   (1, 0)	1.0
#   (1, 3)	60.0
#   (2, 2)	1.0
#   (2, 3)	30.0
```

#### 文本特征

- 单词统计

API

```python
from sklearn.feature_extraction.text import CountVectorizer

# 实例化
vec = CountVectorizer()

vec.fit_transform(X)       
# 参数X:文本或者包含文本字符串的可迭代对象
# 返回：词频矩阵
# fit_transform = fit + transform

vev.get_feature_names()
# 返回值:单词列表
```

实现

```python
from sklearn.feature_extraction.text import CountVectorizer

data = ["life is short, i like python", "lisfe is too long, i disliake python"]

# 特征抽取，抽取词频矩阵
vec = CountVectorizer()
# fit提取特征名
name = vec.fit(data)
# transform根据提取出来的特征词，统计个数
result = vec.transform(data)
# data是文本或包含文本字符串的可迭代对象，返回词频矩阵
# result = vec.fit_transform(data)  # fit_transform = fit + transform
print(vec.get_feature_names())  # 返回单词列表
# ['disliake', 'is', 'life', 'like', 'lisfe', 'long', 'python', 'short', 'too']
print(result)  # 稀疏矩阵
#  (0, 1)	1
#   (0, 2)	1
#   (0, 3)	1
#   (0, 6)	1
#   (0, 7)	1
#   (1, 0)	1
#   (1, 1)	1
#   (1, 4)	1
#   (1, 5)	1
#   (1, 6)	1
#   (1, 8)	1

print(result.toarray())  # sparse矩阵转换为array数组
# [[0 1 1 1 0 0 1 1 0]
#  [1 1 0 0 1 1 1 0 1]]

```

中文文本

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

原始的单词统计会让一些常用词聚集过高的权重，不利于分类算法。

TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
# 返回词的权重矩阵

vec = TfidfVectorizer(stop_words=None,…)

vec.fit_transform(X,y)       
# 参数X:文本或者包含文本字符串的可迭代对象
# 返回值：返回sparse矩阵

vec.inverse_transform(X)
# 参数X:array数组或者sparse矩阵
# 返回值:转换之前数据格式

vec.get_feature_names()
# 返回值:单词列表
```

实现

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = ["life is short, i like python", "lisfe is too long, i disliake python"]

vec = TfidfVectorizer()
X = vec.fit_transform(data)
print(X)
#   (0, 6)	0.35520008546852583
#   (0, 3)	0.4992213265230509
#   (0, 7)	0.4992213265230509
#   (0, 1)	0.35520008546852583
#   (0, 2)	0.4992213265230509
#   (1, 0)	0.4466561618018052
#   (1, 5)	0.4466561618018052
#   (1, 8)	0.4466561618018052
#   (1, 4)	0.4466561618018052
#   (1, 6)	0.31779953783628945
#   (1, 1)	0.31779953783628945
print(X.toarray())
# [[0.         0.35520009 0.49922133 0.49922133 0.         0.         0.35520009 0.49922133 0.]
#  [0.44665616 0.31779954 0.         0.         0.44665616 0.44665616 0.31779954 0.         0.44665616]]
res = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(res)
#    disliake      is      life      like  ...      long  python     short       too
# 0  0.000000  0.3552  0.499221  0.499221  ...  0.000000  0.3552  0.499221  0.000000
# 1  0.446656  0.3178  0.000000  0.000000  ...  0.446656  0.3178  0.000000  0.446656

```

中文

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

#### 图像特征

方法一：使用像素

方法二：HOG特征

### 特征预处理

#### 衍生特征

```python
# 将一个线性回归改为多项式回归，并不通过改变模型来实现，而是通过改变输入数据。也称为基函数回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y, c='r')

# 拟合直线，获得最优解
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.plot(x, yfit, c='g')


# 对数据进行变换，增加额外的多项式特征提升模型复杂度
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)
# [[  1.   1.   1.]
#  [  2.   4.   8.]
#  [  3.   9.  27.]
#  [  4.  16.  64.]
#  [  5.  25. 125.]]
# 第1列表示x，第2列表示x^2，第3列表示x^3

model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.plot(x, yfit, c='y')
plt.show()
```

#### 归一化

- 最值归一化

[API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)

```python
from sklearn.preprocessing import MinMaxScaler

# numpy array格式的数据[n_samples,n_features]
data = ...
# 实例化,每个特征缩放到给定范围(默认[0,1])
mm = MinMaxScaler(feature_range=(0,1)…)
# 归一化，返回转换后的形状相同的array
result = mm.fit_transform(data)  
# 原始数据中每列特征的最小最大值
print(mm.data_min_)
print(mm.data_max_)
```

- 均值方差归一化

[API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)

```python
from sklearn.preprocessing import StandardScaler

data = ...
# 处理之后每列的数据都聚集在均值0，方差1附近
ss = StandardScaler()
# 输入：ndarray，输出:转换后形状相同的array
result = ss.fit_transform(data)
# 原始数据中每列特征的平均值
print(ss.mean_)
```

#### 缺失值

API

```python
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer 
```

示例

```python
import numpy as np
from sklearn.impute import SimpleImputer

X = np.array([[np.nan, 0, 3],
              [3, 7, 9],
              [3, 5, 2],
              [4, np.nan, 6],
              [8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])
imp = SimpleImputer(strategy="mean")
X2 = imp.fit_transform(X)
print(X2)
# [[4.5 0.  3. ]
#  [3.  7.  9. ]
#  [3.  5.  2. ]
#  [4.  5.  6. ]
#  [8.  8.  1. ]]
```

## 自定义转换器

sklearn提供了转换器：`LabelEncoder, OneHotEncoder, LabelBinarizer`等，但是也可以自定义转换器

```python
rooms_ix, bedrooms_ix, populaton_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    由于scickit-learn依赖于鸭子类型的编译，而不是继承，所以可以创建一个类，然后用fit(),transform(),fit_transform()
    实现与其他的转换器类似，可以和scikit-learn自身的功能(如pipline)无缝对接。
    继承BaseEstimator，在构造器中避免*args和**kwargs，可以额外获得get_params()和set_params()两个调整超参数的方法
    继承TransformerMixin，可以直接使用fit_transform()
    """

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, populaton_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

## 管道

需要将多个步骤串联起来使用，可以使用管道对象。

API

```python
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_union
from sklearn.pipeline import FeatureUnion
```

- pipline

Pipeline将多个评估器级联合成一个评估器。这么做的原因是考虑了数据处理过程的一系列前后相继的固定流程，比如：feature selection --> normalization --> classification

在这里，Pipeline提供了两种服务：
```
1. Convenience: 你只需要一次fit和predict就可以在数据集上训练一组estimators。
2. Join parameter selection： 可以把grid search用在pipeline中所有的estimators的参数组合上面。
```
注意： pineline中除了最后一个之外的所有的estimators都必须是变换器（transformers）（也就是说必须要有一个transform方法）。最后一个estimator可以是任意的类型（transformer, classifier, regresser, etc）。

调用pipeline estimator的fit方法，就等于是轮流调用每一个estimator的fit函数一样，不断地变换输入，然后把结果传递到下一个阶段（step）的estimator。Pipeine对象实例拥有最后一个estimator的所有的方法。

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Pipeline
estimators = [('reduce_dim', PCA()), ('clf', SVC()) ]
pipe = Pipeline(estimators) 
# make_pipeline省去名称，程序自动填充
pipe = make_pipeline(PCA(), SVC())

print(pipe)  # 评估器
print('-----------')
print(pipe.steps)  # 评估器执行步骤
print('-----------')
print(pipe.named_steps['clf'])  # 评估器具体步骤

params = dict(reduce_dim__n_components=[2, 5, 10],
              clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=params)  
     

```

- FeatureUnion

FeatureUnion把若干个transformer object组合成一个新的estimators。这个新的transformer组合了他们的输出，一个FeatureUnion对象接受一个transformer对象列表。

在训练阶段，每一个transformer都在数据集上独立的训练。在数据变换阶段，多有的训练好的Trandformer可以并行的执行。他们输出的样本特征向量被以end-to-end的方式拼接成为一个更大的特征向量。

在这里，FeatureUnion提供了两种服务：
```
1. Convenience： 你只需要调用一次fit和transform就可以在数据集上训练一组estimators。
2. Joint parameter selection： 可以把grid search用在FeatureUnion中所有的estimators的参数这上面。
```
FeatureUnion和Pipeline可以组合使用来创建更加复杂的模型。

注意：FeatureUnion无法检查两个transformers是否产生了相同的特征输出，它仅仅产生了一个原来互相分离的特征向量的集合。确保其产生不一样的特征输出是调用者的事情。

```python
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_union
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
# FeatureUnion
estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
combined = FeatureUnion(estimators)
# make_union：省去名称，程序自动填充
combined = make_union(PCA(), KernelPCA())
```

## 验证曲线

API

```python
from sklearn.model_selection import validation_curve
```

示例

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn
import matplotlib.pyplot as plt

# 用交叉检验计算一个模型的验证曲线
# 模型：y=ax^3+bx^2+cx+d
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def PolynomialRegresssion(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


# 创造数据
def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


X, y = make_data(40)

# 拟合曲线
# seaborn.set()
X_test = np.linspace(-0.1, 1.1, 500)[:, None]
#
# plt.scatter(X.ravel(), y, color='black')
# axis = plt.axis()
# for degree in [1, 3, 5]:
#     y_test = PolynomialRegresssion(degree).fit(X, y).predict(X_test)
#     plt.plot(X_test.ravel(), y_test, label='degree={0}'.format(degree))
# plt.xlim(-0.1, 1.0)
# plt.ylim(-2, 12)
# plt.legend(loc='best')
# plt.show()

# 验证曲线:验证模型复杂度的影响
degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegresssion(), X, y, param_name='polynomialfeatures__degree',
                                          param_range=degree, cv=7)
plt.plot(degree, np.median(train_score, 1), color='blue', ls='--', label='training score')
plt.plot(degree, np.median(val_score, 1), color='red', ls='--', label='validation score')
# plt.legend(loc='best')
# plt.ylim(0, 1)
# plt.xlabel('degree')
# plt.ylabel('score')
# plt.show()

# 训练得分总是比验证得分高；训练得分随着模型复杂度的提升而单调递增，验证得分增长到最高点后由于过拟合而开始骤降。
# 得出结果，偏差和方差均衡性最好的是三次多项式
# plt.scatter(X.ravel(), y)
# lim = plt.axis()
# y_test = PolynomialRegresssion(3).fit(X, y).predict(X_test)
# plt.plot(X_test.ravel(), y_test)
# plt.axis(lim)
# plt.show()

# 验证曲线：验证训练数据量和模型复杂度的影响
X2, y2 = make_data(200)
degree = np.arange(21)
train_score2, val_score2 = validation_curve(PolynomialRegresssion(), X2, y2, param_name='polynomialfeatures__degree',
                                            param_range=degree, cv=7)
plt.plot(degree, np.median(train_score2, 1), color='blue', label='training score')
plt.plot(degree, np.median(val_score2, 1), color='red', label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')
plt.show()

```

## 学习曲线

API

```python
from sklearn.model_selection import learning_curve
```

示例

```python
import numpy as np
import matplotlib.pyplot as plt

# 用交叉检验计算一个模型的验证曲线
# 模型：y=ax^3+bx^2+cx+d
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def PolynomialRegresssion(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


# 创造数据
def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


X, y = make_data(40)

# 学习曲线
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for i, degree in enumerate([2, 9]):
    N, train_lc, val_lc = learning_curve(PolynomialRegresssion(degree), X, y, cv=7, train_sizes=np.linspace(0.3, 1, 25))
    ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray', linestyles='dashed')
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title('degree={0}'.format(degree), size=14)
    ax[i].legend(loc='best')

plt.show()

# 展现了模型得分随着训练数据规模的变化而变化，当学习曲线已经收敛时，再增加训练数据不能显著改善拟合效果
# 采用更复杂的模型之后，收敛得分提高了，但是模型的方差也变大了。
```

## 交叉验证

随机抽样

```python
from sklearn.model_selection import train_test_split

train_test_split(*arrays, *options)
# 参数
# X				 x数据集的特征值
# y				 y数据集的特征值
# test_size		 测试集的大小，一般为float
# random_state	 随机数种子
# 返回
# 训练集特征值、测试集特征值、训练集标签、测试集标签
```

分层抽样

```python
from sklearn.model_selection import StratifiedShuffleSplit

# 读取数据
data = pd.read_csv("median_income.csv")
# 查看原始数据在数据中的比例
data["income_cat"].value_counts() / len(data)

# 数据的加工处理，是每个类别里的数据变的均匀
data["income_cat"] = np.ceil(data["median_income"]/1.5)
data["income_cat"].where(data["income_cat"]<5, 5.0, inplace=True)

# 创建对象
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["income_cat"]):
    strat_train_test = data.loc[train_index]
    strat_test_test = data.loc[test_index]

# 计算分层抽样后的各个数据占的比例
strat_train_test["income_cat"].value_counts() / len(strat_train_test)

# 从运行结果很容易发现分层抽样前后各个数据占总数据的比例基本一致
```

交叉验证

```python
# 1.带返回值的cross_val_score
from sklearn.model_selection import cross_val_score, LeaveOneOut
# 执行交叉验证，返回每个折叠的评估分数

cross_val_score(model, X_train, y_train, cv=LeaveOneOut(len(X_train))) 
# 参数
# LeaveOneOut为只留一个测试

from sklearn.model_selection import cross_val_predict
# 执行交叉验证，返回每个折叠的预测

# 2.自定义函数
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predcit(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
```

示例

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


digits = datasets.load_digits()
X = digits.data
y = digits.target

# 数据集留出集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=666)


knn_clf = KNeighborsClassifier()
# 交叉验证
cross_val_score(knn_clf, X_train, y_train)
# cross_val_score(knn_clf, X_train, y_train, cv=5)  # 指定训练集分割份数
best_score, best_p, best_k = 0, 0, 0
for k in range(2, 11):
  	for p in range(1, 6):
      	knn_clf = KNeighborsClassifier(weights="distance", n_neighbors=k, p=p)
        scores = cross_val_score(knn_clf, X_train, y_train)
        score = np.mean(scores)
        if score > best_score:
          	best_score = score
            best_p = p
            best_k = k
print("best_score", best_score)
print("best_p", best_p)
print("bset_k", best_k)
```

## 网格搜索

API

```python
from sklearn.model_selection import GridSearchCV

# 对估计器的指定参数值进行详尽搜索
grid = GridSearchCV(estimator, param_grid=None,cv=None)

# 输入
estimator：估计器对象
param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
cv：指定几折交叉验证
# 方法
fit：输入训练数据
score：准确率
# 属性
best_score_:最佳模型下的分数
best_params_:最佳模型参数
best_estimator_：最好的参数模型
cv_results_:交叉验证的结果
```

示例

```python
import numpy as np
import matplotlib.pyplot as plt

# 用交叉检验计算一个模型的验证曲线
# 模型：y=ax^3+bx^2+cx+d
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def PolynomialRegresssion(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


# 创造数据
def make_data(N, err=1.0, rseed=1):
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


X, y = make_data(40)
X_test = np.linspace(-0.1, 1.1, 500)[:, None]

# 网格搜索
param_grid = {
    'polynomialfeatures__degree': np.arange(21),  # 多项式次数的搜索范围
    'linearregression__fit_intercept': [True, False],  # 是否拟合截距
    'linearregression__normalize': [True, False]  # 是否标准化处理
}
grid = GridSearchCV(PolynomialRegresssion(), param_grid, cv=7)

grid.fit(X, y)
# 最优参数
print(grid.best_params_)
# {'linearregression__fit_intercept': False, 
# 'linearregression__normalize': True, 
# 'polynomialfeatures__degree': 4}

# 拟合数据并显示
model = grid.best_estimator_  # 最佳评估器
plt.scatter(X.ravel(), y)
lim = plt.axis()
y_test = model.fit(X, y).predict(X_test)
plt.plot(X_test.ravel(), y_test)
plt.axis(lim)
plt.show()
```

## 常用算法

分类

```python
# 逻辑回归
from sklearn.linear_model import LogisticRegression	
# 随机梯度下降
from sklearn.linear_model import SGDClassifier
# SVM
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
# 决策树
from sklearn.tree import DecisionTreeClassifier
# 随机森林
from skleran.ensemble import RandomForestClassifier
# k-近邻算法
sklearn.neighbors
# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB  # 高斯
from sklearn.naive_bayes import MultinomialNB  # 多项式
```

回归

```python
# 线性回归
from sklearn.linear_model import LinearRegression
# 岭回归（L2范数正则化）
from sklearn.linear_model import Ridge
# Lasso回归（L1范数正则化）
from sklearn.linear_model import Lasso
# SVM
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
# 随机森林回归
from sklearn.ensemble import RandomForestRegressor
```

无监督

```python
# PCA
from sklearn.decomposition import PCA
# 流形学习
from sklearn.manifold import MDS, LocallyLinearEmbedding, Isomap, TSNE
# 聚类
from sklearn.cluster import KMeans,SpectralClustering
# 高斯混合模型
from sklearn.mixture import GaussianMixture
# 核密度估计
from sklearn.neighbors import KernelDensity
```

## 算法评价

### 回归

均方误差

```python
from sklearn.metrics import mean_squared_error
```

根均方误差

```python
sqrt(mean_squared_error())
```

平均绝对误差

```python
from sklearn.metrics import mean_absolute_error
```

R方

```python
from sklearn.metrics import r2_score
```

### 分类

准确度

```python
from sklearn.metrics import accuracy_score

res = accuracy_score(y_test, y_predict)
```

混淆矩阵

```python
from sklearn.metrics import confusion_matrix

res = confusion_matrix(y_test, y_predict)  # xlabel是predict,ylabel是true；即行表示实际的列表，列表示预测的列表
```

精准率

```python
from sklearn.metrics import precision_score

res = precision_score(y_test, y_predict)
```

召回率

```python
from sklearn.metics import recall_score

res = recall_score(y_test, y_predict)
```

F1 Score

```python
from sklearn.metics import f1_score

res = f1_score(y_test, y_predict)
```

PR曲线

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)
```

ROC/AUC曲线

```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fprs, tprs, thresholds = roc_curve(Y_test, decision_scores)
roc_auc_score(y_test, decision_scores)
```

分类结果报告

```python
from sklearn.metrics import classification_report

res = classification_report(y_test, y_predict, target_names=data.target_names)
```

### 聚类

轮廓系数

```python
from sklearn.metrics import silhouette_score
```

## 模型保存加载

```python
from sklearn.externals import joblib

# 保存
joblib.dump(my_model, "my_model.pkl")

# 加载
my_model_loaded = joblib.load("my_model.pkl")
```

## 使用示例

API使用步骤

```
1.通过从Scikit-Learn中导入适当的评估器类，选择模型类
2.用合适的数值对模型类进行实例化，配置模型超参数
3.整理数据，获取特征矩阵和目标数组
4.调用模型实例的fit()方法对数据进行拟合
5.对新数据应用模型：监督学习中用predict()方法预测新数据的标签；非监督学习用transform()/predict()方法转换或推断数据的性质
```

简单线性回归

```python
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)

# 数据探索
# plt.scatter(x, y)
# plt.show()

# 1.选择模型类
from sklearn.linear_model import LinearRegression

# 2.选择模型超参数
# a.要拟合偏移量(直线的截距)吗？
# b.要做归一化处理吗？
# c.要对特征进行预处理以提高模型灵活性吗
# d.在模型中使用哪种正则化类型
# e.使用多少模型组件
model = LinearRegression(fit_intercept=True)

# 3.整理数据特征矩阵[n_samples, n_features]和目标数组
X = x[:, np.newaxis]

# 4.用模型拟合数据
model.fit(X, y)
k = model.coef_  # 斜率
b = model.intercept_  # 截距

# 5.预测新数据的标签
xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(Xfit, yfit)
plt.show()

```

鸢尾花数据分类

```python
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

# 选择模型类
from sklearn.naive_bayes import GaussianNB

# 实例化模型类
model = GaussianNB()

# 数据矩阵和目标数组
X_iris = iris.data
y_iris = iris.target

# 分测试集和训练集
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)

# 拟合数据
model.fit(Xtrain, ytrain)

# 预测数据
y_model = model.predict(Xtest)

# 验证准确率
from sklearn.metrics import accuracy_score

res = accuracy_score(ytest, y_model)
print(res)  # 0.97

```

鸢尾花数据降维

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris(as_frame=True)

# 选择模型类
from sklearn.decomposition import PCA

# 确定超参数
# 实例化模型类
model = PCA(n_components=2)

# 数据矩阵和目标数组
X_iris = iris.data
y_iris = iris.target

# 拟合数据
model.fit(X_iris)

# 将数据转换为二维
X_2D = model.transform(X_iris)

# 画图
iris_frame = iris.frame
iris_frame['PCA1'] = X_2D[:, 0]
iris_frame['PCA2'] = X_2D[:, 1]

sns.lmplot("PCA1", "PCA2", hue='target', data=iris_frame, fit_reg=False)
plt.show()

```

鸢尾花数据聚类

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets

iris = datasets.load_iris(as_frame=True)

# 选择模型类
from sklearn.decomposition import PCA

# 确定超参数
# 实例化模型类
model = PCA(n_components=2)

# 数据矩阵和目标数组
X_iris = iris.data
y_iris = iris.target

# 拟合数据
model.fit(X_iris)

# 将数据转换为二维
X_2D = model.transform(X_iris)

# 画图
iris_frame = iris.frame
iris_frame['PCA1'] = X_2D[:, 0]
iris_frame['PCA2'] = X_2D[:, 1]

# sns.lmplot("PCA1", "PCA2", hue='target', data=iris_frame, fit_reg=False)

# 选择模型类
from sklearn.mixture import GaussianMixture

# 确定超参数
# 实例化模型类
model = GaussianMixture(n_components=3)

# 拟合数据
model.fit(X_iris)

# 确定簇标签
y_gmm = model.predict(X_iris)

# 画图
iris_frame['cluster'] = y_gmm
print(iris_frame[['target', 'cluster']])

sns.lmplot("PCA1", "PCA2", hue='target', data=iris_frame, col='cluster', fit_reg=False)
plt.show()

```





