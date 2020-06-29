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

## 数据集

### 内置数据集

加载

```python
# 加载获取流行数据集
from sklearn import datasets


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

属性

```python
# 数据集属性
DESCR			 	数据集描述
feature_names	 	特征名
data			 	特征值数据数组，是[n_samples*n_features]的二维numpy.ndarry数组
target_names	 	标签名，回归数据集没有
target				目标值数组
```

### 数据集分离

```python
sklearn.model_selection.train_test_split(*arrays, *options)
# 参数
X				 x数据集的特征值
y				 y数据集的特征值
test_size		 测试集的大小，一般为float
random_state	 随机数种子
# 返回
训练集特征值、测试集特征值、训练集标签、测试集标签
```

示例

```python
from sklearn.model_selection import train_test_split

# 特征值
x = np.arange(0,10).reshape([5,2])
# 目标值
y = range(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
```

## 特征工程

### 特征提取

#### 分类特征

- DictVectorizer

API

```python
# 字典向量化
from sklearn.feature_extraction import DictVectorizer

# 实例化
vec = DictVectorizer(sparse=True,…)

# 方法
vec.fit_transform(X)       
# 参数X:字典或者包含字典的迭代器
# 返回值：返回sparse矩阵

# 等价于vec.fit_transform(X)
vec.fit(X)
vec.transform(X)

vec.inverse_transform(X)
# 参数X:array数组或者sparse矩阵
# 返回值:转换之前数据格式

vec.get_feature_names()  # 返回类别名称
```

实现

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

## 特征管道

需要将多个步骤串联起来使用，可以使用管道对象。

API

```python
from sklearn.pipeline import make_pipeline
```

示例

```python
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[np.nan, 0, 3],
              [3, 7, 9],
              [3, 5, 2],
              [4, np.nan, 6],
              [8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])
model = make_pipeline(SimpleImputer(strategy='mean'), PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)
print(y)
# [14 16 -1  8 -5]
print(model.predict(X))
# [14. 16. -1.  8. -5.]

```



## 交叉验证

API

```python
from sklearn.model_selection import cross_val_score, LeaveOneOut

cross_val_score(model, X_train, y_train, cv=LeaveOneOut(len(X_train))) 

# LeaveOneOut为只留一个测试
```

示例

```python
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


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
best_score_:最好结果
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
# k-近邻算法
sklearn.neighbors	
# 高斯朴素贝叶斯
from sklearn.naive_bayes import GaussianNB 
# 逻辑回归
sklearn.linear_model.LogisticRegression	
```

回归

```python
# 线性回归
from sklearn.linear_model import LinearRegression
# 岭回归
from sklearn.linear_model import Ridge
# Lasso回归
from sklearn.linear_model import Lasso
```

聚类

```python

```

## 算法评价

### 分类

准确度

```python
from sklearn.metrics import accuracy_score
```

混淆矩阵

```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_predict)
```

精准率

```python
from sklearn.metrics import precision_score

precision_score(y_test, y_predict)
```

召回率

```python
from sklearn.metics import recall_score

recall_score(y_test, y_predict)
```

F1 Score

```python
from sklearn.metics import f1_score

f1_score(y_test, y_predict)
```

precision-recall曲线

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, decision_scores)
```

ROC曲线

```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fprs, tprs, thresholds = roc_curve(Y_test, decision_scores)
roc_auc_score(y_test, decision_scores)
```

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

### 聚类

轮廓系数

```python
from sklearn.metrics import silhouette_score
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







