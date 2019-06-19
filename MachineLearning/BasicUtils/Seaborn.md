# Seaborn

Seaborn 是基于 Python 且非常受欢迎的图形可视化库，在 Matplotlib 的基础上，进行了更高级的封装，使得作图更加方便快捷。即便是没有什么基础的人，也能通过极简的代码，做出具有分析价值而又十分美观的图形。

Seaborn 可以实现 Python 环境下的绝大部分探索性分析的任务，图形化的表达帮助你对数据进行分析，而且对 Python 的其他库（比如 Numpy/Pandas/Scipy）有很好的支持。

安装

```python
pip install seaborn
```

使用

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
```

## 图像设置

### 主题风格

Seaborn有五个预设好的主题： darkgrid , whitegrid , dark , white ,和 ticks  默认：darkgrid

`set_style( )`是用来设置主题的

```python
import matplotlib.pyplot as plt  
import seaborn as sns  
sns.set_style("whitegrid")  
plt.plot(np.arange(10))  
plt.show()
```

`set( )`通过设置参数可以用来设置背景，调色板等，更加常用

```python
import seaborn as sns  
import matplotlib.pyplot as plt  
sns.set(style="white", palette="muted", color_codes=True)       
plt.plot(np.arange(10))  
plt.show()
```

### 调色板

设置颜色

```python
color_paletter()  
# 能传入任何matplotlib所支持的颜色

set_palette()
# 设置所有图的颜色
```

分类色板

```python
current_palette = sns.color_palette()
sns.palplot(current_palette)
# 6个默认的颜色循环主题：deep,muted,pastel,bright,dark,colorblind
```

圆形画板

```python
# 有六个以上的分类需要区分时，最简单的方法就是在一个圆形的颜色空间中画出均匀间隔的颜色。常用方法是使用hls的颜色空间，这厮RGB的一个简单转换
sns.palplot(sns.color_palette("hls", 8))  # 有8种颜色
sns.palplot(sns.hls_palette(8, l=.3, s=.8))  # 控制颜色的亮度和饱和度，l表示光亮度，s表示饱和度

# 使用
data = np.random.normal(size=(20, 8) + np.arrange(8) / 2)
sns.boxplot(data=data, palette=sns.color_palette("hls", 8))
```

成对画板

```python
sns.paplot(sns.color_palette("Paired", 8))  # 4对调色板
```

使用xkcd颜色命名

```python
# xkcd包含了一套众包努力的针对随机RGB色的命名。产生了954个可以随时通过xdcd_rgb字典中调用的命名颜色
plt.plot([0, 1], [0, 1], sns.xkcd_rgb["pale red"], lw=3)
plt.plot([0, 1], [0, 2], sns.xkcd_rgb["medium green"], lw=3)
plt.plot([0, 1], [0, 3], sns.xkcd_rgb["denim blue"], lw=3)
```

连续色板

```python
# 色彩随数据变换，比如数据越重要则颜色越深
sns.paplot(sns.color_palette("Blues"))

# 若要翻转渐变，由深到浅，在面板名称中添加_r后缀
sns.paplot(sns.color_palette("BuGn_r"))
```

线性变换调色板

```python
sns.palplot(sns.color_palette("cubehelix", 8))

sns.palplot(sns.cubehelix_palette(8, start=.5, rot=-.75))

sns.palplot(sns.cubehelix_palette(8, start=.75, rot=-.150))
```

定制连续调色板

```python
# 绿色由浅至深
sns.palplot(sns.light_palette('green'))
# 绿色由深到浅
sns.palplot(sns.light_palette('green'， reverse=True))

# 紫色由深到浅
sns.palplot(sns.dark_palette("purple"))

# 使用
x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
pal = sns.dark_palette("green", as_cmap=True)
sns.kdeplot(x, y, cmap=pal)
```

使用颜色空间

```python
sns.paplot(sns.light_palette((210, 90, 60), input="husl"))
```

### 细节设置

去除ticks主题下无关线框

```python
sns.set_style("ticks") 
sns.despine()
```

偏移轴线距离

```python
sns.despine(offset=10)
```

隐藏轴线

```python
sns.despine(left=True)
```

多风格

```python
with sns.axes_style("darkgrid"):
		plt.subplot(211)  # 风格darkgrid
		sinplot()
plt.subplot(212)  # 风格1
sinplot(-1)
```

布局

```python
sns.set_context("paper")
plt.figure(figsize=(8, 6))
sinplot()

sns.set_context("talk")
plt.figure(figsize=(8, 6))
sinplot()

sns.set_context("poster")
plt.figure(figsize=(8, 6))
sinplot()

sns.set_context(
  "notebook",
  font_scale=1.5,
  rc={"lines.linewidth": 2.5}
)
sinplot()
```

## 常用图形

### 直方图

单变量分析

```python
sns.distplot(x, bins=20, kde=False, fit=stats.gamma)
# 参数
x  统计的数字样本
bins 方块数
kde	
fit	统计的指标

```

示例

```python
%matplotlib inline
import numpy as np
import seaborn as sns

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))

x = np.random.normal(size=100)
sns.distplot(x, kde=False)
```

### 散点图

观测两个变量之间的分布关系

```python
sns.jointplot(x="x", y="y", data=df)
# 参数
x			x轴方向显示的特征
y			y轴方向显示的特征
data	包含x、y轴所指定的特征名的数据Dataframe
```

示例

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
%matplotlib inline
sns.set(color_codes=True)
mean, cov = [0.5, 1], [(1, .5),(.5, 1)]#设置均值(一组参数)和协方差（两组参数）
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
print(df.head())
sns.jointplot(x="x", y="y", data=df)
plt.show()
```

数据量大时，用hex散点图

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
%matplotlib inline
sns.set(color_codes=True)
mean, cov = [0, 1], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("ticks"):
    sns.jointplot(x=x, y=y, kind="hex", color="k")
plt.show()
```

### 比较图

观察变量两两之间的关系

```python
sns.pairplot(iris)
# 对角线是直方图(统计数量)，其他的是散点图
# 对角线是单变量，其他是两两之间关系分布
```

示例

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats, integrate
%matplotlib inline
sns.set(color_codes=True)
iris = sns.load_dataset("iris")
sns.pairplot(iris)#对角线是直方图(统计数量)，其他的是散点图
print(iris.head(2))
plt.show()
```

### 条形图

显示值的集中趋势

```python
sns.barplot(x="sex", y="survived", hue="class", data=titanic)
```

示例

```python
f, ax=plt.subplots(figsize=(12,20))

#orient='h'表示是水平展示的，alpha表示颜色的深浅程度
sns.barplot(
  y=group_df.sub_area.values, 	
  x=group_df.price_doc.values,
  orient='h', 
  alpha=0.8, 
  color='red'
)

#设置y轴、X轴的坐标名字与字体大小
plt.ylabel('price_doc', fontsize=16)
plt.xlabel('sub_area', fontsize=16)

#设置X轴的各列下标字体是水平的
plt.xticks(rotation='horizontal')

#设置Y轴下标的字体大小
plt.yticks(fontsize=15)
plt.show()

# 注：如果orient='v'表示成竖直显示的话，一定要记得y=group_df.sub_area.values, x=group_df.price_doc.values调换一下坐标轴，否则报错
```

示例二

```python
import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', family='SimHei', size=13)

num = np.array([13325, 9403, 9227, 8651])
ratio = np.array([0.75, 0.76, 0.72, 0.75])
men = num * ratio
women = num * (1-ratio)
x = ['聊天','支付','团购\n优惠券','在线视频']

width = 0.5
idx = np.arange(len(x))
plt.bar(idx, men, width, color='red', label='男性用户')
plt.bar(idx, women, width, bottom=men, color='yellow', label='女性用户')  #这一块可是设置bottom,top，如果是水平放置的，可以设置right或者left。
plt.xlabel('应用类别')
plt.ylabel('男女分布')
plt.xticks(idx+width/2, x, rotation=40)

#bar图上显示数字

for a,b in zip(idx,men):

    plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)
for a,b,c in zip(idx,women,men):
    plt.text(a, b+c+0.5, '%.0f' % b, ha='center', va= 'bottom',fontsize=12)

plt.legend()
plt.show()
```

### 盒图

```
IQR即统计学概念四分位距，第1/4分位与3/4分位之间的距离
N = 1.5IQR，如果一个值>Q3+N或<Q1-N则为离群点
```

便于查看离群点

```python
sns.boxplot(x="day", y="total_bill", hue="time", data=tips)

sns.boxplot(
  	x='is_overdue',
  	y='gjj_loan_balance',
 		hue='education',
  	hue_order=['HighSchool','Colege','University','Master','other'],
  	data=df,
  	showmeans=True,
  	fliersize=1,
  	order=['0','1-15','15-30','30-45','45+'])
# 参数
#fliersize=1将异常点虚化，showmeans=True显示平均值，order表示按x轴显示进行排序

sns.boxplot(data=iris, orient="h")  # 横着画
```

示例

```python
import matplotlib.pyplot as plt  
import seaborn as sns  
df_iris = sns.load_dataset("iris")
sns.boxplot(x = df_iris['class'],y = df_iris['sepal width'])  
plt.show() 
```

### 密度曲线

```python
sns.kdeplot()
```

示例

```python
import matplotlib.pyplot as plt  
import seaborn as sns  
df_iris = pd.read_csv('../input/iris.csv')  
fig, axes = plt.subplots(1,2)  
sns.distplot(df_iris['petal length'], ax = axes[0], kde = True, rug = True)        # kde 密度曲线  rug 边际毛毯  
sns.kdeplot(df_iris['petal length'], ax = axes[1], shade=True)                     # shade  阴影                         
plt.show() 
```

### 小提琴图

```python
sns.violinplot(x="total_bill", y="day", hue="time", data=tips, split=True)
```

### 点图

更好地描述变化差异

```python
sns.pointplot(x="class", y="survived", hue="sex", data=titanic, palette={"male": "g", "female": "m"}, markers=["^", "o"], linestyles=["-", "--"])
```

示例

```python
grouped_df = train_df.groupby('floor')['price_doc'].aggregate(np.median).reset_index()

plt.figure(figsize=(12,8))

sns.pointplot(grouped_df.floor.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])

plt.ylabel('Median Price', fontsize=12)

plt.xlabel('Floor number', fontsize=12)

plt.xticks(rotation='vertical') plt.show()
```

### 多层面板分类图

````python
sns.factorplot()
# 参数
x,y,hue		数据变量变量名
date			数据集
row,col		更多分类变量进行平铺显示 变量名
col_wrap	每行的最高平铺数 整数
estimator	在每个分类中进行矢量到标量的映射 矢量
ci				置信区间 浮点数或None			
n_boot		计算置信区间时使用的引导迭代次数 整数
units			采样单元的标识符，用于执行多级引导和重复测量设计 数据变量或向量数据
order,hue_order 对应排序列表 字符串列表
row_order,col_order 对应排序列表 字符串列表
kind  可选，poin默认
````

示例

```python
# 多层面板分类图
sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips)  # 默认折线图

sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips, kind="bar")  # 条形图

sns.factorplot(x="day", y="total_bill", hue="smoker", col="time" data=tips, kind="swarm")  # 树群图

sns.factorplot(x="time", y="total_bill", hue="smoker", col="day", data=tips, kind="box", size=4, aspect=.5)  # 盒图
```

### FacetGrid

facetGrid可以根据类别特征各种不同组合进行显示，下面就是根据婚姻与学历情况进行分成了10组，横（row）的表示按婚姻分类显示，竖（col）的表示按学历分类显示

```python
grid=sns.FacetGrid(df,row='martial_status',col='education',palette='seismic',size=4)
grid.map(plt.scatter,'gjj_loan_balance','max_overduer_days')
grid.add_legend()
plt.show()
```

示例

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Categorical

sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "axis_girds")))

tips = sns.load_dataset("tips")
tips.head()

# 实例化预留线框
g = sns.FacetGrid(tips, col="time")
g.map(plt.hist, "tip")  # 画条形图
# 散点图
g = sns.FacetGrid(tips, col="sex", hue="smoker")
g.map(plt.scatter, "total_bill", "tip", alpha=.7)
g.add_legend()
# 回归线
g = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
g.map(sns.regplot, "size", "total_bill", color=".3", fit_reg=False, x_jitter=.1)
# 条形图
g = sns.FacetGrid(tips, col="day", size=4, aspect=.5)
g.map(sns.barplot, "sex", "total_bill")
# 指定顺序
ordered_days = tips.day.value_counts().index
# ordered_days = Categorical(['Thur', "Fri", "Sat", "Sun"])
g = sns.FacetGrid(tips, row="day", row_order=ordered_days, size=1.7, aspect=4)
g.map(sns.boxplot, "total_bill")
# 调色板
pal = dict(Lunch="seagreen", Dinner="gray")
g = sns.FacetGrid(tips, hue="time", palette=pal, size=5)
g.map(plt.scatter, "total_bill", "tip", s=50, alpha=.7, linewidth=.5, edgecolor="white")
g.add_legend()
# 指定标识
g = sns.FacetGrid(tips, hue="sex", palette="Set1", size=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "total_bill", "tip", s=100, alpha=.5, linewidth=.5, edgecolor="white")
g.add_legend()
# 轴处理
with sns.axes_style("white"):
  	g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, size=2.5)
g.map(plt.scatter, "total_bill", "tip", color="#334488", edgecolor="white", lw=.5)
g.set_axis_lables("Total bill (US Dollars)", "Tip")  # 轴标签
g.set(xticks=[10, 30, 50], yticks=[2, 6, 10])  # 轴取值范围
g.fit.subplots_adjust(wspace=.02, hspace=.02)  # 子图与子图间间隔
# g.fit.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=.02, haspace=.02)
# 多变量比较图
# 指定对角线点图
iris = sns.load_dataset("iris")
g = sns.PairGrid(iris)
g.map(plt.scatter)
# 指定对角线条形图
g = sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
# 添加变量
g = sns.PairGrid(iris, hue="spaces")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()
# 取部分特征
g = sns.PairGrid(iris, vars=["space_length", "sepal_width"], hue="species")
g.map(plt.scatter)
# 设置调色板
g = sns.PairGrid(iris, hue="species", palette="GnBu_d")
g.map(plt.scatter, s=50, edgecolor="white")
g.add_legend()
```

### 热力图

```python
sns.heatmap(corrmat, square=True, linewidths=.5, annot=True)
```

示例

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(0)
sns.set()

uniform_data = np.random.rand(3, 3)
heatmap=sns.heatmap(uniform_data)


```

## 常见分析

### 单变量分析

```python
%matplotlib inline
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "distributions")))

# 直方图
x = np.random.normal(size=100)
sns.distplot(x, kde=False)

sns.distplot(x, bins=20, kde=False)  # 设置直方图条数 

X = np.random.gamma(6, size=200)
sns.distplot(x, bins=20, kde=False, fit=stats.gamma)  # 查看数据分布状况

# 散点图
mean, cov = [0.5, 1], [(1, .5),(.5, 1)]#设置均值(一组参数)和协方差（两组参数）
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
print(df.head())
sns.jointplot(x="x", y="y", data=df)
plt.show()

np.random.multivariate_normal(mean, cov, 1000).T  # 大批量数据使用hex散点图
with sns.axes_style("ticks"):
    sns.jointplot(x=x, y=y, kind="hex", color="k")
    
# 比较图
iris = sns.load_dataset("iris")
sns.pairplot(iris)#对角线是直方图(统计数量)，其他的是散点图
print(iris.head(2))
```

### 回归分析

```python
sns.regplot(x="total_bill", y="tip",  data=tips, x_jitter=.05)
# 参数
x				x轴变量
y				y轴变量
data		数据样本
x_jitter 干扰

sns.implot(x="total_bill", y="tip", data=tips)
```

示例

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns

sns.set(color_codes=True)
np.random.seed(sum(map(ord, "regression")))

tips = sns.load_dataset("tips")
tips.head()  # 前5行

# regplot()和implot()都可以绘制回归关系
sns.regplot(x="total_bill", y="tip", data=tips)
# sns.implot(x="total_bill", y="tip", data=tips)
sns.regplot(x="size", y="tip",  data=tips)
sns.regplot(x="size", y="tip",  data=tips, x_jitter=.05)
```

### 多变量分析

```python
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)

sns.swarmplot(x="day", y="total_bill", data=tips, hue="sex")
```

示例

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "categorical")))

titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

sns.stripplot(x="day", y="total_bill", data=tips)  # 重叠现象严重
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)  # 添加偏动

# 类树群图
sns.swarmplot(x="day", y="total_bill", data=tips)
sns.swarmplot(x="day", y="total_bill", data=tips, hue="sex")
sns.swarmplot(x="total_bill", y="day", data=tips, hue="time")

# 盒图
sns.boxplot(x="day", y="total_bill", hue="time", data=tips)

sns.boxplot(data=iris, orient="h")  # 横着画

# 小提琴图
sns.violinplot(x="total_bill", y="day", hue="time", data=tips)

sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True)

# 组合图
sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
sns.swarmplot(x="day", y="total_bill", data=tips, color="w", alpha=.5)

# 条形图
sns.barplot(x="sex", y="survived", hue="class", data=titanic)

# 点图
sns.pointplot(x="sex", y="survived", hue="class", data=titanic)

sns.pointplot(x="class", y="survived", hue="sex", data=titanic, palette={"male": "g", "female": "m"}, markers=["^", "o"], linestyles=["-", "--"])

# 多层面板分类图
sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips)  # 默认折线图

sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips, kind="bar")  # 条形图

sns.factorplot(x="day", y="total_bill", hue="smoker", col="time" data=tips, kind="swarm")  # 树群图

sns.factorplot(x="time", y="total_bill", hue="smoker", col="day", data=tips, kind="box", size=4, aspect=.5)  # 盒图

# FacetGrid

```



