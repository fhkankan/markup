[TOC]

# Matplotlib

Matplotlib 是一个 Python 的 2D绘图库，通过 Matplotlib，开发者可以仅需要几行代码，便可以生成绘图，直方图，功率谱，条形图，错误图，散点图等。

- 用于创建出版质量图表的绘图工具库
- 目的是为Python构建一个Matlab式的绘图接口
- pyploy模块包含了常用的matplotlib API函数，承担了大部分的绘图任务。

**参考学习**

<http://matplotlib.org>

<http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>

> 安装

```python
pip install marplotlib
```

> 引用

```python
import matplotlib as mpl  # 绘制复杂图形
import matplotlib.pyplot as plt  # 绘制简单图形
```

> 交互模式下测试

```
ipython --pylab
```

## 配置参数

### 设置方式

- 代码中

```python
# 方法一：使用参数字典rcParams
import matplotlib as mp1
mp1.rcParams['lines.linewidth'] = 2
mp1.rcParams['lines.color'] = 'r'

# 方法二：调用matplotlib.rc()
import matplotlib as mp1
mp1.rc('lines', linewidth=2, color='r')

# 重置动态修改后的配置参数
matplotlib.rcdefaults()
```

- 项目中

```python
# 配置文件位置决定了应用范围
# 1.当前工作目录
# 代码运行目录
./
# 2.用户级
# 通常是在用户的$HOME目录下(在windows系统中Documents and Settings目录)。
print(matplotlib.get_configdir())
# 3.安装配置文件
# 通常在python的site-packages目录下。是系统级配置，但每次重新安装matplotlib后，配置文件被覆盖。
print(matplotlib.matplotlib_fname())
```

配置文件中的配置项

```
axes: 设置坐标轴边界和表面的颜色、坐标刻度值大小和网格的显示
backend:设置目标输出TkAgg和GTKAgg
figure:控制dpi、边界颜色、图形大小和子区(subplot)设置
font:字体集(font family)、字体大小和样式设置
grid:设置网格颜色和线型
legend:设置图例和其中文本的显示
line:设置线条（颜色、线型、宽度等）和标记
patch:是填充2D空间的图形对象，如多边形和圆。控制线宽、颜色和抗锯齿设置等。
savefig:可以对保存的图形进行单独设置。如设置渲染的文件背景为白色
text:设置字体颜色、文本解析(纯文本或latex标记)等
verbose:设置matplotlib在执行期间信息输出，如silent/helpful/debug/debug-annoying
xticks/yicks：为x,y轴的主刻度和次刻度设置颜色、大小、方向，以及标签大小
```

### 中文异常

- 字体集

查看支持的字体集

```python
import matplotlib  
a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])  
  
for i in a:  
    print(i)  
```

增加支持的字体集

```python
# 查找配置文件
print(matplotlib.get_configdir())
# 修改配置中字体文件fontList.json，在ttflist列表中, 添加中文字体集
"ttflist": [
    {
      "style": "normal",
      "name": "Heiti",  # 可引用的名字
      "weight": 400,
      "fname": "/System/Library/Fonts/STHeiti Medium.ttc",  # 字体文件路径
      "stretch": "normal",
      "_class": "FontEntry",
      "variant": "normal",
      "size": "scalable"
    },
...
```

一些中文字体的英文名

```
宋体 SimSun
黑体 SimHei
微软雅黑 Microsoft YaHei
微软正黑体 Microsoft JhengHei
新宋体 NSimSun
新细明体 PMingLiU
细明体 MingLiU
标楷体 DFKai-SB
仿宋 FangSong
楷体 KaiTi
隶书：LiSu
幼圆：YouYuan
华文细黑：STXihei
华文楷体：STKaiti
华文宋体：STSong
华文中宋：STZhongsong
华文仿宋：STFangsong
方正舒体：FZShuTi
方正姚体：FZYaoti
华文彩云：STCaiyun
华文琥珀：STHupo
华文隶书：STLiti
华文行楷：STXingkai
华文新魏：STXinwei
```

- 配置中文支持

FontProperties

```python
# 方法一：硬编码
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 步骤一

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 步骤二
plt.xlabel("x轴", fontproperties=font) # 步骤三
plt.ylabel("y轴", fontproperties=font)
plt.title("标题", fontproperties=font)
plt.show()

# 方法二：引用
plt.xlabel("x轴")   # 使用默认字体
plt.ylabel("y轴", fontproperties="SimSun") # 使用宋体
plt.title("标题", fontproperties="SimHei") # 使用黑体
plt.show()
```

rcParams

```python
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
#...

plt.xlabel("x轴")
plt.ylabel("y轴")
plt.title("标题")
plt.show()
```

rc

```python
import matplotlib.pyplot as plt

font = {'family' : 'SimHei',
        'weight' : 'bold',
        'size'   : '16'}
plt.rc('font', **font)               # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）

#...

plt.xlabel("x轴")
plt.ylabel("y轴")
plt.title("标题")
plt.show()
```

## figure对象

在Matplotlib中，整个图像为一个figure对象

Matplotlib 的图像均位于figure对象中

```python
# 创建figure
fig = plt.figure()
# 参数
figsize=(a,b),figure的大小，a表示width，b表示height

#  如果不创建figure对象，matplotlib会自动创建一个figure对象。
```

示例

```python
import numpy as np
import matplotlib.pyplot as plt

# 指定为黑体中文字体，防止中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 默认大小(6 * 4) * 72 = (432 * 288)
# 1.创建画布，绘图就回执在画布上
plt.figure(figsize=(10.24, 7.68), dpi=100)
arr = np.arange(1000)
# 2.调用相关的绘图工具绘制图形
plt.plot(arr)
# 3.保存画布的资源图片
plt.savefig("./test.png")
plt.show()
```

## 画布设置

```python
import matplotlib.pyplot as plt

# 坐标的范围
plt.xlim(-5, 15)  # x轴坐标的范围
plt.ylim(-1, 1)  # y轴坐标的范围
plt.axis([-5, 15, -1, 1])  # x,y轴坐标的范围

# 标签：表示x轴和y轴的名称
plt.xlabel("月份", fontsize=15)
plt.ylabel("销售额/万", fontsize=15)

# 题目
plt.title("2017年水果销售额汇总", fontsize=20)

# 图示
plt.plot(arr1, "ro-",  label="苹果")
plt.legend(loc="best")  # 在最合适的位置显示图例

# 刻度
plt.xticks(
    [0,1,2,3,4,5,6,7,8,9,10,11],
    ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
)
plt.yticks([],[])
plt.xticks(rotation=45)  # 逆时针旋转45度
```

## 网格列表

```python
plt.subplots(rows,colums,num)
# 返回新创建的figure和subplot对象数组
```

示例

```python
# 生成2行2列subplot
fig, subplot_arr = plt.subplots(2,2)
# bins 为显示个数，一般小于等于数值个数
subplot_arr[1,0].hist(np.random.randn(100), bins=10, color='b', alpha=0.3)
plt.show()
```

样例

```python
from matplotlib.pyplot import *

# 样本数据
x = [1,2,3,4]
y = [5,4,3,2]
# 创建画布
figure()
# 创建网格子图
subplot(231)
plot(x, y)

subplot(232)
bar(x, y)

subplot(233)
barh(x, y)

subplot(234)
bar(x, y)
y1 = [7,8,5,3]
bar(x, y1, bottom=y, color = 'r')

subplot(235)
boxplot(x)

subplot(236)
scatter(x,y)

show()
```

## 分割子图

subplot命令是将图片窗口划分成若干区域,按照一定顺序使得图形在每个小区域内呈现其图形。

在figure对象中可以包含一个或者多个Axes对象。

每个Axes(ax)对象都是一个拥有自己坐标系统的绘图区域

plot 绘图的区域是最后一次指定subplot的位置 (jupyter notebook里不能正确显示)

```python
fig.add_subplot(a, b, c)

# 参数
a, b 表示将fig分割成 a * b 的区域
c 表示当前选中要操作的区域，
# 注意：从1开始编号（不是从0开始）
```

示例

```python
# 分隔子图需要保留画布
fig = plt.figure(figsize=(8, 6), dpi=100)
arr1 = np.random.randn(100)
arr2 = np.random.randn(100)
arr3 = np.random.randn(100)
arr4 = np.random.randint(-5, 10, (10, 10))
# 分布位置
ax1 = fig.add_subplot(2,2,1)
# ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
# 绘图
ax1.plot(arr1)
ax1.set_title("ax1线型图")
ax2.hist(arr2)
ax2.set_title("ax2直方图")
ax3.pie(arr3)
ax3.set_title("ax3饼图")
ax4_im = ax4.imshow(arr4)
ax4.set_title("ax4混淆矩阵图")
plt.colorbar(ax4_im)
# 显示
plt.show()
```

## 常用图表

### 线型图

```python
plt.plot(arr1, color="r", marker="o", linestyle="--", markerfacecolor="yellow", markersize=10, alpha=0.4, label="苹果")

# 参数
arr1						特征数据
color						线条颜色
marker					标记
linestyle				线型
markerfacecolor	标记的颜色
markersize			标记大小
alpha						透明度
label						标签（通过legend()显示）
```

示例

```python
# 指定为黑体中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 统一设置字体大小
plt.rcParams["font.size"] = 12.0
# 查看主题
# print(plt.style.available)
# 使用主题
plt.style.use("ggplot")
# 创建800*600的Figure画布
plt.figure(figsize=(8, 6), dpi=100)

# 获取需要绘制的三组数据
# 苹果 今年每个月的销售额
arr1 = np.random.randint(10, 30, 12)
# 香蕉 今年每个月的销售额
arr2 = np.random.randint(30, 50, 12)
# 梨子 今年每个月的销售额
arr3 = np.random.randint(50, 60, 12)

# 颜色、标记、线型 可以简写
plt.plot(arr1, "ro-", markerfacecolor="yellow", markersize=5, alpha=0.4, label="苹果")  # label 表示每个线的标签，通过legend() 图例显示出来
plt.plot(arr2, "ko-", markerfacecolor="yellow", markersize=5, alpha=0.4, label="香蕉") 
plt.plot(arr3, "go-", markerfacecolor="yellow", markersize=5, alpha=0.4, label="梨子") 

# 添加画布图像的属性和参数(所有绘图图形共享)
# 1.标题
plt.title("2017年水果销售额汇总", fontsize=20)
# 2.刻度：在指定客堵上绘制文字
# X轴刻度：
plt.xticks(
    [0,1,2,3,4,5,6,7,8,9,10,11],
    ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
)
# y轴刻度：
# plt.yticks([],[])
# 3.标签：表示x轴和y轴的名称
plt.xlabel("月份", fontsize=15)
plt.ylabel("销售额/万", fontsize=15)
# 4.图例：显示线形图的label属性
plt.legend()
# 5.添加背景网格线
plt.grid()
# 6.保存画布图片到磁盘文件
plt.savefig("2017年水果销售额汇总.png")
# 7.输出画布图片
plt.show()
```

> 正弦/余弦图

```python
from pylab import *
import numpy as np

# generate uniformly distributed 
# 256 points from -pi to pi, inclusive
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)

# these are vectorised versions
# of math.cos, and math.sin in built-in Python maths
# compute cos for every x
y = np.cos(x)

# compute sin for every x
y1 = np.sin(x)

# plot cos
plot(x, y)

# plot sin
plot(x, y1)

# define plot title
title("Functions $\sin$ and $\cos$")

# set x limit
xlim(-3.0, 3.0)
# set y limit
ylim(-1.0, 1.0)

# format ticks at specific values
xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
          [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
yticks([-1, 0, +1],
          [r'$-1$', r'$0$', r'$+1$'])

show()
```

### 直方图

```python
plt.hist(arr, bins=100, range(4, 5), color="r", alpha=0.4)
# 参数
arr	x轴表示值大小，y轴表示每个值的个数
bins	直方的个数
range()	显示x值的范围
color 颜色
alpha	透明度
```
示例
```python
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False
# arr = np.arange(10)
# arr = np.randint(-5, 10, 100)
arr = np.random.uniform(-5, 10, 100)
plt.figure(figsize=(8, 6), dpi=100)

plt.hist(arr, bins=100, color="r", alpha=0.4)
plt.show()
```

### 散点图

```python
plt.scatter(x_data, y_data, c="r", alpha=0.4, s=100)
# 参数
x_data		特征数据1
y_data		特征数据2
c					颜色	
alpha			透明度
s 				散点大小
```

示例

```python
x_data = np.arange(100)
# randn()标准正态分布
y_data = x_data * np.random.randn(100)
plt.figure(figsize=(8, 6), dpi=100)
# 两组数据分别表示x轴和y轴的值，元素个数必须相同

plt.scatter(x_data, y_data, c="r", alpha=0.4, s=100)
plt.show()
```

### 柱形图

```python
# 纵向
plt.bar(x, y, width, color="blue", alpha=0.5, label="男")
# 参数
x	表示该组每个柱子的x轴刻度位置
y	表示该组的每个数据y轴刻度位置
width	表示该组柱子的宽度
color	颜色
alpha	透明度
label	标签

# 横向
plt.barh(x, y, width, color="blue", alpha=0.5, label="男")
# 参数
x	表示该组每个柱子的x轴刻度位置
y	表示该组的每个数据y轴刻度位置
width	表示该组柱子的宽度
color	颜色
alpha	透明度
label	标签
```

纵向

```python
plt.figure(figsize=(8, 6), dpi=100)
# 表示x轴的四组数据的刻度：四个城市
x = np.arange(4)
# 四个城市：男性的人口数据
y1 = np.random.randint(10, 50, 4)
# 四个城市：女性的人口数据
y2 = np.random.randint(10, 50, 4)
# 柱子的宽度
width = 0.25
plt.bar(x, y1, width, color="blue", alpha=0.5, label="男")
# 每绘制一组新的数据，x轴必须右移避免形状重叠，y轴不用修改
# x轴右移宽度 小于 柱子宽度，配合alpha可以做到板重叠效果
plt.bar(x + width - 0.1, y2, width, color="red", alpha=0.5, label="女")
# 标题
plt.title("各城市人口分布数量", fontsize=20)
# x坐标文字
plt.xticks(x + width/2, ["北京", "上海", "广州", "深圳"])
# y坐标文字
plt.xlabel("城市")
# y坐标单位
plt.ylabel("人口数量/万")
# 显示分类
plt.legend()
plt.grid()
plt.show()
```

横向

```python
plt.figure(figsize=(8, 6), dpi=100)
# 表示x轴的四组数据的刻度：四个城市
x = np.arange(4)
# 四个城市：男性的人口数据
y1 = np.random.randint(10, 50, 4)
# 四个城市：女性的人口数据
y2 = np.random.randint(10, 50, 4)
# 柱子的宽度
width = 0.25
plt.barh(x, y1, width, color="blue", alpha=0.5, label="男")
plt.barh(x + width - 0.1, y2, width, color="red", alpha=0.5, label="女")
# 每绘制一组新的数据，x轴必须右移避免形状重叠，y轴不用修改
# x轴右移宽度 小于 柱子宽度，配合alpha可以做到板重叠效果
plt.title("各城市人口分布数量", fontsize=20)
plt.yticks(x + width/2, ["北京", "上海", "广州", "深圳"])
plt.ylabel("城市")
plt.xlabel("人口数量/万")

plt.legend()
plt.grid()
plt.show()
```

### 混淆矩阵

```python
plt.imshow()
```

示例

```python
plt.figure(figsize=(8, 6), dpi=100)
arr = np.random.randint(-5, 10, (10, 10))
# cmap表示颜色主题
plt.imshow(arr, cmap=plt.cm.Blues)
# 显示颜色条
plt.colorbar()
# 关闭刻度显示
plt.axis("off")
# 显示
plt.show()
```

### 饼图

```
plt.pie()
```

示例

```python
arr = np.random.randint(10, 100, 5)
print(arr)
plt.figure(figsize=(8, 6), dpi=100)
plt.pie(
    # 数据
    arr,
    # 标签 
    labels = ["北京", "上海", "广州", "深圳", "杭州"],
    # 颜色
    colors = ["red", "yellow", "green", "gray", "blue"],
    # 凸显部分(>0.0)
    explode = [0.1, 0.0, 0.0, 0.2, 0.0],
    # 显示立体阴影
    shadow = True,
    # 表示显示百分比数字
    autopct = "%2.2f%%",
)
# 显示表针的原型(默认椭圆)tt
plt.axis("equal")
plt.title("各城市GDP占比情况", fontsize=20)
plt.legend()
plt.show()
```

### 盒图

```
plt.boxplot()
```

示例

```python
# 单个盒图
fig, ax = plt.subplots()
ax.boxplot(norm_revews['RT_user_norm'])
ax.set_xticklabels(['Rotten Tomatoes'])
ax.set_ylim(0, 5)
plt.show()

# 多个盒图
num_cols = ['RT_user_norm', 'Metacritic_user_normal', 'IMDB_norm', 'Fandango_Ratingvalue']
fig, ax = plt.subplots()
ax.boxplot(norm_revews[num_cols].values)
ax.set_xticklabels(num_cols, rotation=90)
ax.set_ylim(0, 5)
plt.show()
```

## 设置样式

### 坐标轴

坐标轴范围

```
1. 不使用axis()或者其他参数设置
matplotlib会自动使用最小值，可以让我们在一个图中看到所有的数据点
2. 设置axis(v)
若范围比数据集合中的最大值小，则按照设置执行，会无法在图中看到所有的数据点
3. 自适应
maplotlib.pyplot.autoscale()会计算坐标轴的最佳大小以适应数据的显示
```

设定坐标轴范围

```
axis(*v, **kwargs)

参数
不带参数		返回坐标的默认值[xmin,xmax,ymin,ymax]
[xmin,xmax,ymin,ymax]	设置坐标轴范围
```

在相同图形中添加新的坐标轴

```
用途：需要几个不同的视图来表达相同的数据的不同属性值，可以在一张图中组合显示多个图表

matplotlib.pyplot.axes(arg=None, **kwargs)

参数
rect
left/bottom/width/height
axisbg		指定坐标轴的背景颜色
sharex/sharey	接收其他坐标轴的值，并让当前坐标轴(x/y)共享相同的值
polar		指定是否使用极坐标轴
```

对当前图形添加一条线

```
matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1. hold=None, **kwargs)
matplotlib.pyplot.axvline(x=0, ymin=0, hold=None, **kwargs)

根据给定的x/y值相应地绘制出相对于坐标轴的水平线和垂直线

参数
不传参数	默认x=0,y=0
```

对当前图形添加一个矩形

```
matplotlib.pyplot.axhspan(ymin, ymax, xmin=0, xmax=1, hold=None, **kwargs)
matplotlib.pyplot.axvsapn(xmin, ymax, ymin=0, ymax=1, hold=None, **kwargs)
```

网格属性，默认关闭

```
matplotlib.pylot.grid(b=None, which='major', axis='both', **kwargs)

参数
不传参数	切换网格的显示状态
which	指定绘制的网格刻度类型(major, minor, both)
axis	指定绘制哪组网格线(both, x, y)
```

坐标轴内部实现，高级控制

```
matplotlib.axes.Axes	包含了操作坐标轴的大多数方法
matplotlib.axes.Axis	表示单独一个坐标轴
matplotlib.axes.XAxix	表示x轴
matplotlib.axes.YAxis	表示y轴
```

### 图表样式

<http://matplotlib.org/gallery.html>

#### 颜色标价线型

改变线的属性方法

```
方法一：向方法中传入参数
plot(x, y, linewidth=1.5)

方法二：调用方法返回实例后使用setter方法
line = plot(x, y)
line.set_linewidth(1.5)

方法三：setp()方法
line = plot(x, y)
setp(line, 'linewidth', 1.5)
```

线条属性

```
所有属性都包含在matplotlib.lines.Line2D类中
```

常用属性

| 属性                 | 类型                                          | 描述                                                         |
| -------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| alpha                | float                                         | 用来设置混色，并不是所有的后端都支持                         |
| color或c             | 任意matplotlib颜色/十六进制颜色值/归一化的rgb | 设置线条颜色                                                 |
| dashes               | 以点为单位的on/off序列                        | 设置破折号序列，若seq为空或若seq=[None, None],linestyle将被设置为solid |
| label                | 任意字符串                                    | 为图例设置标签值                                             |
| linestyle或ls        | 线条形状参数                                  | 设置线条风格(也接收drawstyles的值)                           |
| linewidth或lw        | 以点为单位的浮点值                            | 设置以点为单位的线宽                                         |
| marker               | 线条标记参数                                  | 设置线条标记                                                 |
| markeredgecolor或mec | 任意matplotlib颜色/十六进制颜色值/归一化的rgb | 设置标记的边缘颜色                                           |
| markeredgewidth或mew | 以点为单位的浮点值                            | 设置以嗲那位单位的标记边缘宽度                               |
| markerfacecolor或mfc | 任意matplotlib颜色/十六进制颜色值/归一化的rgb | 设置标记的颜色                                               |
| markersize或ms       | float                                         | 设置以点为单位的标记大小                                     |
| solid_capstyle       | ['butt'\|'round'\|'projecting']               | 设置实线的线端风格                                           |
| solid_joinstyle      | ['miter'\|'round'\|'bevel']                   | 设置实线的连接风格                                           |
| visible              | bool                                          | 显示或隐藏artist                                             |
| xdata                | np.array                                      | 设置x的np.array值                                            |
| ydata                | np.array                                      | 设置y的np.array值                                            |
| Zorder               | 任意数字                                      | 为artist设置z轴顺序，低Zorder的artist会先绘制，若在屏幕上x轴水平向右，y轴处置向上，则z轴将指向观察者。这样，0表示在屏幕上，1表示上面的一层，以此类推 |

颜色

| 简写 | 全名   |
| ---- | ------ |
| b    | 蓝色   |
| g    | 绿色   |
| r    | 红色   |
| c    | 青色   |
| m    | 洋红色 |
| y    | 黄色   |
| k    | 黑丝   |
| w    | 白色   |

标记

| marker             | description | marker | description    |
| ------------------ | ----------- | ------ | -------------- |
| 'o'                | 圆圈        | '.'    | 点             |
| 'D'                | 菱形        | 's'    | 正方形         |
| 'h'                | 六边形1     | '*'    | 星号           |
| 'H'                | 六边形2     | 'd'    | 小菱形         |
| '_'                | 水平线      | 'v'    | 一角朝下三角形 |
| '','None',' ',None | 无          | '<'    | 一角超左三角形 |
| '8'                | 八边形      | '>'    | 一角超右三角形 |
| 'p'                | 五边形      | '^'    | 一角朝上三角形 |
| ','                | 像素        | '\|'   | 竖线           |
| '+'                | 加号码      | 'x'    | X              |

线型

| linestyle      | description |
| -------------- | ----------- |
| '-'/'solid'    | 实现        |
| '--'/'dashed'  | 破折线      |
| '-.'/'dashdot' | 点划线      |
| ':'/'dotted'   | 虚线        |
| 'None','',' '  | 什么都不画  |

#### 刻度标签图例

对于简单的图表，使用`figure()`、`plot()`、`subplot()`即可使用，对于更多的高级控制，需要使用`matplotlib.axes.Axes`类的坐标轴实例

> 刻度

刻度是图形的一部分，由刻度定位器(tick locator)指定刻度锁在的位置和刻度格式器(tick formatter)指定刻度显示的样式组成。刻度有主刻度(major ticks)和此刻度(minor ticks)，默认不显示次刻度。朱刻度和此刻度可以独立地指定位置和格式化

可以使用`matplotlib.pyplot.locator_params()`控制刻度定位器的行为。尽管刻度位置通常会自动被确定下来，还是可以控制刻度的数目、在plot比较小时使用一个紧凑视图(tight view)

```python
from pylab import *

# get current axis
ax = gca()

# set view to tight, and maximum number of tick intervals to 10
ax.locator_params(tight=True, nbins = 10)

# generate 100 normal distribution values
ax.plot(np.random.normal(10, .1, 100))

show()
```

也可以使用locator类完成相同的设置

```
# 设置主定位器为10的倍数
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
```

刻度格式器的配置很简单，格式器规定了值(数字)的显示方式，如用`matplotlib.ticker.FormatStrFormatter`可以方便地指定`%2.1f`或`%1.1f cm`的字符格式作为刻度标签

```
matplotlib用浮点值表示日期，其值从0001-01-01 UTC起的天数加1。001-01-01 UTC 06:00 的值为1.25

可以使用matplotlib.dates.date2num()/matplotlib.dates.num2data()/matplotlib.dates.drange()对日期进行不同形式的转换
```

示例

```python
from pylab import *
import matplotlib as mpl
import datetime

fig = figure()

# get current axis
ax = gca()

# set some daterange
start = datetime.datetime(2013, 01, 01)
stop = datetime.datetime(2013, 12, 31)
delta = datetime.timedelta(days = 1)

# convert dates for matplotlib
dates = mpl.dates.drange(start, stop, delta)

# generate some random values
values = np.random.rand(len(dates))

ax = gca()

# create plot with dates
ax.plot_date(dates, values, linestyle='-', marker='')

# specify formater
date_format = mpl.dates.DateFormatter('%Y-%m-%d')

# apply formater
ax.xaxis.set_major_formatter(date_format)

# autoformat date labels
# rotates labels by 30 degrees by default
# use rotate param to specify different rotation degree 
# use bottom param to give more room to date labels
fig.autofmt_xdate()

show()
```

> 图例与注解

```python
from matplotlib.pyplot import *

# generate different normal distributions
x1 = np.random.normal(30, 3, 100)
x2 = np.random.normal(20, 2, 100)
x3 = np.random.normal(10, 3, 100)

# plot them
plot(x1, label='plot')
plot(x2, label='2nd plot')
plot(x3, label='last plot')

# generate a legend box
# 列数为3，位置lower left,指定边界框起始位置(0.0, 1.02)，并设置宽度为1，高度为0.102，基于归一化轴坐标系。参数node可设置为None或expend,当为expend时，图例框会水平扩展至整个坐标轴区域。参数borderaxespad指定了坐标轴和图例边界之间的间距
legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode="expand", borderaxespad=0.)

# annotate an important value
# 注解，设置xycoords='data'，可以指定注解和数据使用相同的坐标系，注解文本的起始位置童工xytext指定，箭头由xytext指向xy坐标位置。arrowprops字典中定义了很多箭头属性
annotate("Important value", (55,20), xycoords='data',
         xytext=(5, 38), 
         arrowprops=dict(arrowstyle='->')) 
show()
```

位置参数

| 字符串      | 数值 | 字符串       | 数值 |
| ----------- | ---- | ------------ | ---- |
| upper right | 1    | center left  | 6    |
| upper left  | 2    | center right | 7    |
| Lower left  | 3    | lower center | 8    |
| Lower right | 4    | upper center | 9    |
| right       | 5    | Center       | 10   |

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2)
axes[0].plot(np.random.randint(0, 100, 50), 'ro--')
# 等价
axes[1].plot(np.random.randint(0, 100, 50), color='r', marker='o', linestyle='dashed')
plt.show()
```

- 设置刻度范围

```
plt.xlim(), plt.ylim()

ax.set_xlim(), ax.set_ylim()
```

- 设置显示的刻度

```
plt.xticks(), plt.yticks()

ax.set_xticks(), ax.set_yticks()
```

- 设置刻度标签

```
ax.set_xticklabels(), ax.set_yticklabels()
```

- 设置坐标轴标签

```
ax.set_xlabel(), ax.set_ylabel()
```

- 设置标题

```
ax.set_title()
```

- 图例

```
ax.plot(label=‘legend’)

ax.legend(), plt.legend()

loc=‘best’：自动选择放置图例最佳位置
```

#### 边框线

轴线定义了数据区域的边界，把坐标轴刻度标记连接起来。一共有四个轴线，可以把它们放置在任何位置。默认情况下，它们被放置在坐标轴的边界，故看到数据图表有一个框

移动轴线到图中央

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 500, endpoint=True) 
y = np.sin(x)

plt.plot(x, y)

ax = plt.gca()

# hide two spines 
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# move bottom and left spine to 0,0
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

# move ticks positions
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.show()
```

轴线可以被限制在数据结束的地方结束，如调用`set_smart_bounds(True)`。此时，matplotlib会尝试以一种复杂的方式设置边界。如处理颠倒的界限或在数据延伸出视图的情况下裁剪线条以适应视图