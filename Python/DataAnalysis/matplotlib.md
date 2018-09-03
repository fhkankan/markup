# Matplotlib

Matplotlib 是一个 Python 的 2D绘图库，通过 Matplotlib，开发者可以仅需要几行代码，便可以生成绘图，直方图，功率谱，条形图，错误图，散点图等。

- 用于创建出版质量图表的绘图工具库
- 目的是为Python构建一个Matlab式的绘图接口
- pyploy模块包含了常用的matplotlib API函数，承担了大部分的绘图任务。

**参考学习**

<http://matplotlib.org>

<http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot>

> 引用

```
import matplotlib.pyplot as plt
```

## 配置参数

- 代码中

```
# 方法一：使用参数字典rcParams
import matplotlib as mp1
mp1.rcParams['lines.linewidth'] = 2
mp1.rcParams['lines.color'] = 'r'

# 方法二：调用matplotlib.rc()
import matplotlib as mp1
mp1.rc('lines', linewidth=2, color='r')

# 重置参数
matplotlib.rcdefaults()
```

- 项目中

```python
# 配置文件位置决定了应用范围
# 1.当前工作目录matplotlibrc
代码运行目录，在当前目录下，可以为目录所包含的当前项目代码定制matplotlib配置项
# 2.用户级.matplotlib/matplotlibrc
通常是在用户的$HOME目录下(在windows系统中Documents and Settings目录)。可以用matplotlib.get_configdir()命令来找到当前用户的配置文件目录
# 3.安装配置文件
通常在python的site-packages目录下。是系统级配置，但每次重新安装matplotlib后，配置文件被覆盖。

# 打印出配置文件目录的位置
python -c 'import matplotlib as mpl; print mpl.get_configdir()'
```

- 配置项

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

## figure对象

- 在Matplotlib中，整个图像为一个figure对象
- Matplotlib 的图像均位于figure对象中
- 创建figure：`fig = plt.figure()`
- 如果不创建figure对象，matplotlib会自动创建一个figure对象。

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

## 线型图

`plt.plot()`

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

# 绘制线型图以及相关的属性
#                颜色       标记         线型           标记的颜色                标记大小       透明度     标签（通过legend()显示）
# plt.plot(arr1, color="r", marker="o", linestyle="--", markerfacecolor="yellow", markersize=10, alpha=0.4, label="苹果")  # label 表示每个线的标签，通过legend() 图例显示出来
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

## 直方图

`plt.hist()`

```python
# 解决负号显示问题
plt.rcParams["axes.unicode_minus"] = False
# arr = np.arange(10)
# arr = np.randint(-5, 10, 100)
arr = np.random.uniform(-5, 10, 100)
plt.figure(figsize=(8, 6), dpi=100)
# x轴表示值大小，y轴表示每个值的个数
# bins表示直方的个数
plt.hist(arr, bins=100, color="r", alpha=0.4)
plt.show()
```

## 散点图

`plt.scatter()`

```python
x_data = np.arange(100)
# randn()标准正态分布
y_data = x_data * np.random.randn(100)
plt.figure(figsize=(8, 6), dpi=100)
# 两组数据分别表示x轴和y轴的值，元素个数必须相同
# s表示每个三点的大小
plt.scatter(x_data, y_data, c="r", alpha=0.4, s=100)
plt.show()
```

## 纵向柱形图

`plt.bar()`

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
# 参数1：表示该组每个柱子的x轴刻度位置
# 参数2：表示该组的每个数据y轴刻度位置
# 参数3：表示改组柱子的宽度
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

## 横向柱形图

`plt.barh()`

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

## 混淆矩阵

`plt.imshow()`

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

## 饼图

`plt.pie()`

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

## 分割子图

- subplot命令是将图片窗口划分成若干区域,按照一定顺序使得图形在每个小区域内呈现其图形。
- 在figure对象中可以包含一个或者多个Axes对象。
- 每个Axes(ax)对象都是一个拥有自己坐标系统的绘图区域
- `fig.add_subplot(a, b, c)`
  - a, b 表示将fig分割成 a * b 的区域
  - c 表示当前选中要操作的区域，
  - 注意：从1开始编号（不是从0开始）
- plot 绘图的区域是最后一次指定subplot的位置 (jupyter notebook里不能正确显示)

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

## plt.subplots()

- 同时返回新创建的`figure`和`subplot`对象数组
- 生成2行2列subplot:`fig, subplot_arr = plt.subplots(2,2)`
- 在jupyter里可以正常显示，推荐使用这种方式创建多个图表

```
fig, subplot_arr = plt.subplots(2,2)
# bins 为显示个数，一般小于等于数值个数
subplot_arr[1,0].hist(np.random.randn(100), bins=10, color='b', alpha=0.3)
plt.show()
```

## 图像样式

<http://matplotlib.org/gallery.html>

### 颜色、标价、线型

- ax.plot(x, y, ‘r--’)

> 等价于ax.plot(x, y, linestyle=‘--’, color=‘r’)

```
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2)
axes[0].plot(np.random.randint(0, 100, 50), 'ro--')
# 等价
axes[1].plot(np.random.randint(0, 100, 50), color='r', marker='o', linestyle='dashed')

plt.show()
```

- 颜色

| 简写 | 全名    |
| ---- | ------- |
| b    | blue    |
| g    | green   |
| r    | red     |
| c    | cyan    |
| m    | magenta |
| y    | yellow  |
| k    | black   |
| w    | white   |

- 标记

| marker | description   |
| ------ | ------------- |
| .      | point         |
| ,      | pixel         |
| o      | circle        |
| v      | triangle_down |
| ^      | triangle_up   |
| <      | triangle_left |

- 线型

| linestyle      | description      |
| -------------- | ---------------- |
| '-'/'solid'    | solid line       |
| '--'/'dashed'  | dashed line      |
| '-.'/'dashdot' | dash-dotted line |
| ':'/'dotted'   | dotted line      |
| 'None'         | draw nothing     |
| ' '            | draw nothing     |
| ''             | draw nothing     |

### 刻度、标签、图例

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

