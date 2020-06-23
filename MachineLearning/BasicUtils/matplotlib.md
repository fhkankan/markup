[TOC]

# Matplotlib

python数据可视化常用库如下：

- matplotlib：最知名的python可视化程序库
- seaborn：对matplotlib做了封装的可视化程序库
- Bokeh：用python做前端的javascript可视化程序库，支持非常强大的交互可视化功能，可以处理非常大的批数据或流数据。python前端会生成一份json数据结构，通过Bokeh的JS引擎进行渲染
- Plotly：设计理念与Bokeh类似，高水平开发支持。
- Vispy：侧重于大数据动态可视化项目，建立在OpenGL接口上且可充分利用显卡。
- Vega/Vega-Lite：采用声明式图形表示方法，最终渲染式Javascript，但是API与编程语言无关。



Matplotlib 是一个 Python 的 2D绘图库，通过 Matplotlib，开发者可以仅需要几行代码，便可以生成绘图，直方图，功率谱，条形图，错误图，散点图等。

- 用于创建出版质量图表的绘图工具库
- 目的是为Python构建一个Matlab式的绘图接口
- pyploy模块包含了常用的matplotlib API函数，承担了大部分的绘图任务。

**参考学习**

<http://matplotlib.org>

> 安装

```python
pip install marplotlib
```

> 引用

```python
import matplotlib as mpl  # 绘制复                                   杂图形
import matplotlib.pyplot as plt  # 绘制简单图形
```

> 交互模式下测试

```
ipython --pylab
```

## 绘图样式

Matplotlib支持多种风格列表

```python
# 查看可用的风格
res = plt.style.available
print(res)
# 使用某种风格
plt.style.use('classic')
plt.stype.use('seaborn-whitegrid')

# 风格上下文管理器，可临时切换风格
with plt.style.context('stylename'):
    make_a_plot()
```

## 显示保存

- 显示

根据不同的环境，Matplotlib图形显示实践有所不同

**脚本**：需要使用`plt.show()`(一般在脚本最后使用，避免出现多个命令)。此命令会启动一个事件循环，并找到所有当前可用的图形对象，然后打开一个或多个交互式窗口显示图形。

**Ipython shell**：需要`%matplotlib`魔法命令，此后的`plt`命令会自动打开一个图形窗口，增加新的命令，图形就会更新。若某些属性没有及时更新，使用`plt.draw()`强制更新。

**Ipython Notebook**：有两种展现方式：`%matplotlib notebook` 会启动交互式图形；`%matplotlib inline` 会启动静态图形。

- 保存

查看操作系统支持保存的文件类型

```python
fig = plt.figure()
res = fig.canvas.get_supported_filetypes()
print(res)
```

保存

```python
# 保存文件
fig.savefig('file_name.type')
# 检查文件正确性(Ipython)
from IPython.display import Image
Image(file_path)
```

## figure/Axes

在画图形时，需要先创建一个图形fig和一个坐标轴ax

```python
fig = plt.figure()
ax = plt.axes()

# figure参数
figsize=(a,b),figure的大小，a表示width，b表示height
```

figure可以被看成是一个能够容纳各种坐标轴、图形、文字和标签的容器。如果不显式创建figure对象，matplotlib会自动创建一个figure对象。

axes是一个带有刻度和标签的矩形，最终会包含所有可视化的图形元素。

## 画图接口

matplotlib有两种画图接口：MATLAB风格接口，面向对象接口

matlab

```python
x = np.linspace(0, 10, 100)
plt.figure()  # 创建图形
plt.subplot(2, 1, 1)  # 子图1
plt.plot(x, np.sin(x))
plt.subplot(2, 1, 2)  # 子图2
plt.plot(x, np.cos(x))
plt.show()
# 特性：有状态的，会持续跟踪当前的图形和坐标轴，所有的plt命令均可用。可以使用plt.gcf()获取当前图形，plt.gca()获取当前坐标轴；
# 缺点：在创建第二图时，想修改第一个图实现复杂
```

面向对象

```python
fig, ax = plt.subplots(2)  # 创建图形网格,ax是包含两个Axes对象的数组
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))
plt.show()
# 可以适应更复杂的场景，更好地控制图形。画图函数不再受到当前活动图形或坐标轴的限制，而变成了显式的Figure和Axes的方法
```

## 多子图

有时需从多个角度对数据进行对比，这时需要子图：在较大的图形中同时放置一组较小的坐标轴。这些子图可能是画中画、网格图或其他更复杂的布局。如下有4种创建子图的方法

- `plt.axes` 手动创建子图

```python
# 画中画
ax1 = plt.axes()
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])  # 图形坐标系统[bottom, left, width, height]
plt.show()

# 两个竖直排列的坐标轴
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], ylim=(-1.2, 1.2))
x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x))
plt.show()
```

- `plt.subplot` 简易网格子图

```python
# matlab
plt.subplot(a, b, c)
# 面向对象
fig.add_subplot(a, b, c)

# 参数
a, b 表示将fig分割成 a * b 的区域
c 表示当前选中要操作的区域 # 注意：从1开始编号（不是从0开始）
```

示例

```python
# plt.subplot
for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')

# fig.add_subplot
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # 调整图与图之间的间隔
for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i)
    ax.text(0.5, 0.5, str((2, 3, i)), fontsize=18, ha='center')

plt.show()
```

- `plt.subplots`一行创建网格

```python
plt.subplots(rows,colums,num)
# 返回新创建的figure和subplot对象数组
```

示例

```python
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
for i in range(2):
    for j in range(3):
        ax[i, j].text(0.5, 0.5, str((i, j)), fontsize=18, ha='center')

plt.show()
```

- `plt.GridSpec`实现复杂排列

若要实现不规则的多行多列子图网络，`plt.GridSpec()`是最好的，它本身不能直接创建一个图例，只是`plt.subplot()` 可以识别的简易接口。

```python
# 一个带行列间距的2*3网格
grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
# 通过切片语法设置子图的位置和尺寸
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])

plt.show()
```

示例

```python
# 多轴频次直方图(seaborn中实现更简单)
# 创建一些正态分布数据
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T
# 设置坐标轴和网格配置方式
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
# 主坐标轴画散点图
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
# 次坐标轴画频次图
x_hist.hist(x, 40, histtype='stepfilled', orientation='vertical', color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, 40, histtype='stepfilled', orientation='horizontal', color='gray')
y_hist.invert_xaxis()

plt.show()
```

## 常用图表

### 线型图

```python
plt.plot(arr1, color="r", marker="o", linestyle="--", markerfacecolor="yellow", markersize=10, alpha=0.4, label="苹果")

# 参数
arr1					特征数据
color					线条颜色
marker					标记
linestyle				线型
markerfacecolor			标记的颜色
markersize				标记大小
alpha					透明度
label					标签（通过legend()显示）
```

示例

```python
fig = plt.figure()
ax = plt.axes()
x = np.linspace(0 , 10, 100)
# 方法一
ax.plot(x, np.sin(x))
# 方法二
plt.plot(x, np.sin(x))
```

### 散点图

方法一：使用`plt.plot`，尤其是在数据量比较大时，性能更佳

```
x = np.linsapce(0 ,10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black')
```

方法二：`plt.scatter`，可以单独控制每个散点与数据匹配，更具灵活性

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

### 误差可视化

- 基本误差线

```python
plt.errorbar(x, y, yerr=None, xerr=None, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
```

示例

```python
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy + np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='.k')  # 误差线
plt.show()
```

- 连续误差

通过使用`plt.plot`和`plt.fill_between`来实现。

```python
plt.fill_between(x, y1, y2=0, where=None, interpolate=False, step=None, *, data=None, **kwargs)
# 参数
x 	x轴坐标
y1	y轴下边界
y2	y轴上边界

plt.fill_betweenx(y, x1, x2=0, where=None, step=None, interpolate=False, *,data=None, **kwargs)
# 参数
y	y轴坐标
x1  x轴左边界
x2	x轴右边界
```

示例

```python
from sklearn.gaussian_process import GaussianProcessRegressor

# 定义模型和数据
model = lambda x: x * np.sin(x)
xdata = np.array([1, 3, 5, 6, 8])
ydata = model(xdata)

# 计算高斯过程拟合结果
gp = GaussianProcessRegressor()
gp.fit(xdata[:, np.newaxis], ydata)

xfit = np.linspace(0, 10, 1000)
yfit, y_std = gp.predict(xfit[:, np.newaxis], return_std=True)
dyfit = 2 * np.sqrt(y_std)  # 2*sigma ~ 95% confidence region

# 可视化
plt.plot(xdata, ydata, 'or')
plt.plot(xfit, yfit, '-', color='gray')
plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2)
plt.xlim(0, 10)
```

### 直方图

```python
plt.hist(arr, bins=100, range(4, 5), color="r", alpha=0.4)
# 参数
arr		x轴表示值大小，y轴表示每个值的个数
bins	直方的个数
range()	显示x值的范围
color 	颜色
alpha	透明度

# 不需要画图，只计算数据
counts, bin_edges = np.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)
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

### 数据区间划分

- 二维频次直方图

二维频次直方图是由坐标轴正交的方块风格而成。

```python
plt.hist2d(x, y, bins=10, range=None, density=False, weights=None, cmin=None, cmax=None, *, data=None, **kwargs)

# 不需画图，只计算数据
counts, xedges, yedges = np.histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None)
```

示例

```python
# 多元高斯分布生成x轴与y轴的样本数据
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.show()
```

- 六边形区间划分

六边形区间划分可将二维数据分割成蜂窝状

```python
plt.hexbin(x, y, C=None, gridsize=100, bins=None, xscale='linear',
        yscale='linear', extent=None, cmap=None, norm=None, vmin=None,
        vmax=None, alpha=None, linewidths=None, edgecolors='face',
        reduce_C_function=np.mean, mincnt=None, marginals=False, *,
        data=None, **kwargs)
```

示例

```python
# 多元高斯分布生成x轴与y轴的样本数据
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')
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

### 密度图/等高线

有时在二维图上用等高线图或彩色图来表示三维数据是不错的选择。

`plt.contour` 画等高线图；`plt.contourf`画带有填充色的等高线图的色彩；`plt.imshow` 显示图形。

```python
def f(x, y):
    res = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
    return res


x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)  # 将一维数组构建二维网格数据
Z = f(X, Y)

# 1.默认等高线
# plt.contour(X, Y, Z, colors='black')  # 标准等高线，虚线表示负数，实线表示正数
# 2.配色等高线
# plt.contour(X, Y, Z, 20, cmap='RdGy')  # 将数据范围等分为20份，用不同颜色表示，配色方案用RdGy
# # 查看配色方案：在IPython中用Tab键浏览plt.cm.<Tab>
# 3.填充等高线
# plt.contourf(X, Y, Z, 20, cmap='RdGy')  # 填充等高线，但是由于颜色改变是离散过程，图像效果不好

# 4.颜色渐变
# imshow()不支持x轴和y轴数据设置网格，而是必须通过extend参数设置图形的坐标范围[xmin, xmax, ymin, ymax]
# imshow()默认使用标准的图形数组定义，就是原点位于左上角，而不是左下角，显示时需调整
# plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy')

# 5.等高线与彩色图组合
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower', cmap='RdGy', alpha=0.5)

plt.colorbar()  # 自动创建一个表示图形各种颜色对应标签信息的颜色条

plt.show()
```

## 画布设置

### 常规

matlab

```python
# 坐标的范围
plt.xlim(-5, 15)  # x轴坐标的范围
plt.ylim(-1, 1)  # y轴坐标的范围
plt.clim(-1, 1)  # z轴坐标返回(颜色)
plt.axis([-5, 15, -1, 1])  # x,y轴坐标的范围

# 标签：表示x轴和y轴的名称
plt.xlabel("月份", fontsize=15)
plt.ylabel("销售额/万", fontsize=15)
# 题目
plt.title("2017年水果销售额汇总", fontsize=20)
# 刻度
plt.xticks(
    [0,1,2,3,4,5,6,7,8,9,10,11],
    ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
)
plt.yticks([],[])
plt.xticks(rotation=45)  # 逆时针旋转45度
# 图例
plt.plot(arr1, "ro-",  label="苹果")
plt.legend(loc="best")  # 在最合适的位置显示图例
```

面向对象

```python
# 坐标的范围
ax.set_xlim(-5, 15)  # x轴坐标的范围
ax.set_ylim(-1, 1)  # y轴坐标的范围
ax.set_clim(-1, 1)  # z轴坐标返回(颜色)
# 坐标轴标签
ax.set_xlabel("月份", fontsize=15)
ax.set_ylabel("销售额/万", fontsize=15)
# 题目
ax.set_title("2017年水果销售额汇总", fontsize=20)
# 刻度
ax.set_xticks()
ax.set_yticks()
# 刻度标签
ax.set_xticklabels()
ax.set_yticklabels()
# 图例
ax.plot(arr1, "ro-",  label="苹果")
ax.legend(loc="best")  # 在最合适的位置显示图例

# 实际使用时一次性设置
ax.set(xlim=(0, 10), ylim=(-2, 2), xlabel='x', ylable='sin(x)', title='A sample plot')
```

### 图例

可以使用`plt.legend()`创建最简单的图例，会自动创建一个包含每个图形元素的图例

```python
x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='cosine')
ax.axis('equal')
# 默认图例:位置右上，带边框，1列
leg = ax.legend()
# 设置图例位置并取消外边框
ax.legend(loc='upper left', frameon=False)
# 设置标签列数
ax.legend(loc='lower center', frameon=False, ncol=2)
# 定义圆角边框、增加阴影、改变外边框透明度、文字间距
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
```

图例参数-loc

| 字符串      | 数值 | 字符串       | 数值 |
| ----------- | ---- | ------------ | ---- |
| upper right | 1    | center left  | 6    |
| upper left  | 2    | center right | 7    |
| Lower left  | 3    | lower center | 8    |
| Lower right | 4    | upper center | 9    |
| right       | 5    | Center       | 10   |

- 选择图例显示的元素

图例会默认显示所有元素的标签。若不想显示全部，有如下方法：

方法一：将需要显示的线条传入`plt.legend()`

```python
x = np.linspace(0, 10, 1000)
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
lines = plt.plot(x, y)  # 一组plt.line2D实例
plt.legend(lines[:2], ['first', 'second'])
```

方法二：只为需要在图例中显示的线条设置标签

```python
plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2])
plt.legend(framealpha=1, frameon=True)
```

- 在图例中显示不同尺寸的点

用不同尺寸的点来表示数据的特征，通过这样的图例来反映这些特征

```python
cities = pd.read_csv('./data/california_cities.csv')
print(cities.head(5))
#    Unnamed: 0         city  ...  area_water_km2  area_water_percent
# 0           0     Adelanto  ...           0.046                0.03
# 1           1  AgouraHills  ...           0.076                0.37
# 2           2      Alameda  ...          31.983               53.79
# 3           3       Albany  ...           9.524               67.28
# 4           4     Alhambra  ...           0.003                0.01

# 提取感兴趣的数据
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']

# 用不同尺寸和颜色
plt.scatter(lon, lat, label=None, c=np.log10(population), cmap='viridis', s=area, linewidths=0, alpha=0.5)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)

# 创建一个图例
# 画一些带标签和尺寸的空列表
for area in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.3, s=area, label=str(area) + 'km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City area')
plt.title('California Cities: Area and Population')
plt.show()
```

- 同时显示多个图例

```python
fig, ax = plt.subplots()
lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)

for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2), styles[i], color='black')

# 设置第一个图例要显示的线条和标签
ax.legend(lines[:2], ['line A', 'line B'], loc='upper right', frameon=False)

# 创建第二个图例，通过add_artist方法添加到图上
from matplotlib.legend import Legend

leg = Legend(ax, lines[2:], ['line C', 'line D'], loc='lower right', frameon=False)
ax.add_artist(leg)

plt.show()
```

### 颜色条

可以通过`plt.colorbar()`创建颜色条

```python
x = np.linspace(0, 10 ,1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])
plt.imshow(I)
plt.colorbar()
plt.clim(-1, 1)
plt.show()
```

- 选择配色方案

```python
plt.imshow(I, cmap='gray')
# 在IPython中使用plt.cm<TAB>查找可用配色方案
```

查看配色与黑白灰度对比

```python
def grayscale_cmap(cmap):
    """为配色方案显示灰度图"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # 将RGBA色转换为不同亮度的灰度值
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
    return LinearSegmentedColormap.from_list(cmap.name + '_gray', colors, cmap.N)


def view_colormap(cmap):
    """用等价的灰度图表示配色方案"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2), subplot_kw=dict(xticks=[], yticks=[]))

    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])


view_colormap('jet')
view_colormap('viridis')
view_colormap('cubehelix')
view_colormap('RdBu')
plt.show()
```

- 颜色条刻度

可以将颜色条本身看做一个`plt.Axes` 实例。

```python
x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])
# 为图形像素设置1%噪点
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))

plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')  # 缩短颜色取值的上下限，对于超出上下限的数据，通过extend参数用三角箭头表示比上限大的数或比下限小的数
plt.clim(-1, 1)
plt.show()
```

- 离散型颜色条

颜色条默认是连续的，若是需要离散的，则使用`plt.cm.get_cmap()`

```python
plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
plt.colorbar()
plt.clim(-1, 1)

plt.show()
```

手写数字可视化案例

```python
from sklearn.datasets import load_digits
from sklearn.manifold import Isomap

# 加载0～5的图形
digits = load_digits(n_class=6)
# 展示0～5的图形
# fig, ax = plt.subplots(8, 8, figsize=(6, 6))
# for i, axi in enumerate(ax.flat):
#     axi.imshow(digits.images[i], cmap='binary')
#     axi.set(xticks=[], yticks=[])

# 用IsoMap方法将数字投影到二维空间
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)

# 画图
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
            c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks=range(6), label='digit value')

plt.clim(-0.5, 5.5)

plt.show()

# 由图可知5和3有大面积重叠，说明较难区分
```

### 文字注释

原始数据图

```python
import datetime

births = pd.read_csv('./data/births.csv')

quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

births['day'] = births['day'].astype(int)

births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
births_by_date.index = [datetime.datetime(2012, month, day)
                        for (month, day) in births_by_date.index]

fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

plt.show()
```

- 坐标点文字注释

在特定位置上添加文字注释，使用`ax.text()`

```python
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# 图上增加文字标签
style = dict(size=10, color='gray')

# ha是水平对齐方式
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)

# 设置坐标轴标题
ax.set(title='USA births by day of year (1969-1988)',
       ylabel='average daily births')

# 设置x轴刻度值，让月份居中显示
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));

plt.show()
```

- 坐标转换与文字位置

有时需要将文字放在与数据无关的位置，可通过坐标转换来实现

```python
ax.transData	  # 以数据为基准的坐标变换
ax.transAxes	  # 以坐标轴为基准的坐标变换
fig.transFigure	  #	以图形为基准的坐标变换
```

示例

```python
fig, ax = plt.subplots(facecolor='lightgray')
ax.axis([0, 10, 0, 10])

# 以三种变换方式将文字画在不同的位置
# transData是默认转换，坐标用x轴和y轴的标签作为数据坐标
ax.text(1, 5, ". Data: (1, 5)", transform=ax.transData)
# transAxes坐标以坐标轴(白色矩形)左下角的位置为原点，按坐标轴尺寸的比例呈现坐标
ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes)
# transFigure坐标以图形(灰色矩形)左下角的位置为原点，按图形尺寸的比例呈现坐标
ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure);

# 若改变了坐标轴上下线，图中位置只有transData坐标会受影响
ax.set_xlim(0, 2)
ax.set_ylim(-6, 6)

plt.show()
```

- 箭头注释

`plt.arrow()`可以画箭头，但是其箭头是SVG向量图对象，会随着图形分辨率而改变。

`plt.annotate()` 既可以创建文字，也可以创建箭头，其箭头可以灵活配置。

```python
fig, ax = plt.subplots()

x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"));

plt.show()
```

对原始数据增加箭头

```python
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)

# 图上添加箭头标签
ax.annotate("New Year's Day", xy=('2012-1-1', 4100),  xycoords='data',
            xytext=(50, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-0.2"))

ax.annotate("Independence Day", xy=('2012-7-4', 4250),  xycoords='data',
            bbox=dict(boxstyle="round", fc="none", ec="gray"),
            xytext=(10, -40), textcoords='offset points', ha='center',
            arrowprops=dict(arrowstyle="->"))

ax.annotate('Labor Day', xy=('2012-9-4', 4850), xycoords='data', ha='center',
            xytext=(0, -20), textcoords='offset points')
ax.annotate('', xy=('2012-9-1', 4850), xytext=('2012-9-7', 4850),
            xycoords='data', textcoords='data',
            arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })

ax.annotate('Halloween', xy=('2012-10-31', 4600),  xycoords='data',
            xytext=(-80, -40), textcoords='offset points',
            arrowprops=dict(arrowstyle="fancy",
                            fc="0.6", ec="none",
                            connectionstyle="angle3,angleA=0,angleB=-90"))

ax.annotate('Thanksgiving', xy=('2012-11-25', 4500),  xycoords='data',
            xytext=(-120, -60), textcoords='offset points',
            bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=80,rad=20"))


ax.annotate('Christmas', xy=('2012-12-25', 3850),  xycoords='data',
             xytext=(-30, 0), textcoords='offset points',
             size=13, ha='right', va="center",
             bbox=dict(boxstyle="round", alpha=0.1),
             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1));

# 设置坐标标题
ax.set(title='USA births by day of year (1969-1988)',
       ylabel='average daily births')

# 设置x轴刻度值，让月份居中显示
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));

ax.set_ylim(3600, 5400);

plt.show()
```

### 坐标轴刻度

- 主要/次要刻度

每一个坐标轴都有主要刻度线与次要刻度线。

```python
ax = plt.axes(xscale='log', yscale='log')
plt.show()
```

每个主要刻度都显示为一个较大的刻度线和标签，次要刻度都显示为一个较小的刻度线，且不显示标签

可以设置每个坐标轴的`formatter`和`locator`对象，自定义这些刻度属性(包括刻度线的位置和标签)

- 隐藏刻度与标签

可以通过`plt.NullLocator(),plt.NullFormatter()`实现

```python
ax = plt.axes()
ax.plot(np.random.rand(50))
# 移除了y轴的刻度(标签一并移除了)
ax.yaxis.set_major_locator(plt.NullLocator()) 
# 移除了x轴的标签
ax.xaxis.set_major_formatter(plt.NullFormatter()) 

plt.show()
```

- 增减刻度数量

可以用`plt.MaxNLocator()`增减刻度数量

```python
fig, ax = plt.subplots(4, 4, sharex=True, sharey=True)

# 为每个坐标轴设置主要刻度定位器
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(2))
    axi.yaxis.set_major_locator(plt.MaxNLocator(2))

plt.show()
```

- 花哨的刻度格式

```python
# 画正弦曲线和余弦曲线
fig, ax = plt.subplots()
x = np.linspace(0, 3 * np.pi, 1000)
ax.plot(x, np.sin(x), lw=3, label='Sine')
ax.plot(x, np.cos(x), lw=3, label='Cosine')
# 设置网格、图例和坐标轴上下限
ax.grid(True)
ax.legend(frameon=False)
ax.set_xlim(0, 3 * np.pi)


# 将刻度和网格限画在pi的倍数上
# ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))  # 主要刻度
# ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi/4))  # 次要刻度

# 不太直观，直接显示与pi关系
def format_func(value, tick_number):
    # 找到pi/2的倍数刻度
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return '0'
    elif N == 1:
        return r'$\pi/2$'
    elif N == 2:
        return r'$\pi$'
    elif N % 2 > 0:
        return r'${0}\pi/2$'.format(N)
    else:
        return r'${0}\pi$'.format(N // 2)


ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

plt.show()
```

定位器类

```python
NullLocator		# 无刻度
FixedLocator	# 刻度位置固定
IndexLocator	# 用索引作为定位器
LinearLocator	# 从min到max均匀分布刻度
LogLocator		# 从min到max按对数分布刻度
MultipleLocator	# 刻度和范围都是基数(base)的倍数
MaxNLocator		# 为最大刻度找到最优位置
AutoLocator		# 默认以MaxNLocator进行简单配置
AutoMinorLocator	# 次要刻度的定位器
```

格式生成器类

```python
NullFormatter	# 刻度上无标签
IndexFormatter	# 将一组标签设置为字符串
FixedFormatter	# 手动为刻度设置标签
FuncFormatter	# 用自定义函数设置标签
FormatStrFormatter	# 为每个刻度值设置字符串格式
ScalarFormatter	# 默认为标量值设置标签
LogFormatter	# 对数坐标轴的默认格式生成器
```


### 颜色标记线型

改变线的属性方法

```python
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

## 配置修改

### 手动配置

手动在图中一个个修改

```python
plt.style.use('classic')
x = np.random.randn(1000)
plt.hist(x)  # 默认配置

# 使用灰色背景
ax = plt.axes(facecolor='#E6E6E6')
ax.set_axisbelow(True)
# 画上白色的网格线
plt.grid(color='w', linestyle='solid')
# 隐藏坐标轴的线条
for spine in ax.spines.values():
    spine.set_visible(False)
# 隐藏上边与右边的刻度
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
# 弱化刻度与标签
ax.tick_params(colors='gray', direction='out')
for tick in ax.get_xticklabels():
    tick.set_color('gray')
for tick in ax.get_yticklabels():
    tick.set_color('gray')
# 设置频次直方图轮廓色与填充色
ax.hist(x, edgecolor='#E6E6E6', color='#EE6666')

plt.show()
```

使用`rc`配置

```python
plt.style.use('classic')
x = np.random.randn(1000)

# 复制rcParams字典，修改之后可还原回来
default = plt.rcParams.copy()
from matplotlib import cycler

colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', 'FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='E6E6E6')
plt.rc('lines', linewidth=2)

plt.hist(x)
plt.show()
```

### 默认配置

- 代码中

```python
# 方法一：使用参数字典rcParams
mp1.rcParams['lines.linewidth'] = 2
mp1.rcParams['lines.color'] = 'r'

# 方法二：调用matplotlib.rc()
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
