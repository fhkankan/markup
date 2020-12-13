# Matplotlib2

## 动态图

### 动态气泡图

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def tracker(cur_num):
    # 获取当前索引
    cur_index = cur_num % num_points

    # 定义数据点颜色 
    datapoints['color'][:, 3] = 1.0

    # 更新圆圈的大小
    datapoints['size'] += datapoints['growth']

    # 更新集合中最老的数据点的位置 
    datapoints['position'][cur_index] = np.random.uniform(0, 1, 2)
    datapoints['size'][cur_index] = 7
    datapoints['color'][cur_index] = (0, 0, 0, 1)
    datapoints['growth'][cur_index] = np.random.uniform(40, 150)

    # 更新三点图的参数 
    scatter_plot.set_edgecolors(datapoints['color'])
    scatter_plot.set_sizes(datapoints['size'])
    scatter_plot.set_offsets(datapoints['position'])

if __name__=='__main__':
    # 生成一个图像
    fig = plt.figure(figsize=(9, 7), facecolor=(0,0.9,0.9))
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1), ax.set_xticks([])
    ax.set_ylim(0, 1), ax.set_yticks([])

    # 在随机位置创建和初始化数据点，并以随机的增长率进行初始化
    num_points = 20
    datapoints = np.zeros(num_points, dtype=[('position', float, 2),
            ('size', float, 1), ('growth', float, 1), ('color', float, 4)])
    datapoints['position'] = np.random.uniform(0, 1, (num_points, 2))
    datapoints['growth'] = np.random.uniform(40, 150, num_points)

    # 创建一个每一帧都会更新的散点图
    scatter_plot = ax.scatter(datapoints['position'][:, 0], datapoints['position'][:, 1],
                      s=datapoints['size'], lw=0.7, edgecolors=datapoints['color'],
                      facecolors='none')

    # 用tracker函数启动动态模拟
    animation = FuncAnimation(fig, tracker, interval=10)

    plt.show()
```

### 动态信号模拟

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 生成信号数据
def generate_data(length=2500, t=0, step_size=0.05):
    for count in range(length):
        t += step_size
        signal = np.sin(2*np.pi*t)
        damper = np.exp(-t/8.0)
        yield t, signal * damper 

# 初始化函数
def initializer():
    peak_val = 1.0
    buffer_val = 0.1
    ax.set_ylim(-peak_val * (1 + buffer_val), peak_val * (1 + buffer_val))
    ax.set_xlim(0, 10)
    del x_vals[:]
    del y_vals[:]
    line.set_data(x_vals, y_vals)
    return line

def draw(data):
    # 更新数据
    t, signal = data
    x_vals.append(t)
    y_vals.append(signal)
    x_min, x_max = ax.get_xlim()

    if t >= x_max:
        ax.set_xlim(x_min, 2 * x_max)
        ax.figure.canvas.draw()

    line.set_data(x_vals, y_vals)

    return line

if __name__=='__main__':
    # 创建画图
    fig, ax = plt.subplots()
    ax.grid()

    # 提取线
    line, = ax.plot([], [], lw=1.5)

    # 创建变量
    x_vals, y_vals = [], []

    # 定义动画器对象
    animator = animation.FuncAnimation(fig, draw, generate_data, 
            blit=False, interval=10, repeat=False, init_func=initializer)

    plt.show()


```

## 三维图

导入`mplot3d`工具箱可以画三维图

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')
```

### 要素函数

- 点线

```python
# 三维线的数据
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
# 三维散点的数据
zdata = 15*np.random.random(100)
xdata = np.cos(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
plt.show()
```

- 等高线

```python
# 三维等高线
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# 调整观察角度和方位角
ax.view_init(60, 35)

plt.show()
```

- 线框曲线

```python
# 线框图
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('wireframe')
# 曲面图
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_title('surface')
# 极坐标网络
r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
r, theta = np.meshgrid(r, theta)
X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = f(X, Y)
ax = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
plt.show()
```

- 曲面三角剖分

```python
# 曲面三角剖分
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)

# 为数据点创建一个散点图
# ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
# 使用三角形创建曲面
ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
plt.show()
```

### 案例

莫比乌斯带

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')

# 莫比乌斯带是一条二维带，需要两个内在维度
theta = np.linspace(0, 2 * np.pi, 30)
w = np.linspace(-0.25, 0.25, 8)
w, theta = np.meshgrid(w, theta)
# 环的一半扭转180度
phi = 0.5 * theta
# 将坐标转换成三维直线坐标
r = 1 + w * np.cos(phi)  # x-y平面内的半径
x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(r * np.sin(phi))
# 用基本参数化方法定义三角部分
from matplotlib.tri import Triangulation

tri = Triangulation(np.ravel(w), np.ravel(theta))
ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap='viridis', linewidths=0.2)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

plt.show()
```

## 地图

matplotlib中地图可视化包为Basemap。实际使用中，可考虑`leaflet`开发库、谷歌地图、百度地图、高德地图等API。

### Basemap

安装

```python
conda install basemap
```

引用

```python
from mpl_toolkits.basemap import Basemap
```

使用

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap

# 地球
# plt.figure(figsize=(8, 8))
# m = Basemap(projection='ortho', resolution=None, lat_0=40, lon_0=115)
# m.bluemarble(scale=0.5)
# plt.show()

# 地图投影放大到亚洲，定位北京位置
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None, width=8E6, height=8E6, lat_0=40, lon_0=115)
m.etopo(scale=0.5, alpha=0.5)
# 地图上的经纬度对应图上xy坐标
x, y = m(116, 40)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, 'Beijing', fontsize=12)
plt.show()

```

### 地图投影

Basemap程序包里面实现了几十种投影类型。

```python
from itertools import chain


def draw_map(m, scale=0.2):
    # 画地貌晕染图
    m.shadedrelief(scale=scale)
    # 用字典表示经纬度
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))
    # 字典的键是plt.line2D示例
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    # 用循环将所有线设置城需要的样式
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


# 圆柱投影
# fig = plt.figure(figsize=(8, 6), edgecolor='w')
# m = Basemap(projection='cyl', resolution=None, llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
# # cyl:等距圆柱投影，merc:墨卡托投影，cea:圆柱等积投影
# draw_map(m)
# plt.show()

# 伪圆柱投影
# fig = plt.figure(figsize=(8, 6), edgecolor='w')
# m = Basemap(projection='moll', resolution=None, lat_0=0, lon_0=0)
# # moll:摩尔威德投影，sinu:正弦投影，robin:罗宾森投影 
# draw_map(m)
# plt.show()

# 透视投影
# fig = plt.figure(figsize=(8, 8))
# m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=0)
# # ortho:正射投影，gnom:球心投影，stere:球极投影
# draw_map(m)
# plt.show()

# 圆锥投影
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution=None, lat_0=50, lat_1=45, lat_2=55, lon_0=0, width=1.6E7, height=1.2E7)
# lcc:兰勃特等角圆锥投影，eqdc:等距圆锥投影，aea:阿尔伯斯等积圆锥投影
draw_map(m)
plt.show()
```

### 地图背景

常用画图函数

```python
# 物理边界与水体
drawcoastlines()	# 绘制大陆海岸线
drawlsmask()		# 为陆地和海洋设置填充色，从而在陆地或海洋之投影其他图像
drawmapboundary()	# 绘制地图边界，包括为海洋填充颜色
drawrivers()		# 绘制河流
fillcontinents()	# 用一种颜色填充大陆，用另一种功能颜色填充湖泊
# 政治边界
drawcountries()		# 绘制国界线
drawstates()		# 绘制美国州界线
drawcounties()		# 绘制美国县界线
# 地图功能
drawgreatcircle()	# 在两点之间绘制一个大圆
drawparallels()		# 绘制维线
drawmeridians()		# 绘制经线
drawmapscale()		# 在地图上绘制一个线性比例尺
# 地球影像
bluemarble()		# 绘制NASA蓝色弹珠地球投影
shadedrelief()		# 在地图上绘制地貌晕染图
etopo()				# 在地图上绘制地形晕染图
warpimage()			# 将用户提供的图像投影到地图上
```

示例

```python
# 画地图背景
fig, ax = plt.subplots(1, 2, figsize=(12, 8))

for i, res in enumerate(['l', 'h']):
    m = Basemap(projection='gnom', lat_0=57.3, lon_0=-6.2, width=90000, height=120000, resolution=res, ax=ax[i])
    # 分辨率：c原始分辨率，l低分辨率，i中等分辨率，h高分辨率，f全画质分辨率，None表示不使用边界线
    m.fillcontinents(color='#FFDDCC', lake_color='#DDEEFF')
    m.drawmapboundary(fill_color='#DDEEFF')
    m.drawcoastlines()
    ax[i].set_title("resolution='{0}'".format(res))
plt.show()
```

### 地图数据

使用任意的`plt`函数就可以在地图上画出简单的图形和文字，可以将维度和经度坐标投影为直角坐标系。`Basemap`实例中的许多方法都是与地图相关的函数。这些函数与标准Matplotlib函数用法类似，只是都多了一个布尔参数`latlon`，若是将它设置为`True`，就表示使用原来的经度纬度表示，而不是投影为`(x,y)`坐标。

部分与地图有关的方法

```python
contour()/contourf()		# 绘制等高线/填充等高线
imshow()					# 绘制一个图像
pcolor()/pcolormesh()		# 绘制带规则/不规则网格的伪彩图
plot()						# 绘制线条和/或标签
scatter()				    # 绘制带标签的点
quiver()					# 绘制箭头
barbs()						# 绘制风羽
drawgreatecircle()			# 绘制大圆圈
```

### 案例

美国加州城市数据

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd

cities = pd.read_csv('data/california_cities.csv')

# 提取需要的数据
lat = cities['latd'].values
lon = cities['longd'].values
population = cities['population_total'].values
area = cities['area_total_km2'].values
# 绘制地图投影，绘制数据散点，并创建颜色条与图例
# 1.绘制地图背景
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='h', lat_0=37.5, lon_0=-119, width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
# 2.绘制城市数据散点，用颜色表示人口数据
m.scatter(lon, lat, latlon=True, c=np.log10(population), s=area, cmap='Reds', alpha=0.5)
# 3.创建颜色条与图例
plt.colorbar(label=r'$\log_{10}({\rm population})$')
plt.clim(3, 7)
# 用虚拟点绘制图例
for a in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.5, s=a, label=str(a) + 'km$^2$')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower left')
plt.show()

```



