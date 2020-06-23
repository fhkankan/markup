# Matplotlib2

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

### 

