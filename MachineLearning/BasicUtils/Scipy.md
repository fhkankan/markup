# Scipy

Scipy ：基于Numpy提供了一个在Python中做科学计算的工具集，专为科学和工程设计的Python工具包。主要应用于统计优化、线性代数模块、傅里叶变换、信号和图像处理、常微分方程求解、积分方程、稀疏矩阵等，在数学系或者工程系相对用的多一些，和数据处理的关系不大。

- 在NumPy库的基础上增加了众多的数学、科学及工程常用的库函数
- 线性代数、常微分方程求解、信号处理、图像处理
- 一般的数据处理numpy已经够用

安装

```
pip install scipy
```

引用

```
import scipy as sp
```

>主要模块

```python
scipy.cluster			# 矢量量化/K-均值
scipy.constants			# 物理和数学常数
scipy.fftpack			# 傅里叶变换
scipy.integrate			# 积分
scipy.interpolate		# 插值
scipy.io				# 数据输入输出
scipy.linalg			# 线性代数程序
scipy.ndimage			# n维图像包
scipy.odr				# 正交距离回归
scipy.optimize			# 优化
scipy.signal			# 信号处理
scipy.sparse			# 稀疏矩阵
scipy.spatial			# 空间数据结构和算法
scipy.special			# 特殊数学函数
scipy.stats				# 统计
```

## 数字与物理常数

```python
import scipy.constants as C

print(C.pi)
```

常用常数、物理、换算常数

```python
# 数学
golden	 # 黄金分割比

# 物理
c					# 光速
epsilon_0	# 真空电容率
h					# 普朗克常数
e					# 基本电荷
R					# 普适气体常数
alpha			# 精细结构常数
N_A				# 阿伏伽德罗常数
k					# 玻尔兹曼常数
Rydberg		# 里德伯常数
m_e				# 电子静止质量
m_p				# 质子静止质量
m_n				# 中子静止质量
atm				# 标准气压（帕斯卡）

# 换算
pound			# 英镑/千克
ounce			# 盎司/千克
degree		# 角度/弧度
inch			# 英寸/米
foot			# 英尺/米
yard			# 码/米
mile			# 英里/米
acre			# 英亩/平方米
gallon		# 加仑/立方米
hp				# 马力/瓦特
```

## 特殊函数库

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spl

if __name__ == "__main__":
    x = np.linspace(0, 20, 500)
    for i in range(3):
        y = spl.jv(i, x)
        plt.plot(x, y, '-', label="J%d"%i)
    plt.show()
```

常用函数库

```python
airy()			# 艾里函数
airye()
ai_zeros()
bi_zeros()
bi_zeros()
itairy()
jv()				# 贝塞尔函数
```

## 积分

定积分

```python
from scipy import integrate
y = lambda x: x**2 + 3  # 定义被积分的函数
res = integrate.quad(y, -2, 4)  # 计算积分结果，参数1被积分函数，参数2下界，参数3上界
print(res)
"""
(42.0, 4.662936703425657e-13)  # 返回值，参数1定积分计算结果，参数2对计算误差的估计
"""

integrate.quad			# 一重积分
integrate.dblquad		# 二重积分
integrate.tplquad		# 三重积分
```

数值采样计算函数，数值积分是一种对定积分的近似计算方法，输入不是一个函数，而是该函数的若干采样点

```python
import numpy as np
from scipy import integrate
y = lambda x: x**2 + 3  # 定义函数，用于采样
x = np.linspace(-2, 4, 10)  # 在[-2, 4]内的10个采样点
y = y(x)  # 计算采样点的y值
res = integrate.trapz(y, x)  # 计算数值积分
print(res)  # 42.4444444444

# 数值计算的函数有：trapz(),simps(),romb()等
```

## 优化

`scipy.optimize`主要实现了三类功能：求函数最小值点、线性和非线性拟合、方程组求解。

求函数最小值点

```python
import numpy as np
import scipy.optimize as opt

# 定义被求最小值函数
func = lambda x: x[0]**3 + x[1]**3 + np.cos(x[2] + 1) 
# 猜测的结果
x0 = np.array([0, 0, 0])
# 调用函数求最小值，参数1函数对象，参数2初始猜测值
res = opt.minimize(func, x0)
# 用res.x读取结果
print("y={} when x={}".format(func(res.x), res.x))
"""
y=-0.999999999989 when x=[ 0.          0.          2.14159739]
"""
```

拟合

```python
import numpy as np
from scipy.optimize import curve_fit

# 定义函数y= a*e^(-bx)+c
def func(x ,a, b, c):
    res = a * np.exp(-b * x) + c
    return res

# 生成带噪声的一组拟合数据
xdata = np.linspace(0, 4, 50)  # x轴数据
y = func(xdata, 2.5, 1.3, 0.5)  # y轴数据
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)  # 生成噪声
ydata = y + y_noise  # 在y轴上加入噪声
popt, pcov = curve_fit(func, xdata, ydata)  # 拟合数据，得到a,b,c参数
print(popt)
"""
[ 2.55423706  1.35190947  0.47450618]
"""
```

方程组求解

```python
import numpy as np
import scipy.optimize as opt

# 定义方程组
def fun(x):
    res =  [
        x[0]**2 + x[1]**2 - x[2]/3 - 3,
        x[0]**2 + x[1]/5 - x[2] + 1,
        x[0] + x[1] + x[2] - 7
    ]
    return res

# 用root函数求解，参数1是函数对象，参数2是初始猜测
sol = opt.root(fun, [0, 0, 0])  
print(sol.x)
"""
[ 1.68344818  1.23546169  4.08109013]
"""
```

## 插值

是一种根据已有数据生成新数据的方法，新数据服从于已有数据相同的算法或分布。

与拟合的区别：1.拟合允许有噪声的存在，而插值在计算时认为已有数据时绝对正确的；2.使用拟合需要先知道函数/分布形式，而插值只需要有样本数据即可。

```python
import numpy as np
from scipy import interpolate
import scipy.constants as C
import matplotlib.pyplot as plt

x = np.linspace(0, C.pi*2, num=10, endpoint=True)
y = np.sin(x)

# 一维插值
interp_line = interpolate.interp1d(x, y)
interp_cubic = interpolate.interp1d(x, y, kind='cubic')

xnew = np.linspace(0, C.pi*2, num=33, endpoint=True)
ynew_line = interp_line(xnew)
ynew_cubic = interp_cubic(xnew)

plt.plot(x, y, 'o', xnew, ynew_line, '-', xnew, ynew_cubic, '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()


# 一维插值函数
interp1d(),BarycentricInterpolator(),PchipInterpolator()		
# 多维插值函数
interpn(),Rbf(),NearestNDInterpolator(),RegularGridInterpolator()
# 样条插值函数
BSpline(),BivariateSpline(),splrep()
```

## 离散傅立叶

傅立叶变换时一种可以将任何连续信号转换为无限多个周期性函数相加形式信号的转换方法。

```python
import numpy as np
from scipy.fftpack import fft, ifft

x = np.array([2.0, 3.0, -1.0, -3.0, 0.5])  # 原始信号

# 快速傅立叶变换
y = fft(x)
# 反傅立叶变换
yinv = ifft(y)

print(y, yinv)
"""
[ 1.50000000+0.j         6.31762746-3.5532118j -2.06762746+0.4326499j
 -2.06762746-0.4326499j  6.31762746+3.5532118j]
 
[ 2.0+0.j  3.0+0.j -1.0+0.j -3.0+0.j  0.5+0.j
"""
```

## 卷积

`scipy.signal`包含了卷积计算、B样条变换、小波变换、光谱分析等。

```python
import numpy as np
from scipy import signal

x = np.array([1.0, 2.5, 3.0, 2.0])  # 被卷积信号
h = np.array([0.7, 1.3])  # 卷积信号

res = signal.convolve(x, h)  # 卷积结果
print(res)
"""
[ 0.7   3.05  5.35  5.3   2.6 
"""


# 一维以上限号上进行计算
x = np.array([[1.0, 2.5, 3.0, 3.0], [2.0, -0.3, 9.1, 5.8], [3.7, 2.5, 2.0, 4.2]])
h = np.array([[0.7, 1.3], [4.7, 5.0]])
res = signal.convolve(x, h)
print(res)
"""
[[  0.7    3.05   5.35   6.     3.9 ]
 [  6.1   19.14  32.58  44.99  22.54]
 [ 11.99  15.15  45.92  78.3   34.46]
 [ 17.39  30.25  21.9   29.74  21.  ]
"""
```

## 线性分析

对于向量、矩阵等线性空间使用Numpy数组表达，对于在线性空间上最简单的加减乘除等运算直接使用相应运算符即可；而对于更复杂的运算，可以使用`scipy.linalg`中提供的功能，包括矩阵转置、求特征值、各种分解计算（奇异值分解、QR分解、QZ分解、矩阵三角化）等

基本运算

```python
import numpy as np
from scipy import linalg

a = np.array([[1, 2], [3, 4]])
b = np.array([[-1, 0], [1, -2]])

c = a * b  # 矩阵乘
d = a - b  # 矩阵减
e = 3 * a  # 标量与矩阵乘

f = linalg.inv(a)  # 转置

print(c, d, e, f)
```

特征值与特征向量

```python
import numpy as np
from scipy import linalg

A = np.array([[1, 2], [3, 4]])  # 定义矩阵
la ,v = linalg.eig(A)  # eig()返回特征值和特征向量
print(la, v)
"""
(
	array([-0.37228132+0.j,  5.37228132+0.j]),
	array([[-0.82456484, -0.41597356],[ 0.56576746, -0.90937671]])
)
"""
```

奇异值分解

```python
import numpy as np
from scipy import linalg

A = np.array([[1,2,3], [-1, -2, -3]])  # 原矩阵
m, n = A.shape
U, s, Vh = linalg.svd(A)  # 奇异值分解
Sig = linalg.diagsvd(s, m, n)  # 生成奇异值对脚阵
print(U, Sig, Vh)  # 左正交基矩阵、奇异值矩阵、右正交基矩阵的共轭转置
"""
(
	array([[-0.70710678,  0.70710678],[ 0.70710678,  0.70710678]]),
	array([[  5.29150262e+00,   0.00000000e+00,   0.00000000e+00],
       [  0.00000000e+00,   5.78711299e-16,   0.00000000e+00]]),
	array([[-0.26726124, -0.53452248, -0.80178373],
       [-0.94816592, -0.00256504,  0.31776533],
       [ 0.17190932, -0.84515036,  0.50613047]])
)
"""
```

## 概率统计

`scipy.stats`中主要是各种随机分布的样本生成与统计函数。

PDF概率密度函数，用来衡量某个值或某个区间事件发生可能性大小的函数

CDF累积概率分布函数，是从负无穷到某个值的区间内时间发生可能的大小。

常用分布

```python
# 连续分布
beta  	# B分布，定义域范围在[0,1]之间
cosine	# 余弦分布
expon		# 指数分布，用于表示独立随机事件发生的事件间隔
gamma		# 伽马分布
laplace	# 拉普拉斯分布，两个背靠背指数分布组成的尖顶分布
norm		# 正态分布，高斯分布

# 离散分布
bernoulli  	# 伯努利分布，只包括两个值0、1的概率分布
binom  			# 二项分布，n次独立伯努利分布实验后结果为1的次数的分布
logser			# 对数离散分布，满足麦克拉伦数列的对数分布
poisson			# 泊松分布，单位时间内随机事件发生次数的概率分布

# 多元分布
multivariate_normal  	# 多元正态分布
dirichlet							# 狄获克雷分布，是B分布的多元版本
multinomial						# 多项式分布，是二项分布的多元版本
```

分布上的常用方法

```python
rvs()			# 生成随机序列
pdf()			# 计算概率密度函数
cdf()			# 计算累积概率分布函数
sf()			# 残存函数，即1-cdf()
ppf()			# 百分比函数，即cdf()的反函数
isf()			# 反残存函数，即sf()的反函数
stats()		# 获取分布的均值、方差、偏度、峰度
fit()			# 输入样本，拟合计算分布参数
```

代码

```python
import numpy as np
from scipy.stats import gamma

a = 2  # 定义gammma分布的形状参数a
res1 = gamma.rvs(a, size=5)  # 在gamma上生成5个随机值
res2 = gamma.pdf(np.linspace(0, 9, 10), a = a)  # 在区间0～9上计算10个概率密度
res3 = gamma.stats(a, moments='mvsk')  # 计算均值、方差、偏度、峰度
print(res1, res2, res3)
```

核密度估计

```python
# 以上是已知分布类型的情况下进行样本生成或分布计算。而在概率统计中其逆向运算也非常实用，即不知道分布类型的情况下从已有样本数据产生分布的PDF。核密度估计(KDE)就是一种方法
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

x = np.array([0, 1, 1.5, 2, 4, 6.5, 7, 7.9, 8, 9, 10])  # 样本数据

kde = gaussian_kde(x)  # 用样本数据估计分布
print(kde)  # 查看核对象类型
plt.plot(x, np.zeros(x.shape), 'b+', ms=20)  # 绘制样本数据
x_eval = np.linspace(0, 10, num=200)  # 生成评估数据
y_eval = kde(x_eval)  # 用kde计算评估数据的pdf
plt.plot(x_eval, y_eval, '-') # 绘制评估数据
plt.show()
```



