[TOC]

# Numpy

[官网](https://numpy.org/)

Numpy：提供了一个在Python中做科学计算的基础库，重在数值计算，主要用于多维数组（矩阵）处理的库。用来存储和处理大型矩阵，比Python自身的嵌套列表结构要高效的多。本身是由C语言开发，是个很基础的扩展，Python其余的科学计算扩展大部分都是以此为基础。

- 高性能科学计算和数据分析的基础包
- ndarray对象，表示多维数组（矩阵），具有矢量运算能力
- 矩阵运算，无需循环，可完成类似Matlab中的矢量运算
- 线性代数、随机数生成

安装

```
pip install numpy
```

版本

```
import numpy
numpy.__version__
```

引用

```
import numpy as np
```

> 与普通库的区别

可以不用循环便可对数据执行矢量化运算，大小相等数组之间的任何算术运算都会应用到元素，数组与标量之间的运算也会“广播”到数组的各个元素，并具有执行速度快和节省空间的特点。

```python
# 示例1
def python_multi(n):
    a = range(n)
    b = range(n)
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i]*b[i])
    return c

def numpy_mult(n):
    c = np.arrange(n)**2*np.arrange(n)**3
    return c

# 测试执行时间
%timeit python_multi(10000)  # 常规实现
%timeit numpy_multi(10000)  # 向量化运算

# 示例2
a = np.random.rand(10000)
b = np.random.rand(10000)
c = 0
for i inrange(10000):  # 常规实现
    c += a[i] * b[i]
  
c = np.dot(a, b)  # 向量化运算
```

帮助

```python
np.random?

help(np.random)
```

## ndarry

NumPy数组是一个多维的数组对象（矩阵），称为ndarray。数组的元素一般是同质的，但可以有异质数组元素存在(即结构数组)
注意：ndarray的下标从0开始

### 随机抽样创建

| 函数                                  | 参数                                       | 说明                                                         |
| ------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| `np.random.rand(3,4)`                 | 行数，列数                                 | 生成指定维度的随机多维浮点型数组，数据固定区间[0.0,1.0)      |
| `np.random.randn(3,4)`                | 行数，列数                                 | 生成指定维度的随机多维浮点型数组，服从标准正态分布，数据固定区间[0.0,1.0) |
| `np.random.randint(-1,5，size=(3,4))` | [起始值，结束值)，行数，列数               | 生成指定维度大小的随机多维整型数组，可以指定数字区间（左闭右开） |
| `np.random.random(size=(3,4))`        | 行数，列数                                 | 生成指定维度的随机多维浮点型数组，数据固定区间 [0.0,1.0)，当size=None时返回一个随机数 |
| `np.random.uniform(-1,5，(3,4))`      | [起始值，结束值)，行数，列数               | 生成指定维度大小的随机多维浮点型数组，可以指定数字区间       |
| `np.random.normal(0,1,size=(3,5))`    | 均值，方差，行数，列数                     | 生成指定均值方差和维度大小的随机多维浮点型数组，所有不指定时随机生成一个均值0方差1的浮点数 |
| `np.random.seed(666)`                 | Any                                        | 设置随机数种子，旧式方式                                     |
| `np.random.RandomState(0)`            | Union[None, int, ndarray, Iterable, float] | 伪随机数生成器，底层调用seed，新式方式                       |
| `np.random.permutation(300)`          | Union[int, ndarray, Iterable, float]       | 随机排列序列，或返回排列范围。若是整数，则返回`np.range(x)`；若是数组，返回的是乱序后的复制体；若是多维数组，只对第一维索引重排 |
| `np.random.shuffle(ndarry)`           | Union[ndarray, Iterable, int, float]       | 对原数组进行随机重排，无返回值。若是多维数组，只对第一维索引重排 |

示例

```python
np.random.randint(4, 8, size=10)
np.random.random((3, 4))
np.random.normal(0,1, (3,4))

# 指定随机种子(只对下行有效)
np.random.seed(666)
np.random.randint(4, 8, size=(3, 5))
np.random.randint(4, 8, size=(3, 5))  # 重新随机

rng = np.random.RandomState(40)
rng.rand(50)
rng.rand(50)  # 重新随机

# 乱序
a = np.random.permutation(5)
a = np.random.permutation(np.arange(9))
a = np.random.permutation(np.arange(9).reshape((3,3)))
b = np.arange(9).reshape((3,3))
np.random.shuffle(b)
```

### 序列创建

numpy中创建数组的函数主要有

```python
array
# 将输入数据(列表、元组、数组或其他序列)转换成ndarray
asarray
# 将输入转换为ndarray，如果输入数据本身是ndarray就不进行复制
arange
# 类似python内置的arange，但返回一个ndarray而不是列表
linspace
# 通过指定初始值、终止值和元素个数创建等差数列一维数组，可以通过endpoint参数指定是否包含终止值，默认为True，即包含终止值
logspace 
# 与linspace类似，创建的数组时等比数列。基数可以通过base参数指定，默认值为10
ones
# 根据指定形状和dtype创建一个数据全部为1的数组
ones_like
# 以另一个数组为参考，并根据其形状和dtype创建一个数组
zeros、zeros_like
# 与ones和ones_like类似，产生数据全部为0的数组
empty、empty_like
# 创建一个只分配内存空间但不填充任何值的数组
eye、identity
# 创建单位阵
frombuffer、fromstring、fromfile
# 可以从字节序列或文件创建数组
fromfunction
# 通过指定的函数创建数组
```

| 函数                                 | 参数                                                         | 说明                                                         |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `np.array(list, dtype)`              | list 为 序列型对象(list)、嵌套序列对象(list of list)，dtype表示数据类型 （int、float、str） | 将序列对象转换为数组                                         |
| `np.arange(star, end, step)`         | 可以指定区间[star, end)，步长可以为浮点数                    | arange() 类似 python 的 range() ，用来创建一个一维 ndarray 数组，结果等同于 np.array(range()) |
| `np.linspace(star, end, n)`          | 指定区间[star, end]，个数n                                   | 在star和end之间等量截取n个数据形成ndarray数组，可以通过endpoint参数指定是否包含终止值，默认true |
| `np.zeros()`                         | 第一个参数是元组，用来指定大小，如(3, 4)，第二个参数可以指定类型，如int，默认为float | 指定大小的全0数组                                            |
| `np.ones()`                          | 第一个参数是元组，用来指定大小，如(3, 4)，第二个参数可以指定类型，如int，默认为float | 指定大小的全1数组                                            |
| `np.full(shape=(a,b), fill_value=c)` | 第一个参数是元组，用来指定大小，如(3, 4)，第二个参数可以指定类型 | 创建所有值均为c的a行b列的数组                                |
| `ndarry.reshape()`                   | 总大小不变，指定新数组的维度大小                             | 重新调整数组的维度                                           |


示例

```python
np.ones((3, 5))
np.ones((2, 3, 4), dtype=np.int32)
np.zeros((3,5))
np.zeros((3, 5), dtype=int)
np.full((3, 5), 12)
np.arange(0, 20, 2)  # 左闭右开
np.arange(0, 1, 0.2)
np.linspace(0, 20, 10)  # 左闭右闭
vector = numpy.array([1,2,3,4])  # 一维
matrix = np.array([[5, 10, 15], [20, 25, 30]])  # 二维
```

## 向量

向量(vector)即一维数组

```python
# arange创建
v = np.arange(10)
vstep = np.arange(0, 10, 0.5)

# linspace创建等差
np.linspace(1, 19, 10)
np.linspace(1, 19, 10, endpoint=False)

# logspace创建等比
from math inport e
np.logspace(1, 20, 10, endpoint=False, base=e)

# zeros
np.zeros(20, np.int)
# empty
np.empty(20, np.int)

# random
np.random.randn(10)  # 正态分布随机数

# fromstring
s = "Hello, python"
np.fromstring(s, dtype=np.int8)  # ascii

# fromfunction
def multiply99(i, j):
    return (i+1)*(j+1)
# 第一个参数为计算每个数组元素的函数名称，第二个参数指定数组的形状，指定的是数组的下标，下标作为实参通过遍历的方式传递给函数的形参
np.fromfunction(multiply99, (9,9))
```

## 数组

数组(ndaary)由实际数据和描述这些数据的元数据组成

### 数据类型

| 类型代码 | 类型          | 说明                                        |
| -------- | ------------- | ------------------------------------------- |
| i1、u1   | int8、uint8   | 有符号和无符号的8位(1个字节长度)整型        |
| i2、u2   | int16、uint16 | 有符号和无符号的16位(2个字节长度)整型       |
| i4、u4   | int32、uint32 | 有符号和无符号的32位(4个字节长度)整型       |
| f2       | float16       | 半精度浮点数                                |
| f4或f    | float32       | 标准单精度浮点数                            |
| f8或d    | float64       | 双精度浮点数                                |
| ?        | bool          | 布尔类型                                    |
| O        | object        | Python对象类型                              |
| <U1或U   | unicode       | 固定长度的unicode类型，跟字符串定义方式一样 |
|          | datetime64    | 时间类型                                    |

```python
# dtype属性
print(arr1.dtype)

# dtype参数
# 指定数组的数据类型，类型名+位数，如float64, int32
float_arr = np.array([1, 2, 3, 4], dtype = np.float64)
date = np.array('2019-04-08', dtype=np.datetime64)

# astype方法
# 转换数组的数据类型
int_arr = float_arr.astype(np.int32)

# 完整的ndarray数据类型查看
set(np.typeDict.values())
```
### 结构化数组

数组数据的类型可以由用户自定义，自定义数据类型是一种异质结构数据类型，通常用于记录一行数据或一系列数据，即结构数组。

结构化数组可以直接使用字段名进行索引和切片。

- 方法一

```python
# 定义字段类型
goodslist = np.dtype([('name', np.str_, 50), ('location', np.str_, 30), ('price', np.float16), ('volume', np.int32)])

# 构造结构数组
goods = np.array([('Gree Airconditioner', 'JD.com', 6245, 1),
                  ('Sony Blueray Player', 'Amazon.com', 3210, 2),
                  ('Apple Mackbook pro 13', 'Tamll.com', 12388, 5),
                  ('iPhoneSE', 'JD.com', 4588, 2)
                 ], dtype=goodslist)
```

- 方法二

```python
# 定义结构数组
goodsdict = np.dtype({'names': ['name', 'location', 'price', 'volume'],
                      'formats': ['S50', 'S30', 'f', 'i']})

goods_new = np.array([('Gree Airconditioner', 'JD.com', 6245, 1),
                      ('Sony Blueray Player', 'Amazon.com', 3210, 2),
                      ('Apple Mackbook pro 13', 'Tamll.com', 12388, 5),
                      ('iPhoneSE', 'JD.com', 4588, 2)
                     ], dtype=goodsdict)
```

### 数组属性

常用的数组属性

| 属性       | 含义                                                         |
| ---------- | ------------------------------------------------------------ |
| `shape`    | 返回数组的形状，如行、列、层等                               |
| `dtype`    | 返回数组中各元素的类型                                       |
| `ndim`     | 返回数组的维数或数组轴的个数                                 |
| `size`     | 返回数组元素的总个数                                         |
| `itemsize` | 返回数组中的元素在内存中所占的字节数                         |
| `nbyte`    | 返回数组所占的存储空间，即itemsize与size的乘积               |
| `T`        | 返回数组的转置数组                                           |
| `flat`     | 返回一个`numpy.flatier`对象，即展平迭代器。可以像遍历一维数组一样去遍历任意多维数组，也可从迭代器中获取制定数组元素。可赋值 |

示例

```python
ac = np.arange(12)
ac.shape = (2, 2, 3)
ac.shape  # (2, 2, 3)
ac.dtype  # dtype('int64)
ac.ndim  # 3
ac.size  # 12
ac.itemsize  # 8
ac.nbytes  # 96
ac.T  # [[[0 6][3 9]][[1 7][4 10]][[2 8][5 11]]]
acf = ac.flat  # <numpy.flatier at xxx>
acf[5:]  # array([5, 6, 7, 8, 9, 10, 11])
acf[[1, 3, 11]] = 100  # [[[0 100 2][100 4 5]]]
ac  # [[6 7 8][9 10 100]]

x = np.arange(10)
X = np.arange(15).reshape(3, 5)
# 维度
x.ndim  # 1
X.ndim  # 2
# shape
x.shape  # (10,)
X.shape  # (3, 5)
# 个数
x.size  # 10
X.size  # 15
```

### 索引与切片

#### 基本索引

一维数组

```python
arr1 = np.arange(5)

# 取某个下标的值
print(arr1[3])
# 取连续的多个值
print(arr1[4:10])
print(arr1[:8])
print(arr1[2:])
# 加步长
print(arr1[::2])
print(arr1[::-1])
print(arr1[1:10:3])
```

二维数组

```python
arr2 = arr1.random.randint(-5,10,(4, 4))

# 下标为2的一行
print(arr2[2])

# 下标为2的行，下标为2的列
print(arr2[2][2])
print(arr2[(2, 2)])
print(arr2[2, 2])  # 推荐

# 取连续多行
print(arr2[1:3])

# 取所有行的指定多列
print(arr2[:, 1:3])

# 取不连续的多行
print(arr2[[1,3], :])
print(arr2[[1,3]])

# 取不连续的多列
print(arr2[:, [1,3]])

# 取不同行不同列的数据
print(arr2[[1,3], [2,3]])

# 取连续行连续列的数据
print(arr2[:2, :3]) 
# arr2[:2][:3]表示arr[:2]中前三行
```

#### 逻辑索引

逻辑索引即布尔型索引，可以通过制定布尔数组或者条件进行索引

```python
# 对数组进行条件判断，返回bool类型的数组
is_after6_arr = arr1 > 6
print(is_after6_arr)

# 将一个相同大小的bool数组映射到另一个数组里，
# 会返回所有为True的结果（一维数组）
print(arr1[arr1 > 6])

# 如果有多个条件， 通过 & | ~
is_arr = (arr1 > 6)&(arr1 < 13)
print(is_arr)

# 条件索引，如果有多个条件， 通过 & | ~
year_arr = np.array([[1020,2011,2012],[2013,2014,2015],[2016,2017,201]])
print(year_arr[(year_arr >= 2012) & (year_arr <= 2016)])
```

示例

```python
vector = np.array([5, 10, 15, 20])
equal_to_ten = (vector == 10)
print(vector[equal_to_ten])

matrix = np.array([
  	[5, 10, 15],
  	[20, 25, 30],
  	[35, 40, 45]
])
second_column_25 = (matrix[:, 1] == 25)
print(matrix[second_column_25, :])  # 25所在的行
```

#### 花式索引

fancy indexing即利用整数数组进行索引，结果的形状与索引数组的形状一致，而不是与被索引数组的形状一致

```python
import numpy as np

# 一维数组
x = np.arange(16)
# 取索引不连续的值
[x[3], x[5], x[8]]  # 常规方法
ind = [3, 5, 8]  # Fancy方法
x[ind]
# 二维索引矩阵
ind = np.array([0, 2], 
               [1, 3])
x[ind]

# 二维数组
X = x.reshape(4, -1)
row = np.array([0, 1, 2])
col = np.array([1, 2, 3])
X[row, col]  # 3个值
X[0, col]
X[:2, col]

# 布尔数组
col = [True, False, True, True]  # 对索引为1的值不关心
X[1:3, col]  # 对第1列不关心
```

### 数据修改复制

不同于python中的list(子列表是对原列表的复制)，在ndarry中修改某个位置的值，子ndarry和父ndarry均会修改。采用的是引用的方式。

引用

```python
import numpy as np

arr1 = np.arange(15).reshape(3,5)
# 数据的修改是对原数据进行修改
subx = arr1[:2, :3]
subx[1, 0] = 30  # 子和父均变化
arr1[0, 0] = 20  # 子和父均变化
```

view

```python
arr2 = arr1.view()
arr2.shape = 5, 3
print(arr1.shape)  # 数组引用不同
arr2[0, 0] = 1234
print(arr1) # 值的引用相同
print(id(arr1), id(arr2))
```

copy

```python
arr3 = arr1.copy()
arr3.shape = 5, 3
print(arr1.shape)  # 数组引用不同
arr3[0, 0] = 1234
print(arr1) # 值的引用不同
print(id(arr1), id(arr3))
```

### 数组维度

数组的维度可以进行变换，如行列互换、降维等。numpy中可以使用reshape函数改变数组的维数，使用ravel函数、flatten函数等把数组展平维一维数组

```python
x = 10 * np.random.rand(100)
x.shape  # (100,)
x.ndim  # 1

y = x[:, None]
y.shape  # (100, 1)
y.ndim  # 2


y = x[:, np.newaxis]
y.shape  # (100, 1)
y.ndim  # 2
```

- 展平

```python
b = np.array(24).reshape(2, 3, 4)
b.ndim  # 3

# ravel
br = np.ravel(b)  # 返回数组的一个视图
br.ndim  # 1

# flatten
bf = b.flatten()  # 返回源数据的副本
bf.ndim  # 1

# reshape
brsh = b.reshape(1, 1, 24)
brsh.ndim  # 3
```

- 维度改变

```python
# reshape
bd = b.reshape(4, 6)  # 返回数组的一个视图，b数组没有变化
bd = b.reshape(4, -1)  # 只关心行数，列数自动计算
bd = b.reshape(-1, 6)  # 只关心列数，行数自动计算

# shape
b.shape(1, 1, 24)  # 修改数组b

# resize
b.resize(1, 1, 24)  # 修改数组b
```

- 转置

```python
# T
b.T

# transpose
np.transpose(b)



二维数组直接使用转换函数：transpose()
高维数组转换要指定维度编号参数 (0, 1, 2, …)，注意参数是元组
transpose() 不会更改维度个数，只会修改维度大小。不同于 reshape()。

# 二维数组
arr1 = np.random.randint(-5, 10, (4, 8))
t_arr1 = arr1.transpose()
print(t_arr1)

# 多维数组
arr2 = np.random.randint(-5, 10, (3,4,5))
t_arr2 = arr2.transpose(1,2,0)
print(t_arr2)
```

### 数组组合

数组组合可分为水平组合(hstack)、垂直组合(vstack)、深度组合(dstack)、列组合(column_stack)、行组合(row_stack)等

```python
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.full((2, 2), 100)

# concatenate, 需要相同维度
arr_con = np.concatenate([x, y])
arr_con = np.concatenate([A, A]) # 默认axis=0,行增加
arr_con = np.concatenate([A, A], axis=1)  # 列增加
arr_con = np.concatenate([A, x.reshape(1, -1)])

# vstack，垂直向堆叠，行增加
arr_con = np.vstack([A, x])

# hstack，水平向堆叠，列增加
arr_con = np.hstack([A, B])

# np.r_，行增加
arr_con = np.r_[x,y]  # 两个shape一致
# np.c_，列增加
arr_con = np.c_[A,B]

# tile, 单个向量堆叠
a = np.tile(x, (2, 1))
```

- 水平组合

把所有参加组合的数组拼接起来，各数组行数应相等

```python
a = np.arange(9).reshape(3, 3)
b = np.array([[0,11,22,33],[44,55,66,77],[88,99,00,11]])

# hstack
np.hstack((a, b))  # 参数只有一个，应把要参加组合的数组对象以元组的形式传参

# concatenate
np.concatenate((a, b), axis=1)  # 实现同样功能
```

- 垂直组合

把所有参加组合的数组追加在一起，各数组列数应一致

```python
c = np.array([[0, 11, 22], [44, 55, 66], [88, 99, 00], [22, 33, 44]])

# vstack
np.vstack((a, c))

# concatenate
np.concatenate((a,c), axis=0)
```

- 深度组合

将参加组合的各数组相同位置的数组组合在一起。要求所有数组维度属性要相同，类似于数组叠加

```python
d = np.delete(b, 3, axis=1)  # 1表示列，0表示行

np.dstack((a, d))
```

- 列组合

```python
# column_stack函数对于一维数组按列方向进行组合
a1 = np.arange(4)
a2 = np.arange(4)*2
np.column_stack((a1, a2))

# 二维数组，column_stack与hstack效果相同
```

- 行组合

```python
# row_stack函数对于一维数组按行方向进行组合
np.row_stack((a1, a2))

# 二维数组，row_stack和vstack效果相同
```

### 数组分拆

数组分拆分为水平分拆(hsplit)、垂直分拆(vsplit)、深度分拆(dsplit)，数组分拆的结果是一个由数组作为元素构成的列表

```python
x = np.arange(10)
A = np.arange(16).reshape((4, 4))

# split
# 第2参数为分割点
x1, x2, x3 = np.split(x, [3, 7])
x1, x2 = np.split(x, [2])
# 第2参数为分割行数
A1, A2 = np.split(A, [2])  # 默认axis=0，行分割
# 第3参数为分割列数
A1, A2 = np.split(A, [2]， axis=1)

# vsplit, 垂直向分割
np.vsplit(A, [2])

# hsplit, 水平向分割
np.hsplit(A, [2])
np.hsplit(A, [-1])  # 分隔特征和最后一列的label
```

- 水平分拆

把数组沿着水平方向进行分拆，数组分拆结果返回时列表，列表中的元素时numpy数组

```python
a = np.arange(9).reshape(3, 3)

# hsplit
ahs = a.hsplit(a, 3)
type(ahs)  # list
type(ahs[1])  # numpy.ndarray

# split
np.split(a, 3, axis=1)
```

- 垂直分拆

```python
# vsplit
np.vsplit(a, 3)

# split
np.split(a, 3, axis=0)
```

- 深度拆分

```python
ads = np.arange(12)
ads.shape = (2, 2, 3)
np.dsplit(ads, 3)  # 按照深度方向拆分3个维度以上的数组
```

### ufunc

ufunc(universal function)是一种能对数组中每个元素进行操作的函数。这些函数可以进行四则运算、比较运算及布尔运算等。numpy内置了许多ufunc函数，也可以使用frompyfunc函数来自定义ufunc函数

正是通用函数的向量化和广播使得numpy计算较快

#### 常规运算

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

print(arr1 + 100)
print(arr1 - 10)
print(arr2 * 2)
print(arr2 / 5)
print(arr2 // 2)
print(arr2 ** 2)
print(arr2 % 2)
print(1/arr2)
print(arr1<4)
```

#### 元素计算函数

```python
# 一元ufunc：
np.ceil(x): 向上最接近的整数，参数是 number 或 ndarray
np.floor(x): 向下最接近的整数，参数是 number 或 ndarray
np.rint(x): 四舍五入，参数是 number 或 ndarray
np.around(x, decimals): 四舍五入，decimals舍入的小数位数。默认为0，如果为负，整数将四舍五入到小数点左侧的位置 
    
np.negative(x): 元素取反，参数是 number 或 ndarray
np.abs(x)：元素的绝对值，参数是 number 或 ndarray
np.square(x)：元素的平方，参数是 number 或 ndarray
np.aqrt(x)：元素的平方根，参数是 number 或 ndarray

np.sign(x)：计算各元素的正负号, 1(正数)、0（零）、-1(负数)，参数是 number 或 ndarray
np.modf(x)：将数组的小数和整数部分以两个独立数组的形式返回，参数是 number 或 ndarray
np.isnan(x): 判断元素是否为 NaN(Not a Number)，返回bool，参数是 number 或 ndarray
    
np.sin(x):计算各元素的正弦
np.cos(x):计算各元素的余弦
np.tan(x):计算各元素的正切
np.degress(x):将弧度转换为角度 
    
np.exp(x)：计算各元素的e^x
np.power(3,x)：计算各元素的3^x
np.log(x):计算loge(x)
np.log2(x)：计算log2(x)
np.log10(x):计算log10(x)

# 二元ufunc：
np.add(x, y): 元素相加，x + y，参数是 number 或 ndarray
np.subtract(x, y): 元素相减，x - y，参数是 number 或 ndarray
np.multiply(x, y): 元素相乘，x * y，参数是 number 或 ndarray
np.divide(x, y): 元素相除，x / y，参数是 number 或 ndarray
np.floor_divide(x, y): 元素相除取整数商(丢弃余数)，x // y，参数是 number 或 ndarray
np.mod(x, y): 元素求余数，x % y，参数是 number 或 array
np.power(x, y): 元素求次方，x ** y，参数是 number 或 array

# 三元ufunc：
np.where(condition, x, y): 三元运算符，x if condition else y，条件满足返回x，否则返回y，参数condition 是条件语句，参数 x 和 y 是 number 或 ndarray
```

#### 数据统计函数

多维数组默认统计全部数据，添加axis参数可以按指定轴心统计，值为0则行运算，列标不变，值为1则列运算，行标

```python
# 方式一：
np.mean(x [, axis])：所有元素的平均值，参数是 number 或 ndarray
np.sum(x [, axis])：所有元素的和，参数是 number 或 ndarray 比python的sum效率高
np.percentile(x, q=50): 对百分位点取值
np.median(x [, axis]): 所有元素中位数
np.max(x [, axis])：所有元素的最大值，参数是 number 或 ndarray
np.min(x [, axis])：所有元素的最小值，参数是 number 或 ndarray
np.std(x [, axis])：所有元素的标准差，参数是 number 或 ndarray
np.var(x [, axis])：所有元素的方差，参数是 number 或 ndarray

# 方式二
arr1.mean()
arr1.sum()
arr1.max()
arr1.min()
arr1.std()
arr1.var()

# 索引
np.argmax(x [, axis])：最大值的下标索引值，参数是 number 或 ndarray
np.argmin(x [, axis])：最小值的下标索引值，参数是 number 或 ndarray
np.argwhere(condition):符合指定条件的元素的下标（二维数组）

# 累加乘
np.cumsum(x [, axis])：返回一个一维数组，每个元素都是之前所有元素的 累加和，参数是 number 或 ndarray
np.cumprod(x [, axis])：返回一个一维数组，每个元素都是之前所有元素的 累乘积，参数是 number 或 ndarray
```

#### 数组增删合并

```python
append()：在数组后面追加元素,总是返回一维数组
参数1表示需处理的数组，参数2表示新增的数据

insert()：在指定下标插入元素,若不指定轴方向，做一维数组处理
axis=0，行方向运算，axis=0，列方向运算
参数1表示数组，参数2表示下标，参数3表示插入的数据

delete()：删除指定行/列数据,默认情况将数组按一维数组处理
为了保证数组结构完整性，多维ndarray不能删除单个元素

concatenate((arr1, arr2, ...), axis=0)：将多个维度大小相同的数组合并为一个新的数组
默认情况下，axis=0，行变化，列不变
axis=1,行不变，列变化
```

#### 集合运算

```python
np.unique(x) : 对x里的数据去重，并返回有序结果，默认升序
np.unique(x)[::-1] :对x里的数据去重，并返回降序结果

np.intersect1d(x, y) :计算x和y中的公共元素，并返回有序结果, x & y
np.union1d(x, y) :计算x和y的并集，并返回有序结果, x | y
np.setdiff1d(x, y): 集合的差，即元素在x中且不在y中. x - y, y - x
np.in1d(x, y) : 得到一个表示“x的元素是否包含于y”的布尔型数组.
np.setxor1d(x, y): 对称差集，两个数组中互相不包含的元素。x ^ y
```

#### 排序函数

```python
x = np.arange(16).reshape(4, 4)
# 乱序处理
np.random.shuffle(x)
# 升序排序后的新数组，默认列变化，原数组不变。
np.sort(x)
np.sort(x, axis=0)
# 在原数组上进行升序排序，默认列变化
x.sort()
x.sort(axis=0)
# 返回对升序排序后的值的索引值,默认axis=1
np.argsort(x)
np.argsort(x, axis=0)
# 标定点前的数字比它小，之后的数字比它大，但是并非有序,默认axis=1
np.partition(x, 3)
np.argpartition(x, 3)
np.partition(x, 3 , axis=0)
np.argpartition(x, 3, axis=0)
```

#### 比较运算

```python
# 比较两个array误差是否在误差范围内
np.allclose()
np.allclose([1e10,1e-7], [1.00001e10,1e-8])  # False
np.allclose([1e10,1e-8], [1.00001e10,1e-9])  # True

# 对每个值进行比较，返回布尔数组， 可以添加axis参数指定轴方向
x = np.arange(16)
X = x.reshape(4, 4)
x < 3
x > 3
x <= 3
x >= 3
x == 3
x != 3
2 * x = 24 - 4 * x
# 对应通用函数
np.equal
np.not_equal
np.less
np.less_equal
np.greater
np.greater_equal

# 返回为True的个数
np.sum(x <= 3)  # 满足条件的元素的个数的和
np.count_nonzero(x <= 3)  # 满足条件的非零元素(True)个数
np.all(x>0)  # 验证所有元素是否为真
np.any(x==0)  # 验证任何一个元素是否为真
np.sum(X%2 == 0)
np.sum(X%2 == 0, axis=1)  # 沿着列统计，显示每行的和
np.sum(X%2 == 0, axis=0)  # 沿着行统计，显示每列的和
np.all(X>0, axis=0)
np.all(X>0, axis=1)

# 运算符
np.sum((x > 3) & （x < 10))  # 与，位运算
np.sum((x % 2 == 0)|(x > 10))  # 或
np.sum(~(x == 0))  # 非

# 至少有一个元素满足指定条件返回True
np.any(x==0)
# 所有的元素满足指定条件才返回True
np.all(x>0)
```

#### 布尔运算

布尔运算符

```python
&|^~
# 对应函数
np.bitwise_and
np.bitwise_or
np.bitwise_xor
np.bitwise_not
```

布尔数组作为掩码

```python
# 选取数据
x[x<5]
x[x % 2 == 0]
X[X[:,3] %3 == 0, :]  # 第4列的值可以被3整除的行的所有值
```

#### 自定义运算

通过numpy提供的标准ufunc函数，可以组合出复杂的表达式。需要自定义函数对数组元素进行操作时，可以用frompyfun函数将一个计算单个元素的函数转换城ufunc函数。

- Frompyfunc

调用格式

```python
frompyfunc(func, n_in, n_out)
# n_in表示函数输入参数的个数
# n_out表示函数返回值的个数
```

示例

```python
def liftscore(n):
    n_new = np.sqrt((n^2)*100)
    return n_new

score = np.array([87, 77, 56, 100, 60])
score_1 = np.frompyfunc(liftscore, 1, 1)(score)
```

frompyfunc转换的ufunc函数所返回数组的元素类型是object。因此，还需要再调用数组的astype方法将其转换为浮点数组

```python
score_1 = score_1.astype(float)
```

- vectorize

使用vectorize也可以实现和frompyfunc类似的功能。但它可以通过otypes参数指定返回数组的元素类型。

```python
score_2 = np.vectorize(liftscore, otypes=[float])(score)
any(score1==score_2)  # True
```

#### 广播

当使用ufunc函数对两个数组进行计算时，ufunc函数会对这两个数组的对应元素进行计算，因此要求这两个数组的形状相同。如果形状不同，会进行如下规则的广播(broadcasting)

```
1.如果两个数组的维度数不相同，那么小维度数组的形状将会在最左边补1
2.如果两个数组的形状在任何一个维度上都不匹配，那么数组的形状会沿着维度为1的维度扩展以匹配另外一个数组的形状
3.如果两个数组的形状在任何一个维度上都不匹配并且没有任何一个维度等于1
```

示例

```python
# 示例1
M = np.ones((2, 3))  # (2,3)
a = np.arange(3)  # (3,)
M + a = array([[1., 2., 3.],[1., 2., 3.]])  # a->(1,2)->(2,3)

# 示例2
a = np.arange(3).reshape(3,1)  # (3,1)
b = np.arange(3)  # (3,)
a + b = array([[0, 1, 2],[1, 2, 3],[2, 3, 4]])  # b->(1,3)->(3,3);a->(3,3)

# 示例3
a = np.ones((3,2))  # (3,2)                                                  
b = np.arange(3)  # (3,)
a + b # error;b->(1,2)->(3,3)与a(3,2)不匹配
```

- 快速广播

numpy提供了快速构造可进行广播元素数组的ogrid对象。ogrid和多维数组一样，用切片元组作为下标，返回一组可用来广播计算的数组

```python
x,y = np.ogrif[:5, :5]
```

mgrid对象的用法和ogrid对象类似，但它返回进行广播之后的数组

```
x2 = np.mgrid[:5, :5]
```

- 应用

数组的归一化

```python
X = np.random.random((10, 3))
Xmean = X.mean(0)
X_centered= X-Xmean  # 广播
```

画二维函数

```python
x = np.linspace(0, 5, 50) 
y = np.linspace(0, 5, 50)[:, np.newaxis] 
z = np.sin(x)**10 + np.cos(10+y*x)*np.cos(x) 
import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar()
```

#### ufunc方法

ufunc函数还有只对两个输入一个输出的ufunc函数有效的方法。如`reduce,accumulate,reduceat,outer`等

```python
# reduce
# 沿着axis轴对数组元素进行操作
np.add.reduce(np.arange(5))  # 10
np.add.reduce([[1,2,3,4],[5,6,7,8]], axis=1)  # array([10, 26])

# accumulate
# 类似reduce，返回的数组和输入的数组的形状相同，同时保存所有中间结果
np.add.accumulate(np.arange(5))  # array([0, 1, 3, 6, 10])
np.add.accumulate([[1, 2, 3, 4], [5, 6, 7, 8]], axis=1)  # array([[1, 3, 6, 10],[5, 11, 18, 26]])
np.add.accumulate([[1,2,3,4],[5, 6, 7, 8]], axis=0)  # array([[1, 2, 3, 4],[6, 8, 10, 12]])

# reduceat
# 可以通过indices参数指定多对reduce的起始和终止位置，从而计算多组reduce的结果
ara = np.arange(8)  # array([0,1,2,3,4,5,6,7])
np.add.reduceat(ara, indices=[0,4,1,5,2,6,3,7])  # array([6,4,10,5,14,6,18,7])
np.add.reduceat(ara, [0,4,1,5,2,6,3,7])[::2]  # array([6,10,14,18])

# outer
# 对其作为两个参数的数组的每对元素的组合进行运算
np.add.outer([1,2,3,4],[5,6,7,8])  # array([[6,7,8,9],[7,8,9,10],[8,9,10,11],[9,10,11,12]])
np.multiply.outer([1,2,3],[5,6,7,8])  # array([[5,6,7,8],[10,12,14,16],[15,18,21,24]])
```

## 矩阵点

创建网格点矩阵

```python
import numpy as np
import matplotlib.pyplot as plt

# 方法一
X = np.array([[0, 0.5, 1],[0, 0.5, 1]])
Y = np.array([[0, 0, 0],[1, 1, 1]])
print("X的维度:{},shape:{}".format(X.ndim, X.shape))
print("Y的维度:{},shape:{}".format(Y.ndim, Y.shape))
plt.plot(X, Y, 'o--')
plt.grid(True)

# 方法二
x = np.array([0, 0.5, 1])
y = np.array([0,1])
xv,yv = np.meshgrid(x, y)
print("xv的维度:{},shape:{}".format(xv.ndim, xv.shape))
print("yv的维度:{},shape:{}".format(yv.ndim, yv.shape))
plt.plot(xv, yv, 'o--')
plt.grid(True)

plt.show()
```

## 线性代数

| 函数              | 描述                                                |
| :---------------- | :-------------------------------------------------- |
| `dot`             | 两个数组的点积，即元素对应相乘。                    |
| `vdot`            | 两个向量的点积                                      |
| `inner`           | 两个数组的内积                                      |
| `matmul`          | 两个数组的矩阵积                                    |
| `np.linalg.det`   | 计算输入矩阵的行列式                                |
| `np.linalg.solve` | 求解线性矩阵方程                                    |
| `np.linalg.inv`   | 计算矩阵的乘法逆矩阵                                |
| `np.linalg.svd`   | 对矩阵(M,N)进行奇异值分解，返回u(M,M),s(M,N),v(N,N) |

- 矩阵

矩阵(matrix)是numpy提供的另一种数据类型，可以使用mat或matrix函数将数组转化为矩阵

```python
m1 = np.mat([[1,2,3],[4,5,6]])  # matrix([[1,2,3],[4,5,6]])

m1*8  # matrix([[8,16,24],[32,40,48]])

m2 = np.matrix([[1,2,3],[4,5,6],[7,8,9]])

m1*m2  # matrix([[30, 36, 42],[66, 81,96]])

m2.I  # 逆矩阵
```

- 运算

矩阵与矩阵(多维数组与多维数组)

```python
# 多维数组和多维数组之间的运算
# 将两个数组大小相同的数据，按下标依次对应进行运算，并返回新的数组

# 矩阵乘、加
A = np.arange(4).reshape(2, 2)
B = np.full((2, 2), 10)
A.dot(B)  # 乘法
A * B  # 不是数学意义上的乘法
A + B  # 加法

# 矩阵的转置
A.T

# 矩阵的逆(只有方阵才有逆矩阵)
invA = np.linalg.inv(A)
A.dot(invA) = invA.dot(A)  # 得到单位矩阵

# 矩阵的伪逆矩阵
X = np.arange(16).reshape((2, 8))
pinvX = np.linalg.pinv(X)
X.dot(pinvX)  # 单位矩阵
```

向量与矩阵(一维数组与多维数组)

```python
# 一维数组和多维数组之间运算
# 多维数组的尾部和一位数组的列数相同时，可以相加
# 多维数组的拆分为多个一维数组，然后与一维数组相加
v = np.array([1, 2])
A = np.arange(4).reshape(2, 2)

v+A  # 不是数学意义上的加法
		 # 在numpy中运算，等同与如下计算
  np.vstack([v]*A.shape[0]) + A
  np.tile(v, (2, 1)) + A
  
v*A  # 不是数学意义上的乘法

v.dot(A)  # 数学意义上的矩阵乘法1*2*2*2
A.dot(v)  # numpy自动转换向量为合适列向量或行向量，但结果不一定是数学意义上的样子
```

向量与向量(一维数组与一维数组)

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

# 一维数组和一维数组之间的运算
# 将两个大小相同的数组， 按元素下标对应，进行运算，
# 返回新的数组对象(数组结构和原数组保持一致)
print(arr1 + arr2)
print(arr1 - arr2)
print(arr1 * arr2)
print(arr1 / arr2)
```

数字与数组

```python
# 数组和数字之间的运算
# 将数组的每个元素按照下标依次和数值进行运算, 
# 返回新的数组
print(arr1 + 100)
print(arr1 - 10)
print(arr2 * 2)
print(arr2 / 5)
print(arr2 // 2)
print(arr2 ** 2)
print(arr2 % 2)
print(1/arr2)
```

## 文件读写

- 写入

```python
np.save()
# 将数组数据写入磁盘文件
# 默认情况下，数组是以原始二进制格式保存在扩展名为.npy的文件中。如果在保存文件时没有指定扩展名.npy，则该扩展名会被自动加上。
# 参数1为文件名，参数2为保存的数组

np.savez()
# 可以将多个数组保存到同一个文件中，将数组以关键字参数的形式传入，每个数组通过别名来区分，默认为npz格式


np.savetxt()
# 将数据保存到磁盘文件csv文件里。
# 参数1：文件名，参数2：保存的数组，参数3：dilimiter=分隔符，参数4：fmt=占位符'%s'
```

- 读取

```python
np.genfromtxt()
# 将数据加载到普通的Numpy数组中，这些函数都有许多选项可供使用：指定各种分隔符、读取指定列，指定数据类型等。
# 参数1：文件名，参数2：delimiter=分隔符，参数3：dtype=数组的类型（str/bytes），参数4：usecols=指定读取列的下标

np.load()
# 读取磁盘文件中的数组数据
# 若是单数组，返回该文件保存的数组对象
# 若是多数组，返回一个类字典的文件对象，字典的键就是保存数组的别名

np.loadtxt()
# 读取csv格式的文件，自动切分字段，并将数据载入numpy数组
# 参数1:文件名，参数2:delimiter=分隔符， 参数3:dtype=数组的类型
```

示例

```python
import numpy as np

world_alcohol = np.gefromtxt("world_alcohol.txt", delimiter=",", dtype=str)
```



