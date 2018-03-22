#Scipy

Scipy ：基于Numpy提供了一个在Python中做科学计算的工具集，专为科学和工程设计的Python工具包。主要应用于统计优化、线性代数模块、傅里叶变换、信号和图像处理、常微分方程求解、积分方程、稀疏矩阵等，在数学系或者工程系相对用的多一些，和数据处理的关系不大，我们知道即可，这里不做讲解。

- 在NumPy库的基础上增加了众多的数学、科学及工程常用的库函数
- 线性代数、常微分方程求解、信号处理、图像处理
- 一般的数据处理numpy已经够用
- `import scipy as sp`

# Numpy

Numpy：提供了一个在Python中做科学计算的基础库，重在数值计算，主要用于多维数组（矩阵）处理的库。用来存储和处理大型矩阵，比Python自身的嵌套列表结构要高效的多。本身是由C语言开发，是个很基础的扩展，Python其余的科学计算扩展大部分都是以此为基础。

- 高性能科学计算和数据分析的基础包
- ndarray对象，表示多维数组（矩阵），具有矢量运算能力
- 矩阵运算，无需循环，可完成类似Matlab中的矢量运算
- 线性代数、随机数生成
- `import numpy as np`

**参考学习资料**：

Python、NumPy和SciPy介绍：<http://cs231n.github.io/python-numpy-tutorial>

NumPy和SciPy快速入门：<https://docs.scipy.org/doc/numpy-dev/user/quickstart.html>

##ndarry

```
NumPy数组是一个多维的数组对象（矩阵），称为ndarray，具有高效的算术运算能力和复杂的广播能力，并具有执行速度快和节省空间的特点。
注意：ndarray的下标从0开始，且数组里的所有元素必须是相同类型

# 属性
ndim属性：维度个数
shape属性：维度大小
dtype属性：数据类型
```

##随机抽样创建

| 函数                             | 参数                       | 说明                                                     |
| -------------------------------- | -------------------------- | -------------------------------------------------------- |
| `np.random.rand(3,4)`            | 行数，列数                 | 生成指定维度的随机多维浮点型数组，数据固定区间 0.0 ~ 1.0 |
| `np.random.uniform(-1,5，(3,4))` | 起始值，结束值，行数，列数 | 生成指定维度大小的随机多维浮点型数组，可以指定数字区间   |
| `np.random.randint(-1,5，(3,4))` | 起始值，结束值，行数，列数 | 生成指定维度大小的随机多维整型数组，可以指定数字区间     |

##序列创建

| 函数                    | 参数                                                         | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `np.array(list, dtype)` | list 为 序列型对象(list)、嵌套序列对象(list of list)，dtype表示数据类型 （int、float、str） | 将序列对象转换为数组                                         |
| `np.zeros()`            | 第一个参数是元组，用来指定大小，如(3, 4)，第二个参数可以指定类型，如int，默认为float | 指定大小的全0数组                                            |
| `np.ones()`             | 第一个参数是元组，用来指定大小，如(3, 4)，第二个参数可以指定类型，如int，默认为float | 指定大小的全1数组                                            |
| `np.arange()`           | 可以指定区间，步长                                           | arange() 类似 python 的 range() ，用来创建一个一维 ndarray 数组，结果等同于 np.array(range()) |
| `reshape()`             | 总大小不变，指定新数组的维度大小                             | 重新调整数组的维度                                           |
| `random.shuffle()`      | 参数为数组                                                   | 打乱数组序列（随机重新排列）                                 |

## 数据类型

| 类型          | 类型代码 | 说明                                        |
| ------------- | -------- | ------------------------------------------- |
| int8、uint8   | i1、u1   | 有符号和无符号的8位(1个字节长度)整型        |
| int16、uint16 | i2、u2   | 有符号和无符号的16位(2个字节长度)整型       |
| int32、uint32 | i4、u4   | 有符号和无符号的32位(4个字节长度)整型       |
| float16       | f2       | 半精度浮点数                                |
| float32       | f4或f    | 标准单精度浮点数                            |
| float64       | f8或d    | 双精度浮点数                                |
| bool          | ?        | 布尔类型                                    |
| object        | O        | Python对象类型                              |
| unicode       | <U1或U   | 固定长度的unicode类型，跟字符串定义方式一样 |

```
# dtype属性
print(arr1.dtype)

# dtype参数
# 指定数组的数据类型，类型名+位数，如float64, int32
float_arr = np.array([1, 2, 3, 4], dtype = np.float64)

# astype方法
# 转换数组的数据类型
int_arr = float_arr.astype(np.int32)
```

## 矩阵运算

```
import numpy as np

arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([10, 20, 30, 40, 50])

# 1. 数组和数组之间的运算
print(arr1 * arr2)
# [ 10  40  90 160 250]
print(arr1 + arr2)
# [11 22 33 44 55]


# 2. 数组和数字之间的运算
print(arr1 + 100)
# [101 102 103 104 105]
print(arr2 / 5)
# [ 2.  4.  6.  8.  10.]


# 3. 多维数组和多维数组之间的运算
arr3 = np.arange(9).reshape((3, 3))
arr4 = np.arange(9).reshape((3, 3))

print(arr3)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

print(arr4)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]

print(arr3 + arr4)
# [[ 0  2  4]
#  [ 6  8 10]
#  [12 14 16]]

# 4. 一维数组和多维数组之间运算
arr5 = np.arange(5)
print(arr5)
# [0 1 2 3 4]

arr6 = np.arange(10).reshape((2, 5))
print(arr6)
# [[0 1 2 3 4]
#  [5 6 7 8 9]]

print(arr5 + arr6)
# [[ 0  2  4  6  8]
#  [ 5  7  9 11 13]]
```

## 索引与切片

```
# 一维数组的索引与切片
与Python的列表索引功能相似

# 多维数组的索引与切片
arr[r1:r2, c1:c2]
arr[1, 1] 等价 arr[1][1]
[ : ] 代表某个维度的数据

#  条件索引
布尔值多维数组：arr[condition]，condition也可以是多个条件组合。
条件索引将返回一个一维数组
注意，多个条件组合要使用 & | ~ 连接，而不是Python的 and or not。
```

## 维度转置

```
二维数组直接使用转换函数：transpose()

高维数组转换要指定维度编号参数 (0, 1, 2, …)，注意参数是元组

transpose() 不会更改维度个数，只会修改维度大小。不同于 reshape()。

```

## ufunc

### 元素计算函数

```
一元ufunc：
ceil(x): 向上最接近的整数，参数是 number 或 ndarray
floor(x): 向下最接近的整数，参数是 number 或 ndarray
rint(x): 四舍五入，参数是 number 或 ndarray
negative(x): 元素取反，参数是 number 或 ndarray
abs(x)：元素的绝对值，参数是 number 或 ndarray
square(x)：元素的平方，参数是 number 或 ndarray
aqrt(x)：元素的平方根，参数是 number 或 ndarray
sign(x)：计算各元素的正负号, 1(正数)、0（零）、-1(负数)，参数是 number 或 ndarray
modf(x)：将数组的小数和整数部分以两个独立数组的形式返回，参数是 number 或 ndarray
isnan(x): 判断元素是否为 NaN(Not a Number)，返回bool，参数是 number 或 ndarray


二元ufunc：
add(x, y): 元素相加，x + y，参数是 number 或 ndarray
subtract(x, y): 元素相减，x - y，参数是 number 或 ndarray
multiply(x, y): 元素相乘，x * y，参数是 number 或 ndarray
divide(x, y): 元素相除，x / y，参数是 number 或 ndarray
floor_divide(x, y): 元素相除取整数商(丢弃余数)，x // y，参数是 number 或 ndarray
mod(x, y): 元素求余数，x % y，参数是 number 或 array
power(x, y): 元素求次方，x ** y，参数是 number 或 array

三元ufunc：
where(condition, x, y): 三元运算符，x if condition else y，条件满足返回x，否则返回y，参数condition 是条件语句，参数 x 和 y 是 number 或 ndarray
```

### 数据统计函数

```
多维数组默认统计全部数据，添加axis参数可以按指定轴心统计，值为0则按列统计，值为1则按行统计。

np.mean(x [, axis])：所有元素的平均值，参数是 number 或 ndarray
np.sum(x [, axis])：所有元素的和，参数是 number 或 ndarray
np.max(x [, axis])：所有元素的最大值，参数是 number 或 ndarray
np.min(x [, axis])：所有元素的最小值，参数是 number 或 ndarray
np.std(x [, axis])：所有元素的标准差，参数是 number 或 ndarray
np.var(x [, axis])：所有元素的方差，参数是 number 或 ndarray
np.argmax(x [, axis])：最大值的下标索引值，参数是 number 或 ndarray
np.argmin(x [, axis])：最小值的下标索引值，参数是 number 或 ndarray
np.cumsum(x [, axis])：返回一个一维数组，每个元素都是之前所有元素的 累加和，参数是 number 或 ndarray
np.cumprod(x [, axis])：返回一个一维数组，每个元素都是之前所有元素的 累乘积，参数是 number 或 ndarray
```

### 条件判断函数

```
返回bool值，可以添加axis参数指定轴方向

np.any(): 至少有一个元素满足指定条件，返回True
np.all(): 所有的元素满足指定条件，返回True
```

### 数组增删合并

```
append()：在数组后面追加元素
注意：append总是返回一维数组

insert()：在指定下标插入元素

delete()：删除指定行/列数据
为了保证数组结构完整性，多维ndarray不能删除单个元素

concatenate((arr1, arr2, ...), axis=0)：合并多个数组
注意：合并的两个数组必须维度相同。
```

### 集合运算

```
unique(x) : 对x里的数据去重，并返回有序结果.
intersect1d(x, y) :计算x和y中的公共元素，并返回有序结果, x & y
union1d(x, y) :计算x和y的并集，并返回有序结果, x | y
setdiff1d(x, y): 集合的差，即元素在x中且不在y中. x - y, y - x
in1d(x, y) : 得到一个表示“x的元素是否包含于y”的布尔型数组.
setxor1d(x, y): 对称差集，两个数组中互相不包含的元素。x ^ y
```

### 排序函数

```
ndarray.sort()在原数组上进行排序

np.sort()将返回排序后的新数组，原数组不变。
```

##文件的读写

```
np.save()
将数组数据写入磁盘文件
默认情况下，数组是以原始二进制格式保存在扩展名为.npy的文件中。如果在保存文件时没有指定扩展名.npy，则该扩展名会被自动加上。

np.savez()
可以将多个数组保存到同一个文件中，将数组以关键字参数的形式传入即可

np.load()
读取磁盘文件中的数组数据

savetxt()
将数据保存到磁盘文件里。

np.genfromtxt()
将数据加载到普通的Numpy数组中，这些函数都有许多选项可供使用：指定各种分隔符、读取指定列，指定数据类型等。

np.loadtxt()
```







