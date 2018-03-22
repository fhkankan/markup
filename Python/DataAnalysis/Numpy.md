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

##ndarry创建

```
NumPy数组是一个多维的数组对象（矩阵），称为ndarray，具有高效的算术运算能力和复杂的广播能力，并具有执行速度快和节省空间的特点。
注意：ndarray的下标从0开始，且数组里的所有元素必须是相同类型

# 属性
ndarry.ndim			维度个数
ndarry.shape		维度大小
ndarry.dtype		数据类型
```

### 随机抽样创建

| 函数                             | 参数                         | 说明                                                     |
| -------------------------------- | ---------------------------- | -------------------------------------------------------- |
| `np.random.rand(3,4)`            | 行数，列数                   | 生成指定维度的随机多维浮点型数组，数据固定区间 0.0 ~ 1.0 |
| `np.random.uniform(-1,5，(3,4))` | [起始值，结束值)，行数，列数 | 生成指定维度大小的随机多维浮点型数组，可以指定数字区间   |
| `np.random.randint(-1,5，(3,4))` | [起始值，结束值)，行数，列数 | 生成指定维度大小的随机多维整型数组，可以指定数字区间     |

### 序列创建

| 函数                        | 参数                                                         | 说明                                                         |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `np.array(list, dtype)`     | list 为 序列型对象(list)、嵌套序列对象(list of list)，dtype表示数据类型 （int、float、str） | 将序列对象转换为数组                                         |
| `np.zeros()`                | 第一个参数是元组，用来指定大小，如(3, 4)，第二个参数可以指定类型，如int，默认为float | 指定大小的全0数组                                            |
| `np.ones()`                 | 第一个参数是元组，用来指定大小，如(3, 4)，第二个参数可以指定类型，如int，默认为float | 指定大小的全1数组                                            |
| `np.arange()`               | 可以指定区间，步长                                           | arange() 类似 python 的 range() ，用来创建一个一维 ndarray 数组，结果等同于 np.array(range()) |
| `ndarry.reshape()`          | 总大小不变，指定新数组的维度大小                             | 重新调整数组的维度                                           |
| `np.random.shuffle(ndarry)` | 参数为数组                                                   | 打乱数组序列（随机重新排列）                                 |

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

# 1. 一维数组和一维数组之间的运算
# 将两个大小相同的数组， 按元素下标对应，进行运算，
# 返回新的数组对象(数组结构和原数组保持一致)
print(arr1 + arr2)
print(arr1 - arr2)
print(arr1 * arr2)
print(arr1 / arr2)

# 2. 数组和数字之间的运算
# 将数组的每个元素按照下标依次和数值进行运算, 
# 返回新的数组
print(arr1 + 100)
print(arr2 / 5)

# 3. 多维数组和多维数组之间的运算
# 将两个数组大小相同的数据，按下标依次对应进行运算，
# 并返回新的数组
arr3 = np.arange(9).reshape((3, 3))
arr4 = np.arange(9).reshape((3, 3))
print(arr3 + arr4)


# 4. 一维数组和多维数组之间运算
# 多维数组的尾部和一位数组的列数相同时，可以相加
# 多维数组的拆分为多个一维数组，然后与一维数组相加
arr5 = np.arange(5)
arr6 = np.arange(10).reshape((2, 5))
print(arr5 + arr6)
```

## 索引与切片

### 一维数组

```
arr1 = np.arange(5)

# 取某个下标的值
print(arr1[3])

# 取连续的多个值
print(arr1[4:10])

# 对数组进行条件判断，返回bool类型的数组
is_after6_arr = arr1 > 6
print(is_after6_arr)

# 将一个相同大小的bool数组映射到另一个数组里，
# 会返回所有为True的结果（一维数组）
print(arr1[arr1 > 6])

# 如果有多个条件， 通过 & | ~
is_arr = (arr1 > 6)&(arr1 < 13)
print(is_arr)
```

### 二维数组

```
arr2 = arr1.random.randint(-5,10,(4, 4))

# 下标为2的一行
print(arr2[2])

# 下标为2的行，下标为2的列
print(arr2[2][2])
print(arr2[2, 2])

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

# 条件索引，如果有多个条件， 通过 & | ~
year_arr = np.array([[1020,2011,2012],[2013,2014,2015],[2016,2017,201]])
print(year_arr[(year_arr >= 2012) & (year_arr <= 2016)])

```

## 维度转置

```
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

## ufunc

### 元素计算函数

```
一元ufunc：
np.ceil(x): 向上最接近的整数，参数是 number 或 ndarray
np.floor(x): 向下最接近的整数，参数是 number 或 ndarray
np.rint(x): 四舍五入，参数是 number 或 ndarray
np.negative(x): 元素取反，参数是 number 或 ndarray
abs(x)：元素的绝对值，参数是 number 或 ndarray
np.square(x)：元素的平方，参数是 number 或 ndarray
np.aqrt(x)：元素的平方根，参数是 number 或 ndarray
np.sign(x)：计算各元素的正负号, 1(正数)、0（零）、-1(负数)，参数是 number 或 ndarray
np.modf(x)：将数组的小数和整数部分以两个独立数组的形式返回，参数是 number 或 ndarray
np.isnan(x): 判断元素是否为 NaN(Not a Number)，返回bool，参数是 number 或 ndarray


二元ufunc：
np.add(x, y): 元素相加，x + y，参数是 number 或 ndarray
np.subtract(x, y): 元素相减，x - y，参数是 number 或 ndarray
np.multiply(x, y): 元素相乘，x * y，参数是 number 或 ndarray
np.divide(x, y): 元素相除，x / y，参数是 number 或 ndarray
np.floor_divide(x, y): 元素相除取整数商(丢弃余数)，x // y，参数是 number 或 ndarray
np.mod(x, y): 元素求余数，x % y，参数是 number 或 array
np.power(x, y): 元素求次方，x ** y，参数是 number 或 array

三元ufunc：
np.where(condition, x, y): 三元运算符，x if condition else y，条件满足返回x，否则返回y，参数condition 是条件语句，参数 x 和 y 是 number 或 ndarray
```

### 数据统计函数

```
多维数组默认统计全部数据，添加axis参数可以按指定轴心统计，值为0则行运算，列标不变，值为1则列运算，行标

np.mean(x [, axis])：所有元素的平均值，参数是 number 或 ndarray
np.sum(x [, axis])：所有元素的和，参数是 number 或 ndarray
np.max(x [, axis])：所有元素的最大值，参数是 number 或 ndarray
np.min(x [, axis])：所有元素的最小值，参数是 number 或 ndarray
np.std(x [, axis])：所有元素的标准差，参数是 number 或 ndarray
np.var(x [, axis])：所有元素的方差，参数是 number 或 ndarray
np.argmax(x [, axis])：最大值的下标索引值，参数是 number 或 ndarray
np.argmin(x [, axis])：最小值的下标索引值，参数是 number 或 ndarray
np.argwhere(condition):符合指定条件的元素的下标（二维数组）
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

### 集合运算

```
np.unique(x) : 对x里的数据去重，并返回有序结果，默认升序
np.unique(x)[::-1] :对x里的数据去重，并返回降序结果

np.intersect1d(x, y) :计算x和y中的公共元素，并返回有序结果, x & y
np.union1d(x, y) :计算x和y的并集，并返回有序结果, x | y
np.setdiff1d(x, y): 集合的差，即元素在x中且不在y中. x - y, y - x
np.in1d(x, y) : 得到一个表示“x的元素是否包含于y”的布尔型数组.
np.setxor1d(x, y): 对称差集，两个数组中互相不包含的元素。x ^ y
```

### 排序函数

```
ndarray.sort()在原数组上进行排序，默认列变化

np.sort()将返回升序排序后的新数组，默认列变化，原数组不变。

```

##文件的读写

```
np.save()
将数组数据写入磁盘文件
默认情况下，数组是以原始二进制格式保存在扩展名为.npy的文件中。如果在保存文件时没有指定扩展名.npy，则该扩展名会被自动加上。
参数1为文件名，参数2为保存的数组

np.savez()
可以将多个数组保存到同一个文件中，将数组以关键字参数的形式传入，每个数组通过别名来区分，默认为npz格式

np.load()
读取磁盘文件中的数组数据
若是单数组，返回该文件保存的数组对象
若是多数组，返回一个类字典的文件对象，字典的键就是保存数组的别名

savetxt()
将数据保存到磁盘文件csv文件里。
参数1：文件名，参数2：保存的数组，参数3：dilimiter=分隔符，参数4：fmt=占位符'%s'

np.genfromtxt()
将数据加载到普通的Numpy数组中，这些函数都有许多选项可供使用：指定各种分隔符、读取指定列，指定数据类型等。
参数1：文件名，参数2：delimiter=分隔符，参数3：dtype=数组的类型（str/bytes），参数4：usecols=指定读取列的下标

np.loadtxt()
```







