# Pandas

Pandas的名称来自于面板数据（panel data）和Python数据分析（data analysis）。

Pandas是一个强大的分析结构化数据的工具集，基于NumPy构建，提供了 **高级数据结构** 和 **数据操作工具**，它是使Python成为强大而高效的数据分析环境的重要因素之一。

- 一个强大的分析和操作大型结构化数据集所需的工具集
- 基础是NumPy，提供了高性能矩阵的运算
- 提供了大量能够快速便捷地处理数据的函数和方法
- 应用于数据挖掘，数据分析
- 提供数据清洗功能

**参考学习**

<http://pandas.pydata.org>

安装

```
pip install pandas
```

查看版本

```
import pandas
pandas.__version
```

引用

```python
import pandas as pd
```

## 文件读写

- 文件导入

可以使用如下的函数将主流格式的数据文件读物并转化为DataFrame实例对象

| 函数             | 说明                                 |
| ---------------- | ------------------------------------ |
| `read_csv`       | 读入具有分隔符的csv文件              |
| `read_table`     | 读入具有分隔符的文件                 |
| `read_sql`       | 读入SQL，MySQL数据库中的数据         |
| `read_sas`       | 读入SAS的xpt或sas7bdat格式的数据集   |
| `read_stata`     | 读入STATA数据集                      |
| `read_json`      | 读入json数据                         |
| `read_html`      | 读入网页中的表                       |
| `read_clipboard` | 读入剪贴板中数据内容                 |
| `read_fwf`       | 读入固定宽度格式化数据               |
| `read_hdf`       | 读入分布式存储系统(HDFStore)中的文件 |

示例

```python
jddf = pd.read_csv(
    'data.csv', 
    header=None,  # 表示不会把数据的第1行和第1列设置为行列索引
    name=['name', 'time', 'opening_price', 'closing_price', 'lowest_price', 'highest_price', 'volume']  # 指定列索隐，即通常意义下的变量名
)
```

- 数据导出

可以使用如下函数将实例对象输出到外部文件或指定对象中

| 函数           | 说明                     |
| -------------- | ------------------------ |
| `to_csv`       | 输出到csv文件中          |
| `to_excel`     | 输出到excel表中          |
| `to_hdf`       | 输出到HDFS分布式文件系统 |
| `to_json`      | 输出到json文件中         |
| `to_html`      | 输出到网页表中           |
| `to_dict`      | 输出为字典格式           |
| `to_stata`     |                          |
| `to_latex`     | 输出为latex格式          |
| `to_sql`       | 输出为sql格式            |
| `to_clipboard` | 输出到剪贴板中           |

## 数据结构

Pandas有两个最主要也是最重要的数据结构： **Series** 和 **DataFrame**

- Series

Series是一种类似于一维数组的 **对象**，由一组数据（各种NumPy数据类型）以及一组与之对应的索引（数据标签）组成。

```
- 类似一维数组的对象
- 由数据和索引组成
  - 索引(index)在左，数据(values)在右
  - 索引是自动创建的
```

创建

```python
# 通过list创建
ser_obj = pd.Series(range(10))

# 通过ndarry创建
ser_obj = pd.Series(np.arange(10))

# 通过dict创建
ser_obj3 = pd.Series({"age": 18, "id": 1001, "name": "itcast"})

# 对象命名和索引命名
ser_obj4.name = "Stu"
ser_obj4.index.name = "Info"

# 通过index参数指定行索引创建
ser_obj3 = pd.Series(data=range(-3, 3), index=list("ABCDEF"), dtype=np.float64, name="测试数据")
```

查看

```python
# 查看名字
对象名：ser_obj.name
对象索引名：ser_obj.index.name

# 查看对象索引数据
ser_obj3.index

# 查看对象值
ser_obj3.values

# 查看索引对应的值
ser_obj.label1
ser_obj["label1"]
ser_obj[pos]
```

- DataFrame

DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同类型的值。DataFrame既有行索引也有列索引，它可以被看做是由Series组成的字典（共用同一个索引），数据是以二维结构存放的。

```
- 类似多维数组/表格数据 (如，excel, R中的data.frame)
- 每列数据可以是不同的类型
- 索引包括列索引和行索引
```

创建

```python
# 通过嵌套列表/ndarray创建
import numpy as np
arr = np.random.rand(3, 4)
df_pbj1 = pd.DataFrame(arr)


# 通过dict构建DataFrame
dict_data = {'A': 1, 
             'B': pd.Timestamp('20170426'),
             'C': pd.Series(1, index=list(range(4)),dtype='float32'),
             'D': np.array([3] * 4,dtype='int32'),
             'E': ["Python","Java","C++","C"], }
# 字典的键对应列索引，同一列的数据类型必须相同，不同列之间数据类型可以不同
# 如果的单个数据，会自动填充到最多的个数，如果是多个数据，必须个数相同
df_obj3 = pd.DataFrame(dict_data)

# 增加列数据
df_obj[new_col_idx] = data

# 创建行索引与列索引
df_pbj2 = pd.DataFrame(arr, index= ['A', 'B', 'C'], columns=['a', 'b', 'c', 'd'])
```

查看

```python
# 产看DataFrame对象的详细信息
df_obj.info()
# 查看常见统计信息
df_obj.descibe()

# 查看数据的前n行(默认5行)
df_obj.head(n)
# 查看数据的后n行
df_obj.tail(n)

# 查看实例的值
df_obj.values
# 查看实例每列的属性
df_obj.dtypes
# 查看行索引和列索引对象
df_obj.index
df_obj.columns

# 通过索引获取列数据
df_obj[col_idx]
df_obj.col_idx

# 列数据中的不重复的值
df_obj.col_idx.unique()

# 统计列数据出现的频次
df_obj[col_idx].value_counts()
```

## 索引操作

### 查看数据

- 直接索引

Series

```python
# 指定行索引名
ser_obj1 = pd.Series(range(10, 15), index= list("ABCDE"))

# 取单个数据
ser_obj[‘label’]
ser_obj.label
ser_obj[pos]

# 取Series
# 切片索引
ser_obj[‘label1’: ’label3’]  # 显式，包含结束位
ser_obj[2:4]  # 隐式，左闭右开
# 花哨索引
ser_obj[[‘label1’, ’label2’, ‘label3’]]
ser_obj[[pos1, pos2, pos3]]

# 根据条件取值
# 布尔索引
# 对对象做运算，返回新对象，显示对象中每个元素的布尔值
(ser_obj1 > 10) & (ser_obj1 < 14)
# 对索引做布尔，返回符合条件的结果
ser_obj1[(ser_obj1 > 10) & (ser_obj1 < 14)]
ser_obj1[~(ser_obj1 == 12)]
```

DataFrame

```python
# DataFrame对象本身也支持一些索引取值操作，但是通常情况下，会使用规范良好的高级索引方法。

df_obj = pd.DataFrame(np.random.rand(3, 4), index= list("ABC"), columns=list("abcd"))  # 指定index指定行索引，columns指定列索引

# 列行对象
# 取某列(Series对象)
df_obj['column_label']
df_obj.column_label
# 取某列的数据(ndarray对象)
df_obj['column_label'].values
# 取不连续的多列(不能取连续的列)
df_obj[['column_label1', 'column_label2']]
# 连续索引取单行
df_obj["raw_label1":"raw_label1"]
df_obj[pos:pos+1]
# 取连续的多行（不能取不连续的行）
df_obj["raw_label1":"raw_label2"]
df_obj[pos1:pos2]

# 按条件索引(布尔索引)
df_obj[df_obj["column_label] >= 2]

# 取值
# 取某列的某个数据
df_obj['column_label'].values[num]
df_obj['column_label']['raw_label']
```

- 高级索引

有3种：标签索引 loc、位置索引 iloc，混合索引ix
Series结构简单，一般不需要高级索引取值，主要用于DataFrame对象

> loc

```python
# Series
# 1. 取连续多行
print(ser_obj1.loc['B' : 'C'])
# 2. 取多行
print(ser_obj1.loc[['A', 'C', 'D']])

# DataFrame
# 1.取单行
print(df_obj.loc["B"])
# 2.取单列
print(df_obj.loc[:, "d"])
# 3.取单行单列
print(df_obj.loc["B", "c"])
# 4.取连续多行
print(df_obj.loc["B":"D"])
# 5.取连续多列
print(df_obj.loc[:, "b":"d"])
# 6.取连续的多行多列
print(df_obj.loc["B":"D", "b":"d"])
# 7.取不连续的多行
print(df_obj.loc[["B", "D"]])
# 8.取不连续的多列
print(df_obj.loc[:, ["a", "c", "d"]])
# 9.取不连续的多行多列
print(df_obj.loc[["A","D"], ["a", "d"]])
# 10.取布尔值
# 根据某列的数据进行判断，返回为True的行
print(df_obj.loc[df_obj["a"] > -1])
# 根据某列的数据进行判断，返回为True的行，再取出指定的多列
print(df_obj.loc[df_obj["a"] > -1, ["b", "d"]])

# 删除某一列
data = data.loc[~data.variable.str.contains('exploitable'), :]
data = data.loc[~(data.variable == 'exploitable')]
```

> iloc

作用和loc一样，不过是基于索引编号来索引

```python
# Series
print(ser_obj[1:3])
print(ser_obj.iloc[1:3])

# DataFrame
print(df_obj.iloc[0])
print(df_obj.iloc[0:2])
print(df_obj.iloc[0:2, 1:3])
print(df_obj.iloc[[0, 2], [0, 3]])
print(df_obj.iloc[[1,2,3,4], :])
```

> ix

ix是以上二者的综合，既可以使用索引编号，又可以使用自定义索引，但是如果索引既有数字又有英文，容易导致定位的混乱。目前官方已不推荐使用

### 增删改

增

```python
# Series
# 新增行数据
ser_obj["F"] = 3.14

# DataFrame
# 新增列数据
df_obj["f"] = [10, 20, 30, 40]
df_obj["g"] = df_obj["c"] + df_obj["f"]
```

删

```python
# del 删除原数据，
del(df_obj['g'])

# drop() 
# 删除并返回原数据的副本，原数据不删除,默认axis=0
df_obj.drop(["f"], axis=1)
# 加inplace=True后是删除原数据
df_obj.drop(["C", "D"], inplace=True)
```

改

```python
titanic.log[titanic["Sex"] == "male", "Sex"] = 0
titanic.log[titanic["Sex"] == "female", "Sex"] = 1
```

### 类型转换

```python
# 索引对象支持类型转换：list,ndarray,Series
print(df_obj.index)
print(list(df_obj.index))
print(np.array(df_obj.index))
print(pd.Series(df_obj.index))

print(df_obj.columns)
print(list(df_obj.columns))
print(np.array(df_obj.columns))
print(pd.Series(df_obj.columns))
```

### 重命名

```python
df_obj.rename(
    index = {"A" : "AA", "B" : "BB", "C" : "CC"}, 
    columns = {"a" : 'aa', "b" : "bb", "c" : "cc", "d" : "dd"}, 
    inplace=True
)
```

## 层级索引

对于一维和二维数据，用pandas的Series和DataFrame对象存储，若是三维或四维数据，可以通过能够过层级索引(也称多级索引)配合多个有不同等级的一级索引一起使用，将高维数据转换成类似Series和Dataframe对象的形式。

索引对象

```
- Index，自定义索引
- RangeIndex，序列索引
- Int64Index，整数索引
- MultiIndex，层级索引
```

### 多级索引Series

元组表示索引

```python
index = [('California', 2000),('California', 2001),('New York', 2000), ('New York', 2001)]
population = [1000, 1050, 3000, 3050]
pop = pd.Series(population, index=index)
print(pop)
# (California, 2000)    1000
# (California, 2001)    1050
# (New York, 2000)      3000
# (New York, 2001)      3050
```

pandas多级索引

```python
index = pd.MultiIndex.from_tuples(index)
print(index)
# MultiIndex([('California', 2000),
#             ('California', 2001),
#             (  'New York', 2000),
#             (  'New York', 2001)],
#            )
pop = pop.reindex(index)  # 索引重置
print(pop)
# California  2000    1000
#             2001    1050
# New York    2000    3000
#             2001    3050

```

高维数据多级索引

```python
pop_df = pd.DataFrame({'total': pop, 'under18':[200, 210, 300, 310]})
print(pop_df)
#                  total  under18
# California 2000   1000      200
#            2001   1050      210
# New York   2000   3000      300
#            2001   3050      310
```

### 创建方法

直接创建

```python
# 将index设置为二维索引数组
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'],[1,2,1,2]],
                  columns=['data1', 'data2'])
print(df)
#         data1     data2
# a 1  0.860061  0.962663
#   2  0.174399  0.168180
# b 1  0.070034  0.906971
#   2  0.311166  0.071993

# 将元组作为key的字典传递给pandas，默认转为MultiIndex
data = {
    ('California', 2000): 200,
    ('California', 2001): 210,
    ('New York', 2000): 200,
    ('New York', 2001): 210
}
res = pd.Series(data)
print(res)
# California  2000    200
#             2001    210
# New York    2000    200
#             2001    210
```

显式创建

```python
data = [100, 200, 300, 400]
index1 = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
index2 = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
index3 = pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
index4 = pd.MultiIndex(levels=[['a', 'b'], [1, 2]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
res1 = pd.Series(data, index=index4)
print(res1)
# a  1    100
#    2    200
# b  1    300
#    2    400
res2= pd.DataFrame(data , index=index4, columns=['a'])
print(res2)
#        a
# a 1  100
#   2  200
# b 1  300
#   2  400
```

多级索引的等级名称

```python
# 方法一：内部创建
index = pd.MultiIndex.from_product([['California', 'New York'], [2000, 2001]], names=['state', 'year'])
# 方法二：外部更改
pop.index.names = ['state', 'year']
print(pop)
# state       year
# California  2000    1000
#             2001    1050
# New York    2000    3000
#             2001    3050
```

多级列索引

```python
# 4维数据
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']], names=['Subject', 'type'])
# 模拟数据
data = np.round(np.random.randn(4, 6), 1)
data[:, :2] *= 10
data += 37
# 创建DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data)
# Subject      Bob       Guido         Sue
# type          HR  Temp    HR  Temp    HR  Temp
# year visit
# 2013 1      25.0  33.0  37.4  39.3  36.5  36.2
#      2      51.0  35.0  38.0  36.3  35.3  38.9
# 2014 1      43.0  56.0  38.0  36.0  36.4  35.9
#      2      28.0  30.0  35.8  38.7  38.2  34.8
```

### 查看数据

Series

```python
index = pd.MultiIndex.from_product([['California', 'New York'], [2000, 2001]], names=['state', 'year'])
data = [100, 200, 300, 400]
pop = pd.Series(data, index=index)
print(pop)
# state       year
# California  2000    100
#             2001    200
# New York    2000    300
#             2001    400

print(pop['California'])
print(pop.loc['California'])
# year
# 2000    100
# 2001    200
print(pop['California':'California']) 
print(pop.loc['California':'California'])
# 对于使用行索引名字进行切片时，需按序排列
# pop.sor_index()
# pop['California':'New York']
print(pop[0:2])
print(pop[['California']])
# state       year
# California  2000    100
#             2001    200

print(pop[pop > 150])
# state       year
# California  2001    200
# New York    2000    300
#             2001    400

print(pop['California', 2000])
# 100
```

dataFrame

```python
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']], names=['Subject', 'type'])
# 模拟数据
data = np.round(np.random.randn(4, 6), 1)
data[:, :2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data)
# Subject      Bob       Guido         Sue      
# type          HR  Temp    HR  Temp    HR  Temp
# year visit                                    
# 2013 1      32.0  45.0  37.0  36.8  36.9  36.6
#      2      24.0  39.0  37.1  38.0  35.5  36.1
# 2014 1      18.0  44.0  35.5  36.7  34.8  38.2
#      2      40.0  53.0  36.5  35.6  37.2  36.7
print(health_data['Guido', 'HR'])
print(health_data.loc[:, ('Guido', 'HR')])
print(health_data.iloc[:, 2:3])
# year visit      
# 2013 1      37.0
#      2      37.1
# 2014 1      35.5
#      2      36.5
print(health_data.iloc[0, 0])
# 32.0
```

### 行列转换

unstack/stack

```python
index = pd.MultiIndex.from_product([['New York', 'California'], [2000, 2001]], names=['state', 'year'])
data = [100, 200, 300, 400]
pop = pd.Series(data, index=index)
pop = pop.sort_index()
print(pop)
# state       year
# California  2000    300
#             2001    400
# New York    2000    100
#             2001    200

# unstack()
# Series->DataFrame
# 默认1内层索引为列索引，0将最外层做为列索引
print(pop.unstack())
# year        2000  2001
# state
# California   300   400
# New York     100   200
print(pop.unstack(level=0))
# state  California  New York
# year
# 2000          300       100
# 2001          400       200

# stack()
# DataFrame->Series
# 将列索引旋转为行索引，完成层级索引
print(pop.unstack().stack())
# state       year
# California  2000    300
#             2001    400
# New York    2000    100
#             2001    200

# DataFrame.T
# 将行和列索引互相调换
print(pop.unstack().T)
# state  California  New York
# year
# 2000          300       100
# 2001          400       200
```

重置索引

```python
pop_flat = pop.reset_index(name='population')
print(pop_flat)
#         state  year  population
# 0  California  2000         300
# 1  California  2001         400
# 2    New York  2000         100
# 3    New York  2001         200
pop_flat = pop_flat.set_index(['state', 'year'])
print(pop_flat)
#                  population
# state      year            
# California 2000         300
#            2001         400
# New York   2000         100
#            2001         200
```

### 数据累计方法

除了pandas的数据运算函数如`mean(),sum()`等，对于层级索引，可以设置参数level实现对数据子集的累计操作

```python
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]], names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']], names=['Subject', 'type'])
# 模拟数据
data = np.round(np.random.randn(4, 6), 1)
data[:, :2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data)
# Subject      Bob       Guido         Sue      
# type          HR  Temp    HR  Temp    HR  Temp
# year visit                                    
# 2013 1      32.0  45.0  37.0  36.8  36.9  36.6
#      2      24.0  39.0  37.1  38.0  35.5  36.1
# 2014 1      18.0  44.0  35.5  36.7  34.8  38.2
#      2      40.0  53.0  36.5  35.6  37.2  36.7

# 计算每一年各项指标的平均值
data_mean = health_data.mean(level='year')
print(data_mean)
# Subject   Bob        Guido         Sue
# type       HR  Temp     HR  Temp    HR   Temp
# year
# 2013     43.0  36.5  38.05  37.7  37.9  37.55
# 2014     35.0  32.5  35.15  38.7  36.3  38.35

# 对列索引进行类似的累计操作
res = data_mean.mean(axis=1, level='type')
print(res)
# type         HR       Temp
# year                      
# 2013  40.350000  40.683333
# 2014  34.866667  33.383333
```

## 数据运算

### 数值运算

pandas在数值运算时，对于一元运算，通用函数在输出结果中保留索引和列标签；对于二元运算，通用函数会自动对其索引进行计算.

如果存在不同的索引，结果的索引就是所有索引的并集。

如果索引相同，按索引对齐进行运算。如果没对齐的位置则补NaN；但是可以通过指定数据填充缺失值，再参与对齐运算。

- 一元

```python
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
df = pd.DataFrame(rng.randint(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D'])
np.exp(ser)
np.sin(df * np.pi / 4)
```

- 二元

+/-/*//

```python
ser_obj1 = pd.Series(range(10, 15), index=list("ABCDE"))
ser_obj2 = pd.Series(range(10, 15), index=list("CDEFG"))

ser_obj1 + 5  # np.add()
ser_obj1 - 5  # np.sub(), np.subtract()
ser_obj1 * 5  # np.mul(), np.multiply
ser_obj1 / 5  # np.truediv(), np.div(), np.divde()
ser_obj1 // 5  # np.floordiv()
ser_obj1 % 5  # np.mod()
ser_obj1 ** 5  # np.pow()

ser_obj1 + ser_obj2  
ser_obj1 - ser_obj2
ser_obj1 * ser_obj2
ser_obj1 / ser_obj2
```

Series

```python
#  Series 按行、索引对齐
ser_obj1 = pd.Series(range(10, 15), index=list("ABCDE"))
ser_obj2 = pd.Series(range(10, 15), index=list("CDEFG"))
# 索引对齐则进行算术运算；索引未对齐，则填充NaN值
ser_obj3 = ser_obj1.add(ser_obj2)
# 通过fill_value参数填充一个值参与对齐运算(注意不是填充到结果上)
ser_obj4 = ser_obj1.add(ser_obj2, fill_value=100)
```

DataFrame

```python
df_obj1 = pd.DataFrame(np.random.randint(-5, 10, (3, 4)), index=list("ABC"), columns=list("abcd"))
df_obj2 = pd.DataFrame(np.random.randint(-5, 10, (3, 4)), index=list("ABC"), columns=list("cdef"))
df_obj3 = df_obj1.add(df_obj2)
df_obj4 = df_obj1.add(df_obj2, fill_value=100)
```

### 统计计算

```python
# 常用的统计计算
# axis=0 按列统计，axis=1按行统计
# skipna默认为True，表示计算时排除NaN值
df_obj.sum(df_obj)
df_obj.mean(skipna=False)
df_obj.max(skipna=False)
df_obj.min(skipna=False)

# 常用的统计描述
df_obj.describe()

# 特征间关系
df_obj.pivot_table(index="Pclass", values="Survived", aggfunc=np.mean)
```

| 方法          | 说明                                         |
| ------------- | -------------------------------------------- |
| count         | 非NA值得数量                                 |
| describe      | 针对Series或各DataFrame列计算汇总统计        |
| min,max       | 计算最小值和最大值                           |
| argmin,argmax | 计算能够获取到最小值和最大值的索引位置(整数) |
| idxmin,idxmax | 计算能够获取到最小值和最大值的索引值         |
| quantile      | 计算样本的分位数(0到1)                       |
| sum           | 值的总和                                     |
| mean          | 值的平均数                                   |
| median        | 值的算数中位数                               |
| mad           | 根据平均值计算平均绝对离差                   |
| var           | 样本值的方差                                 |
| std           | 样本值的标准差                               |
| skew          | 样本值的偏度(三阶矩)                         |
| kurt          | 样本值的峰度(四阶矩)                         |
| cumsum        | 样本值的累计和                               |
| cummin/cummax | 样本值的累计最大值和累计最小值               |
| cumprod       | 样本值的累计积                               |
| diff          | 计算一阶差分(对时间序列很有用)               |
| pct_change    | 计算百分数变化                               |

### 自定义函数

- Numpy

numpy的ufunc也可以用于pandas对象，即可将函数应用到Series中的每一个元素

```python
print(np.sum(ser_obj))
print(np.abs(ser_obj))
print(np.sum(df_obj))
print(np.sum(df_obj, axis=1))

reversef = lambda x: -x
reversef(def_obj)
```

- apply

使用apply应用自定义函数到DataFrame的对象的每一行/每一列上

返回值由自定义函数决定，若是计算类，返回DataFrame,若是统计类，返回Series

```python
print(df.apply(lambda x : x.max()))
# 指定轴方向，axis=1，方向是行
print(df.apply(lambda x : x.max(), axis=1))
```

- applymap

通过applymap将函数应用到DataFrame的每个元素

返回DataFrame

```python
print(df.applymap(lambda x : '%.2f' % x))
```

## 数据清洗

数据的质量直接关乎最后数据分析出来的结果，如果数据有错误，在计算和统计后，结果也会有误。所以在进行数据分析前，我们必须对数据进行清洗。需要考虑数据是否需要修改、如何修改调整才能适用于之后的计算和分析等。

数据清洗也是一个迭代的过程，实际项目中可能需要不止一次地执行这些清洗操作。

### 处理缺失值

- 形式

在pandas对象中缺失值除了以`NaN`的形式存在之外，还可以用python对象中的`None`来表示。

注意：在数值型数据二者都表示`NaN`，而字符型数据`np.nan`表示为`NaN`，`None`就是表示为其本身

缺失值在默认情况下不参与运算及数据分析过程

对于时间戳的datetime64[ns]数据格式，其默认缺失值是以`NaT`的形式存在

- 判断

```python
# 判断数据集是否有缺失值,返回bool类型的DataFrame
df_obj.isnull()
df_obj.notnull()

# 返回每列特征对应缺失值的个数
df_obj.isnull().sum()
```

- 处理

删除

```python
# 删除缺失值所在的行
df_obj = df_obj.dropna()
# 删除缺失值所在的列
df_obj = df_obj.dropna(axis=1)
```

填充

```python
# 填充缺失值
df_obj.fillna(value)
# 参数
value 	填充缺失值的标量或字典对象
method	指定填充方式：backfill,bfill,pad,ffill,None.默认为ffill.pad/ffill表示前向填充；bfill/backfill表示后向填充
axis	指定待填充的轴：0，1或index,columns.默认axis=0(index)
inplace	指定是否(默认否)修改对象上的任何其他视图
limit	指定ffill和backfill填充可
# ffill
df_obj.ffill(limit=1)  # 当有连续缺失值时，只填充第1个缺失值
# bfill
df_obj.bfill()  # 等价于fillna(method='bfill')
```

插值

```python
# 插值即利用已有数据对数值型缺失值进行估计，并用估计结果来替换缺失值
df_obj.interpolate(method='liner')  # 线性插值

# 参数
liner	默认
time
index
values
nearest
zero
slinear
quadratic
cubic
barycentric
krogh
polynomial
spline
piecewise_polynomial
from_derivatives
pchip
akima
```

### 处理重复值

列中行数据重复

```python
# 判断某列中是否有重复数据,返回bool类型的series/dataframe
df_obj.duplacited('column_label')

# 删除重复数据的行
df_obj.drop_duplicates('column_label')
df_obj[['column_label']].drop_duplicates()
```

列只有唯一值

```python
# 若列只有唯一值和缺失值，则没有有效信息，可删除此列特征
orig_columns = df_obj.columns
drop_columns = []
for col in orig_columns:
  	col_series = df_obj[col].dropna().unique()
    if len(col_series) == 1:
      	drop_columns.append(col)
df_obj = df_obj.drop(drop_columns, axis=1)
```

行数据重复检测

```python
# 对数据是否只出现一次
all_cols_unique_players = df.groupby('playerShort').agg({
  col:'numique' for col in players_cols
})  # 将行中是否出现重复的playerShort数据以出现次数的形式展示
all_cols_unique_players[all_cols_unique_players > 1].dropna().head()  # 显示重复的前几项
all_cols_unique_players[all_cols_unique_players > 1].dropna().shape[0] == 0  # 返回bool，表示是否无重复
```



### 进行替换值

```python
# Series
# 1.单值替换：将所有的参数1的值替换为参数2的值
print(ser_obj.replace(1, 100))
# 2.多值替换：将所有的参数1的值替换为参数2的值
print(ser_obj.replace([0,2,4], 100))
# 3.不同值做不同替换
print(ser_obj.replace({1:100, 3:300}))

# DataFrame
print(df_obj3)
# 参数1是字典：表示指定列需要替换的指定值， 参数2表示替换后的值
print(df_obj3.replace({"b":7}, 700))
# 参数只有一个字典，表示不同的值做不同的替换
print(df_obj3.replace({6: 600, 7:700}))
# 参数1是列表，表示需要替换的值，替换为参数2的值
print(df_obj3.replace([6, 7], 1000))
# 替换指定值
print(df_obj3.replace(6, 600))

# 示例
df_obj = df_obj[(df_obj["loan_status"] == "Fully Paid")|(df_obj["loan_status"] == "Charged Off")]
status_repace = {"loan_status":{"Fully Paid":1, "Charged Off": 0}}
df_obj = df_obj.replace(status_repace)
```

### 过滤列的数据类型

```python
object_columns_df = df_obj.select_dtype(include=["object"])
print(object_columns_df.iloc[0])
```

## 排序排名

- 排序

```python
# 按索引排序
ser_obj.sort_index()
# 默认为行索变动排序，排序规则为升序,
# axis=0表示列变动，ascending=False表示降序
df_obj.sort_index()
 
# 按值排序,默认升序
# by参数指定需要排序的列名(数字、字符串)
# 若有重名的列，不能参与排序
df_obj.sort_values(by='column_name', ascending=False))
```

- 排名

排名即ranking，从1开始一直到数据中有效数据的数量为止，返回数据在排序中的位置

```python
rrank = pd.Series([10,12,9,9,14,4,2,4,9,1])
rrank.rank()

rrank.rank(ascending=False)  # 逆序
```

当有多个数据值是一样的时候(如rrank对象中有3个值为9，2个值为4)，会出现排名相同的情况，这时可使用rank方法的参数method来处理

```
average	 默认选项，在相同排名中，为各个值平均分配排名
min		 使用整个相同排名的最小排名
max		 使用真个相同排名的最大排名
first	 按值在原始数据中的出现顺序分配排名
```

示例

```python
rrank.rank(method='first')
rrank.rank(method='max')
```

## 数据标签

pandas中可以把DataFrame实例对象的数据转化为Categorical类型的数据，以实现类似于一般统计分析软件中的值标签功能，便于分析结果的展示

categories

```python
student_profile = pd.DataFrame({
    'Name':{'Morgan Wang'},
    'Gender':[0],
    'Blood':['A'],
    'Height':[175]
})

# 转换类型
student_profile['Gender_Value'] = student_profile['Gender'].astype('category')
# 挂上标签
student_profile['Gender_Value'].cat.categories = ['Female', 'Male', 'Unconfirmed']

# 删除与增加值标签，或将类别设置为预定的尺度
student_profile['Gender_Value'].cat.se_categories = ['Male', 'Female', 'Unconfirmed']
```

Cut/qcut

```python
# 对数值类型数据分段标签
labels = ["{0}-{1}".fromat(i, i+10) for i in range(160, 200, 10)]   # 指定标签形式
student_profile['Height_Group'] = pd.cut(
  student_profile.Height, range(160, 205, 10), right=False, labels=labels
)

# 让数据按照排序分段标签
height_categories = ["vlow_height", "low_height", "mid_height", "high_height", "vhigh_height"]
players["heightclass"] = pd.qcut(
	players['height'], len(height_categories), height_categories
)
```

## 数据分组累计

对数据集进行分组，然后对每组进行统计分析。pandas能利用groupby虽然名字借用SQL的语言的命令，但是分组运算过程：split->apply->combine

```
1. 拆分：将DataFrame按照指定的键分割层内若干组
2. 应用：对每个组应用函数，通常是累计、转换或过滤函数
3. 合并：把每个分组的计算结果合并成一个输出数组
```

示例

```python
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=["key", "data"])

# groupby()可进行大多数常见的分割-应用-合并操作
df_gb = df.groupby('key')  # 传入分组的列名
print(df_gb)  # DataFrameGroupBy对象，在应用函数前不会计算
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x127955cf8>
print(df_gb.sum())  # 应用累计函数并生成结果
```

GroupBy对象是一种灵活的抽象类型，在大多数情境中，可认为其是DataFrame的集合，在底层解决所有难题。具有基本操作、aggregat、filter、transform、apply等操作。

### 基本操作

```python
# 数据
import seaborn as sns
planets = sns.load_dataset('planets')
print(planets.head(5))
#             method  number  orbital_period   mass  distance  year
# 0  Radial Velocity       1         269.300   7.10     77.40  2006
# 1  Radial Velocity       1         874.774   2.21     56.95  2008
# 2  Radial Velocity       1         763.000   2.60     19.84  2011
# 3  Radial Velocity       1         326.030  19.40    110.62  2007
# 4  Radial Velocity       1         516.220  10.50    119.47  2009
print(planets.shape)  # (1035, 6)
```

按列取值

```python
# 返回一个修改过的GroupBy对象
pl_gb = planets.groupby('method')
print(pl_gb['orbital_period'])  # <pandas.core.groupby.generic.SeriesGroupBy object at 0x11d57e748>
print(pl_gb['orbital_period'].median())  # 累计并生成结果
```

按组迭代

```python
# 每次迭代返回一个元组 (group_name, group_data)，可用于分组数据的具体运算
for (method, group) in pl_gb:
    print('{0:30s} shape={1}'.format(method, group.shape))
```

调用方法

```python
# 借助python的Aclassmethod，可以让任何不由GroupBy对象直接实现的方法应用到每一组，无论DataFrame或Series对象都同样适用
res = pl_gb['year'].describe()  # 对数据进行描述性统计
print(res)
```

### 高级操作

- 累计

内置函数

| 函数名     | 说明                        |
| ---------- | --------------------------- |
| count      | 分组中非NA值的数量          |
| sum        | 非NA值的和                  |
| mean       | 非NA值的平均值              |
| median     | 非NA值的算数中位数          |
| std/var    | 无偏(分母为n-1)标准差和方差 |
| min/max    | 非NA值的最小值和最大值      |
| prod       | 非NA值的积                  |
| first/last | 第一个和最后一个的非NA值    |

示例

```python
print(df_obj.groupby('key1').sum())
print(df_obj.groupby('key1').max())
print(df_obj.groupby('key1').min())
print(df_obj.groupby('key1').mean())
print(df_obj.groupby('key1').size())
print(df_obj.groupby('key1').count())
print(df_obj.groupby('key1').describe())
```

`aggregate()`支持更复杂的操作。函数调用中也可简写为`agg()`

```python
df_obj = pd.DataFrame(
    {
        "key1": list("abbaabba"),
        "key2": ["one", "two", "three", "two", "one", "three", "one", "two"],
        "data1": np.random.randint(-5, 10, 8),
        "data2": np.random.randint(-5, 10, 8),
    }, index=list("ABCDEFGH")
)
# agg()传入自定义函数 直接写函数名； 若是Pandas内置的函数，用字符串表示
func1 = lambda x: x.max() - x.min()
print(df_obj.groupby("key1").agg(func1))
print(df_obj.groupby("key1").agg("sum"))

# 可以同时应用多个聚合函数(默认使用函数名作为列名)， 也可以再修改列名
res = df_obj.groupby("key1").agg([("和", "sum"), ("平均数", "mean"), ("最大值", "max"), ("最小值", "min"), ("大小差值", func1)])
print(res)

# 可以对不同的列使用不同的聚合函数
dict_map = {
    "data1": ["sum", "mean"],
    "data2": ["max", "min", ("max-min", func1)]
}
print(df_obj.groupby("key1").agg(dict_map))

# 对数据是否只出现一次
all_cols_unique_players = df.groupby('playerShort').agg({
  col:'numique' for col in players_cols
})  # 将行中是否出现重复的playerShort数据以出现次数的形式展示
all_cols_unique_players[all_cols_unique_players > 1].dropna().head()  # 显示重复的前几项
all_cols_unique_players[all_cols_unique_players > 1].dropna().shape[0] == 0  # 返回bool，表示是否无重复
```

- 过滤

过滤操作可以按照分组的属性丢弃若干数据

```python
def filter_dunc(x):
    return x['data2'].std() > 4

print(df_obj.groupby('key1').std())
res = df_obj.groupby('key1').filter(filter_dunc)
print(res)
```

- 转换

转换操作会返回一个新的全量数据。数据经过转换后其形状与原来的输入数据一样

```python
key1_tf_df = df_obj[["data1", "data2"]].groupby(df_obj["key1"]).transform("sum").add_prefix("key1_sum_")
print(key1_tf_df)


key1_df_mean = df_obj.groupby("key1").transform(lambda x: x - x.mean())
print(key1_df_mean)
```

- 应用

通过apply方法，将自定义函数在各个分组上分别调用.

```python
def norm_by_data2(x):
    x['data1'] /= x['data2'].sum()
    return x

res = df_obj.groupby('key1').apply(norm_by_data2)
print(res)
```

### 设置分割的键

前面是适用列名分割DataFrame，还有其他的分组操作

- 列表/数组/Series/索引作为分组键

分组键可以是长度与DataFrame匹配的任意Series或列表

```python
L = [0, 1, 0, 1, 2, 0, 1, 2]
print(df_obj)
print(df_obj.groupby(L).sum())
```

- 字典/Series将索引映射到分组名称

```python
df_obj2 = df_obj.set_index('key1')
mapping = {'a': 'u', 'b': 'v'}
print(df_obj2)
print(df_obj2.groupby(mapping).sum())
```

- 任意python函数

将函数传入groupby，函数映射到索引，然后新的分组输出

```python
print(df_obj2.groupby(str.upper).mean())
```

- 多个有效键构成的列表

```python
print(df_obj2.groupby([str.upper, mapping]).mean())
```

## 数据合并

pandas提供了如下函数对pandas的数据对象进行合并

- np.concatenate

```python
import numpy as np
import pandas as pd
arr1 = np.random.randint(0, 10, (3, 4))
arr2 = np.random.randint(0, 10, (3, 4))

# 默认axis=0,行变动
print(np.concatenate([arr1, arr2]))
# axis=1, 列变动
print(np.concatenate([arr1, arr2], axis=1))
```

- pd.concat

注意指定轴方向，默认axis=0，join指定合并方式，默认为outer，Series合并时查看行索引有无重复

series

```python
#  index 没有重复的情况
ser_obj1 = pd.Series(np.random.randint(-5, 10, 4), index=range(4))
ser_obj2 = pd.Series(np.random.randint(-5, 10, 5), index=range(4, 9))
# 默认axis=0表示新增多行
print(pd.concat([ser_obj1, ser_obj2]))
# axis=1表示新增多列
print(pd.concat([ser_obj1, ser_obj2], axis=1))

# index有重复的情况
ser_obj1 = pd.Series(np.random.randint(-5, 10, 4), index=range(4))
ser_obj2 = pd.Series(np.random.randint(-5, 10, 5), index=range(5))
# 索引重复时， pandas在合并时会保留索引
print(pd.concat([ser_obj1, ser_obj2]))
# 同样列索引下的值有缺失时，用NaN表示
print(pd.concat([ser_obj1, ser_obj2], axis=1))  
# 捕捉索引重复的错误
try:
    pd.concat([ser_obj1, ser_obj2], verify_integrity=True)
except ValueError as e:
    print('ValueError:', e)
# 索引不重要时，可忽略重复索引，创建一个新的整数索引
pd.concat([ser_obj1, ser_obj2], ignore_index=True)
# 增加层级索引
pd.concat([ser_obj1, ser_obj2], keys=['ser1', 'ser2'])

# 默认join="outer"表示外连接，inner表示内连接
print(pd.concat([ser_obj1, ser_obj2], axis=1, join="inner"))
```

DataFrame

```python
# 多个DataFrame对象进行合并，注意索引是否一致(若索引不一致，合并没有任何意义)
df_obj1 = pd.DataFrame(np.random.randint(-5, 10, (3, 4)), index=list("ABC"), columns=list("abcd"))
df_obj2 = pd.DataFrame(np.random.randint(-5, 10, (4, 5)), index=list("ABCD"), columns=list("abcde"))
df_obj3 = pd.DataFrame(np.random.randint(-5, 10, (5, 6)), index=list("ABCDE"), columns=list("abcdef"))
# 默认axis=0表示共享列索引，新增多行
print(pd.concat([df_obj1, df_obj2, df_obj3]))
# axis=1表示共享行索引，新增多列
print(pd.concat([df_obj1, df_obj2, df_obj3], axis=1))
# join
print(pd.concat([df_obj1, df_obj2, df_obj3], join="inner"))
print(pd.concat([df_obj1, df_obj2, df_obj3], axis=1, join="inner")) 
```

- pd.append

```python
c1 = pd.DataFrame({
    'Name': {101: 'Zhang San'},
    'Subject': {101: 'Literature'},
    'Score': {101: 98}
})
c2 = pd.DataFrame({
    'Gender': {101: 'Male'}
})

# concat
print(pd.concat([c1, c2], axis=0))

# append
print(c1.append(c2))
```

- pd.merge

根据单个或多个键将不同DataFrame的行连接起来，类似数据库的连接操作

```python
df_obj1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                        'data1' : np.random.randint(0,10,7)})
df_obj2 = pd.DataFrame({'key': ['a', 'b', 'd'],
                        'data2' : np.random.randint(0,10,3)})
 
# 合并列
# 默认将重叠列的列名作为“外键”进行连接
print(pd.merge(df_obj1, df_obj2))
# on显示指定“外键”,尤其对于多个同名列
print(pd.merge(df_obj1, df_obj2, on='key'))
# left_on左侧数据的“外键”，right_on右侧数据的“外键”，how指定连接方式
df_obj3 = df_obj1.rename(columns={'key':'key1'})
df_obj4 = df_obj2.rename(columns={'key':'key2'})
# 默认how="inner"内连接，取交集数据
print(pd.merge(df_obj3, df_obj4, left_on='key1', right_on='key2'))
# how="outer"外连接，取并集结果
print(pd.merge(df_obj3, df_obj4, left_on="key1", right_on="key2", how="outer" ))
# how="left"左连接，保证左表的完整性
print(pd.merge(df_obj3, df_obj4, left_on="key1", right_on="key2", how="left" ))
# how="right"右连接，保证右表的完整性
print(pd.merge(df_obj3, df_obj4, left_on="key1", right_on="key2", how="right" ))
# 删除等价的外键列
print(pd.merge(df_obj3, df_obj4, left_on='key1', right_on='key2').drop('key2', axis=1))

# 处理重复的非外键列名，suffixes添加前缀，默认为_x, _y
# suffixes参数接收一个元组/列表， 两个元素分别表示左表和游标的后缀
df_obj5 = pd.DataFrame({"key": list("ababcacb"),
    "data": np.random.randint(-5, 10, 8)})
df_obj6 = pd.DataFrame({"key": list("cbdcdb"),
    "data": np.random.randint(-5, 10, 6)})
print(pd.merge(df_obj5, df_obj6, on="key", suffixes=["_left", "_right"]))

# 合并行索引
# left_index=True或right_index=True
print(pd.merge(df_obj5, df_obj6, left_index=True, right_index=True))

# 行列混用
df_obj7 = pd.DataFrame({"key" : list("ababcbac"),
    "data" : np.random.randint(-5, 10, 8)})
df_obj8 = pd.DataFrame({"data" : np.random.randint(-5, 10, 6)},
    index = list("cbdcdb"))
# left_on表示指定左表的key列做为外键，right_index=True表示使用右表行索引作为外键，进行关联
new_df = pd.merge(df_obj7, df_obj8, left_on="key", right_index=True, how="outer", suffixes=["_left", "_right"])
```

## 数据透视表

数据透视表奖每一列数据作为输入，输出将数据不断细分成多个维度累计信息的二维数据表。

函数

```python
pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All')

# 参数
data		指定为pandas中的DataFrame
index		对应数据透视表中的行
columns		对应数据透视表中的列
values		对应数据透视表中的值
aggfunc		指定汇总的函数，默认为mean函数
margins		指定分类汇总和总计
fill_value	指定填补的缺失值
dropna		指定是否包含所有数据项都是缺失值的列

crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, dropna=True, normalize=False)
# 不需要使用data参数指定对象，而是直接在index、columns、values等中直接指定分析对象
```

示例

```python
# 读取文件建表
storesales = pd.read_csv('./data/storesales.csv')
print(storesales.head(5))
#      id  store  method  orders  sales
# 0  1001      1       1      78  89000
# 1  1023      2       1      87  98000
# 2  1234      2       2      67  78500
# 3  1002      3       2      87  77500
# 4  1001      3       1      56  67990

p_table = pd.pivot_table(storesales, index=['store'], columns=['method'], values=['sales'], aggfunc=[sum], fill_value=0, margins=True)
print(p_table)
#            sum
#          sales
# method       1       2      All
# store
# 1       312643       0   312643
# 2       251667  176820   428487
# 3       146335  244903   391238
# 4            0  165010   165010
# All     710645  586733  1297378

p_ctab = pd.crosstab(storesales['store'], storesales['method'], values=storesales['sales'],aggfunc=[sum], margins=True)
print(p_ctab)
#              sum
# method         1         2      All
# store
# 1       312643.0       NaN   312643
# 2       251667.0  176820.0   428487
# 3       146335.0  244903.0   391238
# 4            NaN  165010.0   165010
# All     710645.0  586733.0  1297378
```

## 时间序列

时间序列在pandas中是索引比较特殊的Series或DataFrame，其最主要的特点是以时间戳(TimeStamp)为索引

### 创建时间序列

生成pandas中的时间序列，就需要生成以时间序列为主要特征的索引。pandas提供了类Timestamp，类Period以及to_timestamp,to_datetime,date_range,period_range等方法来创建或将其他数据类型转换为时间序列

```python
# 时间戳
pd.Timestamp('now')
pd.Timestamp('2017-07-06')
pd.Timestamp('2017-07-06 10')
pd.Timestamp('2017-07-06 10:15')

# 用时间戳转换为时间序列
dates = [pd.Timestamp('2017-07-05'), pd.Timestamp('2017-07-06'), pd.Timestamp('2017-07-07')
]
ts = pd.Series(np.random.randn(3), dates)
ts.index

# DateTimeIndex
index = pd.DatetimeIndex(['2013-02-11', '2014-03-12'])
data = pd.Series([1,2], index=index)

# date_range创建
dates = pd.date_range('2017-07-05', '2017-07-07')
# dates = pd.date_range('2017/07/01', periods=10, freq='M')  # M:月,D:天,H:小时
ts = ps.Series(np.random.randn(3), dates)
ts.index

# timedelta_range
index = pd.timedelta_range(0, periods=9, freq='2H30T')


# to_timedelta
index = pd.to_timedelta(np.arange(5), unit='s')

# to_datetime
# 将已有形如时间日期的数据转换为时间序列
jd_ts = jddf.set_index(pd.to_datetime(jddf['time']))
jd_ts.index
jd_ts.head()

# to_period
# 将时间戳转换为时间周期
ts = pd.Series(range(10), pd.date_range('07-10-16 8:00', periods=10, freq='H'))
ts_period = ts.to_period()

# Period类实例化
dates = [pd.Period('2017-07-05'), pd.Period('2017-07-06'), pd.Period('2017-07-07')]
ts = pd.Series(np.random.randn(3), dates)
ts.index 
```

### 索引与切片

与普通的Series和DataFrame等数据结果类似，但是可以按照时间戳或时间范围进行索引和切片

如按指定时间范围对数据进行切片，只需在索引号中传入可以解析成日期的字符串即可，这些字符串可以是表示年月日及其组合的内容

```python
# 指定索引
rng = pd.date_range('2016 Jul 1', periods = 10, freq = 'D')
pd.Series(range(len(rng)), index = rng)
periods = [pd.Period('2016-01'), pd.Period('2016-02'), pd.Period('2016-03')]
ts = pd.Series(np.random.randn(len(periods)), index = periods)


jd_ts['2017-02']  # 提取2017年2月份的数据
jd_ts['2017-02-10':'2017-02-20']  # 提取2017年2月10日至20日数据


sample = pd.Series(np.random.randn(20), index=pd.date_range(pd.datetime(2016,1,1),periods=20))
sample.truncate(before='2016-1-10')  # 2016-1-10之前的数据排除，保留当前和之后的
sample.truncate(after='2016-1-10')  # 2016-1-10之后的数据排除，保留之前的当前的
sample.truncate(after='2017-01-06')  # 2017年1月6日前的所有数据
sample.truncate(after='2017-01-20',before='2017-01-13')  # 2017年1月20日前,2017年1月13日后的所有数据
```

### 范围与偏移

有 时处于数据分析的需要会用到生成一定时期或时间范围内不同间隔的时间序列索引, `period_range`,`date_range`等函数可以满足这些需求

```python
pd.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None)

# 参数
start		用表示时间日期的字符串指定起始日期
end			用表示时间日期的字符串指定终止时间日期
periods		指定时间日期的个数
freq		指定时间日期的频率(间隔方式)
tz			指定时区
normalize	在生成日期范围之前，将开始/结束日期标准化为午夜
name		命名时间日期索引
closed		指定生成的时间日期索引是(默认None)/否包含start和end指定的时间日期
```

freq的参数值及其作用

| 参数值(偏移别名) | 功能                      | 参数值(偏移别名) | 功能         |
| ---------------- | ------------------------- | ---------------- | ------------ |
| B                | 工作日                    | QS               | 季度初       |
| C                | 自定义工作日              | BQS              | 季度初工作日 |
| D                | 日历日                    | A                | 年末         |
| W                | 周                        | BA               | 年末工作日   |
| M                | 月末                      | AS               | 年初         |
| SM               | 半月及月末(第15日及月末)  | BAS              | 年初工作日   |
| BM               | 月末工作日                | BH               | 工作小时     |
| CBM              | 自定义月末工作日          | H                | 小时         |
| MS               | 月初                      | T,min            | 分钟         |
| SMS              | 月初及月中(第1日及第15日) | S                | 秒           |
| BMS              | 月初工作日                | L,ms             | 毫秒         |
| CBMS             | 自定义月初工作日          | U,us             | 微秒         |
| Q                | 季度末                    | N                | 纳秒         |
| BQ               | 季度末工作日              | 用户自定义       | 实现特定功能 |

时间偏移后缀

| 偏移后缀                          | 功能                       | 可使用的偏移别名 |
| --------------------------------- | -------------------------- | ---------------- |
| -SUN,-MON,TUE,-WED,-THU,-FRI,-SAT | 分别表示以周几为频率的周   | W                |
| -DEC,-JAN,-FEB,-MAR,-APR,-MAY     | 分别表示以某月为年末的季度 | Q,BQ,QS,BQS      |
| -JUN,-JUL,-AUG,-SEP,-OCT,-NOV     | 分别表示以某月为年末的年   | A,BA,AS,BAS      |

示例

```python
pd.date_range(start='2017/07/07', periods=3, freq='M')

pd.date_range('2017/07/07', '2018/07/07', freq='BMS')

pd.date_range('2017/07/07', '2018/01/22', freq='W-WED')

# 自定义的对象进行设定得到自定义的时序索引
ts_offset = pd.tseries.offsets.Week(1)+pd.tseries.offsets.Hour(8)
pd.date_range('2017/07/07', periods=10, freq=ts_offset)
```

### 时间移动及运算

时序数据可以进行时间上的移动。即，沿着x轴将数据进行前移或后移，其索引保持不变。pandas中的Series和DataFrame都可通过shift方法来进行移动

```python
sample = jd_ts['2017-01-01':'2017-01-10'][['opening_price', 'closing_price']]
sample.shift(2)  # 如需向前移动，把数值修改为负数
sample.shift(-2, freq='1D')  # 使时序索引按天向前移动2日

# 不同索引的时间序列之间可以直接进行算数运算，运算时会自动按时间日期对齐
date = pd.date_range('2017/01/01', '2017/01/08', freq='D')
s1 = pd.DateFrame({
    'opening_price': np.random.randn(8),
    'closing_price':np.random.randn(8),
    index=date
})
s1 + sample  # 索引会自动将时序索引一致的值进行运算，不一致索引的值赋值为缺失值NaN
```

### 频率转换及重采样

- 频率转换

对pandas的时序对象可以采用asfreq方法对已有时序索引按照指定的频率重新进行索引，即频率转换。如sample对象是以工作日为时序索引的，可以把其转换为按照日历日或其他时间日期进行索引

```python
sample.asfreq(freq='D') 
```

在频率转换的过程中，由于索引发生了变化，原索引的数据会跟转换后的索引自动对齐

- 重采样

重采样(resample)也可将时间序列从一个频率转换到另一个频率，但在转换过程中，可以指定提取出原时序数据中的一些信息，其实质就是按照时间索引进行的数据分组。重采样主要有上采样(upsampling)和下采样(downsampling)两种。该两种方式也称数据处理过程中的上卷和下钻

pandas对象可采用resample方法对时序进行重采样，通过指定方法的参数fill_method进行采样，指定参数how进行下采样

```python
# 按照半天频率进行上采样或升采样，并制定缺失值按当日最后一个有效观测值来填充，即指定插值方式
# 插值方法：ffill为空值取前面的值，bfill为空值取后面的值，interpolate线性取值
sample.resample('12H', fill_method='ffill')

# 按照4天频率进行下采样或降采样，how可以指定采样过程中的运算函数(默认mean)
# ohlc分别表示时序初始值、最大值、最小值、时序终止的数据
sample.resample('4D', how='ohlc')

# 通过重采样提取公票交易周均开、收盘价信息
sample.groupby(lambda x: x.week).mean()

# 通过重采样提取股票交易月均开、收盘价信息
jd_ts[['opening_price', 'closing_price']].groupy(lambda x:x.month).mean()
```

### 滑动窗口

```python
df = pd.Series(np.random.randn(600), index=pd.date_range('7/1/2016', freq='D', periods=600))
r = df.rolling(window=10)  # 窗口大小，默认从左到右
r.max  # 最大值
r.median  # 中位数
r.std  # 标准差
r.skew  # 倾斜度
r.sum  # 总和
r.var  # 方差
r.mean()  # 均
```

##高性能

Pandas中的`eval(),query()`函数依赖于Numexpr程序包，提供了类C的速度。

- 设计动机：复合带数式

```python
# numpy向量化操作比for循环快
rng = np.random.RandomState(42)
x = rng.rand(10 ** 6)
y = rng.rand(10 ** 6)
time1 = time.process_time()
sum = x + y
time2 = time.process_time()
print('numpy向量化操作消耗时间:{}'.format(time2-time1))

# numpy复合代数式效率低
sum_np = np.fromiter((xi + yi for xi, yi in zip(x, y)), dtype=x.dtype, count=len(x))
mask = (x > 0.5) & (y < 0.5)  # 会将中间过程显式分配内存

# numexpr可以不为中间过程分配内存,计算更快速
sum_numexpr = numexpr.evaluate('sum_np')
mask_numexpr = numexpr.evaluate('mask')
```

- eval

```python
# 高性能计算
nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols)) for i in range(4))
time1 = time.process_time()
sum_n = df1 + df2 + df3 + df4
time2 = time.process_time()
sum_e = pd.eval('sum_n')
time3 = time.process_time()
print('pandas普通运算消耗times:{}'.format(time2 - time1))
print('eval普通运算消耗times:{}'.format(time3 - time2))

# 列间运算
rng = np.random.RandomState(42)
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
# 普通计算
res1 = (df['A'] + df['B']) / (df['C'] -1)
res2 = pd.eval('res1')
np.allclose(res1, res2)
# 通过列名
res3 = df.eval('(A + B) / (C - 1)')
np.allclose(res1, res3)  # True
# 新增列
df.eval('D = (A + B) / C', inplace=True)
print(df.head(5))
# 使用局部变量
column_mean = df.mean(1)
res1 = df['A'] + column_mean
res2 = df.eval('A + @column_mean')
np.allclose(res1, res2)  # True 
```

- query

```python
res1 = df[(df.A < 0.5) & (df.B < 0.5)]
res2 = pd.eval('res1')  # 不能使用df.eval()
print(np.allclose(res1, res2))  # True
res3 = df.query('A < 0.5 and B < 0.5')
print(np.allclose(res1, res3))  # True

# 支持局部变量引用
Cmean = df['C'].mean()
res1 = df[(df.A < Cmean) & (df.B < Cmean)]
res2 = df.query('A < @Cmean and B < @Cmean')
print(np.allclose(res1, res2))  # True
```

