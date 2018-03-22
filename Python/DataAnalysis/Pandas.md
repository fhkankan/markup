# Pandas

Pandas的名称来自于面板数据（panel data）和Python数据分析（data analysis）。

Pandas是一个强大的分析结构化数据的工具集，基于NumPy构建，提供了 **高级数据结构** 和 **数据操作工具**，它是使Python成为强大而高效的数据分析环境的重要因素之一。

- 一个强大的分析和操作大型结构化数据集所需的工具集
- 基础是NumPy，提供了高性能矩阵的运算
- 提供了大量能够快速便捷地处理数据的函数和方法
- 应用于数据挖掘，数据分析
- 提供数据清洗功能

<http://pandas.pydata.org>

## 数据结构

Pandas有两个最主要也是最重要的数据结构： **Series** 和 **DataFrame**

### Series

Series是一种类似于一维数组的 **对象**，由一组数据（各种NumPy数据类型）以及一组与之对应的索引（数据标签）组成。

- 类似一维数组的对象
- 由数据和索引组成
  - 索引(index)在左，数据(values)在右
  - 索引是自动创建的

```
# 通过list创建
ser_obj = pd.Series(range(10))

# 通过ndarry创建
ser_obj = pd.Series(np.arange(10))

# 通过index参数指定行索引创建
ser_obj3 = pd.Series(data=range(-3, 3), index=list("ABCDEF"), dtype=np.float64, name="测试数据")

# 通过dict创建
ser_obj3 = pd.Series({"age": 18, "id": 1001, "name": "itcast"})
print(ser_obj3["age"])


# 获取索引数据
ser_obj3.index
ser_obj3.values
# 获取名字
对象名：ser_obj.name
对象索引名：ser_obj.index.name
```

### DataFrame

DataFrame是一个表格型的数据结构，它含有一组有序的列，每列可以是不同类型的值。DataFrame既有行索引也有列索引，它可以被看做是由Series组成的字典（共用同一个索引），数据是以二维结构存放的。

- 类似多维数组/表格数据 (如，excel, R中的data.frame)
- 每列数据可以是不同的类型
- 索引包括列索引和行索引

```




```











## 索引操作

## 算数运算与数据对齐

## 数据清洗

## 函数应用

## 排序处理

## 层级索引与数据重构

## 统计计算和描述

## 分组与聚合

