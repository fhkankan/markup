

# R语言概述

## 运行模式

```R
1.交互模式
在终端中输入r进入交互模式
注：windows中需设定环境变量
2.批处理模式
# 建立批处理文件,在文件z.R中输入
pdf("xh.pdf")
hist(rnorm(100))
dev.off()
#调用shell命令
$ R CMD BATCH z.R
```

## 注释

```
# 为注释标识
```

## 常用命令

```shell
# 读取文件内容并执行
> source("z.R")

# 当前工作目录
> getwd()

# 设定工作目录,并将目标目录作为参数
> setwd("R")

# 退出
> q()
```

## 帮助

```
# 获取帮助
> help(seq)
> ?seq

# 样例展示
> example(seq)

# 搜索相关
> help.search(seq)

# 批处理模式的帮助
R CMD command --help

```

##数据结构

```
1.向量
向量的模式必须属于同种数据结构，可以是字符模式或整数模式等，不可混用
注意：索引从1开始
2.字符串
实际上是字符模式的单元素向量
3.矩阵
概念：矩形的数值数组
技术：带行数和列数的向量
4.列表
值的容器，内容各项可属于不同的数据结构，用$可访问其中的各项
5.数据框
实际上是列表，只是列表中的每个组件是由“矩阵”数据的一列所构成的向量
6.类
```

## 运算符优先级

```
:: :::		access variables in a namespace
$ @			component / slot extraction
[ [[		indexing
^			exponentiation (right to left)
- +			unary minus and plus
:			sequence operator
%any%		special operators (including %% and %/%)
* /			multiply, divide
+ -			(binary) add, subtract
< > <= >= == !=	ordering and comparison
!			negation
& &&		and
| ||		or
~			as in formulae
-> ->>		rightwards assignment
<- <<-		assignment (right to left)
=			assignment (right to left)
?			help (unary and binary)
```



