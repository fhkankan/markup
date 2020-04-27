# Spark

是一个开源的、通用的并行计算与分布式计算框架。特点：基于内存计算，适合迭代计算，兼容多种应用场景，同时还兼容Hadoop生态系统中的组建，并且有很强的容错性。设计目的是全栈式解决批处理、结构化数据查询、流计算、图计算和机器学习等业务和应用，适用于需要多次操作特定数据集的应用场合，需要反复操作的次数越多，所需读取的数据量越大，效率提升越大

集成了Sark SQL(分布式SQL查询引擎，提供一个DataFrame编程抽象)、Spark Streaming(把流式计算分解成一系列短小的批处理计算)、MLib(提供机器学习服务)、GraphX(提供图计算服务)、SparkR(R on Spark)等子框架，为不同应用领域的从业者提供了全新的大数据处理方式。

为了适应迭代计算，Spark把经常被重用的数据缓存到内存汇总以提高数据读取和操作速度，比Hadoop快近百倍，支持Java,Scala,Python, R等多种语言。除map和reduce之外，Spark还支持filter,foreach,reduceByKey,aggregate以及SQL查询、流式查询等

## linux管道实现WordCount

Mapper.py

```python
import sys
for line in sys.stdin:
  	ls = line.split()
    for word in ls:
      	iflen(word.strip()) != 0:
          	print(word + ',' + str(1))
```

Reducer.py

```python
import sys 
word_dict = {}
for line in sys.stdin:
		ls = line.split(',')
		word_dict. = setdefault(ls[0], 0)
		word_dict[ls[0] += int(ls[1])]
```

wordcount.input

```
hello
world
hello world
hi world
```

liunx终端命令

```shell
cat wordcount.input | python mapper.py | python reducer.py | sort -k 2r 
```

## pySpark实现WordCount

```python
import sys 
from operator import add 
from pyspark import SparkContext

sc = SparkContext
lines = sc.textFile("stormfswords.csv")
counts = line.flatMap(lambda x:x.split(','))\
						 .map(lambda x: (x, 1)) \
  					 .reduceByKey(add)

output = counts.collect()
output = filter(lambda x: not x[0].isnumeeric(), sorted(output, key=lambda x:x[1], reverse=True))

for (word, count) in output[:10]:
  	print('%s: %i' % (word, count))
    
sc.stop()
```

