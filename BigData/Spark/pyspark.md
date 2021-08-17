# pyspark

## 安装部署

安装

```
pip install pyspark
```

启动

```shell
# 在$SPARK_HOME/sbin目录下执行
./pyspark
```

## 使用

```python
spark = SparkSession.builder.appName('test').getOrCreate()
sc = spark.sparkContext
words = sc.textFile('file:///home/hadoop/tmp/word.txt') \
            .flatMap(lambda line: line.split(" ")) \
            .map(lambda x: (x, 1)) \
            .reduceByKey(lambda a, b: a + b).collect()
            
"""
[('python', 2), ('hadoop', 1), ('bc', 1), ('foo', 4), ('test', 2), ('bar', 2), ('quux', 2), ('abc', 2), ('ab', 1), ('you', 1), ('ac', 1), ('bec', 1), ('by', 1), ('see', 1), ('labs', 2), ('me', 1), ('welcome', 1)]
"""
```

