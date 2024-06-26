# 使用方法

## RESTFull API

### 索引操作

- 新建 Index

可以直接向 Elastic 服务器发出 PUT 请求。下面的例子是新建一个名叫`weather`的 Index。

```bash
$ curl -X PUT 'localhost:9200/weather'
```

服务器返回一个 JSON 对象，里面的`acknowledged`字段表示操作成功。

```javascript
{
  "acknowledged":true,
  "shards_acknowledged":true
}
```

- 删除Index

然后，我们发出 DELETE 请求，删除这个 Index。

```bash
$ curl -X DELETE 'localhost:9200/weather'
```

### 数据操作

- 新增记录

向指定的 /Index/Type 发送 PUT 请求，就可以在 Index 里面新增一条记录。比如，向`/accounts/person`发送请求，就可以新增一条人员记录。

```bash
$ curl -X PUT 'localhost:9200/accounts/person/1' -d '
{
  "user": "张三",
  "title": "工程师",
  "desc": "数据库管理"
}' 
```

服务器返回的 JSON 对象，会给出 Index、Type、Id、Version 等信息。

```javascript
{
  "_index":"accounts",
  "_type":"person",
  "_id":"1",
  "_version":1,
  "result":"created",
  "_shards":{"total":2,"successful":1,"failed":0},
  "created":true
}
```

如果你仔细看，会发现请求路径是`/accounts/person/1`，最后的`1`是该条记录的 Id。它不一定是数字，任意字符串（比如`abc`）都可以。

新增记录的时候，也可以不指定 Id，这时要改成 POST 请求。

```bash
$ curl -X POST 'localhost:9200/accounts/person' -d '
{
  "user": "李四",
  "title": "工程师",
  "desc": "系统管理"
}'
```

上面代码中，向`/accounts/person`发出一个 POST 请求，添加一个记录。这时，服务器返回的 JSON 对象里面，`_id`字段就是一个随机字符串。

```javascript
{
  "_index":"accounts",
  "_type":"person",
  "_id":"AV3qGfrC6jMbsbXb6k1p",
  "_version":1,
  "result":"created",
  "_shards":{"total":2,"successful":1,"failed":0},
  "created":true
}
```

注意，如果没有先创建 Index（这个例子是`accounts`），直接执行上面的命令，Elastic 也不会报错，而是直接生成指定的 Index。所以，打字的时候要小心，不要写错 Index 的名称。

- 查看记录

向`/Index/Type/Id`发出 GET 请求，就可以查看这条记录。

```bash
$ curl 'localhost:9200/accounts/person/1?pretty=true'
```

上面代码请求查看`/accounts/person/1`这条记录，URL 的参数`pretty=true`表示以易读的格式返回。

返回的数据中，`found`字段表示查询成功，`_source`字段返回原始记录。

```javascript
{
  "_index" : "accounts",
  "_type" : "person",
  "_id" : "1",
  "_version" : 1,
  "found" : true,
  "_source" : {
    "user" : "张三",
    "title" : "工程师",
    "desc" : "数据库管理"
  }
}
```

如果 Id 不正确，就查不到数据，`found`字段就是`false`。

```bash
$ curl 'localhost:9200/weather/beijing/abc?pretty=true'

{
  "_index" : "accounts",
  "_type" : "person",
  "_id" : "abc",
  "found" : false
}
```

- 删除记录

删除记录就是发出 DELETE 请求。

```bash
$ curl -X DELETE 'localhost:9200/accounts/person/1'
```

这里先不要删除这条记录，后面还要用到。

- 更新记录

更新记录就是使用 PUT 请求，重新发送一次数据。

```bash
$ curl -X PUT 'localhost:9200/accounts/person/1' -d '

# 结果
{
    "user" : "张三",
    "title" : "工程师",
    "desc" : "数据库管理，软件开发"
}' 

{
  "_index":"accounts",
  "_type":"person",
  "_id":"1",
  "_version":2,
  "result":"updated",
  "_shards":{"total":2,"successful":1,"failed":0},
  "created":false
}
```

上面代码中，我们将原始数据从"数据库管理"改成"数据库管理，软件开发"。 返回结果里面，有几个字段发生了变化。

```bash
"_version" : 2,
"result" : "updated",
"created" : false
```

可以看到，记录的 Id 没变，但是版本（version）从`1`变成`2`，操作类型（result）从`created`变成`updated`，`created`字段变成`false`，因为这次不是新建记录。

### 简单查询

根据文档ID

```shell
# 查询所有数据
curl -X GET 127.0.0.1:9200/articles/article/1 
# 查询source中指定字段
curl -X GET 127.0.0.1:9200/articles/article/1?_source=title,user_id
# 不查询source中的字段
curl -X GET 127.0.0.1:9200/articles/article/1?_source=false
```

查询所有

```shell
$ curl 'localhost:9200/accounts/person/_search'

# 结果
{
  "took":2,  # 该操作的耗时（单位为毫秒）
  "timed_out":false,  # 是否超时
  "_shards":{"total":5,"successful":5,"failed":0},
  "hits":{  # 命中的记录
    "total":2,   # 返回记录数，本例是2条。
    "max_score":1.0,  # 最高的匹配程度，本例是`1.0`。
    "hits":[  # 返回的记录组成的数组
      {
        "_index":"accounts",
        "_type":"person",
        "_id":"AV3qGfrC6jMbsbXb6k1p",
        "_score":1.0,  # 匹配的程序，默认是按照这个字段降序排列
        "_source": {
          "user": "李四",
          "title": "工程师",
          "desc": "系统管理"
        }
      },
      {
        "_index":"accounts",
        "_type":"person",
        "_id":"1",
        "_score":1.0,
        "_source": {
          "user" : "张三",
          "title" : "工程师",
          "desc" : "数据库管理，软件开发"
        }
      }
    ]
  }
}
```

分页

```shell
# from 起始 size 每页数量
curl -X GET 127.0.0.1:9200/articles/article/_search?_source=title,user_id\&size=3

curl -X GET 127.0.0.1:9200/articles/article/_search?_source=title,user_id\&size=3\&from=10
```

全文检索

```shell
# %20 表示空格
curl -X GET 127.0.0.1:9200/articles/article/_search?q=content:python%20web\&_source=title,article_id\&pretty

curl -X GET 127.0.0.1:9200/articles/article/_search?q=title:python%20web,content:python%20web\&_source=title,article_id\&pretty

curl -X GET 127.0.0.1:9200/articles/article/_search?q=_all:python%20web\&_source=title,article_id\&pretty
```

### 高级查询

- 全文搜索

Elastic 的查询非常特别，使用自己的[查询语法](https://www.elastic.co/guide/en/elasticsearch/reference/5.5/query-dsl.html)，要求 GET 请求带有数据体。

```bash
$ curl 'localhost:9200/accounts/person/_search'  -d '
{
  "query" : { "match" : { "desc" : "软件" }}
}'
```

上面代码使用 [Match 查询](https://www.elastic.co/guide/en/elasticsearch/reference/5.5/query-dsl-match-query.html)，指定的匹配条件是`desc`字段里面包含"软件"这个词。返回结果如下。

```javascript
{
  "took":3,
  "timed_out":false,
  "_shards":{"total":5,"successful":5,"failed":0},
  "hits":{
    "total":1,
    "max_score":0.28582606,
    "hits":[
      {
        "_index":"accounts",
        "_type":"person",
        "_id":"1",
        "_score":0.28582606,
        "_source": {
          "user" : "张三",
          "title" : "工程师",
          "desc" : "数据库管理，软件开发"
        }
      }
    ]
  }
}
```

Elastic 默认一次返回10条结果，可以通过`size`字段改变这个设置。

```bash
$ curl 'localhost:9200/accounts/person/_search'  -d '
{
  "query" : { "match" : { "desc" : "管理" }},
  "size": 1
}'
```

上面代码指定，每次只返回一条结果。

还可以通过`from`字段，指定位移。

```bash
$ curl 'localhost:9200/accounts/person/_search'  -d '
{
  "query" : { "match" : { "desc" : "管理" }},
  "from": 1,
  "size": 1
}'
```

上面代码指定，从位置1开始（默认是从位置0开始），只返回一条结果。

- 逻辑运算

如果有多个搜索关键字， Elastic 认为它们是`or`关系。

```bash
$ curl 'localhost:9200/accounts/person/_search'  -d '
{
  "query" : { "match" : { "desc" : "软件 系统" }}
}'
```

上面代码搜索的是`软件 or 系统`。

如果要执行多个关键词的`and`搜索，必须使用[布尔查询](https://www.elastic.co/guide/en/elasticsearch/reference/5.5/query-dsl-bool-query.html)。

```bash
$ curl 'localhost:9200/accounts/person/_search'  -d '
{
  "query": {
    "bool": {
      "must": [
        { "match": { "desc": "软件" } },
        { "match": { "desc": "系统" } }
      ]
    }
  }
}'
```

全文检索 match

```shell
curl -X GET 127.0.0.1:9200/articles/article/_search -d'
{
    "query" : {
        "match" : {
            "title" : "python web"
        }
    }
}'

curl -X GET 127.0.0.1:9200/articles/article/_search?pretty -d'
{
     "from": 0,
     "size": 5,
     "_source": ["article_id","title"],
     "query" : {
         "match" : {
             "title" : "python web"
          }
      }
}'

curl -X GET 127.0.0.1:9200/articles/article/_search?pretty -d'
{
    "from": 0,
    "size": 5,
    "_source": ["article_id","title"],
    "query" : {
        "match" : {
            "_all" : "python web 编程"
        }
    }
}'
```

短语搜索 match_phrase

```shell
curl -X GET 127.0.0.1:9200/articles/article/_search?pretty -d'
{
    "size": 5,
    "_source": ["article_id","title"],
    "query" : {
        "match_phrase" : {
            "_all" : "python web"
        }
    }
}'
```

精确查找 term

```shell
curl -X GET 127.0.0.1:9200/articles/article/_search?pretty -d'
{
    "size": 5,
    "_source": ["article_id","title", "user_id"],
    "query" : {
        "term" : {
            "user_id" : 1
        }
    }
}'
```

范围查找 range

```shell
curl -X GET 127.0.0.1:9200/articles/article/_search?pretty -d'
{
    "size": 5,
    "_source": ["article_id","title", "user_id"],
    "query" : {
        "range" : {
            "article_id": { 
                "gte": 3,
                "lte": 5
            }
        }
    }
}'
```

高亮搜索 highlight

```shell
curl -X GET 127.0.0.1:9200/articles/article/_search?pretty -d '
{
    "size":2,
    "_source": ["article_id", "title", "user_id"],
    "query": {
        "match": {
             "title": "python web 编程"
         }
     },
     "highlight":{
          "fields": {
              "title": {}
          }
     }
}
'
```

组合查询

```shell
# must		文档 *必须* 匹配这些条件才能被包含进来。
# must_not	文档 *必须不* 匹配这些条件才能被包含进来。
# should	如果满足这些语句中的任意语句，将增加 `_score` ，否则，无任何影响。它们主要用于修正每个文档的相关性得分。
# filter	*必须* 匹配，但它以不评分、过滤模式来进行。这些语句对评分没有贡献，只是根据过滤标准来排除或包含文档。

curl -X GET 127.0.0.1:9200/articles/article/_search?pretty -d '
{
  "_source": ["title", "user_id"],
  "query": {
      "bool": {
          "must": {
              "match": {
                  "title": "python web"
              }
          },
          "filter": {
              "term": {
                  "user_id": 2
              }
          }
      }
  }
}
'
```

排序

```shell
curl -X GET 127.0.0.1:9200/articles/article/_search?pretty -d'
{
    "size": 5,
    "_source": ["article_id","title"],
    "query" : {
        "match" : {
            "_all" : "python web"
        }
    },
    "sort": [
        { "create_time":  { "order": "desc" }},
        { "_score": { "order": "desc" }}
    ]
}'
```

boost 提升权重，优化排序

```shell
curl -X GET 127.0.0.1:9200/articles/article/_search?pretty -d'
{
    "size": 5,
    "_source": ["article_id","title"],
    "query" : {
        "match" : {
            "title" : {
                "query": "python web",
                "boost": 4
            }
        }
    }
}'
```

### 联想提示

- 拼写纠错

对于已经建立的articles索引库，elasticsearch还提供了一种查询模式，suggest建议查询模式

```shell
curl 127.0.0.1:9200/articles/article/_search?pretty -d '
{
    "from": 0,
    "size": 10,
    "_source": false,
    "suggest": {
        "text": "phtyon web",
        "word-phrase": {
            "phrase": {
                "field": "_all",
                "size": 1
            }
        }
    }
}'
```

当我们输入错误的关键词`phtyon web`时，es可以提供根据索引库数据得出的正确拼写`python web`

- 自动补全

使用elasticsearch提供的自动补全功能，因为文档的类型映射要特殊设置，所以原先建立的文章索引库不能用于自动补全，需要再建立一个自动补全的索引库

建立索引

```shell
curl -X PUT 127.0.0.1:9200/completions -H 'Content-Type: application/json' -d'
{
   "settings" : {
       "index": {
           "number_of_shards" : 3,
           "number_of_replicas" : 1
       }
   }
}
'

curl -X PUT 127.0.0.1:9200/completions/_mapping/words -H 'Content-Type: application/json' -d'
{
     "words": {
          "properties": {
              "suggest": {
                  "type": "completion",
                  "analyzer": "ik_max_word"
              }
          }
     }
}
'
```

建议查询

```shell
curl 127.0.0.1:9200/completions/words/_search?pretty -d '
{
    "suggest": {
        "title-suggest" : {
            "prefix" : "pyth", 
            "completion" : { 
                "field" : "suggest" 
            }
        }
    }
}
'

curl 127.0.0.1:9200/completions/words/_search?pretty -d '
{
    "suggest": {
        "title-suggest" : {
            "prefix" : "python web", 
            "completion" : { 
                "field" : "suggest" 
            }
        }
    }
}
'
```

### 第三方插件

使用独立的第三方插件[elasticsearch-head](https://github.com/mobz/elasticsearch-head)可以可视化地快速实现对elasticsearch的使用

安装有独立服务式、docker环境、chrome扩展等多种方式

```shell
# 独立服务式
git clone git://github.com/mobz/elasticsearch-head.git
cd elasticsearch-head
npm install
npm run start
open http://localhost:9100/
```

连接elasticsearch

```python
# 默认情况下，elasticsearch在elasticsearch头连接的9200端口上公开http rest API。
# 当不作为Chrome扩展或elasticsearch的插件运行时（从第5版甚至不可能），必须在elasticsearch中启用CORS，否则您的浏览器将因违反同源策略而拒绝elasticsearch head的请求。
# 1.允许跨域
vim config/elasticsearch.yml
http.cors.enabled: true
http.cors.allow-origin: "*"
# 2.重启服务
```

## elasticsearch

[参考](https://www.jianshu.com/p/462007422e65) [参考](https://www.cnblogs.com/xiao987334176/p/10130712.html#autoid-1-1-0)

[elasticsearch](https://github.com/elastic/elasticsearch-py)是python对elasticsearch的客户端

```shell
pip install elasticsearch
```

### 连接集群

注意区分版本

```
from elasticsearch5 import Elasticsearch
```

指定连接

```python
es = Elasticsearch(
    ['172.16.153.129:9200'],
    # 认证信息
    # http_auth=('elastic', 'changeme')
)
```

动态连接

```python
es = Elasticsearch(
    ['esnode1:port', 'esnode2:port'],
    sniff_on_start=True,  # 在做任何操作之前，先进行嗅探
    sniff_on_connection_fail=True,  # 节点没有响应时，进行刷新，重新连接
    sniffer_timeout=60  # 每 60 秒刷新一次
)
```

对不同的节点，赋予不同的参数

```python
es = Elasticsearch([
    {'host': 'localhost'},
    {'host': 'othernode', 'port': 443, 'url_prefix': 'es', 'use_ssl': True},
])
```

假如使用了 ssl

```python
es = Elasticsearch(
    ['localhost:443', 'other_host:443'],
    use_ssl=True,  # 打开SSL 
    verify_certs=True,  # 确保我们验证了SSL证书（默认关闭）
    ca_certs='/path/to/CA_certs',  # 提供CA证书的路径
    client_cert='/path/to/clientcert.pem',  # PEM格式的SSL客户端证书   
    client_key='/path/to/clientkey.pem'  # PEM格式的SSL客户端密钥
)
```

### 集群信息

获取集群信息

```python
# 测试集群是否启动
es.ping()  # True
# 获取集群基本信息
es.info()
# 获取集群的健康状态信息
es.cluster.health()
# 获取当前连接的集群节点信息
es.cluster.client.info()
# 获取集群目前所有的索引
print(es.cat.indices())
# 获取集群的更多信息
es.cluster.stats()
```

利用实例的 cat 属性得到更简单易读的信息

```python
es.cat.health()
es.cat.master()
es.cat.nodes()
es.cat.indices()
es.cat.count()
es.cat.plugins()
es.cat.templates()
```

任务

```python
es.tasks.get()
es.tasks.list()
```

### 数据操作

```python
# 插入数据,index，doc_type名称可以自定义，id可以根据需求赋值,body为内容
es.index(index="my_index",doc_type="test_type",id=0,body={"name":"python","addr":"深圳"})
es.index(index="my_index",doc_type="test_type",id=1,body={"name":"python","addr":"深圳"})
 
#同样是插入数据，create() 方法需要我们指定 id 字段来唯一标识该条数据，而 index() 方法则不需要，如果不指定 id，会自动生成一个 id
es.create(index="my_index",doc_type="test_type",id=1,body={"name":"python","addr":"深圳"})
 
#删除指定的index、type、id的文档
es.delete(index='indexName', doc_type='typeName', id=1)
 
#删除index
es.indices.delete(index='news', ignore=[400, 404])
 
query = {'query': {'match_all': {}}}# 查找所有文档
query1 = {'query': {'match': {'sex': 'famale'}}}# 删除性别为女性的所有文档
query2 = {'query': {'range': {'age': {'lt': 11}}}}# 删除年龄小于11的所有文档
query3 = {'query': {'term': {'name': 'jack'}}}# 查找名字叫做jack的所有文档
 
 
#删除所有文档
es.delete_by_query(index="my_index",doc_type="test_type",body=query)
 
#get：获取指定index、type、id所对应的文档
es.get(index="my_index",doc_type="test_type",id=1)
 
#search：查询满足条件的所有文档，没有id属性，且index，type和body均可为None
result = es.search(index="my_index",doc_type="test_type",body=query)
print(result['hits']['hits'][0])# 返回第一个文档的内容
 
#update：更新指定index、type、id所对应的文档
#更新的主要点：
#1. 需要指定 id
#2. body={"doc": <xxxx>} , 这个doc是必须的
es.update(index="my_index",doc_type="test_type",id=1,body={"doc":{"name":"python1","addr":"深圳1"}})
```

### 简单查询

```python
es.search(index='logstash-2015.08.20', q='http_status_code:5* AND server_name:"web1"', from_='124119')
# 常用参数
index - 索引名
q - 查询指定匹配 使用Lucene查询语法
from_ - 查询起始点  默认0
doc_type - 文档类型
size - 指定查询条数 默认10
field - 指定字段 逗号分隔
sort - 排序  字段：asc/desc
body - 使用Query DSL
scroll - 滚动查询
```

发送查询请求

```bash
es = Elasticsearch(
        ['172.16.153.129:9200']
    )
    
res = es.search(
    index="logstash-2017.11.14", # 索引名
    body={             # 请求体
      "query": {       # 关键字，把查询语句给 query
          "bool": {    # 关键字，表示使用 filter 查询，没有匹配度
                "must": [      # 表示里面的条件必须匹配，多个匹配元素可以放在列表里
                    {
                        "match": {  # 关键字，表示需要匹配的元素
                            "TransId": '06100021650016153'   # TransId 是字段名， 06100021650016153 是此字段需要匹配到的值
                        }
                    },
                    {
                        "match": {
                            "Ds": '2017-05-06'
                        }
                    },
                    {
                        "match": {
                            "Gy": '2012020235'
                        }
                    }, ],
                 "must_not": {   # 关键字，表示查询的结果里必须不匹配里面的元素
                        "match": {  # 关键字
                            "message": "M("    # message 字段名，这个字段的值一般是查询到的结果内容体。这里的意思是，返回的结果里不能包含特殊字符 'M('
                        }
                 }
            }
        },
        
        # 下面是对返回的结果继续排序
        "sort": [{"@timestamp": {"order": "desc"}}],
        "from": start,  # 从匹配到的结果中的第几条数据开始返回，值是匹配到的数据的下标，从 0 开始
        "size": size    # 返回多少条数据
      }
)
```

得到返回结果的总条数

```bash
total = res['hits']['total']
```

循环返回的结果，得到想要的内容

```bash
res_dict={}
for hit in res['hits']['hits']:
    log_time = "%s|%s" % (hit['_source']['Ds'], hit['_source']['Us'])
    res_dict[log_time] = "%s|%s|%s|%s" % (hit['_source']['beat']['hostname'],hit['_source']['FileName'], hit['_source']['FileNum'],hit['_source']['Messager'])
```

实例查询7天之内的流水号为：06100021650016153 的日志信息

```python
query_body={
    'bool': {
        'must_not': {'match': {'message': 'M('}}, 
        'must': [
            {'match': {'TransId': '06100021650016153'}}, 
            {'range': {'@timestamp': {'gte': u'now-7d', 'lte': 'now'}}}
        ]
    }
}

res = es.search(
    index='logstash-2017.11.14',
    body={
        "query": query_body,
        "sort":[{"@timestamp": {"order": "desc"}}]})
    }
```

### 高级查询

可以借助[elasticsearch_dsl](https://github.com/elastic/elasticsearch-dsl-py/blob/master/docs/index.rst) 做更方便且更高级的查询.

简单查询

```python
from elasticsearch import Elasticsearch
client = Elasticsearch()

response = client.search(
    index="my-index",
    body={
      "query": {
        "filtered": {
          "query": {
            "bool": {
              "must": [{"match": {"title": "python"}}],
              "must_not": [{"match": {"description": "beta"}}]
            }
          },
          "filter": {"term": {"category": "search"}}
        }
      },
      "aggs" : {
        "per_tag": {
          "terms": {"field": "tags"},
          "aggs": {
            "max_lines": {"max": {"field": "lines"}}
          }
        }
      }
    }
)

for hit in response['hits']['hits']:
    print(hit['_score'], hit['_source']['title'])

for tag in response['aggregations']['per_tag']['buckets']:
    print(tag['key'], tag['max_lines']['value'])
```

高级查询

```python
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

client = Elasticsearch()

s = Search(using=client, index="my-index") \
    .filter("term", category="search") \
    .query("match", title="python")   \
    .exclude("match", description="beta")

s.aggs.bucket('per_tag', 'terms', field='tags') \
    .metric('max_lines', 'max', field='lines')

response = s.execute()

for hit in response:
    print(hit.meta.score, hit.title)

for tag in response.aggregations.per_tag.buckets:
    print(tag.key, tag.max_lines.value)
```

### 案例

```python
import csv
from elasticsearch import Elasticsearch

es = Elasticsearch(hosts="218.22.29.213", port=9200, timeout=200)
# 1. 先借助游标,将所有结果数据存储到内存中
# 2. 然后将内存中的结果数据写入到磁盘,也就是文件中
query1 = {
    "size": 100
}
query = es.search(index="hefei_camera_info", doc_type="info", scroll='5m', body=query1)
value = query["hits"]["hits"]

# es查询出的结果第一页
results = query['hits']['hits']
# es查询出的结果总量
total = query['hits']['total']
# 游标用于输出es查询出的所有结果
scroll_id = query['_scroll_id']
# 在发送查询请求的时候,就告诉ES需要使用游标,并定义每次返回数据量的大小
# 定义一个list变量results用来存储数据结果,在代码中,可以另其为空list,即results=[],也可以先将返回结果
# 的第一页存尽进来, 即results = query['hits']['hits']
# 对于所有二级果数据写个分页加载到内存变量的循环
for i in range(0, int(total / 100) + 1):
    # scroll参数必须制定否则会报错
    query_scroll = es.scroll(scroll_id=scroll_id, scroll="5m")['hits']['hits']
    results += query_scroll
with open("D://ml/data.csv", 'w', newline='', encoding="gbk") as flow:
    # 获取_source 下的所有字段名
    names = results[0]['_source'].keys()
    csv_writer = csv.writer(flow)
    csv_writer.writerow(names)
    for res in results:
        csv_writer.writerow(res['_source'].values())
print("done!")
```
