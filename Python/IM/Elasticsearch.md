# Elasticsearch

[参考](http://www.ruanyifeng.com/blog/2017/08/elasticsearch.html)

## 概述

- 简介

[全文搜索](https://baike.baidu.com/item/全文搜索引擎)属于最常见的需求，开源的 [Elasticsearch](https://www.elastic.co/) （以下简称 Elastic）是目前全文搜索引擎的首选。

它可以快速地储存、搜索和分析海量数据。维基百科、Stack Overflow、Github 都采用它。

Elastic 的底层是开源库 [Lucene](https://lucene.apache.org/)。但是，你没法直接用 Lucene，必须自己写代码去调用它的接口。Elastic 是 Lucene 的封装，提供了 REST API 的操作接口，开箱即用。

它具有如下特点
```
- 是一个分布式支持REST API的搜索引擎。每个索引都使用可分配数量的完全分片，每个分片可以用多个副本。该搜索引擎可以在任何副本上操作
- 多集群、多种类型，支持多个索引，每个索引支持多种类型。索引级配置(分片数、索引存储)
- 支持多种API，比如HTTP RESTful API、Native Java API，所有的API都执行自动节点操作重新路由
- 面向文件，不需要前期模式定义，可以为每种类型定义模式以定制索引过程
- 可靠，支持长期持续性地异步写入
- 近实时搜索
- 建立在Lucene上，每个分片都是一个功能齐全的Lucene索引，Lucene的所有权利都可以通过简单的配置/插件轻松曝露
- 操作具有高度一致性，单个文档级操作是原子的、一致的、隔离的和耐用的。
```
> **中文分词说明**

搜索引擎在对数据构建索引时，需要进行分词处理。

分词是指将一句话拆解成**多个单字** 或 **词**，这些字或词便是这句话的关键词。

比如：我是中国人。分词后：`我`、`是`、`中`、`国`、`人`、`中国`等等都可以是这句话的关键字。

Elasticsearch 不支持对中文进行分词建立索引，需要配合扩展`elasticsearch-analysis-ik`来实现中文分词处理。

- 概念

**Node 与 Cluster**

Elastic 本质上是一个分布式数据库，允许多台服务器协同工作，每台服务器可以运行多个 Elastic 实例。

单个 Elastic 实例称为一个节点（node）。一组节点构成一个集群（cluster）。

**Index**

Elastic 会索引所有字段，经过处理后写入一个反向索引（Inverted Index）。查找数据的时候，直接查找该索引。

所以，Elastic 数据管理的顶层单位就叫做 Index（索引）。它是单个数据库的同义词。每个 Index （即数据库）的名字必须是小写。

下面的命令可以查看当前节点的所有 Index。

```bash
$ curl -X GET 'http://localhost:9200/_cat/indices?v'
```

**Document**

Index 里面单条的记录称为 Document（文档）。许多条 Document 构成了一个 Index。

Document 使用 JSON 格式表示，下面是一个例子。

```javascript
{
  "user": "张三",
  "title": "工程师",
  "desc": "数据库管理"
}
```

同一个 Index 里面的 Document，不要求有相同的结构（scheme），但是最好保持相同，这样有利于提高搜索效率。

**Type**

Document 可以分组，比如`weather`这个 Index 里面，可以按城市分组（北京和上海），也可以按气候分组（晴天和雨天）。这种分组就叫做 Type，它是虚拟的逻辑分组，用来过滤 Document。

不同的 Type 应该有相似的结构（schema），举例来说，`id`字段不能在这个组是字符串，在另一个组是数值。这是与关系型数据库的表的[一个区别](https://www.elastic.co/guide/en/elasticsearch/guide/current/mapping.html)。性质完全不同的数据（比如`products`和`logs`）应该存成两个 Index，而不是一个 Index 里面的两个 Type（虽然可以做到）。

下面的命令可以列出每个 Index 所包含的 Type。

```bash
$ curl 'localhost:9200/_mapping?pretty=true'
```

根据[规划](https://www.elastic.co/blog/index-type-parent-child-join-now-future-in-elasticsearch)，Elastic 6.x 版只允许每个 Index 包含一个 Type，7.x 版将会彻底移除 Type。

## 安装配置

### 安装

#### 原生安装

Elastic 需要 Java 8 环境。如果你的机器还没安装 Java，可以参考[这篇文章](https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-debian-8)，注意要保证环境变量`JAVA_HOME`正确设置。

- java

```shell
# 1.验证系统中java版本
java -version
# 2.若没有或版本过低，则去java官网下载jdk
# 3.提取文件
cd /path/to/download/
tar -zxvf jdk-8u181-linux-x64.gz
# 4.移动到/usr/local/jdk目录
sudo mkdir /usr/local/jdk
sudo mv jdk1.8.0_181 /usr/local/jdk
# 5.设置终端启动快捷路径
vim ~/.bashrc
export JAVA_HOME=/usr/local/jdk/jdk1.8.0_181
export PATH=$PATH:$JAVA_HOME/bin
 
source ~/.bashrc
# 6.验证是否ok
java -version
```

- elasticsearch

参考[官方文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)

ubuntu/macos

```shell
# 1.下载
# ubuntu
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.0-linux-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.0-linux-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.9.0-linux-x86_64.tar.gz.sha512 
tar -xzf elasticsearch-7.9.0-linux-x86_64.tar.gz
cd elasticsearch-7.9.0/ 
# macos
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.0-darwin-x86_64.tar.gz
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.0-darwin-x86_64.tar.gz.sha512
shasum -a 512 -c elasticsearch-7.9.0-darwin-x86_64.tar.gz.sha512 
tar -xzf elasticsearch-7.9.0-darwin-x86_64.tar.gz
cd elasticsearch-7.9.0/ 

# 2.启动
./bin/elasticsearch

# 3.修复报错信息
# 如果这时报错"max virtual memory areas vm.maxmapcount [65530] is too low"，要运行下面的命令。
sudo sysctl -w vm.max_map_count=262144

# 4.验证
# Elastic就会在默认的9200端口运行。这时，打开另一个命令行窗口，请求该端口，会得到说明信息。
curl localhost:9200  
```

macos方式二

```shell
brew tap elastic/tap  # 添加仓库
brew install elastic/tap/elasticsearch-full  # 安装
```

#### Docker安装

```shell
# 安装
sudo docker image pull delron/elasticsearch-ik:2.4.6-1.0
# 解压教学资料中本地镜像
sudo docker load -i elasticsearch-ik-2.4.6_docker.tar
# 配置
...
# 启动
sudo docker run -dti --name=elasticsearch --network=host -v /home/python/elasticsearch-2.4.6/config:/usr/share/elasticsearch/config delron/elasticsearch-ik:2.4.6-1.0
```

### 配置

[官方文档](https://www.elastic.co/guide/en/elasticsearch/reference/current/settings.html)

文件位置
```
启动文件位置:	$ES_HOME/bin/
日志文件位置：$ES_HOME/logs/
配置文件位置：$ES_HOME/config/elasticsearch.yml
数据位置：	$ES_HOME/data
插件文件位置：$ES_HOME/plugins
```
ip访问

```shell
# 默认只能本地访问，可暂时设置为0.0.0.0，线上设置固定ip
vim config/elasticsearch.yml
network.host: 0.0.0.0
```

允许跨域

```shell
# 默认不允许跨域，添加跨域配置
vim config/elasticsearch.yml
http.cors.enabled: true
http.cors.allow-origin: "*"
```

安装中文插件（[IK ](https://github.com/medcl/elasticsearch-analysis-ik),[SmartCN](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-smartcn.html),[Pinyin](https://github.com/medcl/elasticsearch-analysis-pinyin) ）

```shell
# 1.安装插件
# 安装ik中文插件
./bin/elasticsearch-plugin install https://github.com/medcl/elasticsearch-analysis-ik/releases/download/v6.3.0/elasticsearch-analysis-ik-6.3.0.zip
# 安装SmartCN
./bin/elasticsearch-plugin install analysis-smartcn
# 2.重启软件，自动加载
# 3.配置引擎
# 新建一个 Index，指定需要分词的字段。这一步根据数据结构而异。基本上，凡是需要搜索的中文字段，都要单独设置一下。
$ curl -X PUT 'localhost:9200/accounts' -d '
{
  "mappings": {
    "person": {
      "properties": {
        "user": {  # 字段
          "type": "text",  # 类型为中文文本
          "analyzer": "ik_max_word",  # 字段文本的分词器：插件ik提供
          "search_analyzer": "ik_max_word"  # 搜索词的分词器
        },
        "title": { 
          "type": "text",
          "analyzer": "ik_max_word",
          "search_analyzer": "ik_max_word"
        },
        "desc": {  
          "type": "text",
          "analyzer": "ik_max_word",
          "search_analyzer": "ik_max_word"
        }
      }
    }
  }
}'
```

删除插件

```shell
./bin/elasticsearch-plugin remove analysis-smartcn
```

### 使用

守护启动

```shell
# 要将Elasticsearch作为守护程序运行，请在命令行上指定-d，然后使用-p选项将进程ID记录在文件中
./bin/elasticsearch -d -p pid
```

启动配置

```shell
# 可以在命令行上使用-E语法在配置文件中指定的任何设置也可以如下指定
./bin/elasticsearch -d -Ecluster.name=my_cluster -Enode.name=node_1
```

终止运行

```shell
# 要关闭Elasticsearch，请杀死pid文件中记录的进程ID
pkill -F id
```

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

### 数据查询

- 返回所有记录

使用 GET 方法，直接请求`/Index/Type/_search`，就会返回所有记录。

```bash
$ curl 'localhost:9200/accounts/person/_search'

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
    # 在做任何操作之前，先进行嗅探
    sniff_on_start=True,
    # 节点没有响应时，进行刷新，重新连接
    sniff_on_connection_fail=True,
    # 每 60 秒刷新一次
    sniffer_timeout=60
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
    ＃打开SSL 
    use_ssl=True,
    ＃确保我们验证了SSL证书（默认关闭）
    verify_certs=True,
    ＃提供CA证书的路径
    ca_certs='/path/to/CA_certs',
    ＃PEM格式的SSL客户端证书
    client_cert='/path/to/clientcert.pem',
    ＃PEM格式的SSL客户端密钥
    client_key='/path/to/clientkey.pem'
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
