

# Elasticsearch

[官方文档](https://www.elastic.co/guide/cn/elasticsearch/guide/current/index.html)

[介绍参考](http://www.ruanyifeng.com/blog/2017/08/elasticsearch.html)

## 概述

### 简介

- 搜索引擎

Elasticsearch是一个基于Lucene库的搜索引擎。Lucene 仅仅只是一个库，然而，Elasticsearch 不仅仅是 Lucene，并且也不仅仅只是一个全文搜索引擎。 它可以被下面这样准确的形容：

```
一个分布式的实时文档存储，*每个字段* 可以被索引与搜索
一个分布式实时分析搜索引擎
能胜任上百个服务节点的扩展，并支持 PB 级别的结构化或者非结构化数据
```

它提供了一个分布式、支持多用户的全文搜索引擎，**具有HTTP Web接口和无模式JSON文档。**所有其他语言可以使用 **RESTful API 通过端口 \*9200\* 和 Elasticsearch 进行通信**

**Elasticsearch是用Java开发的**，并在Apache许可证下作为开源软件发布。官方客户端在Java、.NET（C#）、PHP、Python、Apache Groovy、Ruby和许多其他语言中都是可用的。

根据DB-Engines的排名显示，**Elasticsearch是最受欢迎的企业搜索引擎**，其次是Apache Solr，也是基于Lucene。

Elasticsearch可以用于搜索各种文档。它提供可扩展的搜索，具有接近实时的搜索，并支持多租户。

**Elasticsearch是分布式的**，这意味着索引可以被分成分片，每个分片可以有0个或多个副本。每个节点托管一个或多个分片，并充当协调器将操作委托给正确的分片。再平衡和路由是自动完成的。相关数据通常存储在同一个索引中，该索引由一个或多个主分片和零个或多个复制分片组成。一旦创建了索引，就不能更改主分片的数量。

**Elasticsearch 是一个实时的分布式搜索分析引擎，它被用作全文检索、结构化搜索、分析以及这三个功能的组合**

常见使用
```
- Wikipedia 使用 Elasticsearch 提供带有高亮片段的全文搜索，还有search-as-you-type和did-you-mean的建议。
- 卫报使用 Elasticsearch 将网络社交数据结合到访客日志中，实时的给它的编辑们提供公众对于新文章的反馈。
- Stack Overflow 将地理位置查询融入全文检索中去，并且使用 *more-like-this* 接口去查找相关的问题与答案。
- GitHub 使用 Elasticsearch 对1300亿行代码进行查询。
```

- 属于面向文档的数据库

Elasticsearch 是 **面向文档** 的，意味着它存储整个对象或文档。Elasticsearch 不仅存储文档，而且**索引**每个文档的内容使之可以被检索。在 Elasticsearch 中，你对文档进行索引、检索、排序和过滤--而不是对行列数据。

**Elasticsearch 有2.x、5.x、6.x 三个大版本**

> 特点

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

### 搜索原理

#### 倒排索引

倒排索引（英语：**Inverted index**），也常被称为**反向索引**、置入档案或反向档案，是一种索引方法，被用来存储在全文搜索下某个单词在一个文档或者一组文档中的存储位置的映射。**它是文档检索系统中最常用的数据结构。**

假设我们有两个文档，每个文档的 `content` 域包含如下内容：

1. The quick brown fox jumped over the , lazy+ dog
2. Quick brown foxes leap over lazy dogs in summer

正向索引： 存储每个文档的单词的列表

| Doc  | Quick | The  | brown | dog  | dogs | fox  | foxes |  in  | jumped | lazy | leap | over | quick | summer | the  |
| ---- | :---: | :--: | :---: | :--: | :--: | :--: | :---: | :--: | :----: | :--: | :--: | :--: | :---: | :----: | :--: |
| Doc1 |       |  X   |   X   |  X   |      |  X   |       |      |   X    |  X   |      |  X   |   X   |        |  X   |
| Doc2 |   X   |      |   X   |      |  X   |      |   X   |  X   |        |  X   |  X   |  X   |       |   X    |      |

反向索引：

```
Term      Doc_1  Doc_2
-------------------------
Quick   |       |  X
The     |   X   |
brown   |   X   |  X
dog     |   X   |
dogs    |       |  X
fox     |   X   |
foxes   |       |  X
in      |       |  X
jumped  |   X   |
lazy    |   X   |  X
leap    |       |  X
over    |   X   |  X
quick   |   X   |
summer  |       |  X
the     |   X   |
------------------------
```

如果我们想搜索 `quick brown` ，我们只需要查找包含每个词条的文档：

```
Term      Doc_1  Doc_2
-------------------------
brown   |   X   |  X
quick   |   X   |
------------------------
Total   |   2   |  1
```

两个文档都匹配，但是第一个文档比第二个匹配度更高。如果我们使用仅计算匹配词条数量的简单 *相似性算法* ，那么，我们可以说，对于我们查询的相关性来讲，第一个文档比第二个文档更佳。

#### 分析

上面不太合理的地方：
```
- `Quick` 和 `quick` 以独立的词条(token)出现，然而用户可能认为它们是相同的词。
- `fox` 和 `foxes` 非常相似, 就像 `dog` 和 `dogs` ；他们有相同的词根。
- `jumped` 和 `leap`, 尽管没有相同的词根，但他们的意思很相近。他们是同义词。
```
进行**标准化**：
```
- `Quick` 可以小写化为 `quick` 。
- `foxes` 可以 *词干提取* --变为词根的格式-- 为 `fox` 。类似的， `dogs` 可以为提取为 `dog` 。
- `jumped` 和 `leap` 是同义词，可以索引为相同的单词 `jump` 。
```
标准化的反向索引：

```
Term      Doc_1  Doc_2
-------------------------
brown   |   X   |  X
dog     |   X   |  X
fox     |   X   |  X
in      |       |  X
jump    |   X   |  X
lazy    |   X   |  X
over    |   X   |  X
quick   |   X   |  X
summer  |       |  X
the     |   X   |  X
------------------------
```

**对于查询的字符串必须与词条（token）进行相同的标准化处理，才能保证搜索的正确性。**

分词和标准化的过程称为 *分析* （analysis） ：

1. 首先，将一块文本分成适合于倒排索引的独立的 *词条* ， -> **分词**

2. 之后，将这些词条统一化为标准格式以提高它们的“可搜索性” -> **标准化**

分析工作是由**分析器**（ analyzer）完成的：

**字符过滤器**

首先，字符串按顺序通过每个 *字符过滤器* 。他们的任务是在分词前整理字符串。一个字符过滤器可以用来去掉HTML，或者将 `&` 转化成 `and`。

**分词器**

其次，字符串被 *分词器* 分为单个的词条。一个简单的分词器遇到空格和标点的时候，可能会将文本拆分成词条。

**Token 过滤器 （词条过滤器）**

最后，词条按顺序通过每个 *token 过滤器* 。这个过程可能会改变词条（例如，小写化 `Quick` ），删除词条（例如， 像 `a`， `and`， `the` 等无用词），或者增加词条（例如，像 `jump` 和 `leap` 这种同义词）。

#### 相关性排序

默认情况下，搜索结果是按照 **相关性** 进行倒序排序的——最相关的文档排在最前。

相关性可以用相关性评分表示，评分越高，相关性越高。

评分的计算方式取决于查询类型 不同的查询语句用于不同的目的： `fuzzy` 查询（模糊查询）会计算与关键词的拼写相似程度，`terms` 查询（词组查询）会计算 找到的内容与关键词组成部分匹配的百分比，但是通常我们说的 相关性 是我们用来计算全文本字段的值相对于全文本检索词相似程度的算法。

Elasticsearch 的相似度算法被定义为检索词频率/反向文档频率， *TF/IDF* ，包括以下内容：

**检索词频率**

检索词在该字段出现的频率？出现频率越高，相关性也越高。 字段中出现过 5 次要比只出现过 1 次的相关性高。

**反向文档频率**

每个检索词在索引中出现的频率？频率越高，相关性越低。检索词出现在多数文档中会比出现在少数文档中的权重更低。

**字段长度准则**

字段的长度是多少？长度越长，相关性越低。 检索词出现在一个短的 title 要比同样的词出现在一个长的 content 字段权重更大。

## 概念与集群

### 概念

存储数据到 Elasticsearch 的行为叫做 *索引* （indexing）

关于数据的概念

```
Relational DB -> Databases 数据库 -> Tables 表 -> Rows 行 -> Columns 列
Elasticsearch -> Indices 索引库 -> Types 类型 -> Documents 文档 -> Fields 字段/属性
```

一个 Elasticsearch 集群可以包含多个 **索引** （indices 数据库），相应的每个索引可以包含多个 **类型**（type 表） 。 这些不同的类型存储着多个 **文档**（document 数据行） ，每个文档又有多个 **属性** （field 列）。

### 集群

Elasticsearch 尽可能地屏蔽了分布式系统的复杂性。这里列举了一些在后台自动执行的操作：
```
- 分配文档到不同的容器 或分片中，文档可以储存在一个或多个节点中
- 按集群节点来均衡分配这些分片，从而对索引和搜索过程进行负载均衡
- 复制每个分片以支持数据冗余，从而防止硬件故障导致的数据丢失
- 将集群中任一节点的请求路由到存有相关数据的节点
- 集群扩容时无缝整合新节点，重新分配分片以便从离群节点恢复
```
#### node

Elastic 本质上是一个分布式数据库，允许多台服务器协同工作，每台服务器可以运行多个 Elastic 实例。

单个 Elastic 实例称为一个节点（node）。一组节点构成一个集群（cluster）。

**一个运行中的 Elasticsearch 实例称为一个 节点**，而集群是由一个或者多个拥有相同 `cluster.name` 配置的节点组成， 它们共同承担数据和负载的压力。当有节点加入集群中或者从集群中移除节点时，集群将会重新平均分布所有的数据。

当一个节点被选举成为 **主节点（master）时， 它将负责管理集群范围内的所有变更**，例如增加、删除索引，或者增加、删除节点等。 而**主节点并不需要涉及到文档级别的变更和搜索等操作**，所以当集群只拥有一个主节点的情况下，即使流量的增加它也不会成为瓶颈。 任何节点都可以成为主节点。我们的示例集群就只有一个节点，所以它同时也成为了主节点。

作为用户，**我们可以将请求发送到集群中的任何节点 ，包括主节点**。 每个节点都知道任意文档所处的位置，并且能够将我们的请求直接转发到存储我们所需文档的节点。 无论我们将请求发送到哪个节点，它都能负责从各个包含我们所需文档的节点收集回数据，并将最终结果返回給客户端。 Elasticsearch 对这一切的管理都是透明的。

#### shard

一个 *分片* 是一个底层的 *工作单元* ，它仅保存了 全部数据中的一部分。

索引实际上是指向一个或者多个物理 *分片* 的 *逻辑命名空间* 。

文档被存储和索引到分片内，但是应用程序是直接与索引而不是与分片进行交互。

Elasticsearch 是利用分片将数据分发到集群内各处的。分片是数据的容器，文档保存在分片内，分片又被分配到集群内的各个节点里。 当你的集群规模扩大或者缩小时， Elasticsearch 会自动的在各节点中迁移分片，使得数据仍然均匀分布在集群里。

**主分片**（primary shard）

索引内任意一个文档都归属于一个主分片，所以主分片的数目决定着索引能够保存的最大数据量。

**复制分片**（副分片 replica shard)

一个副本分片只是一个主分片的拷贝。 副本分片作为硬件故障时保护数据不丢失的冗余备份，并为搜索和返回文档等读操作提供服务。

**在索引建立的时候就已经确定了主分片数，但是副本分片数可以随时修改.**

初始设置索引的分片方法

```javascript
PUT /blogs
{
   "settings" : {
      "number_of_shards" : 3,
      "number_of_replicas" : 1
   }
}

// number_of_shards
// 每个索引的主分片数，默认值是 `5` 。这个配置在索引创建后不能修改。
// number_of_replicas
// 每个主分片的副本数，默认值是 `1` 。对于活动的索引库，这个配置可以随时修改。
```

分片是一个功能完整的搜索引擎，它拥有使用一个节点上的所有资源的能力。 我们这个拥有6个分片（3个主分片和3个副本分片）的索引可以最大扩容到6个节点，每个节点上存在一个分片，并且每个分片拥有所在节点的全部资源。

修改复制分片数目的方法

```http
PUT /blogs/_settings
{
   "number_of_replicas" : 2
}
```

拥有越多的副本分片时，也将拥有越高的吞吐量。

#### 故障转移

选举新的主节点

提升复制分片为主分片

#### 查看集群健康状态

```javascript
GET /_cluster/health

{
   "cluster_name":          "elasticsearch",
   "status":                "green", 
   "timed_out":             false,
   "number_of_nodes":       1,
   "number_of_data_nodes":  1,
   "active_primary_shards": 0,
   "active_shards":         0,
   "relocating_shards":     0,
   "initializing_shards":   0,
   "unassigned_shards":     0
}

// status字段指示着当前集群在总体上是否工作正常。它的三种颜色含义如下：
// green  	所有的主分片和副本分片都正常运行。
// yellow 	所有的主分片都正常运行，但不是所有的副本分片都正常运行。
// red 		有主分片没能正常运行。
```

## 索引与类型

### 索引

Elastic 会索引所有字段，经过处理后写入一个反向索引（Inverted Index）。查找数据的时候，直接查找该索引。

所以，Elastic 数据管理的顶层单位就叫做 Index（索引）。它是单个数据库的同义词。每个 Index （即数据库）的名字必须是小写。

下面的命令可以查看当前节点的所有 Index。

```bash
$ curl -X GET 'http://localhost:9200/_cat/indices?v'
```

- 查看索引

```shell
curl 127.0.0.1:9200/_cat/indices
```

> 请求`curl 127.0.0.1:9200/_cat`可获取用于查询的名称

- 创建索引

索引可以在添加文档数据时，通过动态映射的方式自动生成索引与类型。

索引也可以手动创建，通过手动创建，可以控制主分片数目、分析器和类型映射。

```http
PUT /my_index
{
    "settings": { ... any settings ... },
    "mappings": {
        "type_one": { ... any mappings ... },
        "type_two": { ... any mappings ... },
        ...
    }
}
```

**注： 在Elasticsearch 5.x版本中，设置分片与设置索引的类型字段需要分两次设置完成。**

- 删除索引

用以下的请求来 删除索引:

```http
DELETE /my_index
```

你也可以这样删除多个索引：

```http
DELETE /index_one,index_two
DELETE /index_*
```

你甚至可以这样删除 *全部* 索引：

```http
DELETE /_all
DELETE /*
```

- 示例

```shell
// 创建头条项目文章索引库
curl -X PUT 127.0.0.1:9200/articles -H 'Content-Type: application/json' -d'
{
   "settings" : {
        "index": {
            "number_of_shards" : 3,
            "number_of_replicas" : 1
        }
   }
}
'
```

### 类型映射

*类型* 在 Elasticsearch 中表示一类相似的文档，类型由 *名称* 和 *映射* （ mapping）组成。

*映射*, mapping， 就像数据库中的 schema ，描述了文档可能具有的字段或 *属性* 、 每个字段的数据类型—比如 `string`, `integer` 或 `date`。

为了能够将时间字段视为时间，数字字段视为数字，字符串字段视为全文或精确值字符串， Elasticsearch 需要知道每个字段中数据的类型。

- 简单字段类型

```
- 字符串: `text` (在elaticsearch 2.x版本中，为string类型)
- 整数 : `byte`, `short`, `integer`, `long`
- 浮点数: `float`, `double`
- 布尔型: `boolean`
- 日期: `date`
```
头条项目文章类型映射

```shell
curl -X PUT 127.0.0.1:9200/articles/_mapping/article -H 'Content-Type: application/json' -d'
{
     "_all": {
          "analyzer": "ik_max_word"
      },
      "properties": {
          "article_id": {
              "type": "long",
              "include_in_all": "false"
          },
          "user_id": {
              "type": "long",
              "include_in_all": "false"
          },
          "title": {
              "type": "text",
              "analyzer": "ik_max_word",
              "include_in_all": "true",
              "boost": 2
          },
          "content": {
              "type": "text",
              "analyzer": "ik_max_word",
              "include_in_all": "true"
          },
          "status": {
              "type": "integer",
              "include_in_all": "false"
          },
          "create_time": {
              "type": "date",
              "include_in_all": "false"
          }
      }
}
'


# - `_all`字段是把所有其它字段中的值，以空格为分隔符组成一个大字符串，然后被分析和索引，但是不存储，也就是说它能被查询，但不能被取回显示。`_all`允许在不知道要查找的内容是属于哪个具体字段的情况下进行搜索。
# - `analyzer`指明使用的分析器
# 	索引时的顺序如下：
#    - 字段映射里定义的 `analyzer`
#    - 否则索引设置中名为 `default` 的分析器，默认为`standard` 标准分析器
#        在搜索时，顺序有些许不同：
#    - 查询自己定义的 `analyzer`
#    - 否则字段映射里定义的 `analyzer`
#    - 否则索引设置中名为 `default` 的分析器，默认为`standard` 标准分析器
# - `include_in_all` 参数用于控制 `_all` 查询时需要包含的字段。默认为 true。
# - `boost`可以提升查询时计算相关性分数的权重。例如`title`字段将是其他字段权重的两倍。
```

- 查看映射

```shell
curl 127.0.0.1:9200/articles/_mapping/article?pretty
```

- 映射修改

一个类型映射创建好后，可以为类型增加新的字段映射

```shell
curl -X PUT 127.0.0.1:9200/articles/_mapping/article -H 'Content-Type:application/json' -d '
{
  "properties": {
    "new_tag": {
      "type": "text"
    }
  }
}
'
```

**但是不能修改已有字段的类型映射，原因在于elasticsearch已按照原有字段映射生成了反向索引数据，类型映射改变意味着需要重新构建反向索引数据，所以并不能再原有基础上修改，只能新建索引库，然后创建类型映射后重新构建反向索引数据。**

例如，将status字段类型由integer改为byte会报错

```shell
curl -X PUT 127.0.0.1:9200/articles/_mapping/article -H 'Content-Type:application/json' -d '
{
  "properties": {
    "status": {
      "type": "byte"
    }
  }
}
'
```

需要从新建立索引

```shell
curl -X PUT 127.0.0.1:9200/articles_v2 -H 'Content-Type: application/json' -d'
{
   "settings" : {
      "index": {
          "number_of_shards" : 3,
          "number_of_replicas" : 1
       }
   }
}
'

curl -X PUT 127.0.0.1:9200/articles_v2/_mapping/article -H 'Content-Type: application/json' -d'
{
     "_all": {
          "analyzer": "ik_max_word"
      },
      "properties": {
          "article_id": {
              "type": "long",
              "include_in_all": "false"
          },
          "user_id": {
               "type": "long",
              "include_in_all": "false"
          },
          "title": {
              "type": "text",
              "analyzer": "ik_max_word",
              "include_in_all": "true",
              "boost": 2
          },
          "content": {
              "type": "text",
              "analyzer": "ik_max_word",
              "include_in_all": "true"
          },
          "status": {
              "type": "byte",
              "include_in_all": "false"
          },
          "create_time": {
              "type": "date",
              "include_in_all": "false"
          }
      }
}
'
```

- 重新索引数据

```shell
curl -X POST 127.0.0.1:9200/_reindex -H 'Content-Type:application/json' -d '
{
  "source": {
    "index": "articles"
  },
  "dest": {
    "index": "articles_v2"
  }
}
'
```

- 为索引起别名

为索引起别名，让新建的索引具有原索引的名字，可以让应用程序零停机。

```shell
curl -X DELETE 127.0.0.1:9200/articles
curl -X PUT 127.0.0.1:9200/articles_v2/_alias/articles
```

查询索引别名

```shell
# 查看别名指向哪个索引
curl 127.0.0.1:9200/*/_alias/articles

# 查看哪些别名指向这个索引
curl 127.0.0.1:9200/articles_v2/_alias/*
```

> 索引库类型修改方法

常规方法

```
1.不能直接修改映射字段的数据类型
2.新建索引库和新的类型映射额
3.PUT /_reindex重新索引数据，将原库的数据添加到新库中
4.为新库起别名(可能需要先删原库)
```

工程实践方法

```
1.创建库时就增加版本号
PUT /articles_v1  ->起别名 /articles
2.建设新库时不必停机即可覆盖旧索引
PUT /articles_v2  ->起别名 /articles
```

## 文档

- Document

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

- Type

Document 可以分组，比如`weather`这个 Index 里面，可以按城市分组（北京和上海），也可以按气候分组（晴天和雨天）。这种分组就叫做 Type，它是虚拟的逻辑分组，用来过滤 Document。

不同的 Type 应该有相似的结构（schema），举例来说，`id`字段不能在这个组是字符串，在另一个组是数值。这是与关系型数据库的表的[一个区别](https://www.elastic.co/guide/en/elasticsearch/guide/current/mapping.html)。性质完全不同的数据（比如`products`和`logs`）应该存成两个 Index，而不是一个 Index 里面的两个 Type（虽然可以做到）。

下面的命令可以列出每个 Index 所包含的 Type。

```bash
$ curl 'localhost:9200/_mapping?pretty=true'
```

根据[规划](https://www.elastic.co/blog/index-type-parent-child-join-now-future-in-elasticsearch)，Elastic 6.x 版只允许每个 Index 包含一个 Type，7.x 版将会彻底移除 Type。

- 一个文档的实例

```json
{
    "name":         "John Smith",
    "age":          42,
    "confirmed":    true,
    "join_date":    "2014-06-01",
    "home": {
        "lat":      51.5,
        "lon":      0.1
    },
    "accounts": [
        {
            "type": "facebook",
            "id":   "johnsmith"
        },
        {
            "type": "twitter",
            "id":   "johnsmith"
        }
    ]
}

// 一个文档不仅仅包含它的数据 ，也包含 *元数据*(metadata) —— 有关文档的信息。 三个必须的元数据元素如下：
// - `_index`  文档在哪存放
// - `_type`   文档表示的对象类别
// - `_id`	文档唯一标识
```

- 索引文档（保存文档数据）

使用自定义的文档id

```http
PUT /{index}/{type}/{id}
{
  "field": "value",
  ...
}


curl -X PUT 127.0.0.1:9200/articles/article/150000 -H 'Content-Type:application/json' -d '
{
  "article_id": 150000,
  "user_id": 1,
  "title": "python是世界上最好的语言",
  "content": "确实如此",
  "status": 2,
  "create_time": "2019-04-03"
}'
```

自动生成文档id

```http
PUT /{index}/{type}
{
  "field": "value",
  ...
}
```

- 获取指定文档

```shell
curl 127.0.0.1:9200/articles/article/150000?pretty

# 获取一部分
curl 127.0.0.1:9200/articles/article/150000?_source=title,content\&pretty
```

注意：`_version` 每次修改文档数据，版本都会增加，可以当作乐观锁的依赖（判断标准）使用

- 判断文档是否存在

```shell
curl -i -X HEAD 127.0.0.1:9200/articles/article/150000

// - 存在 200状态码
// - 不存在 404状态码
```

- 更新文档

在 Elasticsearch 中文档是 *不可改变* 的，不能修改它们。 相反，如果想要更新现有的文档，需要 *重建索引*或者进行替换。我们可以使用相同的 `index` API 进行实现。

例如修改title字段的内容，不可进行以下操作（这样做会自动删除旧文档，生成当前文档）

```shell
curl -X PUT 127.0.0.1:9200/articles/article/150000 -H 'Content-Type:application/json' -d '
{
  "title": "python必须是世界上最好的语言"
}'
```

而是要索引完整文档内容

```shell
curl -X PUT 127.0.0.1:9200/articles/article/150000 -H 'Content-Type:application/json' -d '
{
  "article_id": 150000,
  "user_id": 1,
  "title": "python必须是世界上最好的语言",
  "content": "确实如此",
  "status": 2,
  "create_time": "2019-04-03"
}'
```

注意返回值_version的变化

- 删除文档

```shell
curl -X DELETE 127.0.0.1:9200/articles/article/150000
```

- 取回多个文档

```shell
curl -X GET 127.0.0.1:9200/_mget -d '
{
  "docs": [
    {
      "_index": "articles",
      "_type": "article",
      "_id": 150000
    },
    {
      "_index": "articles",
      "_type": "article",
      "_id": 150001
    }
  ]
}'
```

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
sudo systemctl restart elasticsearch
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

## 导入数据

> 注意

项目运行前，可以使用logstash等工具批量导入索引数据，项目运行中，新增索引需要手动处理（逻辑新增、脚本处理）

- Logstach安装

```shell
sudo rpm --import https://artifacts.elastic.co/GPG-KEY-elasticsearch
```

在 /etc/yum.repos.d/ 中创建logstash.repo文件

```
[logstash-6.x]
name=Elastic repository for 6.x packages
baseurl=https://artifacts.elastic.co/packages/6.x/yum
gpgcheck=1
gpgkey=https://artifacts.elastic.co/GPG-KEY-elasticsearch
enabled=1
autorefresh=1
type=rpm-md
```

执行

```shell
sudo yum install logstash
cd /usr/share/logstash/bin/
sudo ./logstash-plugin install logstash-input-jdbc
sudo ./logstash-plugin install logstash-output-elasticsearch
scp mysql-connector-java-8.0.13.tar.gz python@10.211.55.7:~/
tar -zxvf mysql-connector-java-8.0.13.tar.gz
```

- 从MySQL导入数据到Elasticsearch

> 常规索引

创建配置文件logstash_mysql.conf

```shell
input{
     jdbc {
         jdbc_driver_library => "/home/python/mysql-connector-java-8.0.13/mysql-connector-java-8.0.13.jar"    # java连接mysql
         jdbc_driver_class => "com.mysql.jdbc.Driver"  # 驱动类
         jdbc_connection_string => "jdbc:mysql://127.0.0.1:3306/toutiao?tinyInt1isBit=false"  # mysql地址
         jdbc_user => "root"
         jdbc_password => "mysql"
         jdbc_paging_enabled => "true"
         jdbc_page_size => "1000"
         jdbc_default_timezone =>"Asia/Shanghai"
         statement => "select a.article_id as article_id,a.user_id as user_id, a.title as title, a.status as status, a.create_time as create_time,  b.content as content from news_article_basic as a inner join news_article_content as b on a.article_id=b.article_id"  # 获取数据的sql语句
         use_column_value => "true"  # 
         tracking_column => "article_id"  # 追踪列中的变量值设置为文档唯一标识
         clean_run => true
     }
}
output{
      elasticsearch {
         hosts => "127.0.0.1:9200"  # es数据库地址
         index => "articles"
         document_id => "%{article_id}"
         document_type => "article"
      }
      stdout {
         codec => json_lines  # 显示输出
     }
}
```

执行命令导入数据

```shell
sudo /usr/share/logstash/bin/logstash -f ./logstash_mysql.conf
```

> 自动补全索引

编辑logstash_mysql_completion.conf

```shell
input{
     jdbc {
         jdbc_driver_library => "/home/python/mysql-connector-java-8.0.13/mysql-connector-java-8.0.13.jar"
         jdbc_driver_class => "com.mysql.jdbc.Driver"
         jdbc_connection_string => "jdbc:mysql://127.0.0.1:3306/toutiao?tinyInt1isBit=false"
         jdbc_user => "root"
         jdbc_password => "mysql"
         jdbc_paging_enabled => "true"
         jdbc_page_size => "1000"
         jdbc_default_timezone =>"Asia/Shanghai"
         statement => "select title as suggest from news_article_basic"
         clean_run => true
     }
}
output{
      elasticsearch {
         hosts => "127.0.0.1:9200"
         index => "completions"
         document_type => "words"
      }
}
```

执行命令导入数据

```shell
sudo /usr/share/logstash/bin/logstash -f ./logstash_mysql_completion.conf
```