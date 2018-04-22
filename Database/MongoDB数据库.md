# 概述

MongoDB官方文档：<https://docs.mongodb.com>

MongoDB中文社区：<http://www.mongoing.com>

```
MongoDB (名称来自「humongous (巨大无比的)」)， 是一个可扩展的高性能，开源，模式自由，面向文档的NoSQL，基于 分布式 文件存储，由 C++ 语言编写，设计之初旨在为 WEB 应用提供可扩展的高性能数据存储解决方案。

MongoDB使用的是内存映射存储引擎，它会把磁盘IO操作转换成内存操作，如果是读操作，内存中的数据起到缓存的作用，如果是写操作，内存还可以把随机的写操作转换成顺序的写操作，大幅度提升性能。

MongoDB 既拥有Key-Value存储方式的高性能和高度伸缩性，也拥有传统的RDBMS系统的丰富的功能，集两者的优势于一身。 介于关系数据库和NoSQL之间，也是功能最丰富、最像关系数据库的的NoSQL。
```

## 特点

```
模式自由 :可以把不同结构的文档存储在同一个数据库里

面向集合的存储：适合存储 JSON风格文件的形式，

完整的索引支持：对任何属性可索引，

复制和高可用性：支持服务器之间的数据复制，支持主-从模式及服务器之间的相互复制。复制的主要目的是提供冗余及自动故障转移。

自动分片：支持水平的数据库集群，可动态添加额外的机器。

丰富的查询：支持丰富的查询表达方式，查询指令使用JSON形式的标记，可轻易查询文档中的内嵌的对象及数组。

快速就地更新：查询优化器会分析查询表达式，并生成一个高效的查询计划。
高效的传统存储方式：支持二进制数据及大型对象（如图片等...）。
```

## 适用场景

```
网站数据： 适合实时的插入，更新与查询，并具备网站实时数据存储所需的复制及高度伸缩性。

缓存： 由于性能很高，也适合作为信息基础设施的缓存层。在系统重启之后，搭建的持久化缓存可以避免下层的数据源过载。

大尺寸、低价值的数据： 使用传统的关系数据库存储一些数据时可能会比较贵，在此之前，很多程序员往往会选择传统的文件进行存储。

高伸缩性的场景： 非常适合由数十或者数百台服务器组成的数据库。
用于对象及JSON数据的存储： MongoDB的BSON数据格式非常适合文档格式化的存储及查询。
```

## 不使用场景

```
高度事物性的系统： 例如银行或会计系统。传统的关系型数据库目前还是更适用于需要大量原子性复杂事务的应用程序。

传统的商业智能应用： 针对特定问题的BI数据库会对产生高度优化的查询方式。对于此类应用，数据仓库可能是更合适的选择。

需要使用SQL语句解决的场景： MongoDB不支持SQL语句。
```

#安装与启动

##安装

- 下载mongodb的版本，两点注意
  1. 根据业界规则，偶数为稳定版，如3.2.X；奇数为开发版，如3.3.X
  2. 32bit的mongodb最大只能存放2G的数据，64bit就没有限制
- MongoDB官网安装包下载地址：<http://www.mongodb.org/downloads>
- MongoDB安装文档：<https://docs.mongodb.com/getting-started/shell/installation/>
- Ubuntu下安装MongoDB：

```
python@ubuntu:~$ sudo apt-get install mongodb
```

- mac安装MongoDB:

```
# 预先安装xcode
brew install MongDB

# 后台及登录启动
brew services start mongodb

# 临时启动
redis-server /usr/local/etc/mongodb.conf
```



## 客户端

- 客户端命令为 **mongo**，可以通过help查看所有参数。
- 这个shell即是mongodb的客户端，用来对MongoDB进行操作和管理的交互式环境。

```
python@ubuntu:~$ mongo --help
```

- 终端退出连接

```
> exit

(或Ctrl+C)
```

## 服务端

MongoDB 默认的存储数据目录为 /data/db，默认端口27017

- 服务的命令为mongod，可以通过help查看所有参数

```
python@ubuntu:~$ mongod --help
```

- 相关文件存放路径：默认各个文件存放路径如下所示：

> - 可执行文件存放路径：`/usr/bin/mongod` 和 `/usr/bin/mongo`
> - 数据库文件存放路径：`/data/db`
> - 日志文件存放路径：`/var/log/mongodb/mongod.log`
> - 配置文件存放路径：`/etc/mongod.conf`

- 启动注意事项

1. 首次启动

出错：表示默认的存储数据目录 /data/db 不存在：

`[initandlisten] exception in initAndListen: 29 Data directory /data/db not found., terminating`

创建 /data目录和 /data/db 目录，并指定 读/写/执行 权限

```
python@ubuntu:~$ sudo mkdir -p /data/db
python@ubuntu:~$ sudo chmod 777 /data/db
```

2. 再次启动

启动成功，但是可能会有如下警告：

```
#### 此乃 Warning 1：
[initandlisten] ** WARNING: /sys/kernel/mm/transparent_hugepage/enabled is 'always'.
[initandlisten] **        We suggest setting it to 'never'
[initandlisten] 
[initandlisten] ** WARNING: /sys/kernel/mm/transparent_hugepage/defrag is 'always'.
[initandlisten] **        We suggest setting it to 'never'

#### 此乃 Warning 2：
[initandlisten] ** WARNING: soft rlimits too low. rlimits set to 1024 processes, 64000 files. Number of processes should be at least 32000 : 0.5 times number of files.

#### 此乃 Warning 3：
[initandlisten] ** WARNING: You are running this process as the root user, which is not recommended.
```

**warning1**:

Linux的内存分配默认由内核动态分配，而不是由程序自行管理。而MongoDB对内存占用有那么点...严重，所以为了防止MongoDB占用内存过大而被内核"管理"，官方推荐关闭动态分配。

默认"always"表示允许动态分配，对应的"never"就是不允许，所以我们将这两个文件内容修改为"naver"后就没有warning了。

```
# Ctrl + c 退出 MongoDB 数据库服务
# 然后进入 root 用户下，执行修改命令

python@ubuntu:~$ sudo su
[sudo] python 的密码： 

root@ubuntu:~# sudo echo "never" > /sys/kernel/mm/transparent_hugepage/enabled
root@ubuntu:~# sudo echo "never" >  /sys/kernel/mm/transparent_hugepage/defrag
```

实际上，除非网站DBA对数据库性能有极限要求，在通常情况下系统动态分配的内存页大小足够我们正常使用，而且更能优化整个系统，所以一般不必理会这个warning。而且这样只是临时修改Linux内核的设置，在Linux服务器重启后则会失效。

**waring2**

这个WARNING（如果有的话）含义为： 表示默认分配给MongoDB的进程和文件数量限制过低，需要重新分配值：

- mongodb当前限制：1024 processes, 64000 files
- mongodb建议要求：processes = 0.5*files=32000（至少）

```
# 打开 相关配置文件：
root@ubuntu:~# vi /etc/security/limits.conf

# 在打开的 文件最下方，添加，然后保存退出
mongod  soft  nofile  64000
mongod  hard  nofile  64000
mongod  soft  nproc  32000
mongod  hard  nproc  32000
```

**warining3**

意思是我们在用root权限做这些事，理论上是不安全的。我们可以通过附加`--auth`参数，来使用用户认证来处理这个情况，这个后面会讲到。

###三种启动方式：

- 命令行方式直接启动

MongoDB默认的存储数据目录为`/data/db`（需要事先创建），默认端口27017，也可以修改成不同目录：

```
# 直接启动mongod，默认数据存储目在 /data/db
python@ubuntu:~$ sudo mongod

# 启动mongod，并指定数据存储目录（目录必须存在，且有读写权限）
python@ubuntu:~$ sudo mongod --dbpath=/xxxxx/xxxxx
```

- 配置文件方式启动

启动时加上`-f`参数，并指向配置文件即可，默认配置文件为`/etc/mongodb.cnf`，也可以自行编写配置文件并指定。

```
# 启动mongod，并按指定配置文件执行
python@ubuntu:~$ sudo mongod -f /etc/mongodb.cnf
```

- 守护进程方式启动

**启动**

MongoDB提供了一种后台程序方式启动的选择，只需要加上—fork参数即可。但是注意：如果用到了`--fork`参数，就必须启用`--logpath`参数来指定log文件，这是强制的。

```
python@ubuntu:~$ sudo mongod --logpath=/data/db/mongodb.log --fork

about to fork child process, waiting until server is ready for connections.
forked process: xxxxx
child process started successfully, parent exiting
```

**关闭 **

如果使用`--fork`在后台运行mongdb服务，那么就要通过本机admin数据库向服务器发送shutdownServer()消息来关闭。

```
python@ubuntu:~$ mongo
MongoDB shell version: 3.2.8
connecting to: test

> use admin
switched to db admin

> db.shutdownServer()
server should be down...
2017-05-16T22:34:22.923+0800 I NETWORK  [thread1] trying reconnect to 127.0.0.1:27017 (127.0.0.1) failed
2017-05-16T22:34:22.923+0800 W NETWORK  [thread1] Failed to connect to 127.0.0.1:27017, reason: errno:111 Connection refused
2017-05-16T22:34:22.923+0800 I NETWORK  [thread1] reconnect 127.0.0.1:27017 (127.0.0.1) failed failed 
```

- 启用用户认证方式启动

如果之前未定义过用户，所以mongod将允许本地直接访问操作数据库将使用本地root权限，如果使用`--auth`参数启动，将启用MongoDB授权认证，即启用不同的用户对不同的数据库的操作权限。

> 也可以在配置文件`mongod.conf`中加入`auth = true`按第二种启动方式启动。

[参考阅读](http://www.mongoing.com/docs/tutorial/enable-authentication.html)

```
# 启动mongod，并启用用户认证
python@ubuntu:~$ sudo mongod --auth

# 启动mongo shell
python@ubuntu:~$ mongo

# 1. 切换admin数据库下
> use admin

# 2. 创建一个拥有root权限的超级用户，拥有所有数据库的所有权限
#      用户名：python，密码：chuanzhi，角色权限：root（最高权限）
> db.createUser({user : "python", pwd : "chuanzhi", roles : ["root"]})

# 3. 如果 MongoDB 开启了权限模式，并且某一个数据库没有任何用户时，可以不用认证权限并创建一个用户，但是当继续创建第二个用户时，会返回错误，若想继续创建用户则必须认证登录。
> db.createUser({user : "bigcat", pwd : "bigcat", roles : [{role : "read", db : "db_01"}, {role : "readWrite", db : "db_02"}]})

# 4. 认证登录到python用户（第一次创建的用户）
> db.auth("python","chuanzhi")

# 5. 查看当前认证登录的用户信息
> show users

# 6. 认证登录成功，可以继续创建第二个用户
# 用户名：bigcat，密码：bigcat，角色权限：[对db_01 拥有读权限，对db_02拥有读/写权限]
> db.createUser({user : "bigcat", pwd : "bigcat", roles : [{role : "read", db : "db_01"}, {role : "readWrite", db : "db_02"}]})


# 7. 查看当前数据库下所有的用户信息.
> db.system.users.find()

# 8. 认证登录到 bigcat 用户
> db.auth("bigcat", "bigcat")

# 9. 切换到 数据库db_01，读操作没有问题
> use db_01

> show collections

# 10. 切换到 数据库db_02，读操作没有问题
> use db_02

> show collections

# 11. 切换到 数据库db_03，读操作出现错误，bigcat用户在db_03数据库下没有相关权限
> use db_03

> show collections
2017-05-17T00:26:56.143+0800 E QUERY    [thread1] Error: listCollections failed: {
    "ok" : 0,
    "errmsg" : "not authorized on db_03 to execute command { listCollections: 1.0, filter: {} }",
    "code" : 13
} :
_getErrorWithCode@src/mongo/shell/utils.js:25:13
DB.prototype._getCollectionInfosCommand@src/mongo/shell/db.js:773:1
DB.prototype.getCollectionInfos@src/mongo/shell/db.js:785:19
DB.prototype.getCollectionNames@src/mongo/shell/db.js:796:16
shellHelper.show@src/mongo/shell/utils.js:754:9
shellHelper@src/mongo/shell/utils.js:651:15
@(shellhelp2):1:1

>
# 12. 认证登录到python用户下
> db.auth("python", "chuanzhi")
1
>
# 13. 删除bigcat用户
> db.dropUser("bigcat")
true
>
# 14. 尝试认证登录bigcat失败
> db.auth("bigcat", "bigcat")
Error: Authentication failed.
0
>
# 15. 退出mongo shell
> exit
bye
python@ubuntu:~$
```

###mongod部分参数说明

在源代码中，mongod的参数分为一般参数，windows参数，replication参数，replica set参数以及隐含参数。下面列举的是一般参数。

mongod的参数中，没有设置内存大小的相关参数，因为MongoDB使用os mmap机制来缓存数据文件数据，自身目前不提供缓存机制。mmap在数据量不超过内存时效率很高，但是数据超过内存后，写入的性能不太稳定。

```
dbpath：数据文件存放路径。每个数据库会在其中创建一个子目录，防止同一个实例多次运行的mongod.lock也保存在次目录中。

logpath：错误日志文件

auth：用户认证

logappend：错误日志采用追加模式(默认覆写模式)

bind_ip：对外服务的绑定ip，一般设置为空，及绑定在本机所有可用ip上。如有需要可以单独绑定。

port：对外服务端口。Web管理端口在这个port的基础上+1000。

fork：以后台Daemon形式运行服务。

journal：开启日志功能，通过保存操作日志来降低单机故障的恢复时间。

syncdelay：系统同步刷新磁盘的时间，单位为秒，默认时60秒。

directoryperdb：每个db存放在单独的目录中，建议设置该参数。

repairpath：执行repair时的临时目录。如果没有开启journal，异常down机后重启，必须执行repair操作。


```

###MongoDB 统计信息

要获得关于MongoDB的服务器统计，需要在MongoDB客户端键入命令`db.stats()`。这将显示数据库名称，收集和数据库中的文档信息

输出信息的参数如下：

```
"db" : "test" ,表示当前是针对"test"这个数据库的描述。想要查看其他数据库，可以先运行$ use datbasename
"collections" : 3,表示当前数据库有多少个collections.可以通过运行show collections查看当前数据库具体有哪些collection.
"objects" : 267,表示当前数据库所有collection总共有多少行数据。显示的数据是一个估计值，并不是非常精确。
"avgObjSize" : 623.2322097378277,表示每行数据是大小，也是估计值，单位是bytes
"dataSize" : 16640,表示当前数据库所有数据的总大小，不是指占有磁盘大小。单位是bytes
"storageSize" : 110592,表示当前数据库占有磁盘大小，单位是bytes,因为mongodb有预分配空间机制，为了防止当有大量数据插入时对磁盘的压力,因此会事先多分配磁盘空间。
"numExtents" : 0,没有什么真实意义
"indexes" : 2 ,表示system.indexes表数据行数。
"indexSize" : 53248,表示索引占有磁盘大小。单位是bytes
"ok" : 1,表示服务器正常
```

# 数据库概念

MongoDB 将数据存储为一个文档，数据结构由键值(key=>value)对组成。MongoDB 文档类似于 JSON 对象。字段值可以包含其他文档，数组及文档数组。

## 数据库

数据库是一个集合的物理容器。一个单一的MongoDB服务器通常有多个数据库。如自带的admin、test，或自行创建的数据库。

## 集合

也称为文档组，类似于关系数据库中的表格。

集合存在于数据库中，一个数据库可以包含很多个集合。集合没有固定的结构，这意味着你在对集合可以插入不同格式和类型的数据，但通常情况下我们插入集合的数据都会有一定的关联性。

## 文档

MongoDB使用了BSON（Binary JSON）这种结构来存储数据，并把这种格式转化成了文档这个概念，每个文档是一组 `键 : 值` 的数据。

## 数据类型

```
# MongoDB中常用的几种数据类型：
ObjectID：文档ID
String：字符串，最常用，必须是有效的UTF-8
Boolean：存储一个布尔值，true或false
Integer：整数可以是32位或64位，这取决于服务器
Double：存储浮点值
Arrays：数组或列表，多个值存储到一个键
Object：用于嵌入式的文档，即一个值为一个文档
Null：存储Null值
Timestamp：时间戳，表示从1970-1-1到现在的总秒数
Date：存储当前日期或时间的UNIX时间格式,注意参数的格式为YYYY-MM-DD

# ObjectID
_id是一个12字节的十六进制数，保证每一份文件的唯一性。你可以自己去设置_id插入文档。如果没有提供，那么MongoDB的每个文档提供了一个独特的ID，这12个字节：
前4个字节为当前时间戳；
之后的3个字节的机器ID；
接下来的2个字节的MongoDB的服务进程id；
剩余3个字节是简单的增量值
一个字节等于2位十六进制（一位十六进制的数等于四位二进制的数。一个字节等于8位二进制数）
```

## RDBMS VS MongoDB

| SQL术语/概念 | MongoDB术语/概念 | 解释/说明                                            |
| ------------ | ---------------- | ---------------------------------------------------- |
| database     | database         | 数据库                                               |
| table        | collection       | 数据库表/集合                                        |
| row          | document         | 数据记录行/文档                                      |
| column       | field            | 数据属性/字段(域)                                    |
| index        | index            | 索引                                                 |
| primary key  | primary key      | 主键,MongoDB默认自动将_id字段设置为主键,可以手动设置 |

# 数据库操作

```
# 连接成功后，默认使用test数据库

# 查看当前数据库名称
db

# 查看所有数据库名称，列出所有在物理上存在的数据库
show dbs

# 切换数据库，如果数据库不存在也并不创建，直到插入数据或创建集合时数据库才被创建
use 数据库名称

# 删除当前指向的数据库，如果数据库不存在，则什么也不做
db.dropDatabase()

# 备份
mongodump -h dbhost -d dbname -o dbdirector
-h：服务器地址，也可以指定端口号
-d：需要备份的数据库名称
-o：备份的数据存放位置，此目录中存放着备份出来的数据

# 恢复
mongorestore -h dbhost -d dbname --dir dbdirectory
-h：服务器地址
-d：需要恢复的数据库实例
--dir：备份数据所在位置
```

# 集合操作

```
# 创建集合
db.createCollection(name, options)
- name是要创建的集合的名称
- options是一个文档，用于指定集合的配置，选项参数是可选的，所以只需要到指定的集合名称，可以不手动创建集合，向不存在的集合中第一次加入数据时，集合会被创建出来

# 创建集合stu
#例1：不限制集合大小
db.createCollection("stu")
#例2：限制集合大小
db.createCollection("stu", {capped : true, size : 6142800} )
- capped：默认值为false表示不设置上限，值为true表示设置上限
- size：当capped值为true时，需要指定此参数，表示上限大小，当文档达到上限时，会将之前的数据覆盖，单位为字节

# 查看当前数据库的集合
show collections

# 删除命令
db.stu.drop()
```

# 数据操作

```
# 插入
db.集合名称.insert(document)
# 插入文档时，如果不指定_id参数，MongoDB会为文档分配一个唯一的ObjectId
# 例1
db.stu.insert({name:'gj',gender:1})
# 例2
s1={_id:'20160101',name:'hr'}
s1.gender=0
db.stu.insert(s1)

# 查询
db.集合名称.find()

# 更新
db.集合名称.update(
   <query>,
   <update>,
   {multi: <boolean>}
)
- query:查询条件，类似sql语句update中where部分
- update:更新操作符，类似sql语句update中set部分
- multi:可选，默认是false，表示只更新找到的第一条记录，值为true表示把满足条件的文档全部更新
# 全文档更新
db.stu.update({name:'hr'},{name:'mnc'})
# 指定属性更新，通过操作符$set
db.stu.insert({name:'hr',gender:0})
db.stu.update({name:'hr'},{$set:{name:'hys'}})
# 修改多条匹配到的数据
db.stu.update({},{$set:{gender:0}},{multi:true})

# 保存
db.集合名称.save(document)
# 如果文档的_id已经存在则修改，如果文档的_id不存在则添加
db.stu.save({_id:'20160102','name':'yk',gender:1})

# 删除
db.集合名称.remove(
   <query>,
   {
     justOne: <boolean>
   }
)
- query:可选，删除的文档的条件
- justOne:可选，如果设为true或1，则只删除一条，默认false，表示删除多条
# 只删除匹配到的第一条
db.stu.remove({gender:0},{justOne:true})
# 全部删除
db.stu.remove({})
```

# 数据查询

## 基本查询

```
# 查询全部符合条件数据
db.集合名称.find({条件文档})
# 查询，只返回第一个
db.集合名称.findOne({条件文档})
# 将结果格式化
db.集合名称.find({条件文档}).pretty()

# 比较运算符
- 等于，默认是等于判断，没有运算符
- 小于 $lt
- 小于或等于 $lte
- 大于 $gt
- 大于或等于 $gte
- 不等于 $ne
# 查询名称等于'郭靖'的学生
db.stu.find({name:'郭靖'})
# 查询年龄大于或等于18的学生
db.stu.find({age:{$gte:18}})

# 逻辑运算符
- and 
- or
# 查询年龄大于或等于18，并且性别为1的学生
db.stu.find({age:{$gte:18},gender:true})
# 查询年龄大于18，或性别为0的学生
db.stu.find({$or:[{age:{$gt:18}},{gender:true}]})
# 查询年龄大于18或性别为0的学生，并且学生的姓名为gj
db.stu.find({$or:[{age:{$gte:18}},{gender:true}],name:'郭靖'})

# 范围运算符
- $in
- $nin
# 查询年龄为18、28的学生
db.stu.find({age:{$in:[18,28]}})

# 支持正则表达式
- 使用/ /或$regex编写正则表达式
# 查询姓黄的学生
db.stu.find({name:/^黄/})
db.stu.find({name:{$regex:'^黄'}}})

# 自定义查询
- 使用$where后面写一个函数，返回满足条件的数据
# 查询年龄大于30的学生
db.stu.find({$where : function(){return this.age>20}})
```

## limit/skip

```
# 方法limit()：用于读取指定数量的文档
db.集合名称.find().limit(NUMBER)
- 参数NUMBER表示要获取文档的条数
- 如果没有指定参数则显示集合中的所有文档
# 查询2条学生信息
db.stu.find().limit(2)

# 方法skip()：用于跳过指定数量的文档
db.集合名称.find().skip(NUMBER)
- 参数NUMBER表示跳过的记录条数，默认值为0
# 查询从第3条开始的学生信息
db.stu.find().skip(2)

# 方法limit()和skip()可以一起使用，不分先后顺序
# 创建数据集
for(i=0;i<15;i++){db.nums.insert({_id:i})}
# 查询第5至8条数据
db.nums.find().limit(4).skip(5)
db.nums.find().skip(5).limit(4)
```

## 投影

```
# 在查询到的返回结果中，只选择必要的字段，而不是选择一个文档的整个字段
db.集合名称.find({},{字段名称:1,...})
- 对于需要显示的字段，设置为1即可，不设置即为不显示
- 特殊：对于_id列默认是显示的，如果不显示需要明确设置为0
db.stu.find({},{name:1, gender:true})
db.stu.find({},{_id:0, name:1, gender:true})
```

## 排序

```
# 方法sort()，用于对结果集进行排序
db.集合名称.find().sort({字段:1,...})
- 1为升序排列
- -1为降序排列
# 根据性别降序，再根据年龄升序
db.stu.find().sort({gender:-1,age:1})
```

## 统计个数

```
# 方法count()用于统计结果集中文档条数
db.集合名称.find({条件}).count()
db.集合名称.count({条件})
# 统计男生人数
db.stu.find({gender:true}).count()
db.stu.count({gender:true})
# 统计年龄大于20的男性人数
db.stu.count({age:{$gt:20},gender:true})
```

## 消除重复

```
# 方法distinct()对数据进行去重
db.集合名称.distinct('去重字段',{条件})
# :查找年龄大于18的学生，来自哪些省份
db.stu.distinct('hometown',{age:{$gt:18}})
```

# 聚合 aggregate

```
# 聚合(aggregate)主要用于计算数据，类似sql中的sum()、avg()
db.集合名称.aggregate([ {管道 : {表达式}} ])
```

## 管道

```
# 常用管道
$group：将集合中的文档分组，可用于统计结果
$match：过滤数据，只输出符合条件的文档
$project：修改输入文档的结构，如重命名、增加、删除字段、创建计算结果
$sort：将输入文档排序后输出
$limit：限制聚合管道返回的文档数
$skip：跳过指定数量的文档，并返回余下的文档
$unwind：将数组类型的字段进行拆分

# 表达式
# 处理输入文档并输出
表达式:'$列名'
# 常用表达式
$sum：计算总和，$sum:1同count表示计数
$avg：计算平均值
$min：获取最小值
$max：获取最大值
$push：在结果文档中插入值到一个数组中
$first：根据资源文档的排序获取第一个文档数据
$last：根据资源文档的排序获取最后一个文档数据

```

### $group

- 将集合中的文档分组，可用于统计结果
- _id表示分组的依据，使用某个字段的格式为'$字段'

```
# 统计男生、女生的总人数
db.stu.aggregate([
    {$group:
        {
            _id:'$gender',
            counter:{$sum:1}
        }
    }
])

Group by null
将集合中所有文档分为一组
# 求学生总人数、平均年龄
db.stu.aggregate([
    {$group:
        {
            _id:null,
            counter:{$sum:1},
            avgAge:{$avg:'$age'}
        }
    }
])

透视数据
# 统计学生性别及学生姓名
db.stu.aggregate([
    {$group:
        {
            _id:'$gender',
            name:{$push:'$name'}
        }
    }
])

使用$$ROOT可以将文档内容加入到结果集的数组中
db.stu.aggregate([
    {$group:
        {
            _id:'$gender',
            name:{$push:'$$ROOT'}
        }
    }
])
```

### $match

- 用于过滤数据，只输出符合条件的文档
- 使用MongoDB的标准查询操作

```
# 查询年龄大于20的学生
db.stu.aggregate([
    {$match:{age:{$gt:20}}}
])

# 查询年龄大于20的男生、女生人数
db.stu.aggregate([
    {$match:{age:{$gt:20}}},
    {$group:{_id:'$gender',counter:{$sum:1}}}
])
```

### $project

- 修改输入文档的结构，如重命名、增加、删除字段、创建计算结果

```
# 查询学生的姓名、年龄
db.stu.aggregate([
    {$project:{_id:0,name:1,age:1}}
])

# 查询男生、女生人数，输出人数
db.stu.aggregate([
    {$group:{_id:'$gender',counter:{$sum:1}}},
    {$project:{_id:0,counter:1}}
])
```

### $sort

- 将输入文档排序后输出

```
# 查询学生信息，按年龄升序
b.stu.aggregate([{$sort:{age:1}}])

# 查询男生、女生人数，按人数降序
db.stu.aggregate([
    {$group:{_id:'$gender',counter:{$sum:1}}},
    {$sort:{counter:-1}}
])
```

### $limit

- 限制聚合管道返回的文档数

```
# 查询2条学生信息
db.stu.aggregate([{$limit:2}])
```

### $skip

- 跳过指定数量的文档，并返回余下的文档

```
# 查询从第3条开始的学生信息
db.stu.aggregate([{$skip:2}])

# 统计男生、女生人数，按人数升序，取第二条数据
db.stu.aggregate([
    {$group:{_id:'$gender',counter:{$sum:1}}},
    {$sort:{counter:1}},
    {$skip:1},
    {$limit:1}
])
```

### $unwind

- 将文档中的某一个数组类型字段拆分成多条，每条包含数组中的一个值

```
# 语法1
# 对某字段值进行拆分
db.集合名称.aggregate([{$unwind:'$字段名称'}])

# 构造数据
db.t2.insert({_id:1,item:'t-shirt',size:['S','M','L']})

# 查询
db.t2.aggregate([{$unwind:'$size'}])

# 语法2
# 对某字段值进行拆分
# 处理空数组、非数组、无字段、null情况
db.inventory.aggregate([{
    $unwind:{
        path:'$字段名称',
        preserveNullAndEmptyArrays:<boolean>#防止数据丢失
    }
}])

# 构造数据
db.t3.insert([
{ "_id" : 1, "item" : "a", "size": [ "S", "M", "L"] },
{ "_id" : 2, "item" : "b", "size" : [ ] },
{ "_id" : 3, "item" : "c", "size": "M" },
{ "_id" : 4, "item" : "d" },
{ "_id" : 5, "item" : "e", "size" : null }
])

# 使用语法1查询
db.t3.aggregate([{$unwind:'$size'}])
- 查询结果，发现对于空数组、无字段、null的文档，都被丢弃了

# 使用语法2查询（可以保留空数组、无字段、null的文档）
db.t3.aggregate([{$unwind:{path:'$sizes',preserveNullAndEmptyArrays:true}}])
```

# 索引

```
# 创建索引
# 1表示升序，-1表示降序
db.集合.ensureIndex({属性:1})
# eg:
db.t1.ensureIndex({name:1})

# 对索引属性查询
db.t1.find({name:'test10000'}).explain('executionStats')

# 建立唯一索引，实现唯一约束的功能
db.t1.ensureIndex({"name":1},{"unique":true})

# 联合索引，对多个属性建立一个索引，按照find()出现的顺序
db.t1.ensureIndex({name:1,age:1})

# 查看文档所有索引
db.t1.getIndexes()

# 删除索引
db.t1.dropIndexes('索引名称')
```

