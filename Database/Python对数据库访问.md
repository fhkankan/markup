# MySQL

##数据库驱动

数据库驱动是用于连接 MySQL 服务器，可以执行sql语句的发送，执行结果的获取

ORM是将模型类的操作转换为sql语句，将数据库执行的结果转换为模型类对象

在mysql数据库的使用中，若使用ORM，ORM只识别mysqldb作为数据库驱动名

- PyMySQL是个人维护的第三方库，支持python2.x和python3.x.

```
# 在ORM（Django,sqlalchemy,flask-sqlalchemy）中使用时需执行如下语句
import pymysql
pymsql.install_as_mysqldb()
```

- MySQL-Python是mysql官方维护的库，只支持python2

```
# 官方的库，安装后，ORM自动识别为mysqldb,直接使用
```

- mysql-client，支持python2和python3

```
替代mysql-python,
```



**安装驱动**

```
pip install PyMySQL
pip install MySQL-Python
pip install mysqlclient
```

## 数据库交互

- 引入模块

```
# 在py文件中引入pymysql模块
from pymysql import *
```

- connection对象

用于建立与数据库的连接

创建对象：调用connect()方法

```
conn=connect(参数列表)

参数host：连接的mysql主机，如果本机是'localhost'
参数port：连接的mysql主机的端口，默认是3306
参数database：数据库的名称
参数user：连接的用户名
参数password：连接的密码
参数charset：通信采用的编码方式，推荐使用utf8
```

对象的方法

```
close()			关闭连接

commit()		提交

cursor()		返回Cursor对象，用于执行sql语句并获得结果
```

- Cursor对象

用于执行sql语句，使用频度最高的语句为select、insert、update、delete

获取Cursor对象：调用Connection对象的cursor()方法

```
cs1=conn.cursor()
```

对象方法

```
close()关闭

execute(operation [, parameters ])执行语句，返回受影响的行数，主要用于执行insert、update、delete语句，也可以执行create、alter、drop等语句

fetchone()执行查询语句时，获取查询结果集的第一个行数据，返回一个元组

fetchall()执行查询时，获取结果集的所有行，一行构成一个元组，再将这些元组装入一个元组返回
```

对象的属性

```
rowcount只读属性，表示最近一次execute()执行后受影响的行数

connection获得当前连接对象
```

## 参数化

- sql语句的参数化，可以有效防止sql注入
- 注意：此处不同于python的字符串格式化，全部使用%s占位

```
from pymysql import *

def main():

    find_name = input("请输入物品名称：")

    # 创建Connection连接
    conn = connect(host='localhost',port=3306,user='root',password='mysql',database='jing_dong',charset='utf8')
    # 获得Cursor对象
    cs1 = conn.cursor()


    # # 非安全的方式
    # # 输入 " or 1=1 or "   (双引号也要输入)
    # sql = 'select * from goods where name="%s"' % find_name
    # print("""sql===>%s<====""" % sql)
    # # 执行select语句，并返回受影响的行数：查询所有数据
    # count = cs1.execute(sql)

    # 安全的方式
    # 构造参数列表
    params = [find_name]
    # 执行select语句，并返回受影响的行数：查询所有数据
    count = cs1.execute('select * from goods where name=%s', params)
    # 注意：
    # 如果要是有多个参数，需要进行参数化
    # 那么params = [数值1, 数值2....]，此时sql语句中有多个%s即可 

    # 打印受影响的行数
    print(count)
    # 获取查询的结果
    # result = cs1.fetchone()
    result = cs1.fetchall()
    # 打印查询的结果
    print(result)
    # 关闭Cursor对象
    cs1.close()
    # 关闭Connection对象
    conn.close()

if __name__ == '__main__':
    main()
```



##实例

### 数据库连接

```
import pymysql
 
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
 
# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()
 
# 使用 execute()  方法执行 SQL 查询 
cursor.execute("SELECT VERSION()")
 
# 使用 fetchone() 方法获取单条数据.
data = cursor.fetchone()
 
print ("Database version : %s " % data)

# 关闭数据库连接
db.close()
```

###创建数据库表

```
import pymysql
 
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
 
# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()
 
# 使用 execute() 方法执行 SQL，如果表存在则删除
cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")
 
# 使用预处理语句创建表
sql = """CREATE TABLE EMPLOYEE (
         FIRST_NAME  CHAR(20) NOT NULL,
         LAST_NAME  CHAR(20),
         AGE INT,  
         SEX CHAR(1),
         INCOME FLOAT )"""
 
cursor.execute(sql)
 
# 关闭数据库连接
db.close()
```

### 数据库插入

```
import pymysql
 
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
 
# 使用cursor()方法获取操作游标 
cursor = db.cursor()
 
# SQL 插入语句
sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
         LAST_NAME, AGE, SEX, INCOME)
         VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""
try:
   # 执行sql语句
   cursor.execute(sql)
   # 提交到数据库执行
   db.commit()
except:
   # 如果发生错误则回滚
   db.rollback()
 
# 关闭数据库连接
db.close()
```

###数据库查询

- **etchone():** 该方法获取下一个查询结果集。结果集是一个对象
- **fetchall():** 接收全部的返回结果行.
- **rowcount:** 这是一个只读属性，并返回执行execute()方法后影响的行数。

```
import pymysql
 
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
 
# 使用cursor()方法获取操作游标 
cursor = db.cursor()
 
# SQL 查询语句
sql = "SELECT * FROM EMPLOYEE \
       WHERE INCOME > '%d'" % (1000)
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 获取所有记录列表
   results = cursor.fetchall()
   for row in results:
      fname = row[0]
      lname = row[1]
      age = row[2]
      sex = row[3]
      income = row[4]
       # 打印结果
      print ("fname=%s,lname=%s,age=%d,sex=%s,income=%d" % \
             (fname, lname, age, sex, income ))
except:
   print ("Error: unable to fetch data")
 
# 关闭数据库连接
db.close()
```

### 数据库更新

```
import pymysql
 
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
 
# 使用cursor()方法获取操作游标 
cursor = db.cursor()
 
# SQL 更新语句
sql = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 提交到数据库执行
   db.commit()
except:
   # 发生错误时回滚
   db.rollback()
 
# 关闭数据库连接
db.close()
```

### 删除操作

```
import pymysql
 
# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )
 
# 使用cursor()方法获取操作游标 
cursor = db.cursor()
 
# SQL 删除语句
sql = "DELETE FROM EMPLOYEE WHERE AGE > '%d'" % (20)
try:
   # 执行SQL语句
   cursor.execute(sql)
   # 提交修改
   db.commit()
except:
   # 发生错误时回滚
   db.rollback()
 
# 关闭连接
db.close()
```

# Redis

## 与python交互

```
# 方法一：安装包
# 进入虚拟环境py2_db，联网安装包redis
pip install redis

# 方法二：到中文官网-客户端下载redis包的源码，使用源码安装
unzip redis-py-master.zip
cd redis-py-master
python setup.py install


# 调用模块
# 引入模块
from redis import *
# 这个模块中提供了StrictRedis对象，用于连接redis服务器，并按照不同类型提供了不同方法，进行交互操作
```

### StrictRedis对象方法

```
通过init创建对象，指定参数host、port与指定的服务器和端口连接，host默认为localhost，port默认为6379
client = StrictRedis()

根据不同的类型，拥有不同的实例方法可以调用，与前面学的redis命令对应，方法需要的参数与命令的参数一致
client.命令('key','value')

# string
set
setex
mset
append
get
mget

# key
keys
exists
type
delete
expire
getrange
ttl

# hash
hset
hmset
hkeys
hget
hmget
hvals
hdel

# list
lpush
rpush
linsert
lrange
lset
lrem

# set
sadd
smembers
srem

# zset
zadd
zrange
zrangebyscore
zscore
zrem
zremrangebyscore
```

### string-增加

```
方法set，添加键、值，如果添加成功则返回True，如果添加失败则返回False

创建文件redis_add.py，编写代码如下
#coding=utf-8
from redis import *

if __name__=="__main__":
    try:
        #创建StrictRedis对象，与redis服务器建立连接
        sr=StrictRedis()
        #添加键py1，值为gj
        result=sr.set('py1','gj')
        #输出响应结果，如果添加成功则返回True，否则返回False
        print result
    except Exception as e:
        print e
```

### string-获取

```
方法get，添加键对应的值，如果键存在则返回对应的值，如果键不存在则返回None

创建文件redis_get.py，编写代码如下
#coding=utf-8
from redis import *

if __name__=="__main__":
    try:
        #创建StrictRedis对象，与redis服务器建立连接
        sr=StrictRedis()
        #获取键py1的值
        result = sr.get('py1')
        #输出键的值，如果键不存在则返回None
        print result
    except Exception as e:
        print e
```

### string-修改

```
方法set，如果键已经存在则进行修改，如果键不存在则进行添加

创建文件redis_set.py，编写代码如下
#coding=utf-8
from redis import *

if __name__=="__main__":
    try:
        #创建StrictRedis对象，与redis服务器建立连接
        sr=StrictRedis()
        #设置键py1的值，如果键已经存在则进行修改，如果键不存在则进行添加
        result = sr.set('py1','hr')
        #输出响应结果，如果操作成功则返回True，否则返回False
        print result
    except Exception as e:
        print e
```

### string-删除

```
方法delete，删除键及对应的值，如果删除成功则返回受影响的键数，否则则返回0

创建文件redis_delete.py，编写代码如下
#coding=utf-8
from redis import *

if __name__=="__main__":
    try:
        #创建StrictRedis对象，与redis服务器建立连接
        sr=StrictRedis()
        #设置键py1的值，如果键已经存在则进行修改，如果键不存在则进行添加
        result = sr.delete('py1')
        #输出响应结果，如果删除成功则返回受影响的键数，否则则返回0
        print result
    except Exception as e:
        print e
```

### 获取键

```
方法keys，根据正则表达式获取键

创建文件redis_keys.py，编写代码如下
#coding=utf-8
from redis import *

if __name__=="__main__":
    try:
        #创建StrictRedis对象，与redis服务器建立连接
        sr=StrictRedis()
        #获取所有的键
        result=sr.keys()
        #输出响应结果，所有的键构成一个列表，如果没有键则返回空列表
        print result
    except Exception as e:
        print e
```

## 与Django框架交互

[文档说明](http://django-redis-chs.readthedocs.io/zh_CN/latest/#id8)

优点

```
持续更新
本地化的 redis-py URL 符号连接字符串
可扩展客户端
可扩展解析器
可扩展序列器
默认客户端主/从支持
完善的测试
已在一些项目的生产环境中作为 cache 和 session 使用
支持永不超时设置
原生进入 redis 客户端/连接池支持
高可配置 ( 例如仿真缓存的异常行为 )
默认支持 unix 套接字
支持 Python 2.7, 3.4, 3.5 以及 3.6
```

安装

```
pip install django-redis
```

### 作为cache backend使用配置

为了使用 django-redis , 你应该将你的 django cache setting 改成这样:

```
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    }
}
```

### 作为session backend使用配置

Django 默认可以使用任何 cache backend 作为 session backend, 将 django-redis 作为 session 储存后端不用安装任何额外的 backend

```
SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"
```

### 原生客户端使用

在某些情况下你的应用需要进入原生 Redis 客户端使用一些 django cache 接口没有暴露出来的进阶特性. 为了避免储存新的原生连接所产生的另一份设置, django-redis 提供了方法 `get_redis_connection(alias)` 使你获得可重用的连接字符串.

```
>>> from django_redis import get_redis_connection
>>> con = get_redis_connection("default")
>>> con
<redis.client.StrictRedis object at 0x2dc4510>
```

## 与集群交互

- 安装包如下

```
pip install redis-py-cluster
```

- [redis-py-cluster源码地址](https://github.com/Grokzen/redis-py-cluster)
- 创建文件redis_cluster.py，示例代码如下

```
#coding=utf-8
from rediscluster import StrictRedisCluster

if __name__=="__main__":
    try:
        #构建所有的节点，Redis会使用CRC16算法，将键和值写到某个节点上
        startup_nodes=[
            {'host': '172.16.0.136', 'port': '7000'},
            {'host': '172.16.0.135', 'port': '7003'},
            {'host': '172.16.0.136', 'port': '7001'},
        ]
        
        #构建StrictRedisCluster对象   client=StrictRedisCluster(startup_nodes=startup_nodes,decode_responses=True)
        #设置键为py2、值为hr的数据
        client.set('py2','hr')
        #获取键为py2的数据并输出
        print client.get('py2')
    except Exception as e:
        print e
```

# MongoDB

- 点击查看[官方文档](http://api.mongodb.org/python/current/tutorial.html)
- 安装python包

```
sudo pip install pymongo
```

- 操作

```
from pymongo import *

# MongoClient对象
# 使用init方法创建连接对象
client = MongoClient('主机ip',端口)

# Database对象
# 通过client对象获取获得数据库对象
db = client.数据库名称

# Collections对象
# 通过db对象获取集合对象
collections = db.集合名称
# 主要方法如下
insert：加入多条文档对象
insert_one：加入一条文档对象
insert_many：加入多条文档对象
find_one：查找一条文档对象
find：查找多条文档对象
update_one：更新一条文档对象
update_many：更新多条文档对象
delete_one：删除一条文档对象
delete_many：删除多条文档对象

# Cursor对象
# 当调用集合对象的find()方法时，会返回Cursor对象
# 结合for...in...遍历cursor对象
```

## 增加

- 创建mongodb_insert1.py文件，增加一条文档对象

```
#coding=utf-8

from pymongo import *

try:
    # 接收输入
    name=raw_input('请输入姓名：')
    home=raw_input('请输入家乡：')
    # 构造json对象
    doc={'name':name,'home':home}
    #调用mongo对象，完成insert
    client=MongoClient('localhost',27017)
    db=client.py3
    db.stu.insert_one(doc)
    print 'ok'
except Exception,e:
    print e
```

- 创建mongodb_insert2.py文件，增加多条文档对象

```
#coding=utf-8

from pymongo import *

try:
    # 构造json对象
    doc1={'name':'hr','home':'thd'}
    doc2={'name':'mnc','home':'njc'}
    doc=[doc1,doc2]
    #调用mongo对象，完成insert
    client=MongoClient('localhost',27017)
    db=client.py3
    db.stu.insert_many(doc)
    print 'ok'
except Exception,e:
    print e
```

## 查询

- 创建mongodb_find1.py文件，查询一条文档对象

```
#coding=utf-8

from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    doc=db.stu.find_one()
    print '%s--%s'%(doc['name'],doc['hometown'])
except Exception,e:
    print e
```

- 创建mongodb_find2.py文件，查询多条文档对象

```
#coding=utf-8

from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    cursor=db.stu.find({'hometown':'大理'})
    for doc in cursor:
        print '%s--%s'%(doc['name'],doc['hometown'])
except Exception,e:
    print e
```

## 修改

- 创建mongodb_update1.py文件，修改一条文档对象

```
#coding=utf-8

from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    db.stu.update_one({'gender':False},{'$set':{'name':'hehe'}})
    print 'ok'
except Exception,e:
    print e
```

- 创建mongodb_update2.py文件，修改多条文档对象

```
#coding=utf-8

from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    db.stu.update_many({'gender':True},{'$set':{'name':'haha'}})
    print 'ok'
except Exception,e:
    print e
```

## 删除

- 创建mongodb_delete1.py文件，删除一条文档对象

```
#coding=utf-8

from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    db.stu.delete_one({'gender':True})
    print 'ok'
except Exception,e:
    print e

```

- 创建mongodb_delete2.py文件，删除多条文档对象

```
#coding=utf-8

from pymongo import *

try:
    client=MongoClient('localhost',27017)
    db=client.py3
    db.stu.delete_many({'gender':False})
    print 'ok'
except Exception,e:
    print e
```
