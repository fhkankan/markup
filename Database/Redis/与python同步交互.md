# 与python同步交互

## 连接

安装

```
pip install redis
```

### 常规
redis-py提供两个类Redis和StrictRedis用于实现Redis的命令，StrictRedis用于实现大部分官方的命令，并使用官方的语法和命令，Redis是StrictRedis的子类，用于向后兼容旧版本的redis-py

```python
import redis

r = redis.Redis(host='192.168.0.110', port=6379,db=0)
r.set('name', 'zhangsan') 
print(r.get('name')) 

sr=StrictRedis()
result=sr.set('py1','gj')
print(result)  #输出响应结果，如果添加成功则返回True，否则返回False
```

### 连接池

redis-py使用connection pool来管理对一个redis server的所有连接，避免每次建立、释放连接的开销。默认，每个Redis实例都会维护一个自己的连接池。可以直接建立一个连接池，然后作为参数Redis，这样就可以实现多个Redis实例共享一个连接池。

```python
import redis

pool = redis.ConnectionPool(host='192.168.0.110', port=6379)
r = redis.Redis(connection_pool=pool)
r.set('name', 'zhangsan')   #添加
print (r.get('name'))   #获取
```

## 操作

常用命令

```python
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

### string

redis中的String在在内存中按照一个name对应一个value来存储

- set

`set()`

```python
set(name, value, ex=None, px=None, nx=False, xx=False)
#在Redis中设置值，默认不存在则创建，存在则修改
# 参数：
ex，过期时间（秒）
px，过期时间（毫秒）
nx，如果设置为True，则只有name不存在时，当前set操作才执行,同setnx(name, value)
xx，如果设置为True，则只有name存在时，当前set操作才执行

setex(name, value, time)
# 设置过期时间(秒)
# 参数
time，过期时间（数字秒 或 timedelta对象）

psetex(name, time_ms, value)
# 设置过期时间(豪秒)
# 参数
time_ms，过期时间（数字毫秒 或 timedelta对象）

setnx(name, value)
# 设置值，只有name不存在时，执行设置操作（添加）

mset(*args, **kwargs)
#批量设置值
r.mset(name1='zhangsan', name2='lisi')

setrange(name, offset, value)
#修改字符串内容，从指定字符串索引开始向后替换，如果新值太长时，则向后添加
# 参数
offset，字符串的索引，字节（一个汉字三个字节）
value，要设置的值
r.set("name","zhangsan")
r.setrange("name",1,"z")
print(r.get("name")) #输出:zzangsan
r.setrange("name",6,"zzzzzzz")
print(r.get("name")) #输出:zzangszzzzzzz
```

`setbit(name, offset, value)`

```python
# 对二进制表示位进行操作
''' name:redis的name
    offset，位的索引（将值对应的ASCII码变换成二进制后再进行索引）
    value，值只能是 1 或 0 '''

str="345"
r.set("name",str)
for i in str:
    print(i,ord(i),bin(ord(i)))#输出 值、ASCII码中对应的值、对应值转换的二进制
'''
输出:
    3 51 0b110011
    4 52 0b110100
    5 53 0b110101'''

r.setbit("name",6,0)#把第7位改为0，也就是3对应的变成了0b110001
print(r.get("name"))#输出：145
```

`incr(self, name, amount=1)`

```python
#自增mount对应的值，当mount不存在时，则创建mount＝amount，否则，则自增,amount为自增数(整数)
print(r.incr("mount",amount=2))#输出:2
print(r.incr("mount"))#输出:3
print(r.incr("mount",amount=3))#输出:6
print(r.incr("mount",amount=6))#输出:12
print(r.get("mount")) #输出:12
```

`incrbyfloat(self, name, amount=1.0)`

```python
#类似 incr() 自增,amount为自增数(浮点数)
```

`decr(self, name, amount=1)`

```python
#自减name对应的值,当name不存在时,则创建name＝amount，否则，则自减，amount为自增数(整数)
```

`append(name, value)`

```python
#在name对应的值后面追加内容
r.set("name","zhangsan")
print(r.get("name"))    #输出:'zhangsan
r.append("name","lisi")
print(r.get("name"))    #输出:zhangsanlisi
```

`getset(name, value)`

```python
# 设置新值，打印原值
print(r.getset("name1","wangwu")) #输出:zhangsan
print(r.get("name1")) #输出:wangwu
```
- get

```python
get(name)
# 获取值

getbit(name, offset)
#获取name对应值的二进制中某位的值(0或1)
r.set("name","3") # 对应的二进制0b110011
print(r.getbit("name",5))   #输出:0
print(r.getbit("name",6))   #输出:1

getrange(key, start, end)
#根据字节获取子序列
# 参数
key	redis的name
start	起始位置(字节)
end		结束位置(字节)
r.set("name","zhangsan")
print(r.getrange("name",0,3))#输出:zhan

mget(keys, *args)
#批量获取
print(r.mget("name1","name2"))
#或
li=["name1","name2"]
print(r.mget(li))
```

`bitcount(key, start=None, end=None)`

```python
#获取对应二进制中1的个数
r.set("name","345")#0b110011 0b110100 0b110101
print(r.bitcount("name",start=0,end=1)) #输出:7
''' key:Redis的name
    start:字节起始位置
    end:字节结束位置'''
```

`strlen(name)`

```python
#返回name对应值的字节长度（一个汉字3个字节）
r.set("name","zhangsan")
print(r.strlen("name")) #输出:8
```

### Hash

redis中的Hash 在内存中类似于一个name对应一个dic来存储 

- set

`hset(name, key, value)`

```python
#name对应的hash中设置一个键值对（不存在，则创建，否则，修改） 
# 参数：
name，redis的name
key，name对应的hash中的key
value，name对应的hash中的value

r.hset("dic_name","a1","aa")
```

`hmset(name, mapping)`

```python
# 在name对应的hash中批量设置键值对,mapping:字典
# 参数：
name，redis的name
mapping，字典，如：{'k1':'v1', 'k2': 'v2'}
    
dic={"a1":"aa","b1":"bb"}
r.hmset("dic_name",dic)
print(r.hget("dic_name","b1"))#输出:bb
```

`hincrby(name, key, amount=1)`

```python
# 自增hash中key对应的值，不存在则创建key=amount(amount为整数)
# 参数：
name，redis中的name
key， hash对应的key
amount，自增数（整数）

print(r.hincrby("demo","a",amount=2))
```

`hincrbyfloat(name, key, amount=1.0)`

```python
# 自增hash中key对应的值，不存在则创建key=amount(amount为浮点数)
# 参数：
name，redis中的name
key， hash对应的key
amount，自增数（浮点数）
```

- get

`hget(name,key)`

```python
r.hset("dic_name","a1","aa")
#在name对应的hash中根据key获取value
print(r.hget("dic_name","a1"))#输出:aa
```

`hmget(name, keys, *args)`

```python
# 在name对应的hash中获取多个key的值
# 参数：
name，reids对应的name
keys，要获取key集合，如：['k1', 'k2', 'k3']
*args，要获取的key，如：k1,k2,k3

li=["a1","b1"]
print(r.hmget("dic_name",li))
print(r.hmget("dic_name","a1","b1"))
```

`hgetall(name)`

```python
#获取name对应hash的所有键值
print(r.hgetall("dic_name"))
```

`hlen(name)、hkeys(name)、hvals(name)`

```python
dic={"a1":"aa","b1":"bb"}
r.hmset("dic_name",dic)

#hlen(name) 获取hash中键值对的个数
print(r.hlen("dic_name"))

#hkeys(name) 获取hash中所有的key的值
print(r.hkeys("dic_name"))

#hvals(name) 获取hash中所有的value的值
print(r.hvals("dic_name"))
```

`hexists(name, key)`

```python
#检查name对应的hash是否存在当前传入的key
print(r.hexists("dic_name","a1"))#输出:True
```

`hscan(name, cursor=0, match=None, count=None)`

``` python
# 增量式迭代获取，对于数据大的数据非常有用，hscan可以实现分片的获取数据，并非一次性将数据全部获取完，从而放置内存被撑爆
# 参数：
name，redis的name
cursor，游标（基于游标分批取获取数据）
match，匹配指定key，默认None 表示所有的key
count，每次分片最少获取个数，默认None表示采用Redis的默认分片个数
# 如：
第一次：cursor1, data1 = r.hscan('xx', cursor=0, match=None, count=None)
第二次：cursor2, data1 = r.hscan('xx', cursor=cursor1, match=None, count=None)
...
直到返回值cursor的值为0时，表示数据已经通过分片获取完毕
```

`hscan_iter(name, match=None, count=None)`

```python
# 利用yield封装hscan创建生成器，实现分批去redis中获取数据
# 参数：
match，匹配指定key，默认None 表示所有的key
count，每次分片最少获取个数，默认None表示采用Redis的默认分片个数
# 如：
for item in r.hscan_iter('xx'):
    print item
```

- delete

`hdel(name,*keys)`

```python
#删除指定name对应的key所在的键值对
r.hdel("dic_name","a1")
```

### List

redis中的List在在内存中按照一个name对应一个List来存储 

- set

`linsert(name, where, refvalue, value)`

```python
# 在name对应的列表的某一个值前或后插入一个新值
# 参数：
name: redis的name
where: BEFORE（前）或AFTER（后）
refvalue: 列表内的值
value: 要插入的数据'''
     
r.linsert("list_name","BEFORE","2","SS")#在列表内找到第一个元素2，在它前面插入SS
```

`lpush(name,values)`

```python
# 在name对应的list中添加元素，每个新的元素都添加到列表的最左边
r.lpush("list_name",2)
r.lpush("list_name",3,4,5)#保存在列表中的顺序为5，4，3，2
```

`rpush(name,values)`

```python
# 同lpush，但每个新的元素都添加到列表的最右边
```

`lpushx(name,value)`

```python
# 在name对应的list中添加元素，只有name已经存在时，值添加到列表的最左边
```

`rpushx(name,value)`

```python
# 在name对应的list中添加元素，只有name已经存在时，值添加到列表的最右边
```

`lset(name, index, value)`

```python
# 对list中的某一个索引位置重新赋值
# 参数：
name，redis的name
index，list的索引位置
value，要设置的值

r.lset("list_name",0,"bbb")
```

`lpop(name)`

```python
# 移除列表的左侧第一个元素，返回值则是第一个元素
print(r.lpop("list_name"))
```

`rpop(name)`

```python
# 从右向左操作
```

`rpoplpush(src, dst)`

```python
# 从一个列表取出最右边的元素，同时将其添加至另一个列表的最左边
# 参数
src 要取数据的列表
dst 要添加数据的列表
```

`blpop(keys, timeout)`

```python
# 将多个列表排列,按照从左到右去移除各个列表内的元素
# 参数：
keys，redis的name的集合
timeout，超时时间，当元素所有列表的元素获取完之后，阻塞等待列表内有数据的时间（秒）, 0 表示永远阻塞
 
r.lpush("list_name",3,4,5)
r.lpush("list_name1",3,4,5)

while True:
    print(r.blpop(["list_name","list_name1"],timeout=0))
    print(r.lrange("list_name",0,-1),r.lrange("list_name1",0,-1))
```

`brpop(keys, timeout)`

```python
# 同blpop，将多个列表排列,按照从右像左去移除各个列表内的元素
```

`brpoplpush(src, dst, timeout=0)`

```python
# 从一个列表的右侧移除一个元素并将其添加到另一个列表的左侧
# 参数：
src，取出并要移除元素的列表对应的name
dst，要插入元素的列表对应的name
timeout，当src对应的列表中没有数据时，阻塞等待其有数据的超时时间（秒），0 表示永远阻塞

r.brpoplpush("list_name","l ist_name1",timeout=0)
```

`lrem(name, value, num)`

```python
# 删除name对应的list中的指定值
# 参数：
name:  redis的name
value: 要删除的值
num:   num=0 删除列表中所有的指定值；
       num=2 从前到后，删除2个；
       num=-2 从后向前，删除2个
           
r.lrem("list_name","SS",num=0)
```

`ltrim(name, start, end)`

```python
# 在name对应的列表中移除没有在start-end索引之间的值
# 参数：
name，redis的name
start，索引的起始位置
end，索引结束位置
    
r.ltrim("list_name",0,2)
```

- get

```python
lindex(name, index)
# 根据索引获取列表内元素
print(r.lindex("list_name",1))

llen(name)
# name对应的list元素的个数
print(r.llen("list_name"))

lrange(name, start, end)
# 在name对应的列表分片获取数据
# 参数：
name，redis的name
start，索引的起始位置
end，索引结束位置
print(r.lrange("list_name",0,-1))
```

自定义增量迭代

```python
# 由于redis类库中没有提供对列表元素的增量迭代，如果想要循环name对应的列表的所有元素，那么就需要：
1、获取name对应的所有列表
2、循环列表
# 但是，如果列表非常大，那么就有可能在第一步时就将程序的内容撑爆，所有有必要自定义一个增量迭代的功能：
 
def list_iter(name):
    """
    自定义redis列表增量迭代
    :param name: redis中的name，即：迭代name对应的列表
    :return: yield 返回 列表元素
    """
    list_count = r.llen(name)
    for index in xrange(list_count):
        yield r.lindex(name, index)
 
# 使用
for item in list_iter('pp'):
    print item
```

 ### Set

Set集合就是不允许重复的列表

- set

`sadd(name,values)`

```python
# 给name对应的集合中添加元素
r.sadd("set_name","aa")
r.sadd("set_name","aa","bb")
```

`sdiffstore(dest, keys, *args)`

```python
# 相当于把sdiff获取的值加入到dest对应的集合中
```

`sinterstore(dest, keys, *args)`

```python
# 获取多个name对应集合的并集，再讲其加入到dest对应的集合中
```

`sunionstore(dest,keys, *args)`

```python
# 获取多个name对应的集合的并集，并将结果保存到dest对应的集合中
```

`smove(src, dst, value)`

```python
# 将某个元素从一个集合中移动到另外一个集合
```

`spop(name)`

```python
# 从集合的右侧移除一个元素，并将其返回
```

`srem(name, values)`

```python
# 删除name对应的集合中的某些值
print(r.srem("set_name2","bb","dd"))
```

- get

`scard(name)`

```python
# 获取name对应的集合中的元素个数
r.scard("set_name")
```

`sdiff(keys, *args)`

```python
# 在第一个name对应的集合中且不在其他name对应的集合的元素集合
r.sadd("set_name","aa","bb")
r.sadd("set_name1","bb","cc")
r.sadd("set_name2","bb","cc","dd")

print(r.sdiff("set_name","set_name1","set_name2"))  # 输出:｛aa｝
```

`sinter(keys, *args)`

```python
# 获取多个name对应集合的并集
r.sadd("set_name","aa","bb")
r.sadd("set_name1","bb","cc")
r.sadd("set_name2","bb","cc","dd")

print(r.sinter("set_name","set_name1","set_name2"))#输出:｛bb｝
```

`smembers(name)`

```python
# 获取name对应的集合的所有成员
```

`sismember(name, value)`

```python
# 检查value是否是name对应的集合内的元素
```

`srandmember(name, numbers)`

```python
# 从name对应的集合中随机获取numbers个元素
print(r.srandmember("set_name2",2))
```

`sunion(keys, *args)`

```python
# 获取多个name对应的集合的并集
r.sunion("set_name","set_name1","set_name2")
```

`sscan(name, cursor=0, match=None, count=None)`
`sscan_iter(name, match=None, count=None)`

```python
# 同字符串的操作，用于增量迭代分批获取元素，避免内存消耗太大`
```

### Zset

在集合的基础上，为每元素排序，元素的排序需要根据另外一个值来进行比较，所以，对于有序集合，每一个元素有两个值，即：值和分数，分数专门用来做排序。值不可重复，分数可以重复。

- set

`zadd(name, *args, **kwargs)`

```python
# 在name对应的有序集合中添加元素
r.zadd("zset_name", "a1", 6, "a2", 2,"a3",5)
# 或
r.zadd('zset_name1', b1=10, b2=5)
```

`zincrby(name, value, amount)`

```python
# 自增有序集合内value对应的分数
r.zincrby("zset_name","a1",amount=2)#自增zset_name对应的有序集合里a1对应的分数
```

`zinterstore(dest, keys, aggregate=None)`

```python
# 获取两个有序集合的交集并放入dest集合，如果遇到相同值不同分数，则按照aggregate进行操作
# aggregate的值为: SUM  MIN  MAX
r.zadd("zset_name", "a1", 6, "a2", 2,"a3",5)
r.zadd('zset_name1', a1=7,b1=10, b2=5)

r.zinterstore("zset_name2",("zset_name1","zset_name"),aggregate="MAX")
print(r.zscan("zset_name2"))
```

`zunionstore(dest, keys, aggregate=None)`

```python
# 获取两个有序集合的并集并放入dest集合，其他同zinterstore，
```

`zrem(name, values)`

```python 
# 删除name对应的有序集合中值是values的成员
r.zrem("zset_name","a1","a2")
```

`zremrangebyrank(name, min, max)`

```python
# 根据排行范围删除
```

`zremrangebyscore(name, min, max)`

```python
# 根据分数范围删除
```

- get

`zcard(name)`

```python
# 获取有序集合内元素的数量
```

`zcount(name, min, max)`

```python
# 获取有序集合中分数在[min,max]之间的个数
print(r.zcount("zset_name",1,5))
```

`zrange( name, start, end, desc=False, withscores=False, score_cast_func=float)`

```python
# 按照索引范围获取name对应的有序集合的元素
# 参数：
name    redis的name
start   有序集合索引起始位置
end     有序集合索引结束位置
desc    排序规则，默认按照分数从小到大排序
withscores  是否获取元素的分数，默认只获取元素的值
score_cast_func 对分数进行数据转换的函数

aa=r.zrange("zset_name",0,1,desc=False,withscores=True,score_cast_func=int)
print(aa)
```

`zrevrange(name, start, end, withscores=False, score_cast_func=float)`

```python
# 同zrange，集合是从大到小排序的
```

`zrangebylex(name, min, max, start=None, num=None)`

```python
# 当有序集合的所有成员都具有相同的分值时，有序集合的元素会根据成员的 值 （lexicographical ordering）来进行排序，而这个命令则可以返回给定的有序集合键 key 中， 元素的值介于 min 和 max 之间的成员
# 对集合中的每个成员进行逐个字节的对比（byte-by-byte compare）， 并按照从低到高的顺序， 返回排序后的集合成员。 如果两个字符串有一部分内容是相同的话， 那么命令会认为较长的字符串比较短的字符串要大
 
# 参数：
name，redis的name
min，左区间（值）。 + 表示正无限； - 表示负无限； ( 表示开区间； [ 则表示闭区间
max，右区间（值）
start，对结果进行分片处理，索引位置
num，对结果进行分片处理，索引后面的num个元素
 
# 如：
ZADD myzset 0 aa 0 ba 0 ca 0 da 0 ea 0 fa 0 ga
r.zrangebylex('myzset', "-", "[ca")  # 结果为：['aa', 'ba', 'ca']
```

`zrevrangebylex(name, max, min, start=None, num=None)`

```python
# 从大到小排序
```

`zrank(name, value)、zrevrank(name, value)`

```python
# 获取value值在name对应的有序集合中的排行位置（从0开始）
print(r.zrank("zset_name", "a2"))
# 从大到小排序
print(r.zrevrank("zset_name", "a2"))
```

`zscore(name, value)`

```python
# 获取name对应有序集合中 value 对应的分数
print(r.zscore("zset_name","a1"))
```

`zscan(name, cursor=0, match=None, count=None, score_cast_func=float)`
`zscan_iter(name, match=None, count=None,score_cast_func=float)`

```python
# 同字符串相似，相较于字符串新增score_cast_func，用来对分数进行操作
```

### 通用

- set

```python
delete(*names)	# 根据name删除redis中的任意数据类型
expire(name, time) # 为某个name设置超时时间
rename(src, dst)  # 对redis的name重命名为
move(name, db)  # 将redis的某个值移动到指定的db下
```

- get

```python
exists(name) 	# 检测redis的name是否存在
randomkey()		# 随机获取一个redis的name（不删除）
type(name)		# 获取name对应值的类型

keys(pattern='*')
# 根据* ？等通配符匹配获取redis的name
"""
* 				匹配数据库中所有 key 。
h?llo 		匹配 hello ， hallo 和 hxllo 等。
h*llo 		匹配 hllo 和 heeeeello 等。
h[ae]llo 	匹配 hello 和 hallo ，但不匹配 hillo
"""

scan(cursor=0, match=None, count=None)
scan_iter(match=None, count=None)
# 同字符串操作，用于增量迭代获取key
```

## 管道

redis-py默认在执行每次请求都会创建（连接池申请连接）和断开（归还连接池）一次连接操作，如果想要在一次请求中指定多个命令，则可以使用pipline实现一次请求指定多个命令，并且默认情况下一次pipline 是原子性操作。

- 在客户端统一收集操作指令
- 补充上multi和exec指令，当作一个事务发送到redis服务器执行

```python
import redis
pool = redis.ConnectionPool(host='192.168.0.110', port=6379)
r = redis.Redis(connection_pool=pool)

pipe = r.pipeline(transaction=True)

r.set('name', 'zhangsan')
r.set('name', 'lisi')

pipe.execute()
```

## 发布和订阅

首先定义一个RedisHelper类，连接Redis，定义频道为monitor，定义发布(publish)及订阅(subscribe)方法。

```python
import redis

class RedisHelper(object):
    def __init__(self):
        self.__conn = redis.Redis(host='192.168.0.110',port=6379)#连接Redis
        self.channel = 'monitor' #定义名称

    def publish(self,msg):#定义发布方法
        self.__conn.publish(self.channel,msg)
        return True

    def subscribe(self):#定义订阅方法
        pub = self.__conn.pubsub()
        pub.subscribe(self.channel)
        pub.parse_response()
        return pub
```

- 发布者

```python
#发布
from RedisHelper import RedisHelper

obj = RedisHelper()
obj.publish('hello')#发布
```

- 订阅

```python
#订阅
from RedisHelper import RedisHelper

obj = RedisHelper()
redis_sub = obj.subscribe()#调用订阅方法

while True:
    msg= redis_sub.parse_response()
    print (msg)
```

## 高可用

为了保证redis最大程度上能够使用，redis提供了主从同步+Sentinel哨兵机制。

Sentinel 哨兵，相关文档https://redis.io/topics/sentinel

redis提供的哨兵是用来看护redis实例进程的，可以自动进行故障转移。具有监控、通知、自动故障转移、配置提供等功能。

```python
from redis.sentinel import Sentinel

# redis 哨兵，不需要直接对接redis地址，直接对接哨兵即可
REDIS_SENTINELS = [
    ('127.0.0.1', '26380'),
    ('127.0.0.1', '26381'),
    ('127.0.0.1', '26382'),
]
REDIS_SENTINEL_SERVICE_NAME = 'mymaster'

_sentinel = Sentinel(REDIS_SENTINELS)
redis_master = _sentinel.master_for(REDIS_SENTINEL_SERVICE_NAME)
redis_slave = _sentinel.slave_for(REDIS_SENTINEL_SERVICE_NAME)

# 读数据，master读不到去slave读
try:
    real_code = redis_master.get(key)
except ConnectionError as e:
    real_code = redis_slave.get(key)

# 写数据，只能在master里写
try:
    current_app.redis_master.delete(key)
except ConnectionError as e:
    logger.error(e)
```

## 集群

Reids Cluster集群方案，内部已经集成了sentinel机制来做到高可用。

> 注意

- redis cluster 不支持事务
- redis cluster 不支持多键操作，如mset

安装

```
pip install redis-py-cluster
```

创建文件redis_cluster.py，示例代码如下

```python
from rediscluster import StrictRedisCluster

if __name__=="__main__":
    try:
        #构建所有的节点，Redis会使用CRC16算法，将键和值写到某个节点上
        startup_nodes=[
            {'host': '172.16.0.136', 'port': '7000'},
            {'host': '172.16.0.135', 'port': '7003'},
            {'host': '172.16.0.136', 'port': '7001'},
        ]
        
        # 构建StrictRedisCluster对象   
        client = StrictRedisCluster(startup_nodes=startup_nodes,decode_responses=True)
        #设置键为py2、值为hr的数据
        client.set('py2','hr')
        #获取键为py2的数据并输出
        print client.get('py2')
    except Exception as e:
        print e
```

# 与Django的交互

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

## 配置

- 作为cache backend使用

为了使用 django-redis , 你应该将你的 django cache setting 改成这样

```python
# project/settings

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/0",  # redis://:[password@]host:port/db
      	"TIMEOUT": 86400, # 1day,设置成0缓存将失效
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    }
}
```

- 作为session backend使用配置

Django 默认可以使用任何 cache backend 作为 session backend, 将 django-redis 作为 session 储存后端不用安装任何额外的 backend

```python
SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"
```

## 使用

- 代码中集成

```shell
from django.core.cache import cache

cache.set('user', 'Aaron', 600) # 保存缓存有效时间为600秒，即10分钟
cache.get('user')  # 获取缓存结果
```

- 原生客户端使用

在某些情况下你的应用需要进入原生 Redis 客户端使用一些 django cache 接口没有暴露出来的进阶特性. 为了避免储存新的原生连接所产生的另一份设置, django-redis 提供了方法 `get_redis_connection(alias)` 使你获得可重用的连接字符串.

```python
from django_redis import get_redis_connection 

redis_client = get_redis_connection('default') # 连接cache配置
for i in range(99999):  
    redis_client.set(i,i)  # 普通方法多次写入，会发多次连接多次发送
 
# 实例化一个pipeline对象，提高多次发送的服务器性能
p1 = redis_client.pipeline() 
for i in range(99999):
    p1.set(i,i) # 使用pipeline执行，会一次连接一次发送多个数据
p1.execute()  # 执行命令
```

