# Redis

## 安装

**Ubuntu**

```
sudo apt-get install software-properties-common
sudo apt-add-repository ppa:chris-lea/redis-server
sudo apt-get update
sudo apt-get install redis-server
```

**官方下载**

```
1、下载：打开redis官方网站，推荐下载稳定版本(stable)

2、解压
tar zxvf redis-3.2.5.tar.gz

3、复制，放到usr/local目录下
sudo mv -r redis-3.2.5/* /usr/local/redis/

4、进入redis目录
cd /usr/local/redis/

5、生成
sudo make

6、测试,这段运行时间会较长
sudo make test

7、安装,将redis的命令安装到/usr/bin/目录
sudo make install

8、配置文件，移动到/etc/目录下
配置文件目录为/usr/local/redis/redis.conf
sudo cp /usr/local/redis/redis.conf /etc/redis/
```

## 配置

```
# 绑定ip：如果需要远程访问，可将此行注释，或绑定一个真实ip
bind 127.0.0.1

# 端口，默认为6379
port 6379

# 是否以守护进程运行,如果以守护进程运行，则不会在命令行阻塞，类似于服务;如果以非守护进程运行，则当前终端被阻塞;设置为yes表示守护进程，设置为no表示非守护进程
daemonize yes

# 数据文件
dbfilename dump.rdb

# 数据文件存储路径
dir /var/lib/redis

# 日志文件
logfile /var/log/redis/redis-server.log

# 数据库，默认有16个
database 16
```

## 服务端

```
# 启动服务器
# 服务端的命令
redis-server

# 可以使用help查看帮助文档
redis-server --help

# 启动
sudo service redis start

# 停止
sudo service redis stop
sudo kill -9 进程pid

# 重启
sudo service redis restart
```

## 客户端

```
# 客户端的命令
redis-cli

# 启动本地客户端
redis-cli
# 启动带ip地址客户端
redis-cli -h 192.168.42.87

# help查看帮助文档
redis-cli --help

# 运行测试命令
ping

# 切换数据库,数据库没有名称，默认有16个，通过0-15来标识
select 0
```

## 数据操作

```
redis是key-value的数据结构，每条数据都是一个键值对

键的类型是字符串
注意：键不能重复，空为nil

值的类型分为五种：
字符串string， 值类型是二进制安全的，可以存储任何数据，比如数字、图片等
哈希hash，值类型是string，用于存储对象，对象的结构为属性、值
列表list，值类型为string，按照插入顺序排序
集合set ,元素类型为string，元素具有唯一性，对集合没有修改操作
有序集合zset，元素类型为string，元素有唯一性，每个元素都会关联一个double类型的score，表示权重，通过权重将元素从小到大排序，没有修改操作

keys *
```

### string

```
# 增加、修改
# 设置键值
set key value

# 设置键值及过期时间，以秒为单位
setex key seconds value

# 设置多个键值
mset key1 value1 key2 value2 ...

# 追加值
append key value

# 获取：根据键获取值，如果不存在此键则返回nil
get key

# 根据多个键获取多个值
mget key1 key2 ...
```

### 键命令

```
# 查找键，参数支持正则表达式
keys pattern

# 判断键是否存在，如果存在返回1，不存在返回0
exists key1

# 查看键对应的value的类型
type key

# 删除键及对应的值
del key1 key2 ...

# 设置过期时间，以秒为单位,如果没有指定过期时间则一直存在，直到使用DEL移除
expire key seconds

# 查看有效时间，以秒为单位
ttl key
```

### hash

```
# 增加、修改
# 设置单个属性
hset key field value

# 设置多个属性
hmset key field1 value1 field2 value2 ...

# 获取
# 获取指定键所有的属性
hkeys key

# 获取一个属性的值
hget key field

# 获取多个属性的值
hmget key field1 field2 ...

# 获取所有属性的值
hvals key

# 删除
# 删除整个hash键及值，使用del命令，删除属性，属性对应的值会被一起删除
hdel key field1 field2 ...
```

### list

```
# 增加
# 在左侧插入数据
lpush key value1 value2 ...

# 在右侧插入数据
rpush key value1 value2 ...

# 在指定元素的前或后插入新元素
linsert key before或after 现有元素 新元素

# 获取
# 返回列表里指定范围内的元素,
# start、stop为元素的下标索引,索引从左侧开始，第一个元素为0,
# 索引可以是负数，表示从尾部开始计数，如-1表示最后一个元素
lrange key start stop


# 修改
# 设置指定索引位置的元素值，
# 索引从左侧开始，第一个元素为0，
# 索引可以是负数，表示尾部开始计数，如-1表示最后一个元素
lset key index value

# 删除
# 删除指定元素，
# 将列表中前count次出现的值为value的元素移除，
# count > 0: 从头往尾移除，
# count < 0: 从尾往头移除，
# count = 0: 移除所有
lrem key count value
```

### set

```
# 增加
# 添加元素
sadd key member1 member2 ...

# 获取
# 返回所有的元素
smembers key

# 删除
# 删除指定元素
srem key member
```

### zset

```
# 增加
# 添加
zadd key score1 member1 score2 member2 ...

# 获取
# 返回指定范围内的元素
# start、stop为元素的下标索引
# 索引从左侧开始，第一个元素为0
# 索引可以是负数，表示从尾部开始计数，如-1表示最后一个元素
zrange key start stop

# 返回score值在min和max之间的成员
zrangebyscore key min max

# 返回成员member的score值
zscore key member

# 删除
# 删除指定元素
zrem key member1 member2 ...

# 删除权重在指定范围的元素
zremrangebyscore key min max
```

## 搭建主从

```
一个master可以拥有多个slave，一个slave又可以拥有多个slave，如此下去，形成了强大的多级服务器集群架构

比如，将ip为192.168.1.10的机器作为主服务器，将ip为192.168.1.11的机器作为从服务器

说明：ip可以换为自己机器与同桌机器的地址

# 设置主服务器的配置
bind 192.168.1.10

# 设置从服务器的配置
# 注意：在slaveof后面写主机ip，再写端口，而且端口必须写
bind 192.168.1.11
slaveof 192.168.1.10 6379

# 在master和slave分别执行info命令，查看输出信息

# 在master上写数据
set hello world

# 在slave上读数据
get hello
```

## 搭建集群

当前拥有两台主机192.168.12.107、192.168.12.84，这里的IP在使用时要改为实际值

### 配置机器1

- 在演示中，192.168.12.107为当前ubuntu机器的ip
- 在192.168.12.107上进入Desktop目录，创建redis目录
- 在redis目录下创建文件7000.conf，编辑内容如下

```
port 7000
bind 192.168.12.107
daemonize yes
pidfile 7000.pid
cluster-enabled yes
cluster-config-file 7000_node.conf
cluster-node-timeout 15000
appendonly yes
```

- 在redis目录下创建文件7001.conf，编辑内容如下

```
port 7001
bind 192.168.12.107
daemonize yes
pidfile 7001.pid
cluster-enabled yes
cluster-config-file 7001_node.conf
cluster-node-timeout 15000
appendonly yes
```

- 在redis目录下创建文件7002.conf，编辑内容如下

```
port 7002
bind 192.168.12.107
daemonize yes
pidfile 7002.pid
cluster-enabled yes
cluster-config-file 7002_node.conf
cluster-node-timeout 15000
appendonly yes
```

- 总结：三个文件的配置区别在port、pidfile、cluster-config-file三项
- 使用配置文件启动redis服务

```
redis-server 7000.conf
redis-server 7001.conf
redis-server 7002.conf
```

- 查看进程

```
ps ajx|grep redis
```

### 配置机器2

- 在演示中，192.168.12.84为学生的一台ubuntu机器的ip，为了演示方便，使用ssh命令连接

```
ssh 192.168.12.84
```

- 在192.168.12.84上进入Desktop目录，创建redis目录
- 在redis目录下创建文件7003.conf，编辑内容如下

```
port 7003
bind 192.168.12.84
daemonize yes
pidfile 7003.pid
cluster-enabled yes
cluster-config-file 7003_node.conf
cluster-node-timeout 15000
appendonly yes

```

- 在redis目录下创建文件7004.conf，编辑内容如下

```
port 7004
bind 192.168.12.84
daemonize yes
pidfile 7004.pid
cluster-enabled yes
cluster-config-file 7004_node.conf
cluster-node-timeout 15000
appendonly yes

```

- 在redis目录下创建文件7005.conf，编辑内容如下

```
port 7005
bind 192.168.12.84
daemonize yes
pidfile 7005.pid
cluster-enabled yes
cluster-config-file 7005_node.conf
cluster-node-timeout 15000
appendonly yes

```

- 总结：三个文件的配置区别在port、pidfile、cluster-config-file三项
- 使用配置文件启动redis服务

```
redis-server 7003.conf
redis-server 7004.conf
redis-server 7005.conf
```

- 查看进程

```
ps ajx|grep redis
```

### 创建集群

- redis的安装包中包含了redis-trib.rb，用于创建集群
- 接下来的操作在192.168.12.107机器上进行
- 将命令复制，这样可以在任何目录下调用此命令

```
sudo cp /usr/share/doc/redis-tools/examples/redis-trib.rb /usr/local/bin/
```

- 安装ruby环境，因为redis-trib.rb是用ruby开发的

```
sudo apt-get install ruby
```

- 在提示信息处输入y，然后回车继续安装


- 运行如下命令创建集群

```
redis-trib.rb create --replicas 1 192.168.12.107:7000 192.168.12.107:7001  192.168.12.107:7002 192.168.12.84:7003  192.168.12.84:7004  192.168.12.84:7005
```

- 提示主从信息，输入yes后回车


- 提示完成，集群搭建成功

**数据验证**

- 根据上图可以看出，当前搭建的主服务器为7000、7001、7003，对应的从服务器是7004、7005、7002
- 在192.168.12.107机器上连接7002，加参数-c表示连接到集群

```
redis-cli -h 192.168.12.107 -c -p 7002
```

- 写入数据

```
set hello world
```

- 自动跳到了7000服务器，并写入数据成功


- 7000对应的从服务器为7004，所以在192.168.12.84服务器连接7004，查看数据如下


- 在192.168.12.84服务器连接7005是没有数据的

**在哪个服务器上写数据：CRC16**

- redis cluster在设计的时候，就考虑到了去中心化，去中间件，也就是说，集群中的每个节点都是平等的关系，都是对等的，每个节点都保存各自的数据和整个集群的状态。每个节点都和其他所有节点连接，而且这些连接保持活跃，这样就保证了我们只需要连接集群中的任意一个节点，就可以获取到其他节点的数据 
- Redis集群没有并使用传统的一致性哈希来分配数据，而是采用另外一种叫做哈希槽 (hash slot)的方式来分配的。redis cluster 默认分配了 16384 个slot，当我们set一个key 时，会用CRC16算法来取模得到所属的slot，然后将这个key 分到哈希槽区间的节点上，具体算法就是：CRC16(key) % 16384。所以我们在测试的时候看到set 和 get 的时候，直接跳转到了7000端口的节点
- Redis 集群会把数据存在一个 master 节点，然后在这个 master 和其对应的salve 之间进行数据同步。当读取数据时，也根据一致性哈希算法到对应的 master 节点获取数据。只有当一个master 挂掉之后，才会启动一个对应的 salve 节点，充当 master
- 需要注意的是：必须要3个或以上的主节点，否则在创建集群时会失败，并且当存活的主节点数小于总节点数的一半时，整个集群就无法提供服务了

### python交互

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

