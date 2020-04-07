#keyspace通知

[参考](https://www.cnblogs.com/leguan1314/p/9642859.html)

## 概述

Redis的键盘空间通知从2.8.0版起就可以使用了。对于更改任何Redis键的每个操作，可以配置Redis将消息发布到发布/订阅。然后可以订阅这些通知。值得一提的是，事件仅在确实修改了键的情况下才生成。例如，删除不存在的键将不会生成事件。

Redis的一个常见用例是，应用程序需要能够响应存储在特定键或键中的值可能发生的更改。由于有了键盘空间通知和发布/订阅，可以对Redis数据中的变化做出响应。通知很容易使用，而事件处理器可能在地理位置上分布。

最大的缺点是，Pub/Sub的实现要求发布者和订阅方始终处于启动状态。用户在停止或连接丢失时丢失数据

## 配置

默认情况下，redis的通知事件是关闭的，在终端执行以下命令开启：

```shell
$ redis-cli config set notify-keyspace-events KEA
OK 
```

KEA字符串表示启用了所有可能的事件。要查看每个字符的含义，[请查看文档](https://redis.io/topics/notifications)

CLI 可以在特殊模式下工作，允许您订阅一个通道以接收消息。

现在检查事件是否起作用：

```shell
# 用于检查事件
# psubscribe '*'表示我们想订阅所有带有模式*的事件

$ redis-cli --csv psubscribe '*'     
Reading messages... (press Ctrl-C to quit)  
"psubscribe","*",1
```

开启一个新的终端，设置一个值

```shell
127.0.0.1:6379> set key1 value1  
OK  
```

在上一个终端，会看到：

```shell
$ redis-cli --csv psubscribe '*'     
Reading messages... (press Ctrl-C to quit)  
"psubscribe","*",1
"pmessage","*","__keyspace@0__:key1","set"
"pmessage","*","__keyevent@0__:set","key1
```

发现通知是工作中的

以上收到三个事件:

- 第一个事件意味着已经成功订阅了reply中作为第二个元素给出的通道。1 表示目前订阅的频道数量。
- 第二个事件是键空间通知。在keyspace通道中，接收事件集的名称作为消息。
- 第三个事件是键-事件通知。在keyevent通道中，接收到key key1的名称作为消息。

## Pub/Sub

事件是通过Redis的Pub/Sub层交付的。

为了订阅channel channel1和channel2，客户端发出带有通道名称的subscribe命令

## 在python中订阅通知

第一步，需要python操作redis的包

```shell
$ pip install redis
```

事件循环，请看以下代码：

```python
import time  
from redis import StrictRedis

# 创建连接
# 默认情况下，所有响应都以字节的形式返回。用户负责解码。如果客户机的所有字符串响应都应该被解码，那么用户可以指定decode_responses=True to StrictRedis。在这种情况下，任何返回字符串类型的Redis命令都将用指定的编码进行解码。
redis = StrictRedis(host='localhost', port=6379)

# redis 发布订阅
pubsub = redis.pubsub()  
# 监听通知
pubsub.psubscribe('__keyspace@0__:*')

# 开始消息循环
print('Starting message loop')  
while True:  
   	# 获取消息
    # 如果有数据，get_message()将读取并返回它。如果没有数据，该方法将不返回任何数据。
    message = pubsub.get_message()
    if message:
        print(message)
    else:
        time.sleep(0.01)
```

从pubsub实例中读取的每条消息都是一个字典，其中包含以下键：
```
- type以下之一:订阅，取消订阅，psubscribe, punsubscribe, message, pmessage
- channel  订阅消息的通道或消息发布到的通道
- pattern  与已发布消息的通道匹配的模式(除pmessage类型外，在所有情况下都不匹配)
- data  消息数据
```

现在启动 `python` 脚本，在终端中设置一个key值

```
127.0.0.1:6379> set mykey myvalue  
OK  
```

将看到脚本以下输出

```
$ python subscribe.py
Starting message loop  
{'type': 'psubscribe', 'data': 1, 'channel': b'__keyspace@0__:*', 'pattern': None}
{'type': 'pmessage', 'data': b'set', 'channel': b'__keyspace@0__:mykey', 'pattern': b'__keyspace@0__:*'}
```

## 回调注册

注册回调函数来处理已发布的消息。消息处理程序接受一个参数，即消息。要使用消息处理程序订阅通道或模式，请将通道或模式名称作为关键字参数传递，其值为回调函数。当使用消息处理程序在通道或模式上读取消息时，将创建消息字典并将其传递给消息处理程序。在这种情况下，`get_message()`返回一个None值，因为消息已经被处理

```python
import time  
from redis import StrictRedis

redis = StrictRedis(host='localhost', port=6379)

pubsub = redis.pubsub()

def event_handler(msg):  
    print('Handler', msg)

pubsub.psubscribe(**{'__keyspace@0__:*': event_handler})

print('Starting message loop')  
while True:  
    message = pubsub.get_message()
    if message:
        print(message)
    else:
        time.sleep(0.01)
```

## 事件循环在单独的线程中

```python
import time  
from redis import StrictRedis

redis = StrictRedis(host='localhost', port=6379)

def event_handler(msg):  
    print(msg)
    thread.stop()  

pubsub = redis.pubsub()  
pubsub.psubscribe(**{'__keyevent@0__:expired': event_handler})  
thread = pubsub.run_in_thread(sleep_time=0.01)
```

