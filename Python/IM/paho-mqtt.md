# Paho-mqtt

[参考](https://blog.csdn.net/weixin_43986924/article/details/88354797)

首先，MQ 遥测传输 (MQTT) 是轻量级基于代理的发布/订阅的消息传输协议，设计思想是开放、简单、轻量、易于实现。这些特点使它适用于受限环境。该协议的特点有：

```
使用发布/订阅消息模式，提供一对多的消息发布，解除应用程序耦合。
对负载内容屏蔽的消息传输。
使用 TCP/IP 提供网络连接。
小型传输，开销很小（固定长度的头部是 2 字节），协议交换最小化，以降低网络流量。
使用 Last Will 和 Testament 特性通知有关各方客户端异常中断的机制。
有三种消息发布服务质量：
“至多一次”，消息发布完全依赖底层 TCP/IP 网络。会发生消息丢失或重复。这一级别可用于如下情况，环境传感器数据，丢失一次读记录无所谓，因为不久后还会有第二次发送。
“至少一次”，确保消息到达，但消息重复可能会发生。
“只有一次”，确保消息到达一次。这一级别可用于如下情况，在计费系统中，消息重复或丢失会导致不正确的结果
```

## 服务端

常用的有mosquitto，EMQ

## 客户端

```
pip install paho-mqtt
```

### 匿名用户

订阅

```python
import paho.mqtt.client as mqtt

# 当客户端从服务器接收到connack响应时的回调
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # 在on_connect()上订阅，如果我们失去连接并重新连接，则订阅将被续订。
    client.subscribe("chat")

# 从服务器接收发布消息时的回调.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("mqtt.eclipse.org", 1883, 60)

# 阻止处理网络流量、分派回调和处理重新连接的调用
# 还提供了其他的loop*()函数，它们提供了一个线程接口和一个手动接口.
client.loop_forever()
```

发布

```python
# encoding: utf-8
 
import paho.mqtt.client as mqtt
 
HOST = "101.200.46.138"
PORT = 1883
 
def test():
    client = mqtt.Client()
    client.connect(HOST, PORT, 60)
    client.publish("chat","hello liefyuan",2) # 发布一个主题为'chat',内容为‘hello liefyuan’的信息
    client.loop_forever()
 
if __name__ == '__main__':
    test()
```

### 用户登陆

订阅者

```python
# coding=utf-8
import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("/+")

def on_message(client, userdata, msg):
    print(msg.topic+" "+msg.payload.decode("utf-8"))

def test():
    HOST= 'test.mosquitto.org'
    PORT = 1883
    KEEPALIVE = 60
    TOPIC = '#'

    
    client_id = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
    client = mqtt.Client(client_id)    # ClientId不能重复
    # client = mqtt.Client()    # 不需要设置ClientId时
    client.username_pw_set("admin", "password")  # 用户名密码
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port=port, keepalive=)
    client.loop_forever()

if __name__ == '__main__':
    test()
```

发布者

```python
# coding=utf-8
import paho.mqtt.client as mqtt
import json

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe("chat")
    client.publish("chat", json.dumps({"user": user, "say": "Hello,anyone!"}))


def on_message(client, userdata, msg):
    #print(msg.topic+":"+str(msg.payload.decode()))
    #print(msg.topic+":"+msg.payload.decode())
    payload = json.loads(msg.payload.decode())
    print(payload.get("user")+":"+payload.get("say"))


if __name__ == '__main__':
    client = mqtt.Client()
    client.username_pw_set("admin", "password")  # 必须设置，否则会返回「Connected with result code 4」
    client.on_connect = on_connect
    client.on_message = on_message

    # HOST = "127.0.0.1"
    HOST= 'test.mosquitto.org'

    client.connect(HOST, 1883, 60)
    #client.loop_forever()

    user = input("请输入名称:")
    client.user_data_set(user)

    client.loop_start()

    while True:
        str = input()
        if str:
            client.publish("#", json.dumps({"user": user, "say": str}))
```

### 连接使用

- 发布

第一种保持连接的方式是在`keeplive`的间隔内，发布消息或者调用`loop()`。

```python
client.connect('127.0.0.1', 1883, 5) # keeplive仅为5秒
for i in range(100):
    client.publish('fifa', payload=f'amazing{i}', qos=0)
    # client.loop() # 或者loop()
    time.sleep(4) # 不能超过5秒
```

第二种方式是使用`loop_start()`

```python
client.connect('127.0.0.1', 1883, 5)
client.loop_start()
for i in range(100):
    client.publish('fifa', payload=f'amazing{i}', qos=0)
    time.sleep(6) # 可以超过5秒了
```

- 订阅

一种方法是使用`loop_start()`保持连接，然后写个死循环阻塞程序，保持监听。

```python
client.connect('127.0.0.1', 1883, 5)
client.subscribe('fifa', qos=0)
client.loop_start()
while True:
    pass
```

第二种方法直接使用`loop_forever()`，也能阻塞运行

```python
client.connect('127.0.0.1', 1883, 5)
client.subscribe('fifa', qos=0)
client.loop_forever()
```

## API

[参考1](https://blog.csdn.net/weixin_41656968/article/details/80848542)  [参考2](https://github.com/eclipse/paho.mqtt.python)

### 常用函数

| name              | desc                 |
| ----------------- | -------------------- |
| `connect()`       | 连接Broker           |
| `connect_async()` | 连接Broker           |
| `loop()`          | 保持与Broker网络连接 |
| `loop_start()`    | 调用一个`loop()`进程 |
| `loop_forever()`  | 保持`loop()`调用     |
| `subscribe()`     | 订阅主题并接收消息   |
| `publish()`       | 发布消息             |
| `disconnect()`    | 与Broker断开连接     |

使用回调函数使Broker返回数据可用，例子如下：

```python
def on_connect(client, userdata, flags, rc):
  print("Connection returned " + str(rc))
client.on_connect = on_connect
```

所有的回调都有一个“client”和一个“userdata”参数，“client”是调用回调的客户端实例，“userdata”是任何类型的用户数据，可以在创建新客户端实例时设置或者使用`user_data_set(userdata)`

`on_connect(client, userdata, flags, rc)`

当Broker响应我们请求时调用，“flags” 是一个包含Broker响应参数的字典:flags['session present'] –此标志仅对于干净会话设置为0，如果设置session=0,用于客户端重新连接到之前Broker是否仍然保存之前会话信息，如果设1，会话一直存在。“rc”值用于判断是否连接成功:

```undefined
0: 连接成功
1: 连接失败-不正确的协议版本
2: 连接失败-无效的客户端标识符
3: 连接失败-服务器不可用
4: 连接失败-错误的用户名或密码
5: 连接失败-未授权
6-255: 未定义.
```

`on_disconnect(client, userdata, rc)`

当客户端与Broker断开时调用

`on_message(client, userdata, message)`

在客户端订阅的主题上接收到消息时调用，“message”变量是一个MQTT消息描述所有消息特征

`on_publish(client, userdata, mid)`

当使用publish()发送的消息已经完成传输到代理时调用。对于QoS级别为1和2的消息，这意味着适当的握手已经完成。对于QoS 0，这仅仅意味着消息已经离开客户端。“mid”变量是从相应的publish()调用返回的中间变量。这个回调很重要，因为即使publish()调用返回成功，也并不总是意味着消息已经被发送

`on_subscribe(client, userdata, mid, granted_qos)`

当Broker响应订阅请求时调用，“mid”变量是从相应的subscribe()调用返回的中间变量，“granted_qos”变量是每次发送不同订阅请求Qos级别的列表

`on_unsubscribe(client, userdata, mid)`

当Broker响应取消订阅请求时调用，“mid“变量是从相应的unsubscribe()调用返回的中间变量

`on_log(client, userdata, level, buf)`

当客户端有日志信息时调用，定义允许调试，“level“变量是消息级别包含MQTT_LOG_INFO, MQTT_LOG_NOTICE, MQTT_LOG_WARNING, MQTT_LOG_ERR, MQTT_LOG_DEBUG，消息本身是buf。

`loop()`

一个心跳函数，用来保持客户端与服务器的连接。比如`keepalive`参数为60秒，那么60秒内必须`loop()`一下或者发布一下消息，不然连接会断，就无法继续发布或者接受消息。

`loop_start()`

启用一个进程保持`loop()`的重复调用，就不需要定期心跳了，对应的有`loop_stop()`。

`loop_forever()`

用来保持无穷阻塞调用`loop()`



