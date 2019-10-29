# MQTT

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

## 搭建服务器



## 客户端

```
pip install paho-mqtt
```

### 发布者

```python

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

    HOST = "127.0.0.1"

    client.connect(HOST, 61613, 60)
    #client.loop_forever()

    user = input("请输入名称:")
    client.user_data_set(user)

    client.loop_start()

    while True:
        str = input()
        if str:
            client.publish("chat", json.dumps({"user": user, "say": str}))
```



### 订阅者

```python
import paho.mqtt.client as mqtt

HOST = "192.168.10.8"
PORT = 61613

def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    client.subscribe("/+")

def on_message(client, userdata, msg):
    print(msg.topic+" "+msg.payload.decode("utf-8"))

def test():
    client = mqtt.Client()    # 可能需要设置ClientId
    client.username_pw_set("admin", "password")  # 必须设置，否则会返回「Connected with result code 4」
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(HOST, PORT, 60)
    client.loop_forever()

if __name__ == '__main__':
    test()

```

