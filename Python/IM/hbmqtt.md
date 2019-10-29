# hbmqtt

hbmqtt是一种针对MQTT协议的python实现，基于Python的标准异步IO框架asyncio，提供基于协程的API，可以开发基于MQTT协议的高并发应用程序

hbmqtt基于MQTT3.1.1协议有如下功能

```
支持3个等级(QoS0,QoS1,QoS2)的消息流
当网络断开时客户端自动发起重连
支持加密认证机制
自带基础系统主题
支持TCP和websocket协议
支持SSl
插件系统
```

## 安装

```
pip3 install hbmqtt
```

相关命令

```
hbmqtt		用于运行hbmqtt的broker
hbmqtt_sub	用于订阅相关主题
hbmqtt_pub	用于在相关主题上发布消息
```

## 命令操作

- 服务器发送消息给网关

功能模块与消息流程

```
hbmqtt的broker运行在服务器上，通过hbmqtt命令启动，负责建立网络连接、支持消息订阅与发布等核心功能

网关作为订阅者去服务器的broker上订阅主题"gateway"，这样一来，只要有主题为“gateway”的消息发布，则该消息就被broker推送给所有订阅该主题的客户端，当然网关包含在内

服务器想要发送消息给网关，只需发布主题为“gateway”的消息即可，一旦发布，则该消息将被broker推送给网关
```

代码实现

```
# 服务器上启动hbmqtt
hbmqtt

# 网关订阅消息
# broker的IP地址192.168.0.4，端口1883，-t表示订阅主题
hbmqtt_sub --url mqtt://192.168.0.4:1883 -t/gateway

# 服务器发布消息
# -m表示发布的消息内容
hbmqtt_pub --url mqtt://192.168.0.4:1883 -t/gateway -m Hi,gateway!
```

- 网关发送消息给服务器

功能模块与消息流程

```
hbmqtt的broker运行在服务器上，通过hbmqtt命令启动，负责建立网络连接、支持消息订阅与发布等核心功能

服务器作为订阅者去服务器的broker上订阅主题“server”，这样一来，只要有主题为“server”的消息发布，则该消息就被broker推送给该主题的客户端，当然，服务器自身也包含在内

网关想要发送消息给服务器，只需发布主题为“server"的消息即可，一旦发布，则该消息将被broker推送给服务器
```

代码实现

```
# 服务器上启动hbmqtt
hbmqtt

# 服务器订阅消息
# broker的IP地址192.168.0.4，端口1883，-t表示订阅主题
hbmqtt_sub --url mqtt://192.168.0.4:1883 -t/server

# 网关发布消息
# -m表示发布的消息内容
hbmqtt_pub --url mqtt://192.168.0.4:1883 -t/server -m Hi,server!
```

## API编程

- 订阅者程序

```python
# sub.py
import logging
import asyncio

from hbmqtt.client import MQTTClient, ClientException
from hbmqtt.mqtt.constants import QOS_1,QOS2

async def sub_test():
    # 实例对象
	C = MQTTClient()
    # 与broker建立连接
    await C.connect('mqtt://192.168.0.4:1883/')
    # 订阅主题并设置消息级别
    await C.subscribe(
        [
            ('server', QOS_1),
            ('gateway', QOS_2),
        ]
    )
	
    print('topic | message')
    print('----- | -------')
    try:
        while True:
            # 等待发布的消息
            message = await C.deliver_meassage()
            packet = message.publish_packet
            print("%s => %s" % (packet.variable_header.topic_name, packet.payload.data.decode()))
        # 停止订阅
        await C.unsubscribe(['server', 'gateway'])
        # 断开与broker的连接
        await C.disconnect()
    except ClientException as ce:
        logger.error("Client exception:%s" % ce)
        
if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(sub_test())
```

- 发布者程序

```python
# pub.py
import asyncio

from hbmqtt.client import MQTTClient
from hbmqtt.mqtt.constants import QOS_0, QOS_1, QOS2

async def publish_test():
    try:
        # 实例化对象
        C = MQTTClient()
        # 建立与broker之间的连接
        ret = await C.connect('mqtt://192.168.0.4:1833/')
        # 发布编码后的信息，并设置信息级别
        message = await C.publish('server', 'MESSAGE-QOS_0'.encode(), qos=QOS_0)
        message = await C.publish('server', 'MESSAGE-QOS_1'.encode(), qos=QOS_1)
        message = await C.publish('gateway', 'MESSAGE-QOS_2'.encode(), qos=QOS_2)
        print("messages published")
        # 断开与broker的连接
        await C.disconnect()
    except ConnectException as ce:
        print("Connection failed:%s" % ce)
        asyncio.get_event_loop().stop()
        
        
if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(publish_test())
```

- 运行与测试

```
# 服务器启动hbmqtt服务
hbmqtt

# 执行订阅者程序
./sub.py

# 执行发布者程序
./pub.py
```

