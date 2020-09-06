# TCP

开发一个广播系统来实践基于TCP的网络应用方法。

该广播系统接收任意客户端的链接请求，并且将任意客户端发送给服务器的消息转发给所有其他客户端。是一个基本的实时通信模型。

## 广播服务器

使用Twisted进行 基于传输层的TCP的编程时，无需操作Socket的`bind,send,receive`等基本原语，而是直接针对Twisted的`Protocol,Factory`等类进行编程，定义它们的子类并重写`connectionMade,dataReceived`进行事件化的TCP编程风格。

```python
from twisted.internet.protocol import Protocol

clients = []  # 保存所有客户端的连接（即Protocol子类Spreader的实例）


class Spreader(Protocol):
    """针对每个客户端链接，Twisted框架建立了一个Protocol子类的实例管理该连接。"""
    def __init__(self, factory):
        self.factory = factory
        self.connect_id = None

    def connectionMade(self):
        """
        当连接建立时由Twisted框架调用，在实际应用中，主要用来在系统中注册该连接，方便以后使用
        """
        self.factory.numProtocols = self.factory.numProtocols + 1  # 连接客户端数+1
        self.connect_id = self.factory.numProtocols
        self.transport.write((u"欢迎来到Spread Site, 您是第%d个客户端用户！\n" %
                              (self.connect_id, )).encode('utf8'))
        print("new connect: %d" % self.connect_id)
        clients.append(self)

    def connectionLost(self, reason):
        """
        当连接断开时由Twisted框架调用。在实际应用中，主要用来清理连接占用的资源
        """
        clients.remove(self)
        print("lost connect: %d" % self.connect_id)

    def dataReceived(self, data):
        """
        当收到客户端的数据时，由Twisted框架调用
        """
        print("dataReceived() entered!")
        if data == "close":  # 收到断开要求
            self.transport.loseConnection()
            print("%s closed" % self.connect_id)
        else:  # 轮询所有客户端，将收到的数据分发给除自己外的所有客户端
            print("spreading message from %s : %s" % (self.connect_id, data))
            for client in clients:  
                if client != self:
                    client.transport.write(data)
        print("dataReceived() existed!")
        

from twisted.internet.protocol import Factory
from twisted.internet.endpoints import TCP4ServerEndpoint
from twisted.internet import reactor


class SpreadFactory(Factory):
    """
    Factory子类起到对Protocol类的管理作用，当有新的客户端连接时，框架用`Facory.buildProtocol()`，使得在这里创建Protocol子类的实例。
    """
    def __init__(self):
        """将客户端计数器置0"""
        self.numProtocols = 0

    def buildProtocol(self, addr):
        """建立Protocol子类的实例"""
        return Spreader(self)

 
if __name__ == "__main__":
    # 8007是本服务器的监听端口，建议选择大于1024的端口
    endpoint = TCP4ServerEndpoint(reactor, 8007)
    endpoint.listen(SpreadFactory())
    reactor.run()  # 挂起运行
```

## 广播客户端

```python
from twisted.internet.protocol import Protocol, ClientFactory
from twisted.internet import reactor
import sys
import datetime


class Echo(Protocol):
    """使用Protocol管理连接"""
    def connectionMade(self):
        print("Connected to the server!")

    def dataReceived(self, data):
        print("got message: ", data.decode('utf8'))
        reactor.callLater(5, self.say_hello)  # 每次收到消息后延时调用say_hello

    def connectionLost(self, reason):
        print("Disconnected from the server!")

    def say_hello(self):
        if self.transport.connected:  # 判断当前是否连接状态
            self.transport.write(  # 向服务器发送消息
                (u"hello, I'm %s %s" %
                 (sys.argv[1], datetime.datetime.now())).encode('utf-8'))


class EchoClientFactory(ClientFactory):
    """使用ClientFactory子类用于构造Protocol的子类的实例"""
    def __init__(self):
        self.protocol = None

    def startedConnecting(self, connector):
        """在连接建立时被调用"""
        print('Started to connect.')

    def buildProtocol(self, addr):
        self.protocol = Echo()
        return self.protocol

    def clientConnectionLost(self, connector, reason):
        """断开连接时被调用"""
        print('Lost connection.  Reason:', reason)

    def clientConnectionFailed(self, connector, reason):
        """连接建立失败时被调用"""
        print('Connection failed. Reason:', reason)

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8007
    factory = EchoClientFactory()
    reactor.connectTCP(host, port, factory)  # 指定要连接的服务器地址和端口
    reactor.run()  # 启动事件循环
```

> 注意

客户端在逻辑需要时可以主动关闭连接

```python
Protocol.transport.loseConnection()
```

在连接建立与关闭时相关回调事件函数的执行顺序

```
1.建立连接
ClientFactory.startedConnecting()
Protocol.connectionMade
2.已连接
Protocol.dataReceived()
Protocol.transport.write()
3.断开连接
Protocol.connectionLost()
ClientFactory.clientConnectionFailed()
```

# UDP

## 普通UDP

UDP是一种无连接对等通讯协议，在UDP层面没有客户端和服务端的概念，通信的任何一方均可通过通信原语和其他方通信

### 完全基于Twisted

一个自发自收的程序

```python
from twisted.internet.protocol import DatagramProtocol

from twisted.internet import reactor
import threading
import time
import datetime

host = "127.0.0.1"
port = 8007


class Echo(DatagramProtocol):
    """定义DatagramProtocol子类"""
    def datagramReceived(self, data, address):
        """定义收到UDP报文后如何处理，收到传入的数据和传入数据发送方的地址"""
        print("Got data from: %s: %s" % (address, data.decode('utf8')))


protocol = Echo()  # 实例化Protocol子类


def routine():  
    """每隔5秒向服务器发送消息"""
    time.sleep(1)
    while True:
        protocol.transport.write(
            ("%s: say hello to myself." % (datetime.datetime.now(), )).encode('utf-8'),
            (host, port))
        time.sleep(5)

if __name__ == "__main__":
    threading.Thread(target=routine).start()
    reactor.listenUDP(port, protocol)  # 指定监听的端口，以便接收其他终端发送的数据
    reactor.run()  # 挂起运行

```

### 适配普通Socket对象

有时需要利用在其他模块中已经建立好的socket对象进行UDP编程

```python
from twisted.internet.protocol import DatagramProtocol
import socket
from twisted.internet import reactor


class Echo(DatagramProtocol):
    """DatagramProtocol子类"""
    def datagramReceived(self, data, address):
        """处理接收的数据"""
        print(data.decode('utf8'))


address = ("127.0.0.1", 8008)

# 使用普通socket编程方法初始化socket对象
recvSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recvSocket.setblocking(False)  # 设为阻塞模式
recvSocket.bind(address)  # 绑定指定端口
reactor.adoptDatagramPort(recvSocket.fileno(), socket.AF_INET, Echo())  # 将protocol对象与socket对象绑定在一起
recvSocket.close()  # 关闭普通socket对象

# 新建一个socket作为发送端
sendSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sendSocket.sendto("Hello my friend!".encode('utf-8'), address)
reactor.run()  # 启动事件循环
```

## Connected UDP

虽然UDP本身时无连接协议，但是在编程接口上仍然可以使用`connect()`，用来限制只与某地址、端口通信。在调用了`connect()`后，当需要向该地址发送UDP数据时就不再需要再指定地址和端口了。这样的技术在twisted中被称为connected UDP.

|                                          | UDP  | Connected UDP | TCP  |
| ---------------------------------------- | ---- | ------------- | ---- |
| 是否点对点通信                           | 否   | 是            | 是   |
| 数据包之间是否有序                       | 否   | 否            | 是   |
| 发送是否可靠（发送方是否知晓数据已到达） | 否   | 是            | 是   |
| 是否支持广播组播                         | 是   | 否            | 否   |

```python
from twisted.internet.protocol import DatagramProtocol
from twisted.internet import reactor
import threading
import time
import datetime

host = "127.0.0.1"
port = 8007


class Echo(DatagramProtocol):
    """定义DatagramProtocol子类""" 
    def startProtocol(self):  
        """当Protocol实例第一次作为参数传递给listenUDP时连接成功后被调用"""
        self.transport.connect(host, port)  # 指定对方地址/端口
        self.transport.write(b"Here is the first connected message")
        print("Connection created!")

    def datagramReceived(self, data, address):  
        """收到数据时被调用"""
        print(data.decode('utf8'))

    def connectionRefused(self):  
        """每次通信失败后被调用，用于实现可靠的数据传输"""
        print("sent failed!")

    def stopProtocol(self):
        """当所有连接都关闭后被调用"""
        print("Connection closed!")


protocol = Echo()  # 实例化Protocol子类


def routine(): 
    """每隔5秒向服务器发送消息"""
    time.sleep(1)
    while True:
        protocol.transport.write(("%s: say hello to myself." %
                                  (datetime.datetime.now(), )).encode('utf-8'))
        time.sleep(5)

if __name__ == "__main__":
    threading.Thread(target=routine).start()
    reactor.listenUDP(port, protocol)
    reactor.run()  # 挂起运行

```

## 组播 

传统的网络通信是基于单播点对点模式的，即每个终端一次只能与另外一个终端通信。而UDP组播提供了一种通信方式：当一个终端发送一条消息时，可以有多个终端接收并进行处理。在局域网设备状态监测、视频通信等应用中经常需要用到UDP组播技术。

IPv4中有一个专有的地址范围被用于组播管理，即224.0.0.0~239.255.255.255。组播参与者(发送者和接收者)在实际收发数据之前需要加入该地址范围中的一个IP地址，之后组中的所有终端都可以用UDP方式向组中的其他终端发送消息。

```python
from twisted.internet.protocol import DatagramProtocol
from twisted.internet import reactor

multicast_ip = "224.0.0.1"  # 组播地址
port = 8001  # 端口


class Multicast(DatagramProtocol):
    """DatagramProtocol子类"""
    def startProtocol(self):
        self.transport.joinGroup(multicast_ip)  # 加入组播组，使自己可以接收到同组中其他终端发来的消息
        self.transport.write(b'Notify', (multicast_ip, port))  # 向组内成员发送组播数据

    def datagramReceived(self, datagram, address):
        """address是真实的IP地址"""
        print("Datagram %s received from %s" % (repr(datagram), repr(address)))
        if datagram == b"Notify":
            self.transport.write(b"Acknowlege", (multicast_ip, port))  # 单播回应


if __name__ == "__main__":
    reactor.listenMulticast(port, Multicast(), listenMultiple=True)  # 组播监听
    reactor.run()  # 挂起运行
```

> 注意

退出组播

```python
DatagramProtocol.transport.leaveGroup()
```



