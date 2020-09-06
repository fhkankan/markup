# Socket

```
socket(简称 套接字) 是进程间通信的一种方式，它与其他进程间通信的一个主要不同是：
它能实现不同主机间的进程间通信，我们网络上各种各样的服务大多都是基于 Socket 来完成通信的
```

## 概述

创建socket

```python
import socket
socket.socket(AddressFamily, Type)
```

在 Python 中 使用socket 模块的函数 socket 就可以完成

函数 socket.socket 创建一个 socket，该函数带有两个参数

```
Address Family：可以选择 AF_INET（用于 Internet 进程间通信） 或者 AF_UNIX（用于同一台机器进程间通信）,实际工作中常用AF_INET

Type：套接字类型，可以是 SOCK_STREAM（流式套接字，主要用于 TCP 协议）或者 SOCK_DGRAM（数据报套接字，主要用于 UDP 协议）
```

**内建方法**

| 函数                                 | 描述                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| 服务器端套接字                       |                                                              |
| `s.bind()`                             | 绑定地址（host,port）到套接字， 在AF_INET下,以元组（host,port）的形式表示地址。 |
| `s.listen()`                           | 开始TCP监听。backlog指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。 |
| `s.accept()`                          | 被动接受TCP客户端连接,(阻塞式)等待连接的到来                 |
| 客户端套接字                         |                                                              |
| `s.connect()`                          | 主动初始化TCP服务器连接，。一般address的格式为元组（hostname,port），如果连接出错，返回socket.error错误。 |
| `s.connect_ex()`                       | connect()函数的扩展版本,出错时返回出错码,而不是抛出异常      |
| 公共用途的套接字函数                 |                                                              |
| `s.recv()`                             | 接收TCP数据，数据以字符串形式返回，bufsize指定要接收的最大数据量。flag提供有关消息的其他信息，通常可以忽略。 |
| `s.send()`                             | 发送TCP数据，将string中的数据发送到连接的套接字。返回值是要发送的字节数量，该数量可能小于string的字节大小。 |
| `s.sendall()`                          | 完整发送TCP数据，完整发送TCP数据。将string中的数据发送到连接的套接字，但在返回之前会尝试发送所有数据。成功返回None，失败则抛出异常。 |
| `s.recvfrom()`                         | 接收UDP数据，与recv()类似，但返回值是（data,address）。其中data是包含接收数据的字符串，address是发送数据的套接字地址。 |
| `s.sendto() `                          | 发送UDP数据，将数据发送到套接字，address是形式为（ipaddr，port）的元组，指定远程地址。返回值是发送的字节数。 |
| `s.close()`                            | 关闭套接字                                                   |
| `s.getpeername()`                      | 返回连接套接字的远程地址。返回值通常是元组（ipaddr,port）。  |
| `s.getsockname()`                      | 返回套接字自己的地址。通常是一个元组(ipaddr,port)            |
| `s.setsockopt(level,optname,value)`    | 设置给定套接字选项的值。                                     |
| `s.getsockopt(level,optname[.buflen])` | 返回套接字选项的值。                                         |
| `s.settimeout(timeout) `               | 设置套接字操作的超时期，timeout是一个浮点数，单位是秒。值为None表示没有超时期。一般，超时期应该在刚创建套接字时设置，因为它们可能用于连接的操作（如connect()） |
| `s.gettimeout() `                      | 返回当前超时期的值，单位是秒，如果没有设置超时期，则返回None。 |
| `s.fileno()`                           | 返回套接字的文件描述符。                                     |
| `s.setblocking(flag)`                  | 如果flag为0，则将套接字设为非阻塞模式，否则将套接字设为阻塞模式（默认值）。非阻塞模式下，如果调用recv()没有发现任何数据，或send()调用无法立即发送数据，那么将引起socket.error异常。 |
| `s.makefile() `                        | 创建一个与该套接字相关连的文件                               |

## TCP

- 客户端

创建一个客户端流程

```
1.创建socket
2.连接服务器
3.发送编码后数据
4.接收数据并解码
5.关闭连接
```

实现

```python
from socket import *

# 创建socket
tcp_client_socket = socket(AF_INET, SOCK_STREAM)

# 目的信息
server_ip = input("请输入服务器ip:")
server_port = int(input("请输入服务器port:"))

# 链接服务器
tcp_client_socket.connect((server_ip, server_port))

# 提示用户输入数据
send_data = input("请输入要发送的数据：")

tcp_client_socket.send(send_data.encode("utf8"))

# 接收对方发送过来的数据，最大接收1024个字节
recvData = tcp_client_socket.recv(1024)

# 对接收的数据解码
recvContent = recvData.decode('utf8')

print(recvContent)

# 关闭套接字
tcp_client_socket.close()
```

- 服务端

如果想要完成一个tcp服务器的功能，需要的流程如下：

```
1. socket创建一个套接字
2. bind绑定ip和port
3. listen使套接字变为可以被动链接
4. accept等待客户端的链接
5. recv/send接收发送数据
```

示例

```python
from socket import *
import threading

# 创建socket
tcp_server_socket = socket(AF_INET, SOCK_STREAM)
# 释放端口
tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

# ip地址和端口号，ip一般不用写，表示本机的任何一个ip
address = ('', 7788)
# 绑定地址
tcp_server_socket.bind(address)
# 使用socket创建的套接字默认的属性是主动的，使用listen将其变为被动的，这样就可以接收别人的链接了，
# 128是等待accept处理的最大链接数
tcp_server_socket.listen(128)

while True:
		# 如果有新的客户端来链接服务器，那么就产生一个新的套接字专门为这个客户端服务
		# client_socket用来为这个客户端服务
		# tcp_server_socket就可以省下来专门等待其他新客户端的链接
		client_socket, clientAddr = tcp_server_socket.accept()
    
    # 创建新的线程来处理TCP连接
    t = threading.Thread(target=tcplink(client_socket, clientAddr))
    t.start()

# 关闭监听套接字
# tcp_server_socket.close()
    
def tplink(client_socket, clientAddr):
    print('Accept new connection from %s:%s...' % clientAddr)
  	while True:
		# 接收对方发送过来的数据
		recv_data = client_socket.recv(1024)  # 接收1024个字节
		print('接收到的数据为:', recv_data.decode('utf8'))
		if recv_data=='exit' or not recv_data:
            break	
		# 发送一些数据到客户端
		client_socket.send("thank you !".encode('utf8'))
		# 关闭为这个客户端服务的套接字，只要关闭了，就意味着为不能再为这个客户端服务了，如果还需要服务，只能再次重新连接
		client_socket.close()
```

## UDP

创建一个基于udp的网络程序流程很简单，具体步骤如下：
```
1. 创建客户端套接字
2. 发送/接收数据
3. 关闭套接字
```
发送数据

```python
from socket import *

# 1. 创建udp套接字
udp_socket = socket(AF_INET, SOCK_DGRAM)

# 2. 准备接收方的地址
# '192.168.1.103'表示目的ip地址
# 8080表示目的端口
dest_addr = ('192.168.1.103', 8080)  # 注意 是元组，ip是字符串，端口是数字

# 3. 从键盘获取数据
send_data = input("请输入要发送的数据:")

# 4. 发送数据到指定的电脑上的指定程序中
udp_socket.sendto(send_data.encode('utf-8'), dest_addr)

# 5. 关闭套接字
udp_socket.close()
```

发送接收数据

```python
#coding=utf-8

from socket import *

# 1. 创建udp套接字
udp_socket = socket(AF_INET, SOCK_DGRAM)

# 2. 准备接收方的地址
dest_addr = ('192.168.236.129', 8080)

# 3. 从键盘获取数据
send_data = input("请输入要发送的数据:")

# 4. 发送数据到指定的电脑上
udp_socket.sendto(send_data.encode('utf-8'), dest_addr)

# 5. 等待接收对方发送的数据
recv_data = udp_socket.recvfrom(1024)  # 1024表示本次接收的最大字节数

# 6. 显示对方发送的数据
# 接收到的数据recv_data是一个元组
# 第1个元素是对方发送的数据
# 第2个元素是对方的ip和端口
print(recv_data[0].decode('gbk'))
print(recv_data[1])

# 7. 关闭套接字
udp_socket.close()
```

服务端绑定端口

```python
#coding=utf-8

from socket import *

# 1. 创建套接字
udp_socket = socket(AF_INET, SOCK_DGRAM)

# 2. 绑定本地的相关信息，如果一个网络程序不绑定，则系统会随机分配
local_addr = ('', 7788) 
udp_socket.bind(local_addr)

# 3. 等待接收对方发送的数据
recv_data = udp_socket.recvfrom(1024) #  1024表示本次接收的最大字节数

# 4. 显示接收到的数据
print(recv_data[0].decode('gbk'))

# 5. 关闭套接字
udp_socket.close()
```

## 模拟http

日常使用中，http请求的处理使用web框架或`requests`模块

```python
#requests -> urlib -> socket
import socket
from urllib.parse import urlparse


def get_url(url):
    #通过socket请求html
    url = urlparse(url)
    host = url.netloc
    path = url.path
    if path == "":
        path = "/"

    #建立socket连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.setblocking(False)
    client.connect((host, 80)) #阻塞不会消耗cpu

    #不停的询问连接是否建立好， 需要while循环不停的去检查状态
    #做计算任务或者再次发起其他的连接请求

    client.send("GET {} HTTP/1.1\r\nHost:{}\r\nConnection:close\r\n\r\n".format(path, host).encode("utf8"))

    data = b""
    while True:
        d = client.recv(1024)
        if d:
            data += d
        else:
            brea

    data = data.decode("utf8")
    html_data = data.split("\r\n\r\n")[1]
    print(html_data)
    client.close()

if __name__ == "__main__":
    import time
    start_time = time.time()
    for url in range(20):
        url = "http://shop.projectsedu.com/goods/{}/".format(url)
        get_url(url)
    print(time.time()-start_time)
```

# Asyncio

标准库asyncio提供的BaseTransport,ReadTransport,WriteTransport,DatagramTransport以及BaseSubprocessTransport类对不同类型的信道进行了抽象。一般来说，不建议使用这些类直接实例化对象，而是使用AbstarctEventLoop函数来创建相应的Transport对象并且对底层信道进行初始化。一旦信道创建成功，可以通过一对Protocol对象进行通信了。目前asyncio支持TCP,UDP,SSL和Subprocess管道，不同类型的Transport对象支持的方法略有不同，另外需注意：Transport类不是线程安全的

标准库asyncio还提供了类Protocol,DatagramProtocol和SubprocessProtocl,这些类可用作基类进行二次开发来实现自己的网络协议，创建派生类时只需重写感兴趣的回调函数即可。Protocol类常与Transport类一起使用，Protocol对象解析收到的数据并请求待发出数据的读写操作，而Transport对象则负责实际的I/O操作和必要的缓冲

Protocol对象常用回调函数

| 函数名称                        | 说明                                                         | 适用对象                                     |
| ------------------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| `connection_made(transport)`    | 连接建立后自动调用                                           | Protocol,DatagramProtocol,SubProcessProtocol |
| `connection_lost(exc)`          | 连接丢失或关闭后自动调用                                     | Protocol,DatagramProtocol,SubProcessProtocol |
| `pipe_data_received(fd, data)`  | 子进程往stdout或stderr管道中写入数据时自动调用，fd是管道的标识符，data是要卸乳的非空字节串 | SubProcessProtocol                           |
| `pipe_connection_lost(fd, exc)` | 与子进程通信的管道被关闭时自动调用                           | SubProcessProtocol                           |
| `process_exited()`              | 子进程退出后自动调用                                         | SubProcessProtocol                           |
| `data_received(data)`           | 收到数据(字节串)时自动调用                                   | Protocol                                     |
| `eof_received()`                | 通信对象通过write_eof()或者其他类似方法通知不再发送数据时自动调用 | Protocol                                     |
| `datagram_received(data, addr)` | 收到数据报时自动调用                                         | DatagramProtocol                             |
| `error_received(exc)`           | 前一次发送或接收操作抛出异常OSError时自动调用                | DatagramProtocol                             |
| `pause_writing()`               | Transport对象缓冲区达到上水位线时自动调用                    | Protocol,DatagramProtocol,SubProcessProtocol |
| `resume_writing()`              | Transport对象缓冲区达到下水位线时自动调用                    | Protocol,DatagramProtocol,SubProcessProtocol |

可以在Protocol对象的方法中使用`ensure_future()`来启动协程，但并不保证严格的执行顺序，Protocol对象并不清楚在对象方法中创建的协程，所以也不会等待其执行结束。若需要确定执行顺序的话，可以在协程中通过yield from语句来使用Stream对象

## TCP

服务端代码

```python
# 服务端代码
import asyncio

class EchoServerClientProtocol(asyncio.Protocol):
    # 连接建立成功
    def connection_made(self, transport):
        peername = transport.get_extra_info('peername')
        pritn('Connection from {}'.format(peername))
        self.transport = transport

    # 收到数据
    def data_received(self, data):
        message = data.decode()
        print('Data received:{!r}'.format(message))
        print('Send:{!r}'.format(message))
        self.transport.write(data)

    # 对方发送消息结束
    def eof_received(self):
        print('Close the client socket')
        self.transport.close()
loop = asyncio.get_event_loop()

# 创建服务器，每个客户端的连接请求都会创建一个新的Protocol实例
coro = loop.create_server(EchoServerClientProtocol, '127.0.0.1', 8888)
server = loop.run_until_complete(coro)

# 服务器一直运行，直到用户按下Ctrl+C键
print('Serving on {}'.format(server.sockets[0].getsockname()))
try:
    loop.run_forever()
except KeyboradInerrupt:
    pass

# 关闭服务器
server.close()
loop.run_until_complete(server.wait_closed)
loop.close()
```

客户端代码

```python
# 客户端代码
import asyncio
import time

class EchoClientProtocol(asyncio.Protocol):
    def __init__(self, message, loop):
        self.message = message
        self.loop = loop

    # 连接创建成功
    def connection_made(self, transport):
        for m in message:
            transport.write(m.encode())
            print('Data sent: {!r}'.format(m))
            time.sleep(1)
        # 全部消息发送完毕，通知对方不再发送消息
        transport.write_eof()

    # 收到数据
    def data_received(self, data):
        print('Data received:{!r}'.format(data.decode()))

    # 连接被关闭
    def connection_lost(self, exc):
        print('The server closed the connection')
        print('Stop the event loop')
        self.loop.stop()

loop = asyncio.get_event_loop()
message = ['Hello word!', '你好']
coro = loop.create_connection(lambda: EchoClientProtocol(message, loop), '127.0.0.1', 8888)
loop.run_until_complete(coro)
loop.run_forever()
loop.close()
```

## UDP

监听端代码

```python
# 服务端代码
import asyncio
import datetime
import socket

class EchoServerProtocol:
    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        message = data.decode()
        print('Received from', str(addr))
        now = str(datetiem.datetime.now())[:19]
        self.transport.sendto(now.encode(),addr)
        print('replied')

loop = asyncio.get_event_loop()
print("Starting UDP server")
# 获取本机IP地址
ip = socket.gethostbyname(socket.gethostname())
# 创建Protocol实例，服务所有客户端
listen = loop.create_datagram_endpoint(EchoServerProtocol, local_addr=(ip, 9999))
transport, protocol = loop.run_until_complete(listen)
try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

transport.close()
loop.close()
```

客户端代码

```python
# 客户端代码
import asyncio
import time

class EchoClientProtocol:
    def __init__(self, message, loop):
        self.message = message
        self.loop = loop

    def connection_made(self, transport):
        self.transport = transport
        self.transport.sendto(self.message.encode())

    def datagram_received(self, data, addr):
        print('Now is :', data.decode())
        self.transport.close()

    def error_received(self, exc):
        print('Error received:', exc)

    def connection_lost(self, exc):
        self.loop.stop()

loop = asyncio.get_event_loop()
message = "ask for me"
while True:
    connect = loop.create_datagram_endpoint(
        lambda: EchoClientProtocol(message, loop),
        remote_addr = ('10.2.1.2', 9999)
    )
    transport, protocol = loop.run_until_complete(connect)
    loop.run_forever()
    transport.close()
    time.sleep(1)
loop.close()
```

## socket

注册用于接收数据的socket,并实现两个socket之间的数据传输

```python
import asyncio

try:
    from socket import socketpair
except ImportError:
    from asyncio.windows_utils import socketpair

class MyProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data):
        # 接收数据，关闭Transport对象
        print("Received:", data.encode())
        self.transport.close()

    def connection_lost(self, exc):
        # Socket已经关闭，停止事件循环
        loop.stop()

# 创建一对互相联通的socket
rsock, wsock = socketpair()
loop = asyncio.get_event_loop()
# 注册用来等待接收数据的socket
connect_coro = loop.create_connection(MyProtocol, sock=rsock)
transport, protocol = loop.run_until_complete(connect_coro)
# 往互相连通的socket中的一个写入数据
loop.call_soon(wsock.send, 'hello world.'.encode())
# 启动事件循环
loop.run_forever()
rsock.close()
wsock.close()
loop.close()
```

## StreamReader/StreamWriter

aysncio模块还提供了`open_connection()`函数(对AbstractEventLoop.create_connection()函数的封装)、`open_unix_connection()`函数(对AbstractEventLoop.create_unix_connection()函数的封装)、`start_server()`(对AbstractEventLoop.create_server()函数的封装)、`start_unix_server()`函数，这些函数都是协程函数，其中参数含义与被封装函数基本一致

`open_connection()`函数执行成功的话会返回(reader, writer),其中reader是Sreamreader类的实例，而writer是StreamWriter类的实例。StreamReader类提供了`set_transport(transport),feed_data(data),feed_eof()`方法及协程方法`read(n=-1),readline(),readexactly(n),readuntil(separator=b'\n')`用来从Transport对象中读取数据；封装了Transport类的StreamWriter类则提供了普通方法`close(),get_extra_info(),write(data),writelines(data),write_eof()`和协程方法`drain()`(如果Transport对象的缓冲区达到上水位线就会阻塞写操作，直到缓冲区大小被拉到下水位线时再恢复)。

实现网络聊天程序

服务端代码

```python
import asyncio

message = {
    'Hello': 'nihao',
    'How are you?': 'Fine, thank you.',
    'Did you have breakfast?': 'Yes',
    'Bye':'Bye'
}

@asyncio.coroutine
def handle_echo(reader, writer):
    while True:
        data = yield from reader.read(100)
        message = data.decode()
        addr = writer.get_extra_info('peername')
        print("Reaceived %r from %r" % (message, addr))

        messageReply = message.get(message, 'Sorry')
        print("Send: %r" % messageReply)
        writer.write(messageReply.encode())
        yield from writer.drain()
        if messageReply == 'Bye':
            break
print("Close the client socket")
writer.close()

# 创建事件循环
loop = asyncio.get_event_loop()
# 创建并启动服务器
coro = asyncio.start_server(handle_echo, '10.2.1.2', 8888, loop=loop)
server = loop.run_until_complete(coro)
print('Serving on {}'.format(server.sockets[0].getsockname()))
#  按Ctrl+C键或Ctrl+Break键退出
try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

# 关闭服务器
server.close()
loop.run_until_complete(server.wait_closed())
loop.close()
```

客户端代码

```python
import asyncio

@asyncio.coroutine
def tcp_echo_client(loop):
    reader, writer = yield from asyncio.open_connection(
        '10.2.1.2', 8888, loop=loop
    )
    while True:
        message = input('You said:')
        writer.write(message.encode())
        data = yield from reader.read(100)
        print('Receive: %r' % data.decode())
        if message == 'Bye':
            break

    print('Close the socket')
    writer.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(tcp_echo_client(loop))
loop.close()
```

获取网页头部信息

```python
import asyncio
import urllib.parse 
import sys

@asyncio.coroutine
def print_http_header(url):
    url = urllib.parse.urlsplit(url)
    if url.scheme == 'https':
        connect = asyncio.open_connection(url.hostname, 443, ssl=True)
    else:
        connect = asyncio.open_connection(url.hostname, 80)
    reader, writer = yield from connect

    query = ('HEAD {path} HTTP/1.0\r\nHost: {hostname}\r\n\r\n'
            ).format(path=url.path or '/', hostname = url.hostname)
    writer.write(query.encode('latin-1'))
    while True:
        line = yield from reader.readline()
        if not line:
            break
        line = line.decode('latin1').rstrip()
        if line:
            print('HTTP header> % s' % line)
    writer.close()

url = 'https://docs.python.org/3/library/asyncio-stream.html'
loop = asyncio.get_event_loop()
task = asyncio.ensure_future(print_http_header(url))
loop.run_until_complete(task)
loop.close()
```

注册端口并接收数据

```python
import asyncio

try:
    from socket import socketpair
except ImportError:
    from asyncio.windows_utils import socketpair

@asyncio.coroutine
def wait_for_data(loop):
    # 创建一对互相连通的socket
    rsock, wsock = socketpair()
    # 注册用来接收数据的socket
    reader, writer = yield form asyncio.open_connection(sock=rsock, loop=loop)
    # 通过socket写入数据
    loop.call_soon(wsock.send, 'This is a test.'.encode())
    # 等待接收数据
    data = yield from reader.read(100)
    print("Received:", data.decode())

    writer.close()
    wsock.close()

loop = asyncio.get_event_loop()
loop.run_until_complete(wait_for_data(loop))
loop.close()
```

## 模拟http

```python
import asyncio
import socket
from urllib.parse import urlparse


async def get_url(url):
    #通过socket请求html
    url = urlparse(url)
    host = url.netloc
    path = url.path
    if path == "":
        path = "/"

    #建立socket连接
    reader, writer = await asyncio.open_connection(host,80)
    writer.write("GET {} HTTP/1.1\r\nHost:{}\r\nConnection:close\r\n\r\n".format(path, host).encode("utf8"))
    
    all_lines = []
    # async for将for循环中阻塞式读数据的过程给异步化
    async for raw_line in reader:
        data = raw_line.decode("utf8")
        all_lines.append(data)
    html = "\n".join(all_lines)
    return html

async def main():
    tasks = []
    for url in range(20):
        url = "http://shop.projectsedu.com/goods/{}/".format(url)
        tasks.append(asyncio.ensure_future(get_url(url)))
    # asyncio.as_completed()返回一个值是协程的迭代器
    for task in asyncio.as_completed(tasks):
        result = await task  # task是协程
        print(result)

if __name__ == "__main__":
    import time
    start_time = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    print('last time:{}'.format(time.time()-start_time))
```

