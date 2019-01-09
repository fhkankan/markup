# 网络通信
网络编程就是如何在程序中实现两台计算机的通信。网络通信是两台计算机上的两个进程之间的通信。

用Python进行网络编程，就是在Python程序本身这个进程内，连接别的服务器进程的通信端口进行通信。

- MAC地址：在设备与设备之间数据通信时用来标记收发双方（网卡的序列号）
- IP地址：在逻辑上标记一台电脑，用来指引数据包的收发方向（相当于电脑的序列号）
- 网络掩码：用来区分ip地址的网络号和主机号
- 默认网关：当需要发送的数据包的目的ip不在本网段内时，就会发送给默认的一台电脑，成为网关
- 集线器：已过时，用来连接多态电脑，缺点：每次收发数据都进行广播，网络会变的拥堵
- 交换机：集线器的升级版，有学习功能知道需要发送给哪台设备，根据需要进行单播、广播
- 路由器：连接多个不同的网段，让他们之间可以进行收发数据，每次收到数据后，ip不变，但是MAC地址会变化
- DNS：用来解析出IP（类似电话簿）
- http服务器：提供浏览器能够访问到的数据

## IP

```
ip地址：用来在网络中标记一台电脑，比如192.168.1.1；在本地局域网上是唯一的。

A类IP地址
一个A类IP地址由1字节的网络地址和3字节主机地址组成，网络地址的最高位必须是“0”，
地址范围1.0.0.1-126.255.255.254
二进制表示为：00000001 00000000 00000000 00000001 - 01111110 11111111 11111111 11111110
可用的A类网络有126个，每个网络能容纳1677214个主机

B类IP地址
一个B类IP地址由2个字节的网络地址和2个字节的主机地址组成，网络地址的最高位必须是“10”，
地址范围128.1.0.1-191.255.255.254
二进制表示为：10000000 00000001 00000000 00000001 - 10111111 11111111 11111111 11111110
可用的B类网络有16384个，每个网络能容纳65534主机

C类IP地址
一个C类IP地址由3字节的网络地址和1字节的主机地址组成，网络地址的最高位必须是“110”
范围192.0.1.1-223.255.255.254
二进制表示为: 11000000 00000000 00000001 00000001 - 11011111 11111111 11111110 11111110
C类网络可达2097152个，每个网络能容纳254个主机

D类地址用于多点广播
D类IP地址第一个字节以“1110”开始，它是一个专门保留的地址。
它并不指向特定的网络，目前这一类地址被用在多点广播（Multicast）中
多点广播地址用来一次寻址一组计算机 s 地址范围224.0.0.1-239.255.255.254

E类IP地址
以“1111”开始，为将来使用保留
E类地址保留，仅作实验和开发用

私有ip
在这么多网络IP中，国际规定有一部分IP地址是用于我们的局域网使用，也就是属于私网IP，不在公网中使用的，它们的范围是：
10.0.0.0～10.255.255.255
172.16.0.0～172.31.255.255
192.168.0.0～192.168.255.255
```

## 端口

```
端口是通过端口号来标记的，端口号只有整数，范围是从0到65535
注意：端口数不一样的*nix系统不一样，还可以手动修改

知名端口
知名端口是众所周知的端口号，范围从0到1023
80端口分配给HTTP服务
21端口分配给FTP服务

动态端口
动态端口的范围是从1024到65535
之所以称为动态端口，是因为它一般不固定分配某种服务，而是动态分配。
动态分配是指当一个系统程序或应用程序程序需要网络通信时，它向主机申请一个端口，主机从可用的端口号中分配一个供它使用。
当这个程序关闭时，同时也就释放了所占用的端口号
```

##TCP/IP

```
计算机为了联网，就必须规定通信协议，早期的计算机网络，都是由各厂商自己规定一套协议，IBM、Apple和Microsoft都有各自的网络协议，互不兼容，这就好比一群人有的说英语，有的说中文，有的说德语，说同一种语言的人可以交流，不同的语言之间就不行了。

为了把全世界的所有不同类型的计算机都连接起来，就必须规定一套全球通用的协议，为了实现互联网这个目标，互联网协议簇（Internet Protocol Suite）就是通用协议标准。Internet是由inter和net两个单词组合起来的，原意就是连接“网络”的网络，有了Internet，任何私有网络，只要支持这个协议，就可以联入互联网。

因为互联网协议包含了上百种协议标准，但是最重要的两个协议是TCP和IP协议，所以，大家把互联网的协议简称TCP/IP协议。

通信的时候，双方必须知道对方的标识，好比发邮件必须知道对方的邮件地址。互联网上每个计算机的唯一标识就是IP地址，类似123.123.123.123。如果一台计算机同时接入到两个或更多的网络，比如路由器，它就会有两个或多个IP地址，所以，IP地址对应的实际上是计算机的网络接口，通常是网卡。

IP协议负责把数据从一台计算机通过网络发送到另一台计算机。数据被分割成一小块一小块，然后通过IP包发送出去。由于互联网链路复杂，两台计算机之间经常有多条线路，因此，路由器就负责决定如何把一个IP包转发出去。IP包的特点是按块发送，途径多个路由，但不保证能到达，也不保证顺序到达。

IP地址实际上是一个32位整数（称为IPv4），以字符串表示的IP地址如192.168.0.1实际上是把32位整数按8位分组后的数字表示，目的是便于阅读。

IPv6地址实际上是一个128位整数，它是目前使用的IPv4的升级版，以字符串表示类似于2001:0db8:85a3:0042:1000:8a2e:0370:7334。

TCP协议则是建立在IP协议之上的。TCP协议负责在两台计算机之间建立可靠连接，保证数据包按顺序到达。TCP协议会通过握手建立连接，然后，对每个IP包编号，确保对方按顺序收到，如果包丢掉了，就自动重发。

许多常用的更高级的协议都是建立在TCP协议基础上的，比如用于浏览器的HTTP协议、发送邮件的SMTP协议等。

一个TCP报文除了包含要传输的数据外，还包含源IP地址和目标IP地址，源端口和目标端口。

端口有什么作用？在两台计算机通信时，只发IP地址是不够的，因为同一台计算机上跑着多个网络程序。一个TCP报文来了之后，到底是交给浏览器还是QQ，就需要端口号来区分。每个网络程序都向操作系统申请唯一的端口号，这样，两个进程在两台计算机之间建立网络连接就需要各自的IP地址和各自的端口号。

一个进程也可能同时与多个计算机建立链接，因此它会申请很多端口。
```

## Socket

```
socket(简称 套接字) 是进程间通信的一种方式，它与其他进程间通信的一个主要不同是：
它能实现不同主机间的进程间通信，我们网络上各种各样的服务大多都是基于 Socket 来完成通信的
```

**创建socket**

```
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
| s.bind()                             | 绑定地址（host,port）到套接字， 在AF_INET下,以元组（host,port）的形式表示地址。 |
| s.listen()                           | 开始TCP监听。backlog指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。 |
| s.accept()                           | 被动接受TCP客户端连接,(阻塞式)等待连接的到来                 |
| 客户端套接字                         |                                                              |
| s.connect()                          | 主动初始化TCP服务器连接，。一般address的格式为元组（hostname,port），如果连接出错，返回socket.error错误。 |
| s.connect_ex()                       | connect()函数的扩展版本,出错时返回出错码,而不是抛出异常      |
| 公共用途的套接字函数                 |                                                              |
| s.recv()                             | 接收TCP数据，数据以字符串形式返回，bufsize指定要接收的最大数据量。flag提供有关消息的其他信息，通常可以忽略。 |
| s.send()                             | 发送TCP数据，将string中的数据发送到连接的套接字。返回值是要发送的字节数量，该数量可能小于string的字节大小。 |
| s.sendall()                          | 完整发送TCP数据，完整发送TCP数据。将string中的数据发送到连接的套接字，但在返回之前会尝试发送所有数据。成功返回None，失败则抛出异常。 |
| s.recvfrom()                         | 接收UDP数据，与recv()类似，但返回值是（data,address）。其中data是包含接收数据的字符串，address是发送数据的套接字地址。 |
| s.sendto()                           | 发送UDP数据，将数据发送到套接字，address是形式为（ipaddr，port）的元组，指定远程地址。返回值是发送的字节数。 |
| s.close()                            | 关闭套接字                                                   |
| s.getpeername()                      | 返回连接套接字的远程地址。返回值通常是元组（ipaddr,port）。  |
| s.getsockname()                      | 返回套接字自己的地址。通常是一个元组(ipaddr,port)            |
| s.setsockopt(level,optname,value)    | 设置给定套接字选项的值。                                     |
| s.getsockopt(level,optname[.buflen]) | 返回套接字选项的值。                                         |
| s.settimeout(timeout)                | 设置套接字操作的超时期，timeout是一个浮点数，单位是秒。值为None表示没有超时期。一般，超时期应该在刚创建套接字时设置，因为它们可能用于连接的操作（如connect()） |
| s.gettimeout()                       | 返回当前超时期的值，单位是秒，如果没有设置超时期，则返回None。 |
| s.fileno()                           | 返回套接字的文件描述符。                                     |
| s.setblocking(flag)                  | 如果flag为0，则将套接字设为非阻塞模式，否则将套接字设为阻塞模式（默认值）。非阻塞模式下，如果调用recv()没有发现任何数据，或send()调用无法立即发送数据，那么将引起socket.error异常。 |
| s.makefile()                         | 创建一个与该套接字相关连的文件                               |



# TCP编程

![1521072763226](C:\Users\ADMINI~1\AppData\Local\Temp\1521072763226.png)

```
TCP协议，传输控制协议（英语：Transmission Control Protocol，缩写为 TCP）是一种面向连接的、可靠的、基于字节流的传输层通信协议，由IETF的RFC 793定义。

TCP通信需要经过创建连接、数据传送、终止连接三个步骤。

TCP特点
1.面向连接
2.可靠传输
1）TCP采用发送应答机制
2）超时重传
3）错误校验
4）流量控制和阻塞管理

TCP与UDP的不同点
面向连接（确认有创建三方交握，连接已创建才作传输。）
有序数据传输
重发丢失的数据包
舍弃重复的数据包
无差错的数据传输
阻塞/流量控制

```

![1521041852128](C:\Users\ADMINI~1\AppData\Local\Temp\1521041852128.png)

## 客户端

```
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

tcp_client_socket.send(send_data.encode("gbk"))

# 接收对方发送过来的数据，最大接收1024个字节
recvData = tcp_client_socket.recv(1024)

# 对接收的数据解码
recvContent = recvData.decode('gbk')

print(recvContent)

# 关闭套接字
tcp_client_socket.close()
```

## 服务端

如果想要完成一个tcp服务器的功能，需要的流程如下：

1. socket创建一个套接字
2. bind绑定ip和port
3. listen使套接字变为可以被动链接
4. accept等待客户端的链接
5. recv/send接收发送数据

```
from socket import *

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

# 如果有新的客户端来链接服务器，那么就产生一个新的套接字专门为这个客户端服务
# client_socket用来为这个客户端服务
# tcp_server_socket就可以省下来专门等待其他新客户端的链接
client_socket, clientAddr = tcp_server_socket.accept()

# 接收对方发送过来的数据
recv_data = client_socket.recv(1024)  # 接收1024个字节
print('接收到的数据为:', recv_data.decode('gbk'))

# 发送一些数据到客户端
client_socket.send("thank you !".encode('gbk'))

# 关闭为这个客户端服务的套接字，只要关闭了，就意味着为不能再为这个客户端服务了，如果还需要服务，只能再次重新连接
client_socket.close()
```

## 长短连接

```
短连接一般只会在 client/server 间传递一次读写操作
长连接一次读写完成，连接不关闭，长时间操作之后client发起关闭请求

# 优缺点
长连接可以省去较多的TCP建立和关闭的操作，减少浪费，节约时间。对于频繁请求资源的客户来说，较适用长连接。

client与server之间的连接如果一直不关闭的话，会存在一个问题，随着客户端连接越来越多，server早晚有扛不住的时候，这时候server端需要采取一些策略，如关闭一些长时间没有读写事件发生的连接，这样可以避免一些恶意连接导致server端服务受损；如果条件再允许就可以以客户端机器为颗粒度，限制每个客户端的最大长连接数，这样可以完全避免某个蛋疼的客户端连累后端服务。

短连接对于服务器来说管理较为简单，存在的连接都是有用的连接，不需要额外的控制手段。

但如果客户请求频繁，将在TCP的建立和关闭操作上浪费时间和带宽
```

## 注意点

```
tcp服务器一般情况下都需要绑定，否则客户端找不到这个服务器

tcp客户端一般不绑定，因为是主动链接服务器，所以只要确定好服务器的ip、port等信息就好，本地客户端可以随机

tcp服务器中通过listen可以将socket创建出来的主动套接字变为被动的，这是做tcp服务器时必须要做的

当客户端需要链接服务器时，就需要使用connect进行链接，udp是不需要链接的而是直接发送，但是tcp必须先链接，只有链接成功才能通信

当一个tcp客户端连接服务器时，服务器端会有1个新的套接字，这个套接字用来标记这个客户端，单独为这个客户端服务

listen后的套接字是被动套接字，用来接收新的客户端的链接请求的，而accept返回的新套接字是标记这个新客户端的

关闭listen后的套接字意味着被动套接字关闭了，会导致新的客户端不能够链接服务器，但是之前已经链接成功的客户端正常通信。

关闭accept返回的套接字意味着这个客户端已经服务完毕

当客户端的套接字调用close后，服务器端会recv解堵塞，并且返回的长度为0，因此服务器可以通过返回数据的长度来区别客户端是否已经下线
```

# UDP编程

<<<<<<< HEAD
![1521072696968](C:\Users\ADMINI~1\AppData\Local\Temp\1521072696968.png)

=======
>>>>>>> 7f34c3097d9ce6c09b8420739d7f1470ff789437
创建一个基于udp的网络程序流程很简单，具体步骤如下：

1. 创建客户端套接字
2. 发送/接收数据
3. 关闭套接字

## 发送数据

```
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

## 发送接收数据

```
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

## 服务端绑定端口

```
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

# asyncio

标准库asyncio提供的BaseTransport,ReadTransport,WriteTransport,DatagramTransport以及BaseSubprocessTransport类对不通咧行的信道进行了抽象。一般来说，不建议使用这些类直接实例化对象，而是使用AbstarctEventLoop函数来创建相应的Transport对象并且对底层信道进行初始化。一旦信道创建成功，可以通过一对Protocol对象进行通信了。目前asyncio支持TCP,UDP,SSL和Subprocess管道，不同类型的Transport对象支持的方法略有不同，另外需注意：Transport类不是线程安全的

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

## 使用TCP通信

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

## 使用UDP通信

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

