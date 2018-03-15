# Web开发阶段

```
Web应用开发可以说是目前软件开发中最重要的部分。Web开发也经历了好几个阶段：

1. 静态Web页面：由文本编辑器直接编辑并生成静态的HTML页面，如果要修改Web页面的内容，就需要再次编辑HTML源文件，早期的互联网Web页面就是静态的；

2. CGI：由于静态Web页面无法与用户交互，比如用户填写了一个注册表单，静态Web页面就无法处理。要处理用户发送的动态数据，出现了Common Gateway Interface，简称CGI，用C/C++编写。

3. ASP/JSP/PHP：由于Web应用特点是修改频繁，用C/C++这样的低级语言非常不适合Web开发，而脚本语言由于开发效率高，与HTML结合紧密，因此，迅速取代了CGI模式。ASP是微软推出的用VBScript脚本编程的Web开发技术，而JSP用Java来编写脚本，PHP本身则是开源的脚本语言。

4. MVC：为了解决直接用脚本语言嵌入HTML导致的可维护性差的问题，Web应用也引入了Model-View-Controller的模式，来简化Web开发。ASP发展为ASP.Net，JSP和PHP也有一大堆MVC框架。

目前，Web开发技术仍在快速发展中，异步开发、新的MVVM前端技术层出不穷。
```

# HTTP协议

在Web应用中，服务器把网页传给浏览器，实际上就是把网页的HTML代码发送给浏览器，让浏览器显示出来。而浏览器和服务器之间的传输协议是HTTP，所以：

- HTML是一种用来定义网页的文本，会HTML，就可以编写网页；
- HTTP是在网络上传输HTML的协议，用于浏览器和服务器的通信。

Web采用的HTTP协议采用了非常简单的请求-响应模式，从而大大简化了开发。当我们编写一个页面时，我们只需要在HTTP请求中把HTML发送出去，不需要考虑如何附带图片、视频等，浏览器如果需要请求图片和视频，它会发送另一个HTTP请求，因此，一个HTTP请求只处理一个资源。

HTTP协议同时具备极强的扩展性，虽然浏览器请求的是http://www.sina.com.cn/的首页，但是新浪在HTML中可以链入其他服务器的资源，比如<img src="http://i1.sinaimg.cn/home/2013/1008/U8455P30DT20131008135420.png">，从而将请求压力分散到各个服务器上，并且，一个站点可以链接到其他站点，无数个站点互相链接起来，就形成了World Wide Web，简称WWW。

##HTTP请求

跟踪了新浪的首页，我们来总结一下HTTP请求的流程：

```
1：浏览器首先向服务器发送HTTP请求，请求包括：
方法：GET还是POST，GET仅请求资源，POST会附带用户数据；
路径：/full/url/path；
域名：由Host头指定：Host: www.sina.com.cn
以及其他相关的Header；

如果是POST，那么请求还包括一个Body，包含用户数据。

2：服务器向浏览器返回HTTP响应，响应包括：
响应代码：200表示成功，3xx表示重定向，4xx表示客户端发送的请求有错误，5xx表示服务器端处理时发生了错误；
响应类型：由Content-Type指定；
以及其他相关的Header；

通常服务器的HTTP响应会携带内容，也就是有一个Body，包含响应的内容，网页的HTML源码就在Body中。

3：如果浏览器还需要继续向服务器请求其他资源，比如图片，就再次发出HTTP请求，重复步骤1、2。
```

##HTTP格式

每个HTTP请求和响应都遵循相同的格式，一个HTTP包含Header和Body两部分，其中Body是可选的。

HTTP协议是一种文本协议，所以，它的格式也非常简单。

- HTTP GET请求的格式：

```
GET /path HTTP/1.1
Header1: Value1
Header2: Value2
Header3: Value3
```

每个Header一行一个，换行符是`\r\n`。

- HTTP POST请求的格式：

```
POST /path HTTP/1.1
Header1: Value1
Header2: Value2
Header3: Value3

body data goes here...
```

当遇到连续两个`\r\n`时，Header部分结束，后面的数据全部是Body。

- HTTP响应的格式：

```
200 OK
Header1: Value1
Header2: Value2
Header3: Value3

body data goes here...
```

HTTP响应如果包含body，也是通过`\r\n\r\n`来分隔的。请再次注意，Body的数据类型由`Content-Type`头来确定，如果是网页，Body就是文本，如果是图片，Body就是图片的二进制数据。

当存在`Content-Encoding`时，Body数据是被压缩的，最常见的压缩方式是gzip，所以，看到`Content-Encoding: gzip`时，需要将Body数据先解压缩，才能得到真正的数据。压缩的目的在于减少Body的大小，加快网络传输。

#URL

## 静态

```
静态URL类似 域名/news/2012-5-18/110.html 我们一般称为真静态URL，每个网页有真实的物理路径，也就是真实存在服务器里的。

优点是：
网站打开速度快，因为它不用进行运算；另外网址结构比较友好，利于记忆。

缺点是：
最大的缺点是如果是中大型网站，则产生的页面特别多，不好管理。至于有的开发者说占用硬盘空间大，我觉得这个可有忽略不计，占用不了多少空间的，况且目前硬盘空间都比较大。还有的开发者说会伤硬盘，这点也可以忽略不计。
```

## 动态

```
动态URL类似 域名/NewsMore.asp?id=5 或者 域名/DaiKuan.php?id=17，带有？号的URL，我们一般称为动态网址，每个URL只是一个逻辑地址，并不是真实物理存在服务器硬盘里的。

优点是：
适合中大型网站，修改页面很方便，因为是逻辑地址，所以占用硬盘空间要比纯静态网站小。

缺点是：
因为要进行运算，所以打开速度稍慢，不过这个可有忽略不计，目前有服务器缓存技术可以解决速度问题。最大的缺点是URL结构稍稍复杂，不利于记忆。
```

## 伪静态

```
伪静态URL类似 域名/course/74.html 这个URL和真静态URL类似。他是通过伪静态规则把动态URL伪装成静态网址。也是逻辑地址，不存在物理地址。

优点是：
URL比较友好，利于记忆。非常适合大中型网站，是个折中方案。

缺点是：
设置麻烦，服务器要支持重写规则，小企业网站或者玩不好的就不要折腾了。另外进行了伪静态网站访问速度并没有变快，因为实质上它会额外的进行运算解释，反正增加了服务器负担，速度反而变慢，不过现在的服务器都很强大，这种影响也可以忽略不计。还有可能会造成动态URL和静态URL都被搜索引擎收录，不过可以用robots禁止掉动态地址。
```
# Web静态服务器

##显示固定页面

```
#coding=utf-8
import socket


def handle_client(client_socket):
    "为一个客户端进行服务"
    recv_data = client_socket.recv(1024).decode("utf-8")
    request_header_lines = recv_data.splitlines()
    for line in request_header_lines:
        print(line)

    # 组织相应 头信息(header)
    response_headers = "HTTP/1.1 200 OK\r\n"  # 200表示找到这个资源
    response_headers += "\r\n"  # 用一个空的行与body进行隔开
    # 组织 内容(body)
    response_body = "hello world"

    response = response_headers + response_body
    client_socket.send(response.encode("utf-8"))
    client_socket.close()


def main():
    "作为程序的主控制入口"

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 设置当服务器先close 即服务器端4次挥手之后资源能够立即释放，这样就保证了，下次运行程序时 可以立即绑定7788端口
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("", 7788))
    server_socket.listen(128)
    while True:
        client_socket, client_addr = server_socket.accept()
        handle_client(client_socket)


if __name__ == "__main__":
    main()
```

## 显示需要的页面

```
#coding=utf-8
import socket
import re


def handle_client(client_socket):
    "为一个客户端进行服务"
    recv_data = client_socket.recv(1024).decode('utf-8', errors="ignore")
    request_header_lines = recv_data.splitlines()
    for line in request_header_lines:
        print(line)

    http_request_line = request_header_lines[0]
    get_file_name = re.match("[^/]+(/[^ ]*)", http_request_line).group(1)
    print("file name is ===>%s" % get_file_name)  # for test

    # 如果没有指定访问哪个页面。例如index.html
    # GET / HTTP/1.1
    if get_file_name == "/":
        get_file_name = DOCUMENTS_ROOT + "/index.html"
    else:
        get_file_name = DOCUMENTS_ROOT + get_file_name

    print("file name is ===2>%s" % get_file_name) #for test

    try:
        f = open(get_file_name, "rb")
    except IOError:
        # 404表示没有这个页面
        response_headers = "HTTP/1.1 404 not found\r\n"
        response_headers += "\r\n"
        response_body = "====sorry ,file not found===="
    else:
        response_headers = "HTTP/1.1 200 OK\r\n"
        response_headers += "\r\n"
        response_body = f.read()
        f.close()
    finally:
        # 因为头信息在组织的时候，是按照字符串组织的，不能与以二进制打开文件读取的数据合并，因此分开发送
        # 先发送response的头信息
        client_socket.send(response_headers.encode('utf-8'))
        # 再发送body
        client_socket.send(response_body)
        client_socket.close()


def main():
    "作为程序的主控制入口"
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("", 7788))
    server_socket.listen(128)
    while True:
        client_socket, clien_cAddr = server_socket.accept()
        handle_client(client_socket)


#这里配置服务器
DOCUMENTS_ROOT = "./html"

if __name__ == "__main__":
    main()
```

## 多进程

```
#coding=utf-8
import socket
import re
import multiprocessing


class WSGIServer(object):

    def __init__(self, server_address):
        # 创建一个tcp套接字
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 允许立即使用上次绑定的port
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定
        self.listen_socket.bind(server_address)
        # 变为被动，并制定队列的长度
        self.listen_socket.listen(128)

    def serve_forever(self):
        "循环运行web服务器，等待客户端的链接并为客户端服务"
        while True:
            # 等待新客户端到来
            client_socket, client_address = self.listen_socket.accept()
            print(client_address)  # for test
            new_process = multiprocessing.Process(target=self.handleRequest, args=(client_socket,))
            new_process.start()

            # 因为子进程已经复制了父进程的套接字等资源，所以父进程调用close不会将他们对应的这个链接关闭的
            client_socket.close()

    def handleRequest(self, client_socket):
        "用一个新的进程，为一个客户端进行服务"
        recv_data = client_socket.recv(1024).decode('utf-8')
        print(recv_data)
        requestHeaderLines = recv_data.splitlines()
        for line in requestHeaderLines:
            print(line)

        request_line = requestHeaderLines[0]
        get_file_name = re.match("[^/]+(/[^ ]*)", request_line).group(1)
        print("file name is ===>%s" % get_file_name) # for test

        if get_file_name == "/":
            get_file_name = DOCUMENTS_ROOT + "/index.html"
        else:
            get_file_name = DOCUMENTS_ROOT + get_file_name

        print("file name is ===2>%s" % get_file_name) # for test

        try:
            f = open(get_file_name, "rb")
        except IOError:
            response_header = "HTTP/1.1 404 not found\r\n"
            response_header += "\r\n"
            response_body = "====sorry ,file not found===="
        else:
            response_header = "HTTP/1.1 200 OK\r\n"
            response_header += "\r\n"
            response_body = f.read()
            f.close()
        finally:
            client_socket.send(response_header.encode('utf-8'))
            client_socket.send(response_body)
            client_socket.close()


# 设定服务器的端口
SERVER_ADDR = (HOST, PORT) = "", 8888
# 设置服务器服务静态资源时的路径
DOCUMENTS_ROOT = "./html"


def main():
    httpd = WSGIServer(SERVER_ADDR)
    print("web Server: Serving HTTP on port %d ...\n" % PORT)
    httpd.serve_forever()

if __name__ == "__main__":
    main()
```

## 多线程

```
#coding=utf-8
import socket
import re
import threading


class WSGIServer(object):

    def __init__(self, server_address):
        # 创建一个tcp套接字
        self.listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 允许立即使用上次绑定的port
        self.listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定
        self.listen_socket.bind(server_address)
        # 变为被动，并制定队列的长度
        self.listen_socket.listen(128)

    def serve_forever(self):
        "循环运行web服务器，等待客户端的链接并为客户端服务"
        while True:
            # 等待新客户端到来
            client_socket, client_address = self.listen_socket.accept()
            print(client_address)
            new_process = threading.Thread(target=self.handleRequest, args=(client_socket,))
            new_process.start()

            # 因为线程是共享同一个套接字，所以主线程不能关闭，否则子线程就不能再使用这个套接字了
            # client_socket.close() 

    def handleRequest(self, client_socket):
        "用一个新的进程，为一个客户端进行服务"
        recv_data = client_socket.recv(1024).decode('utf-8')
        print(recv_data)
        requestHeaderLines = recv_data.splitlines()
        for line in requestHeaderLines:
            print(line)

        request_line = requestHeaderLines[0]
        get_file_name = re.match("[^/]+(/[^ ]*)", request_line).group(1)
        print("file name is ===>%s" % get_file_name) # for test

        if get_file_name == "/":
            get_file_name = DOCUMENTS_ROOT + "/index.html"
        else:
            get_file_name = DOCUMENTS_ROOT + get_file_name

        print("file name is ===2>%s" % get_file_name) # for test

        try:
            f = open(get_file_name, "rb")
        except IOError:
            response_header = "HTTP/1.1 404 not found\r\n"
            response_header += "\r\n"
            response_body = "====sorry ,file not found===="
        else:
            response_header = "HTTP/1.1 200 OK\r\n"
            response_header += "\r\n"
            response_body = f.read()
            f.close()
        finally:
            client_socket.send(response_header.encode('utf-8'))
            client_socket.send(response_body)
            client_socket.close()


# 设定服务器的端口
SERVER_ADDR = (HOST, PORT) = "", 8888
# 设置服务器服务静态资源时的路径
DOCUMENTS_ROOT = "./html"


def main():
    httpd = WSGIServer(SERVER_ADDR)
    print("web Server: Serving HTTP on port %d ...\n" % PORT)
    httpd.serve_forever()

if __name__ == "__main__":
    main()
```

## 非堵塞模式

**单进程非堵塞模型**

```
#coding=utf-8
from socket import *
import time

# 用来存储所有的新链接的socket
g_socket_list = list()

def main():
    server_socket = socket(AF_INET, SOCK_STREAM)
    server_socket.setsockopt(SOL_SOCKET, SO_REUSEADDR  , 1)
    server_socket.bind(('', 7890))
    server_socket.listen(128)
    # 将套接字设置为非堵塞
    # 设置为非堵塞后，如果accept时，恰巧没有客户端connect，那么accept会
    # 产生一个异常，所以需要try来进行处理
    server_socket.setblocking(False)

    while True:

        # 用来测试
        time.sleep(0.5)

        try:
            newClientInfo = server_socket.accept()
        except Exception as result:
            pass
        else:
            print("一个新的客户端到来:%s" % str(newClientInfo))
            newClientInfo[0].setblocking(False)  # 设置为非堵塞
            g_socket_list.append(newClientInfo)

        for client_socket, client_addr in g_socket_list:
            try:
                recvData = client_socket.recv(1024)
                if recvData:
                    print('recv[%s]:%s' % (str(client_addr), recvData))
                else:
                    print('[%s]客户端已经关闭' % str(client_addr))
                    client_socket.close()
                    g_socket_list.remove((client_socket,client_addr))
            except Exception as result:
                pass

        print(g_socket_list)  # for test

if __name__ == '__main__':
    main()
```

**web静态服务器非堵塞**

```
import time
import socket
import sys
import re


class WSGIServer(object):
    """定义一个WSGI服务器的类"""

    def __init__(self, port, documents_root):

        # 1. 创建套接字
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 2. 绑定本地信息
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("", port))
        # 3. 变为监听套接字
        self.server_socket.listen(128)

        self.server_socket.setblocking(False)
        self.client_socket_list = list()

        self.documents_root = documents_root

    def run_forever(self):
        """运行服务器"""

        # 等待对方链接
        while True:

            # time.sleep(0.5)  # for test

            try:
                new_socket, new_addr = self.server_socket.accept()
            except Exception as ret:
                print("-----1----", ret)  # for test
            else:
                new_socket.setblocking(False)
                self.client_socket_list.append(new_socket)

            for client_socket in self.client_socket_list:
                try:
                    request = client_socket.recv(1024).decode('utf-8')
                except Exception as ret:
                    print("------2----", ret)  # for test
                else:
                    if request:
                        self.deal_with_request(request, client_socket)
                    else:
                        client_socket.close()
                        self.client_socket_list.remove(client_socket)

            print(self.client_socket_list)


    def deal_with_request(self, request, client_socket):
        """为这个浏览器服务器"""
        if not request:
            return

        request_lines = request.splitlines()
        for i, line in enumerate(request_lines):
            print(i, line)

        # 提取请求的文件(index.html)
        # GET /a/b/c/d/e/index.html HTTP/1.1
        ret = re.match(r"([^/]*)([^ ]+)", request_lines[0])
        if ret:
            print("正则提取数据:", ret.group(1))
            print("正则提取数据:", ret.group(2))
            file_name = ret.group(2)
            if file_name == "/":
                file_name = "/index.html"


        # 读取文件数据
        try:
            f = open(self.documents_root+file_name, "rb")
        except:
            response_body = "file not found, 请输入正确的url"
            response_header = "HTTP/1.1 404 not found\r\n"
            response_header += "Content-Type: text/html; charset=utf-8\r\n"
            response_header += "Content-Length: %d\r\n" % (len(response_body))
            response_header += "\r\n"

            # 将header返回给浏览器
            client_socket.send(response_header.encode('utf-8'))

            # 将body返回给浏览器
            client_socket.send(response_body.encode("utf-8"))
        else:
            content = f.read()
            f.close()

            response_body = content
            response_header = "HTTP/1.1 200 OK\r\n"
            response_header += "Content-Length: %d\r\n" % (len(response_body))
            response_header += "\r\n"

            # 将header返回给浏览器
            client_socket.send( response_header.encode('utf-8') + response_body)


# 设置服务器服务静态资源时的路径
DOCUMENTS_ROOT = "./html"


def main():
    """控制web服务器整体"""
    # python3 xxxx.py 7890
    if len(sys.argv) == 2:
        port = sys.argv[1]
        if port.isdigit():
            port = int(port)
    else:
        print("运行方式如: python3 xxx.py 7890")
        return

    print("http服务器使用的port:%s" % port)
    http_server = WSGIServer(port, DOCUMENTS_ROOT)
    http_server.run_forever()


if __name__ == "__main__":
    main()
```

## epoll

**简单模型**

```
import socket
import select

# 创建套接字
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 设置可以重复使用绑定的信息
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)

# 绑定本机信息
s.bind(("",7788))

# 变为被动
s.listen(10)

# 创建一个epoll对象
epoll = select.epoll()

# 测试，用来打印套接字对应的文件描述符
# print(s.fileno())
# print(select.EPOLLIN|select.EPOLLET)

# 注册事件到epoll中
# epoll.register(fd[, eventmask])
# 注意，如果fd已经注册过，则会发生异常
# 将创建的套接字添加到epoll的事件监听中
epoll.register(s.fileno(), select.EPOLLIN|select.EPOLLET)

connections = {}
addresses = {}

# 循环等待客户端的到来或者对方发送数据
while True:

    # epoll 进行 fd 扫描的地方 -- 未指定超时时间则为阻塞等待
    epoll_list = epoll.poll()

    # 对事件进行判断
    for fd, events in epoll_list:

        # print fd
        # print events

        # 如果是socket创建的套接字被激活
        if fd == s.fileno():
            new_socket, new_addr = s.accept()

            print('有新的客户端到来%s' % str(new_addr))

            # 将 conn 和 addr 信息分别保存起来
            connections[new_socket.fileno()] = new_socket
            addresses[new_socket.fileno()] = new_addr

            # 向 epoll 中注册 新socket 的 可读 事件
            epoll.register(new_socket.fileno(), select.EPOLLIN|select.EPOLLET)

        # 如果是客户端发送数据
        elif events == select.EPOLLIN:
            # 从激活 fd 上接收
            recvData = connections[fd].recv(1024).decode("utf-8")

            if recvData:
                print('recv:%s' % recvData)
            else:
                # 从 epoll 中移除该 连接 fd
                epoll.unregister(fd)

                # server 侧主动关闭该 连接 fd
                connections[fd].close()
                print("%s---offline---" % str(addresses[fd]))
                del connections[fd]
                del addresses[fd]
```

说明

```
- EPOLLIN （可读）
- EPOLLOUT （可写）
- EPOLLET （ET模式）

epoll对文件描述符的操作有两种模式：LT（level trigger）和ET（edge trigger）。LT模式是默认模式，LT模式与ET模式的区别如下：

LT模式：当epoll检测到描述符事件发生并将此事件通知应用程序，应用程序可以不立即处理该事件。下次调用epoll时，会再次响应应用程序并通知此事件。

ET模式：当epoll检测到描述符事件发生并将此事件通知应用程序，应用程序必须立即处理该事件。如果不处理，下次调用epoll时，不会再次响应应用程序并通知此事件。
```

**静态服务器 **

支持http的长连接，即使用了`Content-Length`

```
import socket
import time
import sys
import re
import select


class WSGIServer(object):
    """定义一个WSGI服务器的类"""

    def __init__(self, port, documents_root):

        # 1. 创建套接字
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 2. 绑定本地信息
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("", port))
        # 3. 变为监听套接字
        self.server_socket.listen(128)

        self.documents_root = documents_root

        # 创建epoll对象
        self.epoll = select.epoll()
        # 将tcp服务器套接字加入到epoll中进行监听
        self.epoll.register(self.server_socket.fileno(), select.EPOLLIN|select.EPOLLET)

        # 创建添加的fd对应的套接字
        self.fd_socket = dict()

    def run_forever(self):
        """运行服务器"""

        # 等待对方链接
        while True:
            # epoll 进行 fd 扫描的地方 -- 未指定超时时间则为阻塞等待
            epoll_list = self.epoll.poll()

            # 对事件进行判断
            for fd, event in epoll_list:
                # 如果是服务器套接字可以收数据，那么意味着可以进行accept
                if fd == self.server_socket.fileno():
                    new_socket, new_addr = self.server_socket.accept()
                    # 向 epoll 中注册 连接 socket 的 可读 事件
                    self.epoll.register(new_socket.fileno(), select.EPOLLIN | select.EPOLLET)
                    # 记录这个信息
                    self.fd_socket[new_socket.fileno()] = new_socket
                # 接收到数据
                elif event == select.EPOLLIN:
                    request = self.fd_socket[fd].recv(1024).decode("utf-8")
                    if request:
                        self.deal_with_request(request, self.fd_socket[fd])
                    else:
                        # 在epoll中注销客户端的信息
                        self.epoll.unregister(fd)
                        # 关闭客户端的文件句柄
                        self.fd_socket[fd].close()
                        # 在字典中删除与已关闭客户端相关的信息
                        del self.fd_socket[fd]

    def deal_with_request(self, request, client_socket):
        """为这个浏览器服务器"""

        if not request:
            return

        request_lines = request.splitlines()
        for i, line in enumerate(request_lines):
            print(i, line)

        # 提取请求的文件(index.html)
        # GET /a/b/c/d/e/index.html HTTP/1.1
        ret = re.match(r"([^/]*)([^ ]+)", request_lines[0])
        if ret:
            print("正则提取数据:", ret.group(1))
            print("正则提取数据:", ret.group(2))
            file_name = ret.group(2)
            if file_name == "/":
                file_name = "/index.html"


        # 读取文件数据
        try:
            f = open(self.documents_root+file_name, "rb")
        except:
            response_body = "file not found, 请输入正确的url"

            response_header = "HTTP/1.1 404 not found\r\n"
            response_header += "Content-Type: text/html; charset=utf-8\r\n"
            response_header += "Content-Length: %d\r\n" % len(response_body)
            response_header += "\r\n"

            # 将header返回给浏览器
            client_socket.send(response_header.encode('utf-8'))

            # 将body返回给浏览器
            client_socket.send(response_body.encode("utf-8"))
        else:
            content = f.read()
            f.close()

            response_body = content

            response_header = "HTTP/1.1 200 OK\r\n"
            response_header += "Content-Length: %d\r\n" % len(response_body)
            response_header += "\r\n"

            # 将数据返回给浏览器
            client_socket.send(response_header.encode("utf-8")+response_body)


# 设置服务器服务静态资源时的路径
DOCUMENTS_ROOT = "./html"


def main():
    """控制web服务器整体"""
    # python3 xxxx.py 7890
    if len(sys.argv) == 2:
        port = sys.argv[1]
        if port.isdigit():
            port = int(port)
    else:
        print("运行方式如: python3 xxx.py 7890")
        return

    print("http服务器使用的port:%s" % port)
    http_server = WSGIServer(port, DOCUMENTS_ROOT)
    http_server.run_forever()


if __name__ == "__main__":
    main()
```

## gevent

```
from gevent import monkey
import gevent
import socket
import sys
import re

monkey.patch_all()


class WSGIServer(object):
    """定义一个WSGI服务器的类"""

    def __init__(self, port, documents_root):

        # 1. 创建套接字
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 2. 绑定本地信息
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("", port))
        # 3. 变为监听套接字
        self.server_socket.listen(128)

        self.documents_root = documents_root

    def run_forever(self):
        """运行服务器"""

        # 等待对方链接
        while True:
            new_socket, new_addr = self.server_socket.accept()
            gevent.spawn(self.deal_with_request, new_socket)  # 创建一个协程准备运行它

    def deal_with_request(self, client_socket):
        """为这个浏览器服务器"""
        while True:
            # 接收数据
            request = client_socket.recv(1024).decode('utf-8')
            # print(gevent.getcurrent())
            # print(request)

            # 当浏览器接收完数据后，会自动调用close进行关闭，因此当其关闭时，web也要关闭这个套接字
            if not request:
                new_socket.close()
                break

            request_lines = request.splitlines()
            for i, line in enumerate(request_lines):
                print(i, line)

            # 提取请求的文件(index.html)
            # GET /a/b/c/d/e/index.html HTTP/1.1
            ret = re.match(r"([^/]*)([^ ]+)", request_lines[0])
            if ret:
                print("正则提取数据:", ret.group(1))
                print("正则提取数据:", ret.group(2))
                file_name = ret.group(2)
                if file_name == "/":
                    file_name = "/index.html"

            file_path_name = self.documents_root + file_name
            try:
                f = open(file_path_name, "rb")
            except:
                # 如果不能打开这个文件，那么意味着没有这个资源，没有资源 那么也得需要告诉浏览器 一些数据才行
                # 404
                response_body = "没有你需要的文件......".encode("utf-8")

                response_headers = "HTTP/1.1 404 not found\r\n"
                response_headers += "Content-Type:text/html;charset=utf-8\r\n"
                response_headers += "Content-Length:%d\r\n" % len(response_body)
                response_headers += "\r\n"

                send_data = response_headers.encode("utf-8") + response_body

                client_socket.send(send_data)

            else:
                content = f.read()
                f.close()

                # 响应的body信息
                response_body = content
                # 响应头信息
                response_headers = "HTTP/1.1 200 OK\r\n"
                response_headers += "Content-Type:text/html;charset=utf-8\r\n"
                response_headers += "Content-Length:%d\r\n" % len(response_body)
                response_headers += "\r\n"
                send_data = response_headers.encode("utf-8") + response_body
                client_socket.send(send_data)

# 设置服务器服务静态资源时的路径
DOCUMENTS_ROOT = "./html"

def main():
    """控制web服务器整体"""
    # python3 xxxx.py 7890
    if len(sys.argv) == 2:
        port = sys.argv[1]
        if port.isdigit():
            port = int(port)
    else:
        print("运行方式如: python3 xxx.py 7890")
        return

    print("http服务器使用的port:%s" % port)
    http_server = WSGIServer(port, DOCUMENTS_ROOT")
    http_server.run_forever()


if __name__ == "__main__":
    main()
```

# Web动态服务器

## 请求动态页面

![1521043212066](C:\Users\ADMINI~1\AppData\Local\Temp\1521043212066.png)

Web应用的本质就是：

1. 浏览器发送一个HTTP请求；
2. 服务器收到请求，生成一个HTML文档；
3. 服务器把HTML文档作为HTTP响应的Body发送给浏览器；
4. 浏览器收到HTTP响应，从HTTP Body取出HTML文档并显示。

```
所以，最简单的Web应用就是先把HTML用文件保存好，用一个现成的HTTP服务器软件，接收用户请求，从文件中读取HTML，返回。Apache、Nginx、Lighttpd等这些常见的静态服务器就是干这件事情的。

如果要动态生成HTML，就需要把上述步骤自己来实现。不过，接受HTTP请求、解析HTTP请求、发送HTTP响应都是苦力活，如果我们自己来写这些底层代码，还没开始写动态HTML呢，就得花个把月去读HTTP规范。

正确的做法是底层代码由专门的服务器软件实现，我们用Python专注于生成HTML文档。因为我们不希望接触到TCP连接、HTTP原始请求和响应格式，所以，需要一个统一的接口，让我们专心用Python编写Web业务。

这个接口就是WSGI：Web Server Gateway Interface。

WSGI允许开发者将选择web框架和web服务器分开。可以混合匹配web服务器和web框架，选择一个适合的配对,web服务器必须具备WSGI接口，所有的现代Python Web框架都已具备WSGI接口，它让你不对代码作修改就能使服务器和特点的web框架协同工作。

其他语言也有类似接口：java有Servlet API，Ruby 有 Rack。
```

**定义WSGI接口**

WSGI接口定义非常简单，它只要求Web开发者实现一个函数，就可以响应HTTP请求。

```
def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return 'Hello World!'
```

上面的`application()`函数就是符合WSGI标准的一个HTTP处理函数，`application()`函数必须由WSGI服务器来调用。它接收两个参数：

- environ：一个包含所有HTTP请求信息的dict对象；
- start_response：一个发送HTTP响应的函数。

整个`application()`函数本身没有涉及到任何解析HTTP的部分，也就是说，把底层web服务器解析部分和应用程序逻辑部分进行了分离，这样开发者就可以专心做一个领域了，应用领域的开发者不需要编写服务器底层代码，只负责在更高层次上考虑如何响应请求就可以了。

有了WSGI，我们关心的就是如何从`environ`这个`dict`对象拿到HTTP请求信息，然后构造HTML，通过`start_response()`发送Header，最后返回Body。

## 传递的字典

```
{
    'HTTP_ACCEPT_LANGUAGE': 'zh-cn',
    'wsgi.file_wrapper': <built-infunctionuwsgi_sendfile>,
    'HTTP_UPGRADE_INSECURE_REQUESTS': '1',
    'uwsgi.version': b'2.0.15',
    'REMOTE_ADDR': '172.16.7.1',
    'wsgi.errors': <_io.TextIOWrappername=2mode='w'encoding='UTF-8'>,
    'wsgi.version': (1,0),
    'REMOTE_PORT': '40432',
    'REQUEST_URI': '/',
    'SERVER_PORT': '8000',
    'wsgi.multithread': False,
    'HTTP_ACCEPT': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'HTTP_HOST': '172.16.7.152: 8000',
    'wsgi.run_once': False,
    'wsgi.input': <uwsgi._Inputobjectat0x7f7faecdc9c0>,
    'SERVER_PROTOCOL': 'HTTP/1.1',
    'REQUEST_METHOD': 'GET',
    'HTTP_ACCEPT_ENCODING': 'gzip,deflate',
    'HTTP_CONNECTION': 'keep-alive',
    'uwsgi.node': b'ubuntu',
    'HTTP_DNT': '1',
    'UWSGI_ROUTER': 'http',
    'SCRIPT_NAME': '',
    'wsgi.multiprocess': False,
    'QUERY_STRING': '',
    'PATH_INFO': '/index.html',
    'wsgi.url_scheme': 'http',
    'HTTP_USER_AGENT': 'Mozilla/5.0(Macintosh;IntelMacOSX10_12_5)AppleWebKit/603.2.4(KHTML,likeGecko)Version/10.1.1Safari/603.2.4',
    'SERVER_NAME': 'ubuntu'
}
```

## 快速实现

Python内置了一个WSGI服务器，这个模块叫wsgiref，它是用纯Python编写的WSGI服务器的参考实现。所谓“参考实现”是指该实现完全符合WSGI标准，但是不考虑任何运行效率，仅供开发和测试使用。

**hello.py**

实现Web应用程序的WSGI处理函数

```
# hello.py

def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b'<h1>Hello, web!</h1>']
```

**server.py**

负责启动WSGI服务器，加载`application()`函数

```
# server.py
# 从wsgiref模块导入:
from wsgiref.simple_server import make_server
# 导入我们自己编写的application函数:
from hello import application

# 创建一个服务器，IP地址为空，端口是8000，处理函数是application:
httpd = make_server('', 8000, application)
print('Serving HTTP on port 8000...')
# 开始监听HTTP请求:
httpd.serve_forever()
```

**运行**

```
# 确保两个文件在同一目录下，命令行输入
python server.py

# 浏览器中输入
localhost:8000/
```



## 基本实现

**文件结构**

```
├── web_server.py
├── web
│   └── my_web.py
└── html
    └── index.html
    .....
```

`web/my_web.py`

```
import time

def application(environ, start_response):
    status = '200 OK'
    response_headers = [('Content-Type', 'text/html')]
    start_response(status, response_headers)
    return str(environ) + '==Hello world from a simple WSGI application!--->%s\n' % time.ctime()
```

`web_server.py`

```
import select
import time
import socket
import sys
import re
import multiprocessing


class WSGIServer(object):
    """定义一个WSGI服务器的类"""

    def __init__(self, port, documents_root, app):

        # 1. 创建套接字
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 2. 绑定本地信息
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("", port))
        # 3. 变为监听套接字
        self.server_socket.listen(128)

        # 设定资源文件的路径
        self.documents_root = documents_root

        # 设定web框架可以调用的函数(对象)
        self.app = app

    def run_forever(self):
        """运行服务器"""

        # 等待对方链接
        while True:
            new_socket, new_addr = self.server_socket.accept()
            # 创建一个新的进程来完成这个客户端的请求任务
            new_socket.settimeout(3)  # 3s
            new_process = multiprocessing.Process(target=self.deal_with_request, args=(new_socket,))
            new_process.start()
            new_socket.close()

    def deal_with_request(self, client_socket):
        """以长链接的方式，为这个浏览器服务器"""

        while True:
            try:
                request = client_socket.recv(1024).decode("utf-8")
            except Exception as ret:
                print("========>", ret)
                client_socket.close()
                return

            # 判断浏览器是否关闭
            if not request:
                client_socket.close()
                return

            request_lines = request.splitlines()
            for i, line in enumerate(request_lines):
                print(i, line)

            # 提取请求的文件(index.html)
            # GET /a/b/c/d/e/index.html HTTP/1.1
            ret = re.match(r"([^/]*)([^ ]+)", request_lines[0])
            if ret:
                print("正则提取数据:", ret.group(1))
                print("正则提取数据:", ret.group(2))
                file_name = ret.group(2)
                if file_name == "/":
                    file_name = "/index.html"

            # 如果不是以py结尾的文件，认为是普通的文件
            if not file_name.endswith(".py"):

                # 读取文件数据
                try:
                    f = open(self.documents_root+file_name, "rb")
                except:
                    response_body = "file not found, 请输入正确的url"

                    response_header = "HTTP/1.1 404 not found\r\n"
                    response_header += "Content-Type: text/html; charset=utf-8\r\n"
                    response_header += "Content-Length: %d\r\n" % (len(response_body))
                    response_header += "\r\n"

                    response = response_header + response_body

                    # 将header返回给浏览器
                    client_socket.send(response.encode('utf-8'))

                else:
                    content = f.read()
                    f.close()

                    response_body = content

                    response_header = "HTTP/1.1 200 OK\r\n"
                    response_header += "Content-Length: %d\r\n" % (len(response_body))
                    response_header += "\r\n"

                    # 将header返回给浏览器
                    client_socket.send(response_header.encode('utf-8') + response_body)

            # 以.py结尾的文件，就认为是浏览需要动态的页面
            else:
                # 准备一个字典，里面存放需要传递给web框架的数据
                env = {}
                # 存web返回的数据
                response_body = self.app(env, self.set_response_headers)

                # 合并header和body
                response_header = "HTTP/1.1 {status}\r\n".format(status=self.headers[0])
                response_header += "Content-Type: text/html; charset=utf-8\r\n"
                response_header += "Content-Length: %d\r\n" % len(response_body)
                for temp_head in self.headers[1]:
                    response_header += "{0}:{1}\r\n".format(*temp_head)

                response = response_header + "\r\n"
                response += response_body

                client_socket.send(response.encode('utf-8'))

    def set_response_headers(self, status, headers):
        """这个方法，会在 web框架中被默认调用"""
        response_header_default = [
            ("Data", time.ctime()),
            ("Server", "ItCast-python mini web server")
        ]

        # 将状态码/相应头信息存储起来
        # [字符串, [xxxxx, xxx2]]
        self.headers = [status, response_header_default + headers]


# 设置静态资源访问的路径
g_static_document_root = "./html"
# 设置动态资源访问的路径
g_dynamic_document_root = "./web"

def main():
    """控制web服务器整体"""
    # python3 xxxx.py 7890
    if len(sys.argv) == 3:
        # 获取web服务器的port
        port = sys.argv[1]
        if port.isdigit():
            port = int(port)
        # 获取web服务器需要动态资源时，访问的web框架名字
        web_frame_module_app_name = sys.argv[2]
    else:
        print("运行方式如: python3 xxx.py 7890 my_web_frame_name:application")
        return

    print("http服务器使用的port:%s" % port)

    # 将动态路径即存放py文件的路径，添加到path中，这样python就能够找到这个路径了
    sys.path.append(g_dynamic_document_root)

    ret = re.match(r"([^:]*):(.*)", web_frame_module_app_name)
    if ret:
        # 获取模块名
        web_frame_module_name = ret.group(1)
        # 获取可以调用web框架的应用名称
        app_name = ret.group(2)

    # 导入web框架的主模块
    web_frame_module = __import__(web_frame_module_name)
    # 获取那个可以直接调用的函数(对象)
    app = getattr(web_frame_module, app_name) 

    # print(app)  # for test

    # 启动http服务器
    http_server = WSGIServer(port, g_static_document_root, app)
    # 运行http服务器
    http_server.run_forever()


if __name__ == "__main__":
    main()
```

## 运行

```
# 打开终端，输入命令，开始服务器
python3 web_server.py my_web:application

# 打开浏览器，输入url，开始请求
127.0.0.1:7890/***.py
```

# Web框架

其实一个Web App，就是写一个WSGI的处理函数，针对每个HTTP请求进行响应。

但是如何处理HTTP请求不是问题，问题是如何处理100个不同的URL。

每一个URL可以对应GET和POST请求，当然还有PUT、DELETE等请求，但是我们通常只考虑最常见的GET和POST请求。

一个最简单的想法是从`environ`变量里取出HTTP请求的信息，然后逐个判断：

```
def application(environ, start_response):
    method = environ['REQUEST_METHOD']
    path = environ['PATH_INFO']
    if method=='GET' and path=='/':
        return handle_home(environ, start_response)
    if method=='POST' and path='/signin':
        return handle_signin(environ, start_response)
    ...
```

只是这么写下去代码是肯定没法维护了。

代码这么写没法维护的原因是因为WSGI提供的接口虽然比HTTP接口高级了不少，但和Web App的处理逻辑比，还是比较低级，我们需要在WSGI接口之上能进一步抽象，让我们专注于用一个函数处理一个URL，至于URL到函数的映射，就交给Web框架来做。

有了Web框架，我们在编写Web应用时，注意力就从WSGI处理函数转移到URL+对应的处理函数，这样，编写Web App就更加简单了。

在编写URL处理函数时，除了配置URL外，从HTTP请求拿到用户数据也是非常重要的。Web框架都提供了自己的API来实现这些功能。Flask通过`request.form['name']`来获取表单的内容。

除了Flask，常见的Python Web框架还有：

- [Django](https://www.djangoproject.com/)：全能型Web框架；
- [web.py](http://webpy.org/)：一个小巧的Web框架；
- [Bottle](http://bottlepy.org/)：和Flask类似的Web框架；
- [Tornado](http://www.tornadoweb.org/)：Facebook的开源异步Web框架。

# 模板

由于在Python代码里拼HTML页面字符串是不现实的，所以，模板技术出现了。

使用模板，我们需要预先准备一个HTML文档，这个HTML文档不是普通的HTML，而是嵌入了一些变量和指令，然后，根据我们传入的数据，替换后，得到最终的HTML，发送给用户：

![1521105890630](C:\Users\ADMINI~1\AppData\Local\Temp\1521105890630.png)



除了Jinja2，常见的模板还有：

- [Mako](http://www.makotemplates.org/)：用`<% ... %>`和`${xxx}`的一个模板；
- [Cheetah](http://www.cheetahtemplate.org/)：也是用`<% ... %>`和`${xxx}`的一个模板；
- [Django](https://www.djangoproject.com/)：Django是一站式框架，内置一个用`{% ... %}`和`{{ xxx }}`的模板。

