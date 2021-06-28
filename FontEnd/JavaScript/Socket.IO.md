# Socket.IO

## 概述

[js官网](https://socket.io/docs/v4/)

[py官网](https://python-socketio.readthedocs.io/en/latest/)

Socket.IO 本是一个面向实时 web 应用的 JavaScript 库，现在已成为拥有众多语言支持的Web即时通讯应用的框架。

Socket.IO 主要使用WebSocket协议。但是如果需要的话，Socket.io可以回退到几种其它方法，例如Adobe Flash Sockets，JSONP拉取，或是传统的AJAX拉取，并且在同时提供完全相同的接口。尽管它可以被用作WebSocket的包装库，它还是提供了许多其它功能，比如广播至多个套接字，存储与不同客户有关的数据，和异步IO操作。

**Socket.IO 不等价于 WebSocket**，WebSocket只是Socket.IO实现即时通讯的其中一种技术依赖，而且Socket.IO还在实现WebSocket协议时做了一些调整。

- 优点

Socket.IO 会自动选择合适双向通信协议，仅仅需要程序员对套接字的概念有所了解。

有Python库的实现，可以在Python实现的Web应用中去实现IM后台服务。

- 缺点

Socket.io并不是一个基本的、独立的、能够回退到其它实时协议的WebSocket库，它实际上是一个依赖于其它实时传输协议的自定义实时传输协议的实现。该协议的协商部分使得支持标准WebSocket的客户端不能直接连接到Socket.io服务器，并且支持Socket.io的客户端也不能与非Socket.io框架的WebSocket或Comet服务器通信。因而，Socket.io要求客户端与服务器端均须使用该框架。

- 安装

>javascript

```shell
# 标准客户端
npm install socket.io-client

# 服务端/客户端
npm install socket.io
```

> python

```shell
# 标准客户端
pip install "python-socketio[client]"
pip install "python-socketio[asyncio_client]"

# 服务端/客户端
pip install python-socketio
```

Socket.IO 是一种传输协议，可在客户端（通常但不总是 Web 浏览器）和服务器之间实现基于事件的实时双向通信。客户端和服务器组件的官方实现是用 JavaScript 编写的。这个包提供了两者的 Python 实现，每个都有标准和 asyncio 变体。

- 要求

Socket.IO 协议经过多次修订，其中一些引入了向后不兼容的更改，这意味着客户端和服务器必须使用兼容版本才能正常工作。如果您使用 Python 客户端和服务器，则确保兼容性的最简单方法是为客户端和服务器使用此包的相同版本。如果您在不同的客户端或服务器上使用此包，则必须确保版本兼容。下面的版本兼容性图表将此包的版本映射到 JavaScript 参考实现的版本以及 Socket.IO 和Engine.IO 协议的版本。

| JavaScript Socket.IO version | Socket.IO protocol revision | Engine.IO protocol revision | python-socketio version | python-engineio version |
| ---------------------------- | --------------------------- | --------------------------- | ----------------------- | ----------------------- |
| 0.9.x                        | 1, 2                        | 1, 2                        | Not supported           | Not supported           |
| 1.x and 2.x                  | 3, 4                        | 3                           | 4.x                     | 3.x                     |
| 3.x and 4.x                  | 5                           | 4                           | 5.x                     | 4.x                     |

## javascript实现

- client

```javascript
const socket = io("ws://localhost:3000");

socket.on("connect", () => {
  // either with send()
  socket.send("Hello!");

  // or with emit() and custom event names
  socket.emit("salutations", "Hello!", { "mr": "john" }, Uint8Array.from([1, 2, 3, 4]));
});

// handle the event sent with socket.send()
socket.on("message", data => {
  console.log(data);
});

// handle the event sent with socket.emit()
socket.on("greetings", (elem1, elem2, elem3) => {
  console.log(elem1, elem2, elem3);
});
```

- server

独立版

```javascript
const io = require("socket.io")(3000);

io.on("connection", socket => {
  // either with send()
  socket.send("Hello!");

  // or with emit() and custom event names
  socket.emit("greetings", "Hey!", { "ms": "jane" }, Buffer.from([4, 3, 3, 1]));

  // handle the event sent with socket.send()
  socket.on("message", (data) => {
    console.log(data);
  });

  // handle the event sent with socket.emit()
  socket.on("salutations", (elem1, elem2, elem3) => {
    console.log(elem1, elem2, elem3);
  });
});
```

## python实现

- client

特点

```
1. 可以连接到与 JavaScript Socket.IO 1.x 和 2.x 版本兼容的其他 Socket.IO 服务器。支持 3.x 版的工作正在进行中。 
2. 与 Python 3.5 兼容。
3. 客户端的两个版本，一个用于标准 Python，另一个用于 asyncio。
4. 使用基于事件的架构，其中装饰器隐藏了协议的细节。
5. 实现HTTP 长轮询和WebSocket 传输。
6. 如果连接断开，则自动重新连接到服务器。
```

standard

```python
import socketio

sio = socketio.Client()

@sio.event
def connect():
    print('connection established')

@sio.event
def my_message(data):
    print('message received with ', data)
    sio.emit('my response', {'response': 'my response'})

@sio.event
def disconnect():
    print('disconnected from server')

sio.connect('http://localhost:5000')
sio.wait()
```

asyncio

```python
import asyncio
import socketio

sio = socketio.AsyncClient()

@sio.event
async def connect():
    print('connection established')

@sio.event
async def my_message(data):
    print('message received with ', data)
    await sio.emit('my response', {'response': 'my response'})

@sio.event
async def disconnect():
    print('disconnected from server')

async def main():
    await sio.connect('http://localhost:5000')
    await sio.wait()

if __name__ == '__main__':
    asyncio.run(main())
```

- server

特点

```
1. 可以连接到运行与 JavaScript 客户端版本 1.x 和 2.x 兼容的其他 Socket.IO 客户端的服务器。支持 3.x 版本的工作正在进行中。
2. 与 Python 3.5 兼容。
3. 服务器的两个版本，一个用于标准 Python，另一个用于 asyncio。
4. 由于是异步的，即使在适度的硬件上也支持大量客户端。
5. 可以托管在任何 WSGI 和 ASGI 网络服务器上，包括 Gunicorn、Uvicorn、eventlet 和 gevent。
6. 可与Flask、Django等框架编写的WSGI应用集成。
7. 可与aiohttp、sanic和tornado asyncio应用集成。
8. 向所有连接的客户端或分配给“房间”的客户端子集广播消息。
9. 可选支持多个服务器，通过消息队列（如 Redis 或 RabbitMQ）连接。
10.从外部进程（例如 Celery 工作线程或辅助脚本）向客户端发送消息。
11.使用隐藏协议细节的装饰器实现的基于事件的架构。 
12.支持 HTTP 长轮询和 WebSocket 传输。
13.支持 XHR2 和 XHR 浏览器。
14.支持文本和二进制消息。
15.支持 gzip 和 deflate HTTP 压缩。
16.可配置的 CORS 响应，以避免浏览器的跨域问题。 
```

standard

```python
import eventlet
import socketio

sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={
    '/': {'content_type': 'text/html', 'filename': 'index.html'}
})

@sio.event
def connect(sid, environ):
    print('connect ', sid)

@sio.event
def my_message(sid, data):
    print('message ', data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
```

asyncio

```python
from aiohttp import web
import socketio

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

async def index(request):
    """Serve the client-side application."""
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.event
def connect(sid, environ):
    print("connect ", sid)

@sio.event
async def chat_message(sid, data):
    print("message ", data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
    web.run_app(app)
```

