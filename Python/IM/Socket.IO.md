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

## 客户端

### 概述

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

sio = socketio.Client()  # 创建客户端实例

# 定义事件处理器
@sio.event
def connect():
    print('connection established')

@sio.event
def my_message(data):
    print('message received with ', data)
    sio.emit('my response', {'response': 'my response'})  # 发送消息

@sio.event
def disconnect():
    print('disconnected from server')

sio.connect('http://localhost:5000')  # 连接服务器
sio.wait() # 管理后台任务，等待连接结束
```

asyncio

```python
import asyncio
import socketio

sio = socketio.AsyncClient()  # 创建客户端实例

# 定义事件处理器
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
    await sio.connect('http://localhost:5000')  # 连接服务器
    await sio.wait()  # 管理后台任务，等待连接结束

if __name__ == '__main__':
    asyncio.run(main())
```

### 其他

- 断开连接

客户端主动断开与服务器连接

```python
# standard
sio.disconnect()
# asyncio
await sio.disconnect()
```

- 管理后台任务

开始一个自定义后台任务

```python
# standard
def my_background_task(my_argument):
    # do some background work here!
    pass

task = sio.start_background_task(my_background_task, 123)

# asyncio
async def my_background_task(my_argument):
    # do some background work here!
    pass

task = sio.start_background_task(my_background_task, 123)
# 这个函数不是一个协程，因为它不等待后台函数结束。后台函数必须是协程。
```

协助后台程序暂停

```python
# standard
sio.sleep(2)
# asyncio
await sio.sleep(2)
```

- 事件回调

在客户端定义供服务端掉用的函数，服务端使用`socketio.Server.emit(callback="my_event")`来调用。

```python
@sio.event
def my_event(sid, data):
    # handle the message
    return "OK", 123
```

- 命名空间

基于装饰器

```python
# 建立连接
sio.connect('http://localhost:5000', namespaces=['/chat'])

# 定义事件处理
@sio.event(namespace='/chat')
def my_custom_event(sid, data):
    pass

@sio.on('connect', namespace='/chat')
def on_connect():
    print("I'm connected to the /chat namespace!")
    
# 发送信息
sio.emit('my message', {'foo': 'bar'}, namespace='/chat')
```

基于类

```python
# standard
class MyCustomNamespace(socketio.ClientNamespace):
    def on_connect(self):
        pass

    def on_disconnect(self):
        pass

    def on_my_event(self, data):
        self.emit('my_response', data)

sio.register_namespace(MyCustomNamespace('/chat'))


# asyncio
class MyCustomNamespace(socketio.AsyncClientNamespace):
    def on_connect(self):
        pass

    def on_disconnect(self):
        pass

    async def on_my_event(self, data):
        await self.emit('my_response', data)

sio.register_namespace(MyCustomNamespace('/chat'))
```

- 调试设置

```python
import socketio

# standard Python
sio = socketio.Client(logger=True, engineio_logger=True)

# asyncio
sio = socketio.AsyncClient(logger=True, engineio_logger=True)

# logger参数控制与Socket.IO协议相关的日志记录，而engineio_logger控制源自低级Engine.IO传输的日志记录。
```

## 服务器端

### 概述

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

### 服务实例

- WSGI

使用多进程多线程模式的WSGI服务器对接（如uWSGI、gunicorn)

```python
import socketio  

# create a Socket.IO servers
sio = socketio.Server()
# 打包成WSGI应用，可以使用WSGI服务器托管运行，创建好app对象后，使用uWSGI、或gunicorn服务器运行此对象。
app = socketio.WSGIApp(sio)
```

作为Flask、Django一部分

```python
from wsgi import app  # a Flask, Django, etc. application
import socketio

sio = socketio.Server()
app = socketio.WSGIApp(sio, app)  
# 创建好app对象后，使用uWSGI、或gunicorn服务器运行此对象。
```

- ASGI

```python
# create a Socket.IO server
sio = socketio.AsyncServer()

# wrap with ASGI application
app = socketio.ASGIApp(sio)
```

### 静态文件

映射配置

```python
# 路径与文件
static_files = {
    '/': 'latency.html',
    '/static/socket.io.js': 'static/socket.io.js',
    '/static/style.css': 'static/style.css',
}

# 路径与内容类型
static_files = {
    '/': {'filename': 'latency.html', 'content_type': 'text/plain'},
}

# 路径与目录
static_files = {
    '/static': './public',
}

# 默认值
static_files = {
    '/static': './public',
    '': 'image.gif',
}
static_files = {
    '/static': './public',
    '': {'filename': 'image.gif', 'content_type': 'text/plain'},
}
```

使用

```python
# for standard WSGI applications
sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files=static_files)

# for asyncio-based ASGI applications
sio = socketio.AsyncServer()
app = socketio.ASGIApp(sio, static_files=static_files)
```

### 事件处理

- 连接与断开连接

```python
@sio.event
def connect(sid, environ, auth):
    print('connect ', sid)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)
    
# 传输异常信息
@sio.event
def connect(sid, environ):
    raise ConnectionRefusedError('authentication failed')
```

- 发送事件消息

群发

```python
sio.emit('my event', {'data': 'foobar'})
```

给指定用户发送

```python
sio.emit('my event', {'data': 'foobar'}, room=user_sid)
```

### 命名空间

基于装饰器

```python
@sio.event(namespace='/chat')
def my_custom_event(sid, data):
    pass

@sio.on('my custom event', namespace='/chat')
def my_custom_event(sid, data):
    pass
```

基于类

```python
# standard
class MyCustomNamespace(socketio.Namespace):
    def on_connect(self, sid, environ):
        pass

    def on_disconnect(self, sid):
        pass

    def on_my_event(self, sid, data):
        self.emit('my_response', data)

sio.register_namespace(MyCustomNamespace('/test'))

# asyncio
class MyCustomNamespace(socketio.AsyncNamespace):
    def on_connect(self, sid, environ):
        pass

    def on_disconnect(self, sid):
        pass

    async def on_my_event(self, sid, data):
        await self.emit('my_response', data)

sio.register_namespace(MyCustomNamespace('/test'))
```

### 房间

为了方便服务器向相关客户端组发送事件，应用程序可以将其客户端放入“房间”，然后将消息寻址到这些房间。

进入离开房间

```python
# 当客户端连接后，socketio会自动将客户端添加到以此客户端sid为名的room中
@sio.on('chat')
def begin_chat(sid):
    sio.enter_room(sid, 'chat_users')
    
# 离开房间
@sio.on('exit_chat')
def exit_chat(sid):
    sio.leave_room(sid, 'chat_users')
```

给群组中一个房间发消息

```python
@sio.on('my message')
def message(sid, data):
  sio.emit('my reply', data, room='chat_users')
```

群组发消息时跳过指定客户端

```python
@sio.on('my message')
def message(sid, data):
  sio.emit('my reply', data, room='chat_users', skip_sid=sid)
```

### Session

`get_session(),save_session(),session() `方法采用可选的命名空间参数。如果未提供此参数，会话将附加到默认命名空间。

`get_session(),save_session()`

```python
# standard
@sio.event
def connect(sid, environ):
    username = authenticate_user(environ)
    sio.save_session(sid, {'username': username})

@sio.event
def message(sid, data):
    session = sio.get_session(sid)
    print('message from ', session['username'])
    
# asyncio
@sio.event
async def connect(sid, environ):
    username = authenticate_user(environ)
    await sio.save_session(sid, {'username': username})

@sio.event
async def message(sid, data):
    session = await sio.get_session(sid)
    print('message from ', session['username'])
```

`session()`上下文

```python
# standard
@sio.event
def connect(sid, environ):
    username = authenticate_user(environ)
    with sio.session(sid) as session:
        session['username'] = username

@sio.event
def message(sid, data):
    with sio.session(sid) as session:
        print('message from ', session['username'])
        
# asyncio
@sio.event
def connect(sid, environ):
    username = authenticate_user(environ)
    async with sio.session(sid) as session:
        session['username'] = username

@sio.event
def message(sid, data):
    async with sio.session(sid) as session:
        print('message from ', session['username'])
```

> 注意
>
> 当客户端断开连接时，用户会话的内容将被销毁。特别是，当客户端在与服务器意外断开连接后重新连接时，不会保留用户会话内容。

### 消息队列

使用分布式应用程序时，通常需要从多个进程访问 Socket.IO 的功能。有两个特定的用例：
```
1. 使用工作队列的应用程序（例如 Celery）可能需要在后台作业完成后向客户端发出事件。执行此任务最方便的地方是处理此作业的工作进程。
2. 高可用应用程序可能希望使用 Socket.IO 服务器的水平扩展，以便能够处理大量并发客户端。
```
作为解决方案针对以上问题，可以配置Socket.IO服务器连接到Redis或RabbitMQ等消息队列，与其他相关的Socket.IO服务器或辅助worker进行通信。

- Redis

安装

```python
# socketio.Server class
pip install redis

# socketio.AsyncServer class
pip install aioredis
```

使用

```python
# socketio.Server class
mgr = socketio.RedisManager('redis://')
sio = socketio.Server(client_manager=mgr)

# socketio.AsyncServer class
mgr = socketio.AsyncRedisManager('redis://')
sio = socketio.AsyncServer(client_manager=mgr)

# client_manager 参数指示服务器连接到给定的消息队列，并与连接到队列的其他进程协调
```

- Kombu

安装

```shell
pip install kombu

# 要使用 RabbitMQ 或其他 AMQP 协议兼容队列，这是唯一需要的依赖项。
# 但是对于其他消息队列，Kombu 可能需要额外的包。例如，要通过 Kombu 使用 Redis 队列，还要pip install redis
```
使用

```python
# 通过sio mgr对象可以发布要进行即时消息推送的任务，由socketio服务器从rabbitmq中取出任务，推送消息
mgr = socketio.KombuManager('amqp://')
sio = socketio.Server(client_manager=mgr)

# 通过socket.IO提供的kombu管理对象向rabbitMQ中写入数据，记录需要socketio服务器向客户端推送消息的任务
# socket.IO会自动从tabbitMQ中读取消息，发送给客户端
mgr.emit('task_name', data=data_dict, rooms=target_room)
```

> 注意
>
> 目前只支持同步

- AioPika

安装

```shell
pip install aio_pika

# asyncio 应用程序支持 RabbitMQ 消息队列
```

使用

```python
mgr = socketio.AsyncAioPikaManager('amqp://')
sio = socketio.AsyncServer(client_manager=mgr)
```

- kafaka

安装

```shell
pip install kafka-python
```

使用

```python
mgr = socketio.KafkaManager('kafka://')
sio = socketio.Server(client_manager=mgr)
```

> 注意
>
> 目前只支持同步

- 从外部进程发送消息

要让除服务器之外的进程连接到队列以发出消息，可以将相同的客户端管理器类用作独立对象。在这种情况下， `write_only` 参数应设置为` True` 以禁用侦听线程的创建，这仅在服务器中有意义。

```python
# connect to the redis queue as an external process
external_sio = socketio.RedisManager('redis://', write_only=True)

# emit an event
external_sio.emit('my event', data={'foo': 'bar'}, room='my room')
```

只写客户端管理器对象的一个限制是它在发消息时无法接收回调。当外部进程需要接收回调时，使用具有读写支持的客户端连接到服务器是比只写客户端管理器更好的选择。

### 调试设置

```python
import socketio

# standard Python
sio = socketio.Server(logger=True, engineio_logger=True)

# asyncio
sio = socketio.AsyncServer(logger=True, engineio_logger=True)
```

### 部署策略

Sanic

```python
sio = socketio.AsyncServer(async_mode='sanic')

app = Sanic()
sio.attach(app)


if __name__ == '__main__':
    app.run()

    
# 禁用cors启用sanic-cors控制
sio = socketio.AsyncServer(async_mode='sanic', cors_allowed_origins=[])
app.config['CORS_SUPPORTS_CREDENTIALS'] = True
```

Uvicorn, Daphne, and other ASGI servers

```python
sio = socketio.AsyncServer(async_mode='asgi')
app = socketio.ASGIApp(sio)
```

...

### 跨域控制

出于安全原因，此服务器默认强制执行同源策略。实际上，这意味着以下内容：

- 如果传入的 HTTP 或 WebSocket 请求包含 Origin 标头，则该标头必须与连接 URL 的方案和主机匹配。如果不匹配，将返回 400 状态代码响应并拒绝连接。
-  对不包含 Origin 标头的传入请求没有任何限制。

如有必要，可以使用`cors_allowed_origins`选项来允许其他来源。此参数可以设置为字符串以设置单个允许的来源，或设置为列表以允许多个来源。 `*` 的特殊值可用于指示服务器允许所有来源，但这应该小心完成，因为这可能会使服务器容易受到跨站请求伪造 (CSRF) 攻击。

