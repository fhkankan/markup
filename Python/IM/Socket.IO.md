# Socket.IO

## 概述

**Socket.IO 本是一个面向实时 web 应用的 JavaScript 库，现在已成为拥有众多语言支持的Web即时通讯应用的框架。**

Socket.IO 主要使用WebSocket协议。但是如果需要的话，Socket.io可以回退到几种其它方法，例如Adobe Flash Sockets，JSONP拉取，或是传统的AJAX拉取，并且在同时提供完全相同的接口。尽管它可以被用作WebSocket的包装库，它还是提供了许多其它功能，比如广播至多个套接字，存储与不同客户有关的数据，和异步IO操作。

**Socket.IO 不等价于 WebSocket**，WebSocket只是Socket.IO实现即时通讯的其中一种技术依赖，而且Socket.IO还在实现WebSocket协议时做了一些调整。

> 优点

Socket.IO 会自动选择合适双向通信协议，仅仅需要程序员对套接字的概念有所了解。

有Python库的实现，可以在Python实现的Web应用中去实现IM后台服务。

> 缺点

Socket.io并不是一个基本的、独立的、能够回退到其它实时协议的WebSocket库，它实际上是一个依赖于其它实时传输协议的自定义实时传输协议的实现。该协议的协商部分使得支持标准WebSocket的客户端不能直接连接到Socket.io服务器，并且支持Socket.io的客户端也不能与非Socket.io框架的WebSocket或Comet服务器通信。因而，Socket.io要求客户端与服务器端均须使用该框架。

## Python服务器端

文档https://python-socketio.readthedocs.io/en/latest/server.html

### 安装

```python
pip install python-socketio
```

### 创建服务器

创建服务器有多种使用方式：

- 使用多进程多线程模式的WSGI服务器对接（如uWSGI、gunicorn)

```python
import socketio  

# create a Socket.IO servers
sio = socketio.Server()

# 打包成WSGI应用，可以使用WSGI服务器托管运行
app = socketio.WSGIApp(sio)  # Flask  Django
```

创建好app对象后，使用uWSGI、或gunicorn服务器运行此对象。

- 作为Flask、Django 应用中的一部分

```python
from wsgi import app  # a Flask, Django, etc. application
import socketio

# create a Socket.IO server
sio = socketio.Server()

app = socketio.WSGIApp(sio, app)
```

创建好app对象后，使用uWSGI、或gunicorn服务器运行此对象。

- 使用协程的方式运行 (推荐)

```python
import eventlet
eventlet.monkey_patch()

import socketio
import eventlet.wsgi

sio = socketio.Server(async_mode='eventlet')  # 指明在evenlet模式下
app = socketio.Middleware(sio)
eventlet.wsgi.server(eventlet.listen(('', 8000)), app)
```

> 说明

因为服务器与客户端进行即时通信时，会尽可能的使用长连接，所以若服务器采用多进程或多线程方式运行，受限于服务器能创建的进程或线程数，能够支持的并发连接客户端不会很高，也就是服务器性能有限。采用协程方式运行服务器，可以提升即时通信服务器的性能。

### 事件处理

不同于HTTP服务的编写方式，SocketIO服务编写不再以请求Request和响应Response来处理，而是对收发的数据以**消息（message）**来对待，收发的不同类别的消息数据又以**事件（event）**来区分。

原本HTTP服务编写中处理请求、构造响应的视图处理函数在SocketIO服务中改为编写收发不同事件的事件处理函数。

- 事件处理方法

编写事件处理方法，可以接收指定的事件消息数据，并在处理方法中对消息数据进行处理。

```python
@sio.on('connect')
def on_connect(sid, environ):
    """
    与客户端建立好连接后被执行
    :param sid: string sid是socketio为当前连接客户端生成的识别id
    :param environ: dict 在连接握手时客户端发送的握手数据(HTTP报文解析之后的字典)
    """
    pass

@sio.on('disconnect')
def on_disconnect(sid):
    """
    与客户端断开连接后被执行
    :param sid: string sid是断开连接的客户端id
    """
    pass

# 以字符串的形式表示一个自定义事件，事件的定义由前后端约定
@sio.on('my custom event')  
def my_custom_event(sid, data):
    """
    自定义事件消息的处理方法
    :param sid: string sid是发送此事件消息的客户端id
    :param data: data是客户端发送的消息数据
    """
    pass
```

> 注意

connect 为特殊事件，当客户端连接后自动执行

disconnect 为特殊事件，当客户端断开连接后自动执行

connect、disconnect与自定义事件处理方法的函数传入参数不同

- 发送事件消息

1. 群发

```python
sio.emit('my event', {'data': 'foobar'})
```

2. 给指定用户发送

```python
sio.emit('my event', {'data': 'foobar'}, room=user_sid)
```

3. 给一组用户发送

SocketIO提供了**房间（room）**来为客户端分组

`sio.enter_room(sid, room_name)`

将连接的客户端添加到一个room

```python
@sio.on('chat')
def begin_chat(sid):
    sio.enter_room(sid, 'chat_users')
```

注意：当客户端连接后，socketio会自动将客户端添加到以此客户端sid为名的room中

`sio.leave_room(sid, room_name)`

将客户端从一个room中移除

```python
@sio.on('exit_chat')
def exit_chat(sid):
    sio.leave_room(sid, 'chat_users')
```

`sio.rooms(sid)`

查询sid客户端所在的所有房间

给一组用户发送消息的示例

```python
@sio.on('my message')
def message(sid, data):
  sio.emit('my reply', data, room='chat_users')
```

也可在群组发消息时跳过指定客户端

```python
@sio.on('my message')
def message(sid, data):
  sio.emit('my reply', data, room='chat_users', skip_sid=sid)
```

使用`send`发送`message`事件消息

对于**'message'事件**，可以使用send方法

```python
sio.send({'data': 'foobar'})
sio.send({'data': 'foobar'}, room=user_sid)
```

## Python客户端

```python
import socketio

sio = socketio.Client()

@sio.on('connect')
def on_connect():
    pass

@sio.on('event')
def on_event(data):
    pass

sio.connect('http://10.211.55.7:8000')
sio.wait()
```

## 发送消息队列

在Socket.IO 框架中可以选择使用以下两种方式作为消息中间件：

- 使用Redis

```python
mgr = socketio.RedisManager('redis://')
sio = socketio.Server(client_manager=mgr)
```

- 使用RabbitMQ

安装

```shell
pip install kombu
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