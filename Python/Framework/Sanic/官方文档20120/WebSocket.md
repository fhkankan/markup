# WebSocket

Sanic在websocket之上提供了易于使用的抽象。Sanic支持websocket版本7和8。

## 在app中

- 创建

使用装饰器设置WebSocket

```python
from sanic import Sanic
from sanic.response import json
from sanic.websocket import WebSocketProtocol

app = Sanic("websocket_example")

@app.websocket('/feed')
async def feed(request, ws):
    while True:
        data = 'hello!'
        print('Sending: ' + data)
        await ws.send(data)
        data = await ws.recv()
        print('Received: ' + data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, protocol=WebSocketProtocol)
```

使用`app.add_websocket_route`

```PYTHON
async def feed(request, ws):
    pass

app.add_websocket_route(feed, '/feed')
```

调用WebSocket路由的处理程序时，将请求作为第一个参数，将WebSocket协议对象作为第二个参数。协议对象具有`send`和`recv`方法，分别用于发送和接收数据。

- 配置

您可以通过`app.config`设置自己的WebSocket配置，例如

```python
app.config.WEBSOCKET_MAX_SIZE = 2 ** 20
app.config.WEBSOCKET_MAX_QUEUE = 32
app.config.WEBSOCKET_READ_LIMIT = 2 ** 16
app.config.WEBSOCKET_WRITE_LIMIT = 2 ** 16
app.config.WEBSOCKET_PING_INTERVAL = 20
app.config.WEBSOCKET_PING_TIMEOUT = 20
```

如果在ASGI模式下运行，这些设置将没有影响。

在`Configuration`部分中找到更多信息。

## 在蓝图中

使用装饰器

```python
StudentVideoCallBP = Blueprint(__name__, url_prefix='/api/student/v2/')

@StudentVideoCallBP.websocket('/feed')
async def feed(ws):
    try:
        while True:
            data = 'hello!'
            print('Sending: ' + data)
            await ws.send(data)
            data = await ws.recv()
            print('Received: ' + data)
    except Exception as e:
        print(e)
```

使用`add_websocket_route()`

```python
StudentVideoCallBP = Blueprint(__name__, url_prefix='/api/student/v2/')

async def feed(ws):
	pass


StudentVideoCallBP.add_websocket_route(feed, "/feed")
```

