# WebSocket

## app

### 创建

使用`app.add_websocket_route`

```PYTHON
async def feed(request, ws):
    pass

app.add_websocket_route(feed, '/feed')
```

使用装饰器

```python
@app.websocket('/feed')
async def feed(request, ws):
	pass
```

### 控制

```python
from sanic import Request, Websocket

@app.websocket("/feed")
async def feed(request: Request, ws: Websocket):
    while True:
        data = "hello!"
        print("Sending: " + data)
        await ws.send(data)
        data = await ws.recv()
        print("Received: " + data)
        
     
from sanic import Request, Websocket

@app.websocket("/feed")
async def feed(request: Request, ws: Websocket):
    async for msg in ws:
        await ws.send(msg)
```

### 配置

```python
app.config.WEBSOCKET_MAX_SIZE = 2 ** 20
app.config.WEBSOCKET_MAX_QUEUE = 32
app.config.WEBSOCKET_READ_LIMIT = 2 ** 16
app.config.WEBSOCKET_WRITE_LIMIT = 2 ** 16
app.config.WEBSOCKET_PING_INTERVAL = 20
app.config.WEBSOCKET_PING_TIMEOUT = 20
```
## 蓝图

使用`add_websocket_route()`

```python
bp = Blueprint(__name__, url_prefix='/api/student/v2/')

async def feed(ws):
	pass


bp.add_websocket_route(feed, "/feed")
```

使用装饰器

```python
bp = Blueprint(__name__, url_prefix='/api/student/v2/')

@bp.websocket('/feed')
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

