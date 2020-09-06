# Debug模式

启用Sanic的调试模式时，Sanic将提供更详细的日志记录输出，默认情况下将启用自动重新加载功能。

> 警告
>
> Sanic的更多调试功能会减慢服务器的性能，因此建议仅在开发环境中启用它。

## 配置`debug`模式

通过设置`debug`模式，将输出Sanic的更详细的输出，并激活自动重新加载器。

```python
from sanic import Sanic
from sanic.response import json

app = Sanic()

@app.route('/')
async def hello_world(request):
    return json({"hello": "world"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
```

## 控制自动重载

Sanic提供了一种手动启用或禁用自动重新加载器的方法，`auto_reload`参数将激活或停用自动重新加载器。

```python
from sanic import Sanic
from sanic.response import json

app = Sanic()

@app.route('/')
async def hello_world(request):
    return json({"hello": "world"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, auto_reload=True)
```