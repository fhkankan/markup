# APP

## 实例

```python
from sanic import Sanic

app = Sanic("MyHelloWorldApp")
```

## 上下文

```python
app = Sanic("MyApp")
app.ctx.db = Database()



# 监听器中使用
app = Sanic("MyApp")

@app.before_server_start
async def attach_db(app, loop):
    app.ctx.db = Database()

```

## 注册

注册获取

```python
# ./path/to/server.py
from sanic import Sanic

app = Sanic("my_awesome_server")



# ./path/to/somewhere_else.py
from sanic import Sanic

app = Sanic.get_app("my_awesome_server")
```

特殊情况

```python
# 不存在
app = Sanic.get_app(
    "non-existing",  # 若不存在，默认则抛出sanic.exceptions.SanicException
    force_create=True,  # 不存在时，强制创建一个新的实例
)

# 只有一个实例
Sanic("My only app")

app = Sanic.get_app()  # 不需要参数
```

## 配置

配置项大写

```python
app = Sanic('myapp')

app.config.DB_NAME = 'appdb'
app.config['DB_USER'] = 'appuser'

db_settings = {
    'DB_HOST': 'localhost',
    'DB_NAME': 'appdb',
    'DB_USER': 'appuser'
}
app.config.update(db_settings)
```

## 工厂模式

```python
# ./path/to/server.py
from sanic import Sanic
from .path.to.config import MyConfig
from .path.to.some.blueprint import bp


def create_app(config=MyConfig) -> Sanic:
    app = Sanic("MyApp", config=config)
    app.blueprint(bp)
    return app

```

启动

```shell
sanic path.to.server:create_app
```

## 自定义

### 配置

```python
from sanic.config import Config

class MyConfig(Config):
    FOO = "bar"

app = Sanic(..., config=MyConfig())  # 可以配置sanic内置不支持的情况
```

### 上下文

默认上下文是可以设置任何属性的`SimpleNamespace()`，但是也可以自定义

```python
app = Sanic(..., ctx=1)
app = Sanic(..., ctx={})

class MyContext:
    ...

app = Sanic(..., ctx=MyContext())

```

### 请求

```python
import time

from sanic import Request, Sanic, text

class NanoSecondRequest(Request):
    @classmethod
    def generate_id(*_):
        return time.time_ns()

app = Sanic(..., request_class=NanoSecondRequest)

@app.get("/")
async def handler(request):
    return text(str(request.id))

```

### 异常处理器

```python
from sanic.handlers import ErrorHandler

class CustomErrorHandler(ErrorHandler):
    def default(self, request, exception):
        ''' handles errors that have no error handlers assigned '''
        # You custom error handling logic...
        return super().default(request, exception)

app = Sanic(..., error_handler=CustomErrorHandler())

```

### 序列化

```python
import ujson

dumps = partial(ujson.dumps, escape_forward_slashes=False)
app = Sanic(__name__, dumps=dumps)


# 自定义包
from orjson import dumps

app = Sanic("MyApp", dumps=dumps)
```

### 反序列化

```python
from orjson import loads

app = Sanic("MyApp", loads=loads)
```

### 应用类型

默认类型

```
sanic.app.Sanic[sanic.config.Config, types.SimpleNamespace]  # 参数1：配置类型，参数2上下文类型
```

自定义

```python
from sanic import Sanic
from sanic.config import Config

class CustomConfig(Config):
    pass

class Foo:
    pass

app = Sanic("test", config=CustomConfig(), ctx=Foo())
reveal_type(app)  # N: Revealed type is "sanic.app.Sanic[main.CustomConfig, main.Foo]"

```

### 请求类型

默认类型

```
sanic.request.Request[
    sanic.app.Sanic[sanic.config.Config, types.SimpleNamespace],  # 应用类型
    types.SimpleNamespace	# 上下文类型
]
```

自定义

```python
from sanic import Request, Sanic
from sanic.config import Config

class CustomConfig(Config):
    pass

class Foo:
    pass

class RequestContext:
    foo: Foo

class CustomRequest(Request[Sanic[CustomConfig, Foo], RequestContext]):
    @staticmethod
    def make_context() -> RequestContext:
        ctx = RequestContext()
        ctx.foo = Foo()
        return ctx

app = Sanic(
    "test", config=CustomConfig(), ctx=Foo(), request_class=CustomRequest
)

@app.get("/")
async def handler(request: CustomRequest):
    # Full access to typed:
    # - custom application configuration object
    # - custom application context object
    # - custom request context object
    pass

```

