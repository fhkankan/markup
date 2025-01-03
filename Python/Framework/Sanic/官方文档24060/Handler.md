# 控制器

包含`sanic.request.Request`实例作为参数，返回`sanic.response.HTTPResponse`实例或协程

```
def i_am_a_handler(request):
    return HTTPResponse()

async def i_am_ALSO_a_handler(request):
    return HTTPResponse()

```

一般返回使用如下

```
from sanic import json
from sanic import html
from sanic import redirect
```

`await`提升效率

```python
@app.get("/async")
async def async_handler(request):
    await asyncio.sleep(0.1) 
    return text("Done.")

```

函数命名需唯一

```python
# Handler name will be "foo_handler"
@app.get("/foo")
async def foo_handler(request):
    return text("I said foo!")

# Handler name will be "foo"
@app.get("/foo", name="foo")
async def foo_handler(request):
    return text("I said foo!")


# Two handlers, same function,
# different names:
# - "foo_arg"
# - "foo"
@app.get("/foo/<arg>", name="foo_arg")
@app.get("/foo")
async def foo(request, arg=None):
    return text("I said foo!")
```

