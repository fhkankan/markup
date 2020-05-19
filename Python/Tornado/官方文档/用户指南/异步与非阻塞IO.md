# 异步与非阻塞IO

实时Web功能要求每个用户建立长期存在的空闲连接。在传统的同步Web服务器中，这意味着为每个用户分配一个线程，这可能非常昂贵。

为了最小化并发连接的成本，Tornado使用了单线程事件循环。这意味着所有应用程序代码都应以异步和非阻塞为目标，因为一次只能执行一个操作。

术语“异步”和“非阻塞”密切相关，并且经常互换使用，但它们并非完全相同。

## 阻塞

函数在返回之前等待发生的事情时会阻塞。一个功能可能由于多种原因而被阻塞：网络I / O，磁盘I / O，互斥锁等。实际上，每个功能在运行和使用CPU时都会阻塞至少一点（对于一个极端的例子，为什么必须与其他类型的阻止一样重视CPU阻止，请考虑使用密码散列功能（例如bcrypt），该功能在设计上会花费数百毫秒的CPU时间，远远超过了典型的网络或磁盘访问时间。

函数在某些方面可以是阻塞的，而在其他方面则可以是非阻塞的。在Tornado的上下文中，我们通常谈论在网络I / O上下文中的阻塞，尽管所有类型的阻塞都将被最小化。

## 异步

异步函数会在完成之前返回，通常会导致一些工作在后台发生，然后再触发应用程序中的将来动作（与普通的同步函数相反，普通的同步函数会在返回之前执行所有操作）。异步接口有很多样式：

```
回调参数
返回占位符(Future,Promise,Defered)
传递到队列
回调注册表（例如POSIX信号）
```

无论使用哪种类型的接口，根据定义，异步函数与其调用者的交互方式都不同。没有免费的方法可以使同步函数对其调用方透明（例如gevent之类的系统使用轻量级线程来提供与异步系统可比的性能，但实际上并不能使它们异步）。

Tornado中的异步操作通常返回占位符对象（`Futures`），除了一些使用回调的底层组件（如`IOLoop`）外。`Futures`通常使用`await`或`yield`关键字转换为结果。

## 示例

同步函数

```python
from tornado.httpclient import HTTPClient

def synchronous_fetch(url):
    http_client = HTTPClient()
    response = http_client.fetch(url)
    return response.body
```

作为原生协程异步重写的相同函数

```python
from tornado.httpclient import AsyncHTTPClient

async def asynchronous_fetch(url):
    http_client = AsyncHTTPClient()
    response = await http_client.fetch(url)
    return response.body
```

为了与旧版本的Python兼容，请使用`tornado.gen`模块：

```python
from tornado.httpclient import AsyncHTTPClient
from tornado import gen

@gen.coroutine
def async_fetch_gen(url):
    http_client = AsyncHTTPClient()
    response = yield http_client.fetch(url)
    raise gen.Return(response.body)
```

协程有点神奇，但是它们在内部做的事情是这样的

```python
from tornado.concurrent import Future

def async_fetch_manual(url)
:
    http_client = AsyncHTTPClient()
    my_future = Future()
    fetch_future = http_client.fetch(url)
    def on_fetch(f):
        my_future.set_result(f.result().body)
    fetch_future.add_done_callback(on_fetch)
    return my_future
```

请注意，协程在获取完成之前会返回其`Future`。这就是使协程异步的原因。

协程可以做的任何事情，都可以通过传递回调对象来完成，但是协程通过让您以与同步代码相同的方式组织代码来提供重要的简化。这对于错误处理尤为重要，因为`try / except`块可以像在协程中所期望的那样工作，而用回调很难做到这一点。