# Web开发

```
tornado 5.1
```

## 网站结构

```python
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    """实现web.RequestHandler子类，重载其中的get()函数"""
    def get(self):
        """负责处理相应定位到该RequestHandler的HTTP GET请求"""
        self.write("Hello, world")

def make_app():
    """返回web.Application对象，第一个参数用于定义路由映射"""
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)  # 监听服务器端口
    tornado.ioloop.IOLoop.current().start()  # 启动IOLoop，该函数将一直运行且不退出，用于处理完所有客户端的访问请求
```

## 路由解析

Tornado的路由字符串有两种：固定字符串路径和 参数字符串路径

- 固定字符串 

````python
Hanlders = [
    ("/", Mainhandler), 				// 只匹配根路径
    ("/entry", Enteryhandler), 			// 只匹配/entry
    ("/entry/2005", Entery2005Handler), // 只匹配/enrty/2005
]
````

- 参数字符串路径

参数字符串可以 将具备一定模式的路径映射到同一个RequestHandler中处理，其中路径中的参数部分用()标识

```python
# url handler
handlers = [(r"/entry/([^/]+)", EntryHandler),]

class EntryHandler(tornado.web.RequestHandler):
    def get(self, slug):
        entry = self.db.get("select * from entries where slug=%s", slug)
        if not entry:
            raise tornado.web.HTTPError(404)
        self.render("entry.html", entry=entry)
```

- 带默认值的参数路径

需要匹配客户端未传入时的路径

```python
# url handler
handlers = [(r"/entry/([^/]*)", EntryHandler),]

class EntryHandler(tornado.web.RequestHandler):
    def get(self, slug='default'):
        entry = self.db.get("select * from entries where slug=%s", slug)
        if not entry:
            raise tornado.web.HTTPError(404)
        self.render("entry.html", entry=entry)
```

- 多参数路径

参数路径允许一个URL模式中定义多个可变参数

```python
handlers = [(r'/(\d{4})/(\d{2})/(\d{2})/([a-zA-Z\-0-9\.:,_]+)/?', DetailHandler)]

class DetailHandler(tornado.web.RequestHandler):
    def get(self, year, month, day, slug):
        self.write("%d-%d-%d %s"%(year, month, day, slug))
```

## RequestHandler

### 接入点函数

需要子类即成并定义具体行为的函数在RequestHandler中被称为接入点函数

- `initialize()`

该方法被子类重写，实现了RequestHandler子类实例化的初始化过程，可以为该函数传递参数，参数来源于配置URL映射的定义

```python
from tornado.web import RequestHandler
from tornado.web import Application


class ProfileHandler(RequestHandler):
    def initialize(self, database):
        self.database = database

    def get(self):
        pass

    def post(self):
        pass


app = Application([
    (r'/account', ProfileHandler, dict(databse="./example.db")),
])

```

- `prepare(),on_finish()`

`prepare()`函数用于调用请求处理(`get,post`)方法之前的初始化处理。

`on_finish()`函数用于请求处理结束后的一些清理工作。

- HTTP Action处理函数

```
get()
header()
post()
delete()
patch()
put()
options()
```

### 输入捕获

输入捕获是指在RequestHandler中用于获取客户端输入的工具函数和属性，比如获取URL查询字符串、Post提交参数等

- `get_argument(name), get_arguments(name)`

两个参数都是返回给定参数的值，一个用于单个值；一个用于存在多个值的情况，返回多个值的列表。

获取的是URL查询字符串参数与POST提交参数的参数合集

- `get_query_argument(name), get_query_arguments(name)`

获取URL查询参数重获取参数值

- `get_body_argument(name), get_body_arguments(name)`

从Post提交参数中获取参数值

- `get_cookie(name, default=None)`

根据cookie名称获取cookie值

- `request`

返回`tornado.httputil.HTTPServerRequest`对象实例的属性，通过该对象可以获取关于HTTP请求的一切信息

```python
import tornado.web

class DetialHandler(tornado.web.RequestHandler):
    def get(self):
        remote_ip = self.request.remote_ip
        host = self.request.host
```

常用对象属性

| 属性名    | 说明                                                       |
| --------- | ---------------------------------------------------------- |
| method    | HTTP请求方法                                               |
| uri       | 客户端请求的uri的完整内容                                  |
| path      | uri路径名，不包括请求查询字符串                            |
| query     | uri中查询字符串                                            |
| version   | 客户端请求是的HTTP版本                                     |
| headers   | 以字典方式表达的HTTP Headers                               |
| body      | 以字符串方式表达的HTTP消息体                               |
| remote_ip | 客户端的ip                                                 |
| protocol  | 请求协议，http/https                                       |
| host      | 请求消息中的主机名                                         |
| arguments | 客户端提交的所有参数                                       |
| files     | 以字典方式表达的客户端上传的文件，每个文件对应一个HTTPFile |
| cookies   | 客户端提交的Cookie字典                                     |

### 输出响应函数

输出响应函数是指一组为客户端生成处理结果的工具函数，开发者调用它们以控制URL的处理结果。

- `set_status(status_code, reason=None)`

设置HTTP Response中的返回码，如果有描述性的语句，可以赋值给reason参数

- `set_header(name, value)`

以键值对的方式设置HTTP Response中的HTTP头参数，使用`set_header`配置的Header值将覆盖之前配置的Header

```python
import tornado.web

class DetialHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("NUMBER", 9)
```

- `add_header(name, value)`

以键值对的方式设置HTTP Response中的HTTP头参数，使用`add_header`配置的Header值将不会覆盖之前配置的Header

```python
import tornado.web

class DetialHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("NUMBER", 9)
        self.add_header("NUMBER", 8)
```

- `write(chunk)`

将给定的块作为HTTP Body发送给客户端。在一般情况下，用本函数输出字符串给客户端。如果给定的块是一个字典，则会将这个块以JSON格式发送给客户端，同时将HTTP Header中的Content_Type设置为`application/json`

- `finishe(chunk=None)`

通知`Tornado.Response`的生成工作已经完成，chunk参数是需要传递给客户端的HTTP body。调用`finish()`后，Tornado将客户端发送HTTPResponse。

- `render(template_name, **kwargs)`

用给定的参数渲染模版，可以在本函数中传入模板文件名称和模板参数

```python
import tornado.web

class DetialHandler(tornado.web.RequestHandler):
    def get(self):
        items = ["Python", "C++", "Java"]
        self.render("template.html", title="Tornado Templates", items=items)
```

- `redirect(url, permanent=False, status=None)`

进行页面重定向

```python
import tornado.web
import tornado.escape

class LoginHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("login.html", next=self.get_argument("next", "/"))
    def post(self):
        username = self.get_argument("username", "")
        password = self.get_argument("password", "")
        auth = self.db.authenticate(username, password)
        if auth:
            # 成功则重定向到next参数所指向的URL
            self.set_current_user(username)
            self.redirect(self.get_argument("next", u"/"))
        else:
            # 不成功则重定向到/login页面
            error_msg = u"?error=" + tornado.escape.url_escape("Login incorrect.")
            self.redirect(u"/login" + error_msg)
            
```

- `clear()`

清空所有在本次请求之前写入的Header和Body内容

```python
import tornado.web


class DetailHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("number", 8)
        self.clear()

```

- `set_cookie(name, value)`

按键值对设置Response中的cookie值

- `clear_all_cookie(path="/", domain=None)`

清空本次请求中的所有cookie

## 异步化/协程化

之前的`get/post`等函数都是采用同步的方法处理用户的请求，在函数完成处理，退出函数后马上向客户端返回Response。但是当处理逻辑比较复杂或者需等待外部I/O时，这样就会阻塞服务器线程，并不适合大量客户端的高并发请求场景。

Tronado中有两种方式来改变同步的处理流程

- 异步化：针对RequestHandler的处理函数使用`@tornado.web.asynchronous`修饰器，将默认的同步机制改为异步机制。
- 协程化：针对RequestHandler的处理函数使用`@tornado.gen.coroutine`修饰器，将默认的同步机制改为协程机制。

### 异步化

```python
import tornado.web
import tornado.httpclient


class MainHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    def get(self):
        http = tornado.httpclient.AsyncHTTPClient()
        http.fetch("http://www.bidu.com", callback=self.on_response)

    def on_resposne(self, response):
        if response.error: raise tornado.web.HTTPError(500)
        self.write(response.body)
        self.finish()

```

### 协程化

```python
import tornado.web
import tornado.httpclient


class MainHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def get(self):
        http = tornado.httpclient.AsyncHTTPClient()
        response = yield http.fetch("http://www.bidu.com")
        self.write(response.body)

```

