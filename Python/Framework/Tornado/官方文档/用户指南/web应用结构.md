# web应用结构

Tornado Web应用程序通常由一个或多个`RequestHandler`子类，将传入的请求路由到处理程序的`Application`对象以及启动服务器的`main()`函数组成。

一个最小的“ hello world”示例如下所示：

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

## `Application`对象

`Application`对象负责全局配置，包括将请求映射到处理程序的路由表。

路由表是`URLSpec`对象（或元组）的列表，每个对象都（至少）包含一个正则表达式和一个处理程序类。顺序是有影响的；使用第一个匹配规则。如果正则表达式包含捕获组，则这些组是路径参数，并将传递给处理程序的HTTP方法。如果将字典作为`URLSpec`的第三个元素传递，则它将提供初始化参数，该参数将传递给`RequestHandler.initialize`。最后，`URLSpec`可能有一个名称，这将使其可以与`RequestHandler.reverse_url`一起使用。

例如，在此片段中，根URL`/`映射到`MainHandler`，形式为`/story/`后跟数字的URL映射到`StoryHandler`。该数字（作为字符串）传递给`StoryHandler.get`。

```python
class MainHandler(RequestHandler):
    def get(self):
        self.write('<a href="%s">link to story 1</a>' %
                   self.reverse_url("story", "1"))

class StoryHandler(RequestHandler):
    def initialize(self, db):
        self.db = db

    def get(self, story_id):
        self.write("this is story %s" % story_id)

app = Application([
    url(r"/", MainHandler),
    url(r"/story/([0-9]+)", StoryHandler, dict(db=db), name="story")
    ])
```

`Application`构造函数采用许多关键字参数，这些参数可用于自定义应用程序的行为并启用可选功能。有关完整列表，请参见`Application.settings`。

## `RequestHandler`子类

Tornado Web应用程序的大部分工作都是在`RequestHandler`的子类中完成的。处理程序子类的主要入口点是一种以要处理的HTTP方法命名的方法：`get()`，`post()`等。每个处理程序都可以定义一个或多个这些方法来处理不同的HTTP操作。如上所述，将使用与匹配的路由规则的捕获组相对应的参数来调用这些方法。

在处理程序中，调用诸如`RequestHandler.render`或`RequestHandler.write`之类的方法以产生响应。`render()`按名称加载`Template`，并使用给定的参数进行渲染。`write()`用于基于非模板的输出；它接受字符串，字节和字典（字典将被编码为JSON）。

`RequestHandler`中的许多方法被设计为在子类中重写，并在整个应用程序中使用。定义一个`BaseHandler`类来覆盖诸如`write_error`和`get_current_user`之类的方法是很常见的，然后为您所有的特定处理程序创建您自己的`BaseHandler`而不是`RequestHandler`的子类。

## 处理请求输入

请求处理程序可以使用`self.request`访问表示当前请求的对象。有关属性的完整列表，请参见HTTPServerRequest的类定义。

HTML表单使用的格式的请求数据将为您解析，并在诸如`get_query_argument`和`get_body_argument`之类的方法中可用。

```python
class MyFormHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('<html><body><form action="/myform" method="POST">'
                   '<input type="text" name="message">'
                   '<input type="submit" value="Submit">'
                   '</form></body></html>')

    def post(self):
        self.set_header("Content-Type", "text/plain")
        self.write("You wrote " + self.get_body_argument("message"))
```

由于HTML表单编码对于参数是单个值还是具有一个元素的列表是模棱两可的，因此`RequestHandler`具有不同的方法来允许应用程序指示它是否期望列表。对于列表，请使用`get_query_arguments`和`get_body_arguments`而不是单数形式。

通过表单上传的文件位于`self.request.files`中，该文件将名称（HTML `<input type ="file">`元素的名称）映射到文件列表。每个文件都是格式为`{"filename":..., "content_type":..., "body":...}`的字典。仅当文件是使用表单包装程序（即`multipart / form-data Content-Type`）上传时，`files`对象才存在；如果未使用此格式，则原始上传的数据可在`self.request.body`中获得。默认情况下，上传的文件将完全缓冲在内存中；如果您需要处理太大而无法舒适地保留在内存中的文件，请参见`stream_request_body`类装饰器。

在demos目录中，file_receiver.py显示了两种接收文件上传的方法。

file_receiver

```python
#!/usr/bin/env python

"""Usage: python file_receiver.py
Demonstrates a server that receives a multipart-form-encoded set of files in an
HTTP POST, or streams in the raw data of a single file in an HTTP PUT.
See file_uploader.py in this directory for code that uploads files in this format.
"""

import logging

try:
    from urllib.parse import unquote
except ImportError:
    # Python 2.
    from urllib import unquote

import tornado.ioloop
import tornado.web
from tornado import options


class POSTHandler(tornado.web.RequestHandler):
    def post(self):
        for field_name, files in self.request.files.items():
            for info in files:
                filename, content_type = info["filename"], info["content_type"]
                body = info["body"]
                logging.info(
                    'POST "%s" "%s" %d bytes', filename, content_type, len(body)
                )

        self.write("OK")


@tornado.web.stream_request_body
class PUTHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.bytes_read = 0

    def data_received(self, chunk):
        self.bytes_read += len(chunk)

    def put(self, filename):
        filename = unquote(filename)
        mtype = self.request.headers.get("Content-Type")
        logging.info('PUT "%s" "%s" %d bytes', filename, mtype, self.bytes_read)
        self.write("OK")


def make_app():
    return tornado.web.Application([(r"/post", POSTHandler), (r"/(.*)", PUTHandler)])


if __name__ == "__main__":
    # Tornado configures logging.
    options.parse_command_line()
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
```

file_uploader

```python
#!/usr/bin/env python

"""Usage: python file_uploader.py [--put] file1.txt file2.png ...
Demonstrates uploading files to a server, without concurrency. It can either
POST a multipart-form-encoded request containing one or more files, or PUT a
single file without encoding.
See also file_receiver.py in this directory, a server that receives uploads.
"""

import mimetypes
import os
import sys
from functools import partial
from uuid import uuid4

try:
    from urllib.parse import quote
except ImportError:
    # Python 2.
    from urllib import quote

from tornado import gen, httpclient, ioloop
from tornado.options import define, options


# Using HTTP POST, upload one or more files in a single multipart-form-encoded
# request.
@gen.coroutine
def multipart_producer(boundary, filenames, write):
    boundary_bytes = boundary.encode()

    for filename in filenames:
        filename_bytes = filename.encode()
        mtype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        buf = (
            (b"--%s\r\n" % boundary_bytes)
            + (
                b'Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
                % (filename_bytes, filename_bytes)
            )
            + (b"Content-Type: %s\r\n" % mtype.encode())
            + b"\r\n"
        )
        yield write(buf)
        with open(filename, "rb") as f:
            while True:
                # 16k at a time.
                chunk = f.read(16 * 1024)
                if not chunk:
                    break
                yield write(chunk)

        yield write(b"\r\n")

    yield write(b"--%s--\r\n" % (boundary_bytes,))


# Using HTTP PUT, upload one raw file. This is preferred for large files since
# the server can stream the data instead of buffering it entirely in memory.
@gen.coroutine
def post(filenames):
    client = httpclient.AsyncHTTPClient()
    boundary = uuid4().hex
    headers = {"Content-Type": "multipart/form-data; boundary=%s" % boundary}
    producer = partial(multipart_producer, boundary, filenames)
    response = yield client.fetch(
        "http://localhost:8888/post",
        method="POST",
        headers=headers,
        body_producer=producer,
    )

    print(response)


@gen.coroutine
def raw_producer(filename, write):
    with open(filename, "rb") as f:
        while True:
            # 16K at a time.
            chunk = f.read(16 * 1024)
            if not chunk:
                # Complete.
                break

            yield write(chunk)


@gen.coroutine
def put(filenames):
    client = httpclient.AsyncHTTPClient()
    for filename in filenames:
        mtype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        headers = {"Content-Type": mtype}
        producer = partial(raw_producer, filename)
        url_path = quote(os.path.basename(filename))
        response = yield client.fetch(
            "http://localhost:8888/%s" % url_path,
            method="PUT",
            headers=headers,
            body_producer=producer,
        )
        print(response)


if __name__ == "__main__":
    define("put", type=bool, help="Use PUT instead of POST", group="file uploader")

    # Tornado configures logging from command line opts and returns remaining args.
    filenames = options.parse_command_line()
    if not filenames:
        print("Provide a list of filenames to upload.", file=sys.stderr)
        sys.exit(1)

    method = put if options.put else post
    ioloop.IOLoop.current().run_sync(lambda: method(filenames))
```

由于HTML表单编码的古怪之处（例如，单数和复数参数前后存在歧义），Tornado不会尝试将表单参数与其他类型的输入统一。特别是，我们不解析JSON请求主体。希望使用JSON而不是表单编码的应用程序可能会覆盖`prepare`来解析其请求：

```python
def prepare(self):
    if self.request.headers.get("Content-Type", "").startswith("application/json"):
        self.json_args = json.loads(self.request.body)
    else:
        self.json_args = None
```

##覆写RequestHandler方法

除了`get(),post()`外，`RequestHandler`中的某些其他方法还设计为在必要时被子类覆盖。在每个请求上，都会按以下顺序进行调用：

1. 在每个请求上都会创建一个新的`RequestHandler`对象。
2. 使用`Application`配置中的初始化参数调用`initialize()`。初始化通常应该只保存传递给成员变量的参数；它可能不会产生任何输出或调用方法，例如`send_error`。
3. `prepare()`被调用。这在所有处理程序子类共享的基类中最有用，因为无论使用哪种HTTP方法，都会调用`prepare`。`prepare`可能产生产出；如果调用`finish`（或`redirect`等），则处理在此处停止。
4. HTTP方法之一称为：`get(),post(),put()`等。如果URL正则表达式包含捕获组，则将它们作为参数传递给此方法。
5. 请求完成后，将调用`on_finish()`。通常是在`get()`或另一个HTTP方法返回之后。

在`RequestHandler`文档中会注明所有旨在被覆盖的方法。一些最常用的覆盖方法包括：

- `write_error`-输出HTML以用于错误页面。
- `on_connection_close`-客户端断开连接时调用；应用程序可能选择检测这种情况并停止进一步处理。请注意，不能保证可以立即检测到关闭的连接。
- `get_current_user`-请参阅用户身份验证。
- `get_user_locale`-返回要用于当前用户的Locale对象。
- `set_default_headers`-可用于在响应上设置其他标头（例如自定义`server`标头）。

## 错误处理

如果处理程序引发异常，Tornado将调用`RequestHandler.write_error`生成错误页面。`tornado.web.HTTPError`可用于生成指定的状态码；所有其他异常均返回500状态。

默认错误页面包括调试模式下的堆栈跟踪以及错误的单行描述（例如“ 500：内部服务器错误”）。要生成自定义错误页面，请重写`RequestHandler.write_error`（可能在所有处理程序共享的基类中）。此方法通常可以通过诸如`write`和`render`之类的方法产生输出。如果错误是由异常引起的，则`exc_info`三元组将作为关键字参数传递（请注意，不能保证此异常是`sys.exc_info`中的当前异常，因此`write_error`必须使用例如`traceback.format_exception`而不是`traceback.format_exc`）。

通过调用`set_status`，编写响应并返回，还可以从常规处理程序方法而不是`write_error`生成错误页面。在简单返回不方便的情况下，可能引发特殊异常`tornado.web.Finish`以终止处理程序，而无需调用`write_error`。

对于404错误，请使用`default_handler_class`应用程序设置。该处理程序应该重写`prepare`，而不是像`get()`这样的更具体的方法，因此它可以与任何HTTP方法一起使用。它应如上所述产生其错误页面：通过引发`HTTPError(404)`并重写`write_error`，或调用`self.set_status(404)`并直接`在prepare()`中产生响应。

## 重定向

在Tornado中重定向请求的主要方法有两种：`RequestHandler.redirect`和`RedirectHandler`。

您可以在`RequestHandler`方法中使用`self.redirect()`将用户重定向到其他地方。还有一个可选参数`permanent`，可用于指示重定向被视为永久。`permanent`的默认值为`False`，它将生成`302 Found` HTTP响应代码，适用于在成功的`POST`请求后重定向用户之类的事情。如果`permanent`为`True`，则使用`301 Moved Permanently`HTTP响应代码，这对于例如以SEO友好的方式重定向到页面的规范URL比较有用。

`RedirectHandler`使您可以直接在`Application`路由表中配置重定向。例如，配置单个静态重定向：

```python
app = tornado.web.Application([
    url(r"/app", tornado.web.RedirectHandler,
        dict(url="http://itunes.apple.com/my-app-id")),
    ])
```

`RedirectHandler`还支持正则表达式替换。以下规则将所有以`/pictures/`开头的请求重定向到前缀`/photos/`：

```python
app = tornado.web.Application([
    url(r"/photos/(.*)", MyPhotoHandler),
    url(r"/pictures/(.*)", tornado.web.RedirectHandler,
        dict(url=r"/photos/{0}")),
    ])
```

与`RequestHandler.redirect`不同，`RedirectHandler`默认使用永久重定向。这是因为路由表在运行时不会更改，并且假定为永久性的，而在处理程序中找到的重定向很可能是其他可能更改的逻辑的结果。要使用`RedirectHandler`发送临时重定向，请在`RedirectHandler`初始化参数中添加`permanent=False`。

## 异步处理程序

某些处理程序方法（包括`prepare()`和HTTP动词方法`get(),post()`等）可能被重写为协程，以使处理程序异步。

例如，这是一个使用协程的简单处理程序：

```python
class MainHandler(tornado.web.RequestHandler):
    async def get(self):
        http = tornado.httpclient.AsyncHTTPClient()
        response = await http.fetch("http://friendfeed-api.com/v2/feed/bret")
        json = tornado.escape.json_decode(response.body)
        self.write("Fetched " + str(len(json["entries"])) + " entries "
                   "from the FriendFeed API")
```

对于更高级的异步示例，请看 [chat example application](https://github.com/tornadoweb/tornado/tree/stable/demos/chat),，该应用程序使用长轮询来实现AJAX聊天室。长时间轮询的用户可能希望在客户端关闭连接后重写`on_connection_close()`进行清理（但请注意该方法的文档字符串以进行警告）。

