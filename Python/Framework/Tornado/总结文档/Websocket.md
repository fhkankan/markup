# Websocket

tornado定义了`tornado.websocket.WebsocketHandler`类用于处理Websocket连接的请求。

有Tronado框架自动调用的入口函数：

- `open()`

在一个新的websocket链接建立时，Tornado框架会调用此函数。在本函数中，可以和在`get/post`等函数中一样用`get_argument()`函数获取客户端提交的参数，也可以用`get_secure_cookie/set_secure_cookie`操作Cookie等。

- `on_message(message)`

建立websocket链接后，当收到来自客户端的消息时，Tronado框架会调用本函数。通常，这是服务器端websocket编程的核心函数，通过解析收到的消息作出相应的处理。

- `on_close()`

当websocket链接被关闭时，Tronado框架会调用本函数。可以通过访问`self.close_code，self.close_reason`查询关闭的原因。

开发者主动操作websocket的函数

- `write_message(messagee, binary=False)`

用于向与本链接相对应的客户端写消息。

- `close(code=None,reason=None)`

主动关闭websocket链接。其中的code和reason用于告诉客户端链接被关闭的原因。code必须是一个数值，而reason是一个字符串。

示例

```python
import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado.options import define, options, parse_command_line

define("port", default=8888, help="run on the given port", type=int)

clients = dict()  # 客户端session字典


class IndexHandler(tornado.web.RequestHandler):
    """页面处理器，用于向客户端渲染主页Index.html"""
    @tornado.web.asynchronous
    def get(self):
        self.render("index.html")


class MyWebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self, *args):
        """有新链接时被调用"""
        self.id = self.get_argument("Id")
        self.stream.set_nodelay(True)
        clients[self.id] = {"id": self.id, "object": self}  # 保存session到clients字典中

    def on_message(self, message):
        """收到消息时被调用"""
        print("Client %s received a message: %s" % (self.id, message))

    def on_close(self):
        """关闭链接时被调用"""
        if self.id in clients:
            del clients[self.id]
            print("Client %s is closed" % (self.id))

    def check_origin(self, origin):
        return True


app = tornado.web.Application([
    (r'/', IndexHandler),
    (r'/websocket', MyWebSocketHandler),
])

import threading
import time
import datetime
import asyncio


# 启动单独的线程运行此函数，每隔1s向所有客户端推送当前时间
def sendTime():
    asyncio.set_event_loop(asyncio.new_event_loop())  # 启动异步event loop
    while True:
        for key in clients.keys():
            msg = str(datetime.datetime.now())
            clients[key]["object"].write_message(msg)
            print("write to client %s: %s" % (key, msg))
        time.sleep(1)


if __name__ == '__main__':
    threading.Thread(traget=sendTime).start()  # 启动推送时间线程
    parse_command_line()
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()  # 挂起运行

```

