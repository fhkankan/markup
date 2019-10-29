# Websocket

## 原生实现

### 实现一

[参考](https://www.cnblogs.com/lichmama/p/3931212.html)

协议采用Hybi-10

- 后台客户端服务端

```python
#-*- coding:utf8 -*-

import threading
import hashlib
import socket
import base64

global clients
clients = {}

#通知客户端
def notify(message):
    for connection in clients.values():
        connection.send('%c%c%s' % (0x81, len(message), message))

#客户端处理线程
class websocket_thread(threading.Thread):
    def __init__(self, connection, username):
        super(websocket_thread, self).__init__()
        self.connection = connection
        self.username = username
    
    def run(self):
        print 'new websocket client joined!'
        data = self.connection.recv(1024)
        headers = self.parse_headers(data)
        token = self.generate_token(headers['Sec-WebSocket-Key'])
        self.connection.send('\
HTTP/1.1 101 WebSocket Protocol Hybi-10\r\n\
Upgrade: WebSocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Accept: %s\r\n\r\n' % token)
        while True:
            try:
                data = self.connection.recv(1024)
            except socket.error, e:
                print "unexpected error: ", e
                clients.pop(self.username)
                break
            data = self.parse_data(data)
            if len(data) == 0:
                continue
            message = self.username + ": " + data
            notify(message)
            
    def parse_data(self, msg):
        v = ord(msg[1]) & 0x7f
        if v == 0x7e:
            p = 4
        elif v == 0x7f:
            p = 10
        else:
            p = 2
        mask = msg[p:p+4]
        data = msg[p+4:]
        return ''.join([chr(ord(v) ^ ord(mask[k%4])) for k, v in enumerate(data)])
        
    def parse_headers(self, msg):
        headers = {}
        header, data = msg.split('\r\n\r\n', 1)
        for line in header.split('\r\n')[1:]:
            key, value = line.split(': ', 1)
            headers[key] = value
        headers['data'] = data
        return headers

    def generate_token(self, msg):
        key = msg + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        ser_key = hashlib.sha1(key).digest()
        return base64.b64encode(ser_key)

#服务端
class websocket_server(threading.Thread):
    def __init__(self, port):
        super(websocket_server, self).__init__()
        self.port = port

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', self.port))
        sock.listen(5)
        print 'websocket server started!'
        while True:
            connection, address = sock.accept()
            try:
                username = "ID" + str(address[1])
                thread = websocket_thread(connection, username)
                thread.start()
                clients[username] = connection
            except socket.timeout:
                print 'websocket connection timeout!'

if __name__ == '__main__':
    server = websocket_server(9000)
    server.start()
```

- 测试页面

```html
<!--
@http://www.cnblogs.com/zhuweisky/p/3930780.html
-->
<!DOCTYPE html>
</html>
    <head>
        <meta charset="utf-8">
    </head>
    <body>
        <h3>WebSocketTest</h3>
        <div id="login">
            <div>
                <input id="serverIP" type="text" placeholder="服务器IP" value="127.0.0.1" autofocus="autofocus" />
                <input id="serverPort" type="text" placeholder="服务器端口" value="9000" />
                <input id="btnConnect" type="button" value="连接" onclick="connect()" />
            </div>
            <div>
                <input id="sendText" type="text" placeholder="发送文本" value="I'm WebSocket Client!" />
                <input id="btnSend" type="button" value="发送" onclick="send()" />
            </div>
            <div>
                <div>
                    来自服务端的消息
                </div>
                <textarea id="txtContent" cols="50" rows="10" readonly="readonly"></textarea>
            </div>
        </div>
    </body>
    <script>
        var socket;

        function connect() {
            var host = "ws://" + $("serverIP").value + ":" + $("serverPort").value + "/"
            socket = new WebSocket(host);
            try {

                socket.onopen = function (msg) {
                    $("btnConnect").disabled = true;
                    alert("连接成功！");
                };

                socket.onmessage = function (msg) {
                    if (typeof msg.data == "string") {
                        displayContent(msg.data);
                    }
                    else {
                        alert("非文本消息");
                    }
                };

                socket.onclose = function (msg) { alert("socket closed!") };
            }
            catch (ex) {
                log(ex);
            }
        }

        function send() {
            var msg = $("sendText").value
            socket.send(msg);
        }

        window.onbeforeunload = function () {
            try {
                socket.close();
                socket = null;
            }
            catch (ex) {
            }
        };

        function $(id) { return document.getElementById(id); }

        Date.prototype.Format = function (fmt) { //author: meizz 
            var o = {
                "M+": this.getMonth() + 1, //月份 
                "d+": this.getDate(), //日 
                "h+": this.getHours(), //小时 
                "m+": this.getMinutes(), //分 
                "s+": this.getSeconds(), //秒 
                "q+": Math.floor((this.getMonth() + 3) / 3), //季度 
                "S": this.getMilliseconds() //毫秒 
            };
            if (/(y+)/.test(fmt)) fmt = fmt.replace(RegExp.$1, (this.getFullYear() + "").substr(4 - RegExp.$1.length));
            for (var k in o)
                if (new RegExp("(" + k + ")").test(fmt)) fmt = fmt.replace(RegExp.$1, (RegExp.$1.length == 1) ? (o[k]) : (("00" + o[k]).substr(("" + o[k]).length)));
            return fmt;
        }

        function displayContent(msg) {
            $("txtContent").value += "\r\n" +new Date().Format("yyyy/MM/dd hh:mm:ss")+ ":  " + msg;
        }
        function onkey(event) { if (event.keyCode == 13) { send(); } }
    </script>
</html>
```

### 实现二

[参考](https://www.cnblogs.com/JetpropelledSnake/p/9033064.html)

- 服务端

```python
# 创建主线程，用于实现接受websocket建立请求
def create_socket():
    # 启动socket并监听连接
    socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket.bind(('127.0.0.1', 8001))
        # 操作系统会在服务器socket被关闭或服务器进程终止后马上释放端口
        socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socket.listen(5)
    except Exception as e:
        logging.error(e)
        return
    else:
        logging.info('Server running...')
        
    # 等待访问
    while True:
        conn, addr = sock.accept()
        data = str(conn.recv(1024))
        logging.debug(data)
        header_dict = {}
        header, _ = data.split(r'\r\n\r\n', 1)
        for line in header.split(r'\r\n')[1:]:
            key, val = line.split(':', 1)
            header_dict[key] = val
            
        if 'Sec-WebSocket-Key' not in header_dict:
            logging.error('This socket is not websocket. client close.')
            cnn.close()
            return
        
        magic_key = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
        sec_key = header_dict['Sec-WebSocket-Key'] + magic_key
        key = base64.b64encode(hashlib.sha1(bytes(sec_key, encoding='utf-8')).digest())
        key_str = str(key)[2:30]
        logging.debug(key_str)
        response = 'HTTP/1.1 101 Switching Protocols\r\n'\
        			'Connection: Upgrade\r\n'\
            		'Upgrade: websocket\r\n'\
                	'Sec-WebSocket-Accept: {0}\r\n'\
                    'WebSocket-Protocol: chat\r\n\r\n'\.format(key_str)
        conn.send(bytes(response, encoding='utf-8'))
        logging.debug('Send the handshake data')
        WebSocetThread(conn).start()
        
        
# 服务端对客户端报文进行解析
def read_msg(data):
    logging.debug(data)
    msg_len = data[1] & 127  # 数载荷的长度
    if msg_len == 126:
        mask = data[4:8]  # mask掩码
        content = data[8:]  # 消息内容
    elif msg_len == 126：
    	mask = data[10:14]
        content = data[14:]
    else:
        mask = data[2:6]
        content = data[6:]
        
    raw_str = ''  # 解码后的内容
    for i, d in enumerate(content):
        raw_str += chr(d ^ mask[i % 4])
    return raw_str

# 服务端发送websocket报文
# 返回时不携带掩码，所以Mask为0，再按载荷数据的大小写入长度，最后写入载荷数据
def write_msg(message):
    data = struct.pack('B', 129)  # 写入第一个字节， 10000001
    # 写入包长度
    msg_len = len(message)
    if msg_len <= 125:
        # struct按照给定的格式fmt，把数据封装成字符串 ( 实际上是类似于 C 结构体的字节流 )
    	data += struct.pack('B', msg_len)
    elif msg_len <= (2**16-1):
        data += struct.pack('!BH', 126, msg_len)
    elif msg_len <= (2**64-1):
        data += struct.pack('!BQ', 127, msg_len)
    else:
        logging.error('Message is too long!')
        return
    data += bytes(message, encoding='utf-8')  # 写入消息内容
    logging.debug(data)
    return data
```

### 实现三

[参考](https://www.jb51.net/article/97748.htm)

- 需求：

使用python简单的实现websocket服务器，可以在浏览器上实时显示远程服务器的日志信息。

在web上弹出iframe层来实时显示远程服务器的日志，点击stop按钮，停止日志输出，以便查看相关日志，点start按钮，继续输出日志，点close按钮，关闭iframe层。

资料中发现很多只能在web上显示本地的日志，不能看远程服务器的日志，能看远程日志的是引用了其他框架（例如bottle，tornado）来实现的，而且所有这些都是要重写thread的run方法来实现的。用python简单的实现websocket服务器也可实现。

- 项目启动

```
nohup python manage.py runserver 10.1.12.110 &
nohup python websocketserver.py &
```

- 服务端代码

启动websocket后，接收到请求，起一个线程和客户端握手，然后根据客户端发送的ip和type，去数据库查找对应的日志路径，用paramiko模块ssh登录到远程服务器上tail查看日志，再推送给浏览器，服务端完整代码如下：

```python
# coding:utf-8
import os
import struct
import base64
import hashlib
import socket
import threading
import paramiko


def get_ssh(ip, user, pwd):
  try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, 22, user, pwd, timeout=15)
    return ssh
  except Exception, e:
    print e
    return "False"


def recv_data(conn):  # 服务器解析浏览器发送的信息
  try:
    all_data = conn.recv(1024)
    if not len(all_data):
      return False
  except:
    pass
  else:
    code_len = ord(all_data[1]) & 127
    if code_len == 126:
      masks = all_data[4:8]
      data = all_data[8:]
    elif code_len == 127:
      masks = all_data[10:14]
      data = all_data[14:]
    else:
      masks = all_data[2:6]
      data = all_data[6:]
    raw_str = ""
    i = 0
    for d in data:
      raw_str += chr(ord(d) ^ ord(masks[i % 4]))
      i += 1
    return raw_str


def send_data(conn, data):  # 服务器处理发送给浏览器的信息
  if data:
    data = str(data)
  else:
    return False
  token = "\x81"
  length = len(data)
  if length < 126:
    token += struct.pack("B", length)  # struct为Python中处理二进制数的模块，二进制流为C，或网络流的形式。
  elif length <= 0xFFFF:
    token += struct.pack("!BH", 126, length)
  else:
    token += struct.pack("!BQ", 127, length)
  data = '%s%s' % (token, data)
  conn.send(data)
  return True


def handshake(conn, address, thread_name):
  headers = {}
  shake = conn.recv(1024)
  if not len(shake):
    return False

  print ('%s : Socket start handshaken with %s:%s' % (thread_name, address[0], address[1]))
  header, data = shake.split('\r\n\r\n', 1)
  for line in header.split('\r\n')[1:]:
    key, value = line.split(': ', 1)
    headers[key] = value

  if 'Sec-WebSocket-Key' not in headers:
    print ('%s : This socket is not websocket, client close.' % thread_name)
    conn.close()
    return False

  MAGIC_STRING = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
  HANDSHAKE_STRING = "HTTP/1.1 101 Switching Protocols\r\n" \
            "Upgrade:websocket\r\n" \
            "Connection: Upgrade\r\n" \
            "Sec-WebSocket-Accept: {1}\r\n" \
            "WebSocket-Origin: {2}\r\n" \
            "WebSocket-Location: ws://{3}/\r\n\r\n"

  sec_key = headers['Sec-WebSocket-Key']
  res_key = base64.b64encode(hashlib.sha1(sec_key + MAGIC_STRING).digest())
  str_handshake = HANDSHAKE_STRING.replace('{1}', res_key).replace('{2}', headers['Origin']).replace('{3}', headers['Host'])
  conn.send(str_handshake)
  print ('%s : Socket handshaken with %s:%s success' % (thread_name, address[0], address[1]))
  print 'Start transmitting data...'
  print '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'
  return True


def dojob(conn, address, thread_name):
  handshake(conn, address, thread_name)   # 握手
  conn.setblocking(0)            # 设置socket为非阻塞

  ssh = get_ssh('192.168.1.1', 'root', '123456')  # 连接远程服务器
  ssh_t = ssh.get_transport()
  chan = ssh_t.open_session()
  chan.setblocking(0)  # 设置非阻塞
  chan.exec_command('tail -f /var/log/messages')

  while True:
    clientdata = recv_data(conn)
    if clientdata is not None and 'quit' in clientdata:  # 但浏览器点击stop按钮或close按钮时，断开连接
      print ('%s : Socket close with %s:%s' % (thread_name, address[0], address[1]))
      send_data(conn, 'close connect')
      conn.close()
      break
    while True:
      while chan.recv_ready():
        clientdata1 = recv_data(conn)
        if clientdata1 is not None and 'quit' in clientdata1:
          print ('%s : Socket close with %s:%s' % (thread_name, address[0], address[1]))
          send_data(conn, 'close connect')
          conn.close()
          break
        log_msg = chan.recv(10000).strip()  # 接收日志信息
        print log_msg
        send_data(conn, log_msg)
      if chan.exit_status_ready():
        break
      clientdata2 = recv_data(conn)
      if clientdata2 is not None and 'quit' in clientdata2:
        print ('%s : Socket close with %s:%s' % (thread_name, address[0], address[1]))
        send_data(conn, 'close connect')
        conn.close()
        break
    break


def ws_service():

  index = 1
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(("127.0.0.1", 12345))
  sock.listen(100)

  print ('\r\n\r\nWebsocket server start, wait for connect!')
  print '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'
  while True:
    connection, address = sock.accept()
    thread_name = 'thread_%s' % index
    print ('%s : Connection from %s:%s' % (thread_name, address[0], address[1]))
    t = threading.Thread(target=dojob, args=(connection, address, thread_name))
    t.start()
    index += 1

	
ws_service()
```

- 客户端代码

```html
<!DOCTYPE html>
<html>
<head>
  <title>WebSocket</title>

  <style>
  #log {
    width: 440px;
    height: 200px;
    border: 1px solid #7F9DB9;
    overflow: auto;
  }
  pre {
    margin: 0 0 0;
    padding: 0;
    border: hidden;
    background-color: #0c0c0c;
    color: #00ff00;
  }
  #btns {
    text-align: right;
  }
  </style>

  <script>
    var socket;
    function init() {
      var host = "ws://127.0.0.1:12345/";

      try {
        socket = new WebSocket(host);
        socket.onopen = function () {
          log('Connected');
        };
        socket.onmessage = function (msg) {
          log(msg.data);
          var obje = document.getElementById("log");  //日志过多时清屏
          var textlength = obje.scrollHeight;
          if (textlength > 10000) {
            obje.innerHTML = '';
          }
        };
        socket.onclose = function () {
          log("Lose Connection!");
          $("#start").attr('disabled', false);
          $("#stop").attr('disabled', true);
        };
        $("#start").attr('disabled', true);
        $("#stop").attr('disabled', false);
      }
      catch (ex) {
        log(ex);
      }
    }
    window.onbeforeunload = function () {
      try {
        socket.send('quit');
        socket.close();
        socket = null;
      }
      catch (ex) {
        log(ex);
      }
    };
    function log(msg) {
      var obje = document.getElementById("log");
      obje.innerHTML += '<pre><code>' + msg + '</code></pre>';
      obje.scrollTop = obje.scrollHeight;  //滚动条显示最新数据
    }
    function stop() {
      try {
        log('Close connection!');
        socket.send('quit');
        socket.close();
        socket = null;
        $("#start").attr('disabled', false);
        $("#stop").attr('disabled', true);
      }
      catch (ex) {
        log(ex);
      }
    }
    function closelayer() {
      try {
        log('Close connection!');
        socket.send('quit');
        socket.close();
        socket = null;
      }
      catch (ex) {
        log(ex);
      }
      var index = parent.layer.getFrameIndex(window.name); //先得到当前iframe层的索引
      parent.layer.close(index); //再执行关闭
    }
  </script>

</head>


<body onload="init()">
  <div >
    <div >
      <div id="log" ></div>
      <br>
    </div>
  </div>
  <div >
    <div >
      <div id="btns">
        <input disabled="disabled" type="button" value="start" id="start" onclick="init()">
        <input disabled="disabled" type="button" value="stop" id="stop" onclick="stop()" >
        <input type="button" value="close" id="close" onclick="closelayer()" >
      </div>
    </div>
  </div>
</body>

</html>
```

