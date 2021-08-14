# Websocket

## 概述

HTML5 开始提供的一种浏览器与服务器进行全双工通讯的网络技术，属于应用层协议。它基于 TCP 传输协议，并复用 HTTP 的握手通道。

WebSocket 复用了 HTTP 的握手通道。具体指的是，客户端通过 HTTP 请求与 WebSocket 服务端协商升级协议。协议升级完成后，后续的数据交换则遵照 WebSocket 的协议。

优点：

```
1.较少的控制开销。
在连接创建后，服务器和客户端之间交换数据时，用于协议控制的数据包头部相对较小。在不包含扩展的情况下，对于服务器到客户端的内容，此头部大小只有2至10字节（和数据包长度有关）；对于客户端到服务器的内容，此头部还需要加上额外的4字节的掩码。相对于HTTP请求每次都要携带完整的头部，此项开销显著减少了。
2.更强的实时性。
由于协议是全双工的，所以服务器可以随时主动给客户端下发数据。相对于HTTP请求需要等待客户端发起请求服务端才能响应，延迟明显更少；即使是和Comet等类似的长轮询比较，其也能在短时间内更多次地传递数据。
3. 保持连接状态。
与HTTP不同的是，Websocket需要先创建连接，这就使得其成为一种有状态的协议，之后通信时可以省略部分状态信息。而HTTP请求可能需要在每个请求都携带状态信息（如身份认证等）。 更好的二进制支持。Websocket定义了二进制帧，相对HTTP，可以更轻松地处理二进制内容。
4.可以支持扩展。
Websocket定义了扩展，用户可以扩展协议、实现部分自定义的子协议。如部分浏览器支持压缩等。
5.更好的压缩效果。
相对于HTTP压缩，Websocket在适当的扩展支持下，可以沿用之前内容的上下文，在传递类似的数据时，可以显著地提高压缩率。
6.没有同源限制，客户端可以与任意服务器通信。
7.可以发送文本，也可以发送二进制数据。
```

特点

```
1.websocket适合服务器主动推送的场景
2.相对于Ajax和Long poll等技术，websocket通信模型更高效
3.websocket仍然与HTTP完成Internet通信
4.因为它是HTML5标准协议，所以不受企业防火墙拦截
```

## 通信原理

websocket的通信原理是在客户端 和服务端之间建立TCP持久链接，从而使当服务器有消息需要推送给客户端时能够进行即时通信。

虽然websocket不是HTTP，但是由于在Internet上HTML本身是由HTTP封装并进行传输的，所以websocket仍然需要与HTTP进行协作。IETF在RFC6455中定义了基于HTTP链路建立Websocket信道的标准流程。

![websocket](images/websocket.png)

Websocket使用ws或wss的统一资源标志符，类似于HTTPS，其中wss表示在TLS之上的Websocket。如：

```
ws://example.com/wsapi
wss://secure.example.com/
```

Websocket使用和 HTTP 相同的 TCP 端口，可以绕过大多数防火墙的限制。默认情况下，Websocket协议使用80端口；运行在TLS之上时，默认使用443端口。

![ws与http](images/ws与http.jpg)

WebSocket 是独立的、创建在 TCP 上的协议报文

**Websocket 通过 HTTP/1.1 协议的101状态码进行握手。**

为了创建Websocket连接，需要通过浏览器发出请求，之后服务器进行回应，这个过程通常称为“握手”（handshaking）。

一个典型的Websocket握手请求如下：

- 请求报文

采用的是标准的 HTTP 报文格式，且只支持`GET`方法

```javascript
GET / HTTP/1.1
Host: localhost:8080
Origin: http://127.0.0.1:3000
Connection: Upgrade  		//表示要升级协议
Upgrade: websocket			//表示要升级到websocket协议	
Sec-WebSocket-Version: 13	//表示 websocket 的版本。如果服务端不支持该版本，需要返回一个Sec-WebSocket-Versionheader，里面包含服务端支持的版本号。
Sec-WebSocket-Key: w4v7O6xFTi36lq3RNcgctw==  //浏览器随机生成，与后面服务端响应首部的Sec-WebSocket-Accept是配套的，提供基本的防护，比如恶意的连接，或者无意的连接
```

- 响应报文

备注：每个 header 都以`\r\n`结尾，并且最后一行加上一个额外的空行`\r\n`。此外，服务端回应的 HTTP 状态码只能在握手阶段使用。过了握手阶段后，就只能采用特定的错误码。

注意： Sec-WebSocket-Key/ Sec-WebSocket-Accept 的换算，只能带来基本的保障，但连接是否安全、数据是否安全、客户端 / 服务端是否合法的 ws 客户端、ws 服务端，其实并没有实际性的保证。

```javascript
HTTP/1.1 101 Switching Protocols  //101状态码表示服务器已经理解了客户端的请求，表示协议切换。
Connection:Upgrade  //通过Upgrade消息头通知客户端采用不同的协议来完成这个请求
Upgrade: websocket
Sec-WebSocket-Accept: Oy4NRAQ13jhfONC7bP8dTKb4PTU=  //经过服务器确认，并且加密过后的 Sec-WebSocket-Key
Sec-WebSocket-Protocol: chat  // 表示最终使用的协议
```

Sec-WebSocket-Accept计算方法

```javascript
// 1.将 Sec-WebSocket-Key 跟 258EAFA5-E914-47DA-95CA-C5AB0DC85B11 拼接；
// 2.通过 SHA1 计算出摘要，并转成 base64 字符串
const crypto = require('crypto');
const magic = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const secWebSocketKey = 'w4v7O6xFTi36lq3RNcgctw==';

let secWebSocketAccept = crypto.createHash('sha1')
	.update(secWebSocketKey + magic)
	.digest('base64');

console.log(secWebSocketAccept);
// Oy4NRAQ13jhfONC7bP8dTKb4PTU=
```

- 注意

```
- Connection必须设置Upgrade，表示客户端希望连接升级。
- Upgrade字段必须设置Websocket，表示希望升级到Websocket协议。
- Sec-WebSocket-Key是随机的字符串，服务器端会用这些数据来构造出一个SHA-1的信息摘要。把“Sec-WebSocket-Key”加上一个特殊字符串“258EAFA5-E914-47DA-95CA-C5AB0DC85B11”，然后计算SHA-1摘要，之后进行BASE-64编码，将结果做为“Sec-WebSocket-Accept”头的值，返回给客户端。如此操作，**可以尽量避免普通HTTP请求被误认为Websocket协议。**
- Sec-WebSocket-Version 表示支持的Websocket版本。RFC6455要求使用的版本是13，之前草案的版本均应当弃用。
- Origin字段是可选的，通常用来表示在浏览器中发起此Websocket连接所在的页面，类似于Referer。但是，与Referer不同的是，Origin只包含了协议和主机名称。
- 其他一些定义在HTTP协议中的字段，如Cookie等，也可以在Websocket中使用。
```

## 实现

### 客户端

向 8080 端口发起 WebSocket 连接。连接建立后，打印日志，同时向服务端发送消息。接收到来自服务端的消息后，同样打印日志

```javascript
<html>
<head>
    <title>ws demo</title>
    <meta charset="utf8">
</head>
<body>
  
</body>
<script>
  // 创建WebSocket对象，ws表示Websocket协议，后面接地址及端口
  var ws = new WebSocket('ws://localhost:8080');
  // 此事件发生在websocket链接建立时
  ws.onopen = function (msg) {  
    console.log('client: ws connection is open');
    // 向服务器发送消息
    ws.send('hello');
  };
  // 此事件发生在收到了来自服务器的消息时
  ws.onmessage = function (message) {
    console.log('client: received %s', message.data);
  };
  // 此事件发生在通信过程中有任何错误时
  ws.onerror = function(error){
      console.log('Error: ' + error.name + error.number)
  };
  // 此事件发生在与服务器的连接关闭时
  ws.onclose = function(){
      console.log('Websocket closed!')
  }
  window.onbeforeunload = function(){
      ws.onclose = function(){}  // 首先关闭WebSocket
      // 主动关闭现有链接
      ws.close()
  }
</script>
</html>
```

### 服务端

- node.js

监听 8080 端口。当有新的连接请求到达时，打印日志，同时向客户端发送消息。当收到到来自客户端的消息时，同样打印日志。

```javascript
var app = require('express')();
var server = require('http').Server(app);
var WebSocket = require('ws');

var wss = new WebSocket.Server({ port: 8080 });

wss.on('connection', function connection(ws) {
    console.log('server: receive connection.');
    
    ws.on('message', function incoming(message) {
        console.log('server: received %s', message);
        ws.send('server: reply');
    });

    ws.on('pong', () => {
        console.log('server: received pong from client');
    });

    ws.send('world');
    
    // setInterval(() => {
    //     ws.ping('', false, true);
    // }, 2000);
});

app.get('/', function (req, res) {
  res.sendfile(__dirname + '/index.html');
});

app.listen(3000);
```

- python

[参考1](https://www.cnblogs.com/lichmama/p/3931212.html) [参考2](https://www.cnblogs.com/JetpropelledSnake/p/9033064.html) [参考3](https://www.jb51.net/article/97748.htm)

实现一

```python
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

实现二

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

实现三

```python
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
	
if __name__ == '__main__':
    ws_service()
```

