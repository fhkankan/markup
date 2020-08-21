# Websocket

## 概述

HTML5 开始提供的一种浏览器与服务器进行全双工通讯的网络技术，属于应用层协议。它基于 TCP 传输协议，并复用 HTTP 的握手通道。

WebSocket 复用了 HTTP 的握手通道。具体指的是，客户端通过 HTTP 请求与 WebSocket 服务端协商升级协议。协议升级完成后，后续的数据交换则遵照 WebSocket 的协议。

优点：

```
1.支持双向通信，实时性更强。
2.更好的二进制支持。
3.较少的控制开销。连接创建后，ws 客户端、服务端进行数据交换时，协议控制的数据包头部较小。在不包含头部的情况下，服务端到客户端的包头只有 2~10 字节（取决于数据包长度），客户端到服务端的的话，需要加上额外的 4 字节的掩码。而 HTTP 协议每次通信都需要携带完整的头部。
4.支持扩展。ws 协议定义了扩展，用户可以扩展协议，或者实现自定义的子协议。（比如支持自定义压缩算法等）
```

### 请求报文

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

### 响应报文

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
  ws.onopen = function (msg) {
    console.log('client: ws connection is open');
    ws.send('hello');
  };
  ws.onmessage = function (message) {
    console.log('client: received %s', message.data);
  };
  ws.onerror = function(error){
      console.log('Error: ' + error.name + error.number)
  };
  ws.onclose = function(){
      console.log('Websocket closed!')
  }
  window.onbeforeunload = function(){
      ws.onclose = function(){}  // 首先关闭WebSocket
      ws.close()
  }
</script>
</html>
```

### 服务端

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