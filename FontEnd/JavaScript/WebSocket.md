# WebSocket

## 概述

### 概念

HTTP请求是基于请求响应模式的，客户端永远是主动的一方，这种单向请求的特点，注定了如果服务器有连续的状态变化，客户端要获知就非常麻烦。我们只能使用 **轮询** ：每隔一段时候，就发出一个询问，了解服务器有没有新的信息。轮询的效率低，非常浪费资源（因为必须不停连接，或者 HTTP 连接始终打开）。WebSocket 最大特点就是，服务器可以主动向客户端推送信息，客户端也可以主动向服务器发送信息，是真正的双向平等对话，属于 **服务器推送技术** 的一种。

### 特点

webSocket主要有以下几个特点：

```
1. 建立在 TCP 协议之上，服务器端的实现比较容易。
2. 与 HTTP 协议有着良好的兼容性。默认端口也是80和443，并且**握手阶段采用 HTTP 协议**，因此握手时不容易屏蔽，能通过各种 HTTP 代理服务器。
3. 数据格式比较轻量，性能开销小，通信高效。
4. 可以发送文本，也可以发送二进制数据。
5. 没有同源限制，客户端可以与任意服务器通信。
6. 协议标识符是`ws`（如果加密，则为`wss`），服务器网址就是 URL。
```

工作过程：

```
建立一个 WebSocket 连接，客户端浏览器首先要向服务器发起一个 HTTP 请求，
这个请求和通常的 HTTP 请求不同，包含了一些附加头信息，其中 附加头信息"Upgrade: WebSocket" 表明这是一个申请协议升级的 HTTP 请求，
服务器端解析这些附加的头信息然后产生应答信息返回给客户端，
客户端和服务器端的 WebSocket 连接就建立起来了，
双方就可以通过这个连接通道自由的传递信息，并且这个连接会持续存在直到客户端或者服务器端的某一方主动的关闭连接。
```

http生命周期

```sequence
title:HTTP
participant Client
participant Server

Client-->Server: Request
Server-->Client: Response
Client-->Server: Request
Server-->Client: Response
```

WebSocket生命周期

```sequence
title:WebSocket
participant Client
participant Server

Client-->Server: Handshake
Server-->Client: Acknowledgement
Client-->Server: 
Note right of Client: Bi-directional messages
Server-->Client:
Client-->Server: 
Server-->Client: 
Note right of Client: Connection End
```

## WebSocket API

创建WebSocket对象

```javascript
var Socket = new WebSocket(url, [protocol] );
```

属性

| **属性**              | **描述**                                                     |
| --------------------- | ------------------------------------------------------------ |
| Socket.readyState     | 只读属性 **readyState** 表示连接状态：0 - 连接尚未建立。1 - 连接已建立，可以进行通信。2 -连接正在进行关闭。3 - 连接已经关闭或者连接不能打开。 |
| Socket.bufferedAmount | 只读属性 **bufferedAmount** 已被 send() 放入正在队列中等待传输，但是还没有发出的 UTF-8 文本字节数。 |

事件

| 事件    | 事件处理程序     | 描述                       |
| ------- | ---------------- | -------------------------- |
| open    | Socket.onopen    | 连接建立时触发             |
| message | Socket.onmessage | 客户端接收服务端数据时触发 |
| error   | Socket.onerror   | 通信发生错误时触发         |
| close   | Socket.onclose   | 连接关闭时触发             |

方法

| 方法           | 描述             |
| -------------- | ---------------- |
| Socket.send()  | 使用连接发送数据 |
| Socket.close() | 关闭连接         |

## 使用

前端示例

```javascript
var ws = new WebSocket("ws://localhost:8080");

ws.onopen = function()
{
   // Web Socket 已连接上，使用 send() 方法发送数据
   ws.send("发送数据");
   alert("数据发送中...");
};
 
ws.onmessage = function (evt) 
{ 
   var received_msg = evt.data;
   alert("数据已接收...");
};
 
ws.onclose = function()
{ 
   // 关闭 websocket
   alert("连接已关闭..."); 
};
```

后端示例

```
见各个应用框架
```

