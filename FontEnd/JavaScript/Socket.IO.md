# Socket.IO

## 概述

**Socket.IO 本是一个面向实时 web 应用的 JavaScript 库，现在已成为拥有众多语言支持的Web即时通讯应用的框架。**

Socket.IO 主要使用WebSocket协议。但是如果需要的话，Socket.io可以回退到几种其它方法，例如Adobe Flash Sockets，JSONP拉取，或是传统的AJAX拉取，并且在同时提供完全相同的接口。尽管它可以被用作WebSocket的包装库，它还是提供了许多其它功能，比如广播至多个套接字，存储与不同客户有关的数据，和异步IO操作。

**Socket.IO 不等价于 WebSocket**，WebSocket只是Socket.IO实现即时通讯的其中一种技术依赖，而且Socket.IO还在实现WebSocket协议时做了一些调整。

> 优点

Socket.IO 会自动选择合适双向通信协议，仅仅需要程序员对套接字的概念有所了解。

有Python库的实现，可以在Python实现的Web应用中去实现IM后台服务。

> 缺点

Socket.io并不是一个基本的、独立的、能够回退到其它实时协议的WebSocket库，它实际上是一个依赖于其它实时传输协议的自定义实时传输协议的实现。该协议的协商部分使得支持标准WebSocket的客户端不能直接连接到Socket.io服务器，并且支持Socket.io的客户端也不能与非Socket.io框架的WebSocket或Comet服务器通信。因而，Socket.io要求客户端与服务器端均须使用该框架。

## 前端实现

