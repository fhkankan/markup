# Web服务器

## 开发阶段

```
Web应用开发可以说是目前软件开发中最重要的部分。Web开发也经历了好几个阶段：

1. 静态Web页面：由文本编辑器直接编辑并生成静态的HTML页面，如果要修改Web页面的内容，就需要再次编辑HTML源文件，早期的互联网Web页面就是静态的；

2. CGI：由于静态Web页面无法与用户交互，比如用户填写了一个注册表单，静态Web页面就无法处理。要处理用户发送的动态数据，出现了Common Gateway Interface，简称CGI，用C/C++编写。

3. ASP/JSP/PHP：由于Web应用特点是修改频繁，用C/C++这样的低级语言非常不适合Web开发，而脚本语言由于开发效率高，与HTML结合紧密，因此，迅速取代了CGI模式。ASP是微软推出的用VBScript脚本编程的Web开发技术，而JSP用Java来编写脚本，PHP本身则是开源的脚本语言。

4. MVC：为了解决直接用脚本语言嵌入HTML导致的可维护性差的问题，Web应用也引入了Model-View-Controller的模式，来简化Web开发。ASP发展为ASP.Net，JSP和PHP也有一大堆MVC框架。

目前，Web开发技术仍在快速发展中，异步开发、新的MVVM前端技术层出不穷。
```

## 常见服务器

web服务器可以解析HTTP，当微博、服务器接收到一个HTTP请求时，会根据配置的内容返回一个静态HTML页面或者调用某些代码动态生成返回结果。web服务器把动态响应产生的委托给其他一些程序，如python代码、JSP脚本、Servelets、ASP脚本等。无论它们的目的如何，这些服务器端的程序通常会产生一个HTTP响应让浏览器浏览。

常用的web服务器有

- Apache：世界上应用最多的Web服务器。由于其卓越的性能，Tomcat或JBoss等适用Apache为字节提供HTTP接口服务。
- Nginx：是一款轻量级 、高性能的HTTP和反向代理服务器。因为它的稳定性、丰富的功能集、示例配置文件和低系统资源的消耗而闻名
- IIS：微软的Web服务器产品，对微软ASP.net及其周围产品的支持
- Tomcat：开源服务器，是Java Servelet2.2 和JavaServer Page1.1技术的标准实现
- JBoss：是一个管理EJB的容器和服务器，支持EJB1.1/2.0/3的规范。但核心服务不包括支持Servlet、JSP的web服务器，一般与Tomcat或Jetty绑定使用。

主流服务器都实现了主流语言的可调用接口标准，这些标准有

- CGI：common Gateway Interface，CGI规范允许web服务器执行外部程序，并将它们的输出发送给web服务器，CGI将web服务器的一组简单的静态超媒体文档变成一个完整的新的交互式媒体
- ISAPI：Internet Server Application Program Interface，是微软提供的一套面向web服务的API接口，它能实现CGI提供的全部功能，并在此基础上进行了扩展，如提供了过滤器应用程序的接口
- WSGI：Web Server Gateway Interface，是一套专为python语言制定的网络服务器标准接口。 

## WSGI&ASGI

[参考](https://www.jianshu.com/p/65807220b44a)

- WSGI

先说一下`CGI`，（通用网关接口， Common Gateway Interface/CGI），定义客户端与Web服务器的交流方式的一个程序。例如正常情况下客户端发来一个请求，根据`HTTP`协议Web服务器将请求内容解析出来，进过计算后，再将加us安出来的内容封装好，例如服务器返回一个`HTML`页面，并且根据`HTTP`协议构建返回内容的响应格式。涉及到`TCP`连接、`HTTP`原始请求和相应格式的这些，都由一个软件来完成，这时，以上的工作需要一个程序来完成，而这个程序便是`CGI`。

那什么是`WSGI`呢？[维基](https://link.jianshu.com?t=https://zh.wikipedia.org/wiki/Web服务器网关接口)上的解释为，**Web服务器网关接口(Python Web Server Gateway Interface，WSGI)**，是为`Python`语言定义的Web服务器和Web应用程序或框架之间的一种简单而通用的接口。从语义上理解，貌似`WSGI`就是`Python`为了解决**Web服务器端与客户端**之间的通信问题而产生的，并且`WSGI`是基于现存的`CGI`标准而设计的，同样是一种程序（或者`Web`组件的接口规范？）。

[WSGI](https://link.jianshu.com?t=https://zh.wikipedia.org/wiki/Web服务器网关接口)区分为两部分：一种为“服务器”或“网关”，另一种为“应用程序”或“应用框架”。
 所谓的`WSGI`中间件同时实现了`API`的两方，即在`WSGI`服务器和`WSGI`应用之间起调解作用：从`WSGI`服务器的角度来说，中间件扮演应用程序，而从应用程序的角度来说，中间件扮演服务器。中间件具有的功能：

```
- 重写环境变量后，根据目标URL，将请求消息路由到不同的应用对象。
- 允许在一个进程中同时运行多个应用程序或应用框架
- 负载均衡和远程处理，通过在网络上转发请求和相应消息。
- 进行内容后处理，例如应用`XSLT`样式表。（以上 from 维基）
```
看了这么多，总结一下，其实可以说`WSGI`就是基于`Python`的以`CGI`为标准做一些扩展。

- ASGI

异步网关协议接口，一个介于网络协议服务和`Python`应用之间的标准接口，能够处理多种通用的协议类型，包括`HTTP`，`HTTP2`和`WebSocket`。
 然而目前的常用的`WSGI`主要是针对`HTTP`风格的请求响应模型做的设计，并且越来越多的不遵循这种模式的协议逐渐成为`Web`变成的标准之一，例如`WebSocket`。
 `ASGI`尝试保持在一个简单的应用接口的前提下，提供允许数据能够在任意的时候、被任意应用进程发送和接受的抽象。并且同样描述了一个新的，兼容`HTTP`请求响应以及`WebSocket`数据帧的序列格式。允许这些协议能通过网络或本地`socket`进行传输，以及让不同的协议被分配到不同的进程中。

- 区别

以上，`WSGI`是基于`HTTP`协议模式的，不支持`WebSocket`，而`ASGI`的诞生则是为了解决`Python`常用的`WSGI`不支持当前`Web`开发中的一些新的协议标准。同时，`ASGI`对于`WSGI`原有的模式的支持和`WebSocket`的扩展，即`ASGI`是`WSGI`的扩展。