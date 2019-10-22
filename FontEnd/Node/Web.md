

# Web 模块

##  Web 服务器

Web服务器一般指网站服务器，是指驻留于因特网上某种类型计算机的程序，Web服务器的基本功能就是提供Web信息浏览服务。它只需支持HTTP协议、HTML文档格式及URL，与客户端的网络浏览器配合。

大多数 web 服务器都支持服务端的脚本语言（php、python、ruby）等，并通过脚本语言从数据库获取数据，将结果返回给客户端浏览器。

目前最主流的三个Web服务器是Apache、Nginx、IIS。

## Web 应用架构

![Web 应用架构](https://www.runoob.com/wp-content/uploads/2015/09/web_architecture.jpg)

- **Client** - 客户端，一般指浏览器，浏览器可以通过 HTTP 协议向服务器请求数据。
- **Server** - 服务端，一般指 Web 服务器，可以接收客户端请求，并向客户端发送响应数据。
- **Business** - 业务层， 通过 Web 服务器处理应用程序，如与数据库交互，逻辑运算，调用外部程序等。
- **Data** - 数据层，一般由数据库组成。

## 创建Web服务器

Node.js 提供了 http 模块，http 模块主要用于搭建 HTTP 服务端和客户端，使用 HTTP 服务器或客户端功能必须调用 http 模块，代码如下：

```
var http = require('http');
```

以下是演示一个最基本的 HTTP 服务器架构(使用 8080 端口)，创建 server.js 文件，代码如下所示：
```javascript
var http = require('http'); 
var fs = require('fs'); 
var url = require('url');   

// 创建服务器
http.createServer( function (request, response) {
  // 解析请求，包括文件名
  var pathname = url.parse(request.url).pathname;      
  // 输出请求的文件名
  console.log("Request for " + pathname + " received.");      
  
  // 从文件系统中读取请求的文件内容
  fs.readFile(pathname.substr(1), function (err, data) {      
    	if (err) {
        	console.log(err);
        	// HTTP 状态码: 404 : NOT FOUND
        	// Content Type: text/html
        	response.writeHead(404, {'Content-Type': 'text/html'});
      }else{
        // HTTP 状态码: 200 : OK 
        // Content Type: text/html
        response.writeHead(200, {'Content-Type': 'text/html'});
        
        // 响应文件内容
        response.write(data.toString());
      }
    	//  发送响应数据
    	response.end();
  });
}).listen(8080);

// 控制台会输出以下信息
console.log('Server running at http://127.0.0.1:8080/');
```
接下来我们在该目录下创建一个 index.html 文件，代码如下：
```html
<!DOCTYPE html> 
<html> 
<head> 
<meta charset="utf-8">
<title>菜鸟教程(runoob.com)</title> 
</head>
<body>
		<h1>我的第一个标题</h1>
    <p>我的第一个段落。</p>
</body>
</html>
```
执行 server.js 文件：

```
$ node server.js
Server running at http://127.0.0.1:8080/
```
控制台输出信息如下：
```
Server running at http://127.0.0.1:8080/
Request for /index.html received.     #  客户端请求信息
```
##创建Web客户端

Node 创建 Web 客户端需要引入 http 模块，创建 client.js 文件，代码如下所示：
```javascript
var http = require('http');

// 用于请求的选项
var options = {
		host: 'localhost',
    port: '8080',
    path: '/index.html'
    };
    
// 处理响应的回调函数
var callback = function(response){
		// 不断更新数据
    var body = '';
    response.on('data', function(data) {
    		body += data;
    });
    response.on('end', function() {
    		// 数据接收完成
        console.log(body);
    });
}
// 向服务端发送请求
var req = http.request(options, callback);
req.end();
```
**新开一个终端**，执行 client.js 文件，输出结果如下：

```
$ node  client.js 
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>菜鸟教程(runoob.com)</title>
</head>
<body>
    <h1>我的第一个标题</h1>
    <p>我的第一个段落。</p>
</body>
</html>
```
执行 server.js 的控制台输出信息如下：
```
Server running at http://127.0.0.1:8080/
Request for /index.html received.   # 客户端请求信息
```

# 路由

我们要为路由提供请求的 URL 和其他需要的 GET 及 POST 参数，随后路由需要根据这些数据来执行相应的代码。

因此，我们需要查看 HTTP 请求，从中提取出请求的 URL 以及 GET/POST 参数。这一功能应当属于路由还是服务器（甚至作为一个模块自身的功能）确实值得探讨，但这里暂定其为我们的HTTP服务器的功能。

我们需要的所有数据都会包含在 request 对象中，该对象作为 onRequest() 回调函数的第一个参数传递。但是为了解析这些数据，我们需要额外的 Node.JS 模块，它们分别是 url 和 querystring 模块。

```
                   url.parse(string).query
                                           |
           url.parse(string).pathname      |
                       |                   |
                       |                   |
                     ------ -------------------
http://localhost:8888/start?foo=bar&hello=world
                                ---       -----
                                 |          |
                                 |          |
              querystring.parse(queryString)["foo"]    |
                                            |
                         querystring.parse(queryString)["hello"]
```

当然我们也可以用 querystring 模块来解析 POST 请求体中的参数，稍后会有演示。

现在我们来给 onRequest() 函数加上一些逻辑，用来找出浏览器请求的 URL 路径：

```javascript
// server.js 文件代码：
var http = require("http"); 
var url = require("url");  
function start() {  
		function onRequest(request, response) {   
    		var pathname = url.parse(request.url).pathname; 
        console.log("Request for " + pathname + " received.");  
        response.writeHead(200, {"Content-Type": "text/plain"}); 
        response.write("Hello World");    
        response.end();  
    }   
    
    http.createServer(onRequest).listen(8888);  
    console.log("Server has started."); 
}  

exports.start = start;
```

好了，我们的应用现在可以通过请求的 URL 路径来区别不同请求了--这使我们得以使用路由（还未完成）来将请求以 URL 路径为基准映射到处理程序上。

在我们所要构建的应用中，这意味着来自 /start 和 /upload 的请求可以使用不同的代码来处理。稍后我们将看到这些内容是如何整合到一起的。

现在我们可以来编写路由了，建立一个名为 **router.js** 的文件，添加以下内容：

```javascript
// router.js 文件代码：

function route(pathname) { 
		console.log("About to route a request for " + pathname); 
}  

exports.route = route;
```

如你所见，这段代码什么也没干，不过对于现在来说这是应该的。在添加更多的逻辑以前，我们先来看看如何把路由和服务器整合起来。

我们的服务器应当知道路由的存在并加以有效利用。我们当然可以通过硬编码的方式将这一依赖项绑定到服务器上，但是其它语言的编程经验告诉我们这会是一件非常痛苦的事，因此我们将使用依赖注入的方式较松散地添加路由模块。

首先，我们来扩展一下服务器的 start() 函数，以便将路由函数作为参数传递过去，**server.js** 文件代码如下

```javascript
var http = require("http"); 
var url = require("url"); 

function start(route) {  
		function onRequest(request, response) {    
				var pathname = url.parse(request.url).pathname;    		
				console.log("Request for " + pathname + " received.");     
      	route(pathname);     
      	response.writeHead(200, {"Content-Type": "text/plain"}); 
      	response.write("Hello World");    
      	response.end();  
    }   
  http.createServer(onRequest).listen(8888);  
  console.log("Server has started."); 
}  

exports.start = start;
```

同时，我们会相应扩展 index.js，使得路由函数可以被注入到服务器中：

```javascript
// index.js 文件代码：
var server = require("./server"); 
var router = require("./router");  
server.start(router.route);
```

在这里，我们传递的函数依旧什么也没做。

如果现在启动应用（node index.js，始终记得这个命令行），随后请求一个URL，你将会看到应用输出相应的信息，这表明我们的HTTP服务器已经在使用路由模块了，并会将请求的路径传递给路由：

```javascript
$ node index.js
Server has started.
```

以上输出已经去掉了比较烦人的 /favicon.ico 请求相关的部分。

# GET/POST请求

在很多场景中，我们的服务器都需要跟用户的浏览器打交道，如表单提交。

表单提交到服务器一般都使用 GET/POST 请求。

## GET

由于GET请求直接被嵌入在路径中，URL是完整的请求路径，包括了?后面的部分，因此你可以手动解析后面的内容作为GET请求的参数。

node.js 中 url 模块中的 parse 函数提供了这个功能。

```javascript
var http = require('http'); 
var url = require('url'); 
var util = require('util');  
http.createServer(function(req, res){    
		res.writeHead(200, {'Content-Type': 'text/plain; charset=utf-8'});    		res.end(util.inspect(url.parse(req.url, true))); 
}).listen(3000);
```

### 获取 URL 的参数

我们可以使用 url.parse 方法来解析 URL 中的参数，代码如下：

```javascript
var http = require('http'); 
var url = require('url'); 
var util = require('util');  
http.createServer(function(req, res){  
		// 解析 url 参数
		res.writeHead(200, {'Content-Type': 'text/plain'});         
		var params = url.parse(req.url, true).query;    
		res.write("网站名：" + params.name);    
		res.write("\n");    
		res.write("网站 URL：" + params.url);    res.end();  
}).listen(3000);
```

## POST 

POST 请求的内容全部的都在请求体中，http.ServerRequest 并没有一个属性内容为请求体，原因是等待请求体传输可能是一件耗时的工作。

比如上传文件，而很多时候我们可能并不需要理会请求体的内容，恶意的POST请求会大大消耗服务器的资源，所以 node.js 默认是不会解析请求体的，当你需要的时候，需要手动来做。

- 基本语法结构说明

```javascript
var http = require('http'); 
var querystring = require('querystring'); 
var util = require('util');  

http.createServer(function(req, res){    
  	// 定义了一个post变量，用于暂存请求体的信息    
  	var post = '';          
  	// 通过req的data事件监听函数，每当接受到请求体的数据，就累加到post变量中    
  	req.on('data', function(chunk){            
      	post += chunk;    
    });     
  	// 在end事件触发后，通过querystring.parse将post解析为真正的POST请求格式，然后向客户端返回。    
  	req.on('end', function(){
      post = querystring.parse(post);
      res.end(util.inspect(post));    
    }); 
}).listen(3000);
```

以下实例表单通过 POST 提交并输出数据：

- 实例

```javascript
var http = require('http'); 
var querystring = require('querystring');  
var postHTML =   
    '<html><head><meta charset="utf-8"><title>菜鸟教程 Node.js 实例</title></head>' +  
    '<body>' +  '<form method="post">' +  
    '网站名： <input name="name"><br>' +  
    '网站 URL： <input name="url"><br>' +  
    '<input type="submit">' +  
    '</form>' +  
    '</body></html>';
http.createServer(function (req, res) {  
  var body = "";  
  req.on('data', function (chunk) {
    body += chunk;  
  });
  req.on('end', function () {    
    // 解析参数
    body = querystring.parse(body);
    // 设置响应头部信息及编码
    res.writeHead(200, {'Content-Type': 'text/html; charset=utf8'});
    
    if(body.name && body.url) {
      // 输出提交的数据
      res.write("网站名：" + body.name);
      res.write("<br>");
      res.write("网站 URL：" + body.url);
    } else {
      // 输出表单
      res.write(postHTML);
    }
    res.end();
  });
}).listen(3000);
```

