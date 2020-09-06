[TOC]

# Web开发

## WSGI接口

Web应用的本质就是：

```
1. 浏览器发送一个HTTP请求；
2. 服务器收到请求，生成一个HTML文档；
3. 服务器把HTML文档作为HTTP响应的Body发送给浏览器；
4. 浏览器收到HTTP响应，从HTTP Body取出HTML文档并显示。
```

所以，最简单的Web应用就是先把HTML用文件保存好，用一个现成的HTTP服务器软件，接收用户请求，从文件中读取HTML，返回。Apache、Nginx、Lighttpd等这些常见的静态服务器就是干这件事情的。

如果要动态生成HTML，就需要把上述步骤自己来实现。不过，接受HTTP请求、解析HTTP请求、发送HTTP响应都是苦力活，如果我们自己来写这些底层代码，还没开始写动态HTML呢，就得花个把月去读HTTP规范。

正确的做法是底层代码由专门的服务器软件实现，我们用Python专注于生成HTML文档。因为我们不希望接触到TCP连接、HTTP原始请求和响应格式，所以，需要一个统一的接口，让我们专心用Python编写Web业务。

这个接口就是WSGI：Web Server Gateway Interface。

WSGI允许开发者将选择web框架和web服务器分开。可以混合匹配web服务器和web框架，选择一个适合的配对,web服务器必须具备WSGI接口，所有的现代Python Web框架都已具备WSGI接口，它让你不对代码作修改就能使服务器和特点的web框架协同工作。

其他语言也有类似接口：java有Servlet API，Ruby 有 Rack。

### 接口定义

WSGI接口定义非常简单，它只要求Web开发者实现一个函数，就可以响应HTTP请求。

```python
def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return 'Hello World!'
```

上面的`application()`函数就是符合WSGI标准的一个HTTP处理函数，`application()`函数必须由WSGI服务器来调用。它接收两个参数：

- environ：一个包含所有HTTP请求信息的dict对象；
- start_response：一个发送HTTP响应的函数。

整个`application()`函数本身没有涉及到任何解析HTTP的部分，也就是说，把底层web服务器解析部分和应用程序逻辑部分进行了分离，这样开发者就可以专心做一个领域了，应用领域的开发者不需要编写服务器底层代码，只负责在更高层次上考虑如何响应请求就可以了。

有了WSGI，我们关心的就是如何从`environ`这个`dict`对象拿到HTTP请求信息，然后构造HTML，通过`start_response()`发送Header，最后返回Body。

### 快速实现

Python内置了一个WSGI服务器，这个模块叫wsgiref，它是用纯Python编写的WSGI服务器的参考实现。所谓“参考实现”是指该实现完全符合WSGI标准，但是不考虑任何运行效率，仅供开发和测试使用。

**hello.py**

实现Web应用程序的WSGI处理函数

```python
# hello.py

def application(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/html')])
    return [b'<h1>Hello, web!</h1>']
```

**server.py**

负责启动WSGI服务器，加载`application()`函数

```python
# server.py
# 从wsgiref模块导入:
from wsgiref.simple_server import make_server
# 导入我们自己编写的application函数:
from hello import application

# 创建一个服务器，IP地址为空，端口是8000，处理函数是application:
httpd = make_server('', 8000, application)
print('Serving HTTP on port 8000...')
# 开始监听HTTP请求:
httpd.serve_forever()
```

**运行**

```python
# 确保两个文件在同一目录下，命令行输入
python server.py

# 浏览器中输入
localhost:8000/
```

## Web框架

其实一个Web App，就是写一个WSGI的处理函数，针对每个HTTP请求进行响应。

但是如何处理HTTP请求不是问题，问题是如何处理100个不同的URL。

每一个URL可以对应GET和POST请求，当然还有PUT、DELETE等请求，但是我们通常只考虑最常见的GET和POST请求。

一个最简单的想法是从`environ`变量里取出HTTP请求的信息，然后逐个判断：

```
def application(environ, start_response):
    method = environ['REQUEST_METHOD']
    path = environ['PATH_INFO']
    if method=='GET' and path=='/':
        return handle_home(environ, start_response)
    if method=='POST' and path='/signin':
        return handle_signin(environ, start_response)
    ...
```

只是这么写下去代码是肯定没法维护了。

代码这么写没法维护的原因是因为WSGI提供的接口虽然比HTTP接口高级了不少，但和Web App的处理逻辑比，还是比较低级，我们需要在WSGI接口之上能进一步抽象，让我们专注于用一个函数处理一个URL，至于URL到函数的映射，就交给Web框架来做。

有了Web框架，我们在编写Web应用时，注意力就从WSGI处理函数转移到URL+对应的处理函数，这样，编写Web App就更加简单了。

在编写URL处理函数时，除了配置URL外，从HTTP请求拿到用户数据也是非常重要的。Web框架都提供了自己的API来实现这些功能。Flask通过`request.form['name']`来获取表单的内容。

除了Flask，常见的Python Web框架还有：

- [Django](https://www.djangoproject.com/)：全能型Web框架；
- [web.py](http://webpy.org/)：一个小巧的Web框架；
- [Bottle](http://bottlepy.org/)：和Flask类似的Web框架；
- [Tornado](http://www.tornadoweb.org/)：Facebook的开源异步Web框架。
- Sanic：支持异步的web服务框架
- Aiohttp: 支持异步的http客户端/服务框架

## web客户端

python中实现http请求的客户端常用的有

- [requests](https://github.com/psf/requests)：支持同步请求第三方http请求的模块，见spider/requests文档
- [aiohttp](https://github.com/aio-libs/aiohttp)：支持异步请求的第三方http请求模块，见aiohttp文档
- [httpx](https://github.com/encode/httpx)：支持同步和异步请求的第三方http请求模块，见ThirdPartLibrary/httpx文档

## 模板

由于在Python代码里拼HTML页面字符串是不现实的，所以，模板技术出现了。

使用模板，我们需要预先准备一个HTML文档，这个HTML文档不是普通的HTML，而是嵌入了一些变量和指令，然后，根据我们传入的数据，替换后，得到最终的HTML，发送给用户：

除了Jinja2，常见的模板还有：

- [Mako](http://www.makotemplates.org/)：用`<% ... %>`和`${xxx}`的一个模板；
- [Cheetah](http://www.cheetahtemplate.org/)：也是用`<% ... %>`和`${xxx}`的一个模板；
- [Django](https://www.djangoproject.com/)：Django是一站式框架，内置一个用`{% ... %}`和`{{ xxx }}`的模板。

## 相关知识
###XSS注入式攻击

**前端**

```
# xss.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Template</title>
</head>
<body>
    <form method="post">
    	<textarea name="text"></textarea>
    </form>
</body>
</html>
```

**后端**

```
# xss.py
@app.route("/xss", methods=["GET","POST"])
def xss():
	if request.method == "GET"
		t = ""
	else:
		t = requst.form.get("text")
	return render_template("xss.html", text = t)
```

**用户输入**

```
<script>alter("hhhh")</script>
```

用户输入了脚本进行xss注入式攻击，所以模板带有自动转义功能，转换用户输入的代码指令

```
&lt;script&gt;alert(&#34;hhhh&#34;);&lt;/script&gt
```

从而没有运行此段脚本，直接在页面显示

```
<script>alter("hhhh")</script>
```
### CSRF跨站请求

#### 原理

CSRF跨站点请求伪造(Cross—Site Request Forgery)，跟XSS攻击一样，存在巨大的危害性，你可以这样来理解：
​      攻击者盗用了你的身份，以你的名义发送恶意请求，对服务器来说这个请求是完全合法的，但是却完成了攻击者所期望的一个操作，比如以你的名义发送邮件、发消息，盗取你的账号，添加系统管理员，甚至于购买商品、虚拟货币转账等。 

```
如下：其中Web A为存在CSRF漏洞的网站，Web B为攻击者构建的恶意网站，User C为Web A网站的合法用户。

CSRF攻击攻击原理及过程如下：
1. 用户C打开浏览器，访问受信任网站A，输入用户名和密码请求登录网站A；
2. 在用户信息通过验证后，网站A产生Cookie信息并返回给浏览器，此时用户登录网站A成功，可以正常发送请求到网站A；
3. 用户C未退出网站A之前，在同一浏览器中，打开一个TAB页访问网站B；
4. 网站B接收到用户请求后，返回一些攻击性代码，并发出一个请求要求访问第三方站点A；
5. 浏览器在接收到这些攻击性代码后，根据网站B的请求，在用户C不知情的情况下携带Cookie信息，向网站A发出请求。网站A并不知道该请求其实是由B发起的，所以会根据用户C的Cookie信息以C的权限处理该请求，导致来自网站B的恶意代码被执行。 
```

#### 防御

目前防御 CSRF 攻击主要有三种策略：验证 HTTP Referer 字段；在请求地址中添加 token 并验证；在 HTTP 头中自定义属性并验证。

- 验证 HTTP Referer 字段

 根据 HTTP 协议，在 HTTP 头中有一个字段叫 Referer，它记录了该 HTTP 请求的来源地址。在通常情况下，访问一个安全受限页面的请求来自于同一个网站，比如需要访问 http://bank.example/withdraw?account=bob&amount=1000000&for=Mallory，用户必须先登陆 bank.example，然后通过点击页面上的按钮来触发转账事件。这时，该转帐请求的 Referer 值就会是转账按钮所在的页面的 URL，通常是以 bank.example 域名开头的地址。而如果黑客要对银行网站实施 CSRF 攻击，他只能在他自己的网站构造请求，当用户通过黑客的网站发送请求到银行时，该请求的 Referer 是指向黑客自己的网站。因此，要防御 CSRF 攻击，银行网站只需要对于每一个转账请求验证其 Referer 值，如果是以 bank.example 开头的域名，则说明该请求是来自银行网站自己的请求，是合法的。如果 Referer 是其他网站的话，则有可能是黑客的 CSRF 攻击，拒绝该请求。

​        这种方法的显而易见的好处就是简单易行，网站的普通开发人员不需要操心 CSRF 的漏洞，只需要在最后给所有安全敏感的请求统一增加一个拦截器来检查 Referer 的值就可以。特别是对于当前现有的系统，不需要改变当前系统的任何已有代码和逻辑，没有风险，非常便捷。

​        然而，这种方法并非万无一失。Referer 的值是由浏览器提供的，虽然 HTTP 协议上有明确的要求，但是每个浏览器对于 Referer 的具体实现可能有差别，并不能保证浏览器自身没有安全漏洞。使用验证 Referer 值的方法，就是把安全性都依赖于第三方（即浏览器）来保障，从理论上来讲，这样并不安全。事实上，对于某些浏览器，比如 IE6 或 FF2，目前已经有一些方法可以篡改 Referer 值。如果 bank.example 网站支持 IE6 浏览器，黑客完全可以把用户浏览器的 Referer 值设为以 bank.example 域名开头的地址，这样就可以通过验证，从而进行 CSRF 攻击。

​	即便是使用最新的浏览器，黑客无法篡改 Referer 值，这种方法仍然有问题。因为 Referer 值会记录下用户的访问来源，有些用户认为这样会侵犯到他们自己的隐私权，特别是有些组织担心 Referer 值会把组织内网中的某些信息泄露到外网中。因此，用户自己可以设置浏览器使其在发送请求时不再提供 Referer。当他们正常访问银行网站时，网站会因为请求没有 Referer 值而认为是 CSRF 攻击，拒绝合法用户的访问。

- 在请求地址中添加 token 并验证

​         CSRF 攻击之所以能够成功，是因为黑客可以完全伪造用户的请求，该请求中所有的用户验证信息都是存在于 cookie 中，因此黑客可以在不知道这些验证信息的情况下直接利用用户自己的 cookie 来通过安全验证。要抵御 CSRF，关键在于在请求中放入黑客所不能伪造的信息，并且该信息不存在于 cookie 之中。可以在 HTTP 请求中以参数的形式加入一个随机产生的 token，并在服务器端建立一个拦截器来验证这个 token，如果请求中没有 token 或者 token 内容不正确，则认为可能是 CSRF 攻击而拒绝该请求。

​        这种方法要比检查 Referer 要安全一些，token 可以在用户登陆后产生并放于 session 之中，然后在每次请求时把 token 从 session 中拿出，与请求中的 token 进行比对，但这种方法的难点在于如何把 token 以参数的形式加入请求。对于 GET 请求，token 将附在请求地址之后，这样 URL 就变成 `http://url?csrftoken=tokenvalue`。 而对于 POST 请求来说，要在 form 的最后加上 `<input type=”hidden” name=”csrftoken” value=”tokenvalue”/>`，这样就把 token 以参数的形式加入请求了。但是，在一个网站中，可以接受请求的地方非常多，要对于每一个请求都加上 token 是很麻烦的，并且很容易漏掉，通常使用的方法就是在每次页面加载时，使用 javascript 遍历整个 dom 树，对于 dom 中所有的 a 和 form 标签后加入 token。这样可以解决大部分的请求，但是对于在页面加载之后动态生成的 html 代码，这种方法就没有作用，还需要程序员在编码时手动添加 token。

​         该方法还有一个缺点是难以保证 token 本身的安全。特别是在一些论坛之类支持用户自己发表内容的网站，黑客可以在上面发布自己个人网站的地址。由于系统也会在这个地址后面加上 token，黑客可以在自己的网站上得到这个 token，并马上就可以发动 CSRF 攻击。为了避免这一点，系统可以在添加 token 的时候增加一个判断，如果这个链接是链到自己本站的，就在后面添加 token，如果是通向外网则不加。不过，即使这个 csrftoken 不以参数的形式附加在请求之中，黑客的网站也同样可以通过 Referer 来得到这个 token 值以发动 CSRF 攻击。这也是一些用户喜欢手动关闭浏览器 Referer 功能的原因。

- 在 HTTP 头中自定义属性并验证

​        这种方法也是使用 token 并进行验证，和上一种方法不同的是，这里并不是把 token 以参数的形式置于 HTTP 请求之中，而是把它放到 HTTP 头中自定义的属性里。通过 XMLHttpRequest 这个类，可以一次性给所有该类请求加上 csrftoken 这个 HTTP 头属性，并把 token 值放入其中。这样解决了上种方法在请求中加入 token 的不便，同时，通过 XMLHttpRequest 请求的地址不会被记录到浏览器的地址栏，也不用担心 token 会透过 Referer 泄露到其他网站中去。

​        然而这种方法的局限性非常大。XMLHttpRequest 请求通常用于 Ajax 方法中对于页面局部的异步刷新，并非所有的请求都适合用这个类来发起，而且通过该类请求得到的页面不能被浏览器所记录下，从而进行前进，后退，刷新，收藏等操作，给用户带来不便。另外，对于没有进行 CSRF 防护的遗留系统来说，要采用这种方法来进行防护，要把所有请求都改为 XMLHttpRequest 请求，这样几乎是要重写整个网站，这代价无疑是不能接受的。

#### 程序操作

- Django

1. 重要信息如金额、积分等的获取，采用POST请求
2. 开启CSRF中间件（默认就是开启的）

```
# Django
# 项目下的setting.py
MIDDLEWARE_CLASSES = (
	...
	# 开启csrf中间件（默认是开启的）
    'django.middleware.csrf.CsrfViewMiddleware',
	...
)

# 表单post提交数据时加上 {% csrf_token %} 标签
{% csrf_token %} 
```

- Flask

```
from flask import Flask, 
# 导入wtf中的csrf防护部分
from flask_wtf import CSRFProtect
from flask_wtf import csrf

app = Flask(__name__)

# 补充csrf防护
# flask_wtf表单中的组成部分,应用了请求钩子
# 防护机制：对于包含了请求体的请求(POST,PUT,DELETE),从请求的cookie中读取一个csrf_token的值,
# 从请求体中读取一个csrf_token的值，进行比较，若相同允许访问，若不同返回403信息
CSRFProtect(app)



# Flask用扩展Flask-WTF
from flask import make_response
# 导入csrf生成随机字符串
from flask_wtf import csrf
...
def get_html_file(file_name):
	...
	# send_static_file(静态目录中的文件名)， 函数会自动去静态目录中找文件，返回包含文件内容的响应信息
    # make_response()构造响应对象，接收响应体数据，可以是字符串，文件内容
    resp = make_response(current_app.send_static_file(file_name))

    # 生成csrf_token随机字符串的值
    csrf_token = csrf.generate_csrf()

    # 设置csrf用到的cookie
    # 参数1：键名；参数2：随机字符串的值
    resp.set_cookie("csrf_token", csrf_token)

    return resp
```

### Cookie/Session

- 浏览器请求服务器是无状态的： 无状态指一次用户请求时，浏览器、服务器无法知道之前这个用户做过什么，每次请求都是一次新的请求。无状态的应用层面的原因是：浏览器和服务器之间的通信都遵守HTTP协议。根本原因是：浏览器与服务器是使用Socket套接字进行通信的，服务器将请求结果返回给浏览器之后，会关闭当前的Socket连接，而且服务器也会在处理页面完毕之后销毁页面对象。


- 但是： 有时需要保存用户浏览的状态，比如： 用户是否登录过，浏览过哪些商品等
- 解决：cookie和session

#### Cookie

访问流程
```
客户端请求服务器，如果服务器需要记录该用户状态，就使用response向客户端浏览器颁发一个Cookie。客户端浏览器会把Cookie保存起来。

当浏览器再请求该网站时，浏览器把请求的网址连同该Cookie一同提交给服务器。服务器检查该Cookie，以此来辨认用户状态。服务器还可以根据需要修改Cookie的内容。
```
特性
```
- Cookie是由服务器生成的，存储在浏览器端的少量数据(键值对)
- 服务器生成Cookie后，会在响应请求时发送Cookie数据给浏览器，浏览器接收到后会自动保存
- 浏览器再次请求服务器时，会自动上传该服务器生成的所有的Cookie
- Cookie是有过期时间的，默认关闭浏览器之后Cookie就会过期
- 每个域名下保存的Cookie的个数是有限制的，不同浏览器保存的个数不一样；
- 每个Cookie保存的数据大小是有限制的，不同的浏览器保存的数据大小不一样；
- Cookie是基于域名安全的： 
  - Cookie的存储是以域名的方式进行区分的； 
  - 每个网站只能读取自己生成的Cookie，而无法读取其它网站生成的Cookie； 
  - 浏览器请求某个网站时，会自动携带该网站所有的Cookie数据给服务器，但不会携带其它网站生成的Cookie数据。
```
- 程序操作

**Django**

```
读取数据
request.COOKIE['键名']
或者：
request.COOKIES.get('键名')

保存数据
response.set_cookie('键名', count，max_age, expires)

- max_age是一个整数，表示在指定秒数后过期
- expires是一个datetime或timedelta对象，会话将在这个指定的日期/时间过期
- max_age与expires二选一
- 如果不指定过期时间，在关闭浏览器时cookie会过期
```

**Flask**

```
from flask import Flask,make_response
@app.route('/set_cookie')
def set_cookie():
	# 响应对象
    resp = make_response('set cookie ok')
    # 设置cookie
    # 参数1：key, 参数2：value，临时cookie,浏览器关闭即失效
    resp.set_cookie('username', 'itcast')
    # max_age指明有效期，单位秒
    resp.set_cookie('username2', 'itcast', max_age=3600)
    return resp
    
# set_cookie实质是在响应报头中添加Set-Cookie项   
@app.route('/set_cookie2')
def set_cookie2():
    # 设置cookie
    return "set cookie 2", 200, [("Set-coojie", "user_name3=itcast; path=/")]
    
    
# request是请求上下文对象，cookies是其对象的属性
from flask import Flask,request
@app.route('/get_cookie')
def resp_cookie():
	# 获取cookie
    resp = request.cookies.get('username')
    return resp
    
    
from flask import Flask,make_response
@app.route('/del_cookie')
def set_cookie():
	# 响应对象
    resp = make_response('del cookie ok')
    # 删除cookie,有效期为0
    resp.delete_cookie('username2')
    return resp
```

#### Session

访问流程
```
当浏览器 第一次发送请求时，服务器自动生成了一个Session和一个Session ID用来唯一标识这个Session，并将其通过响应发送到浏览器。

当浏览器第二次发送请求，会将前一次服务器响应中的Session ID放在请求中一并发送到服务器上，服务器从请求中提取出Session ID，并和保存的所有Session ID进行对比，找到这个用户对应的Session。
```
特性
```
Session 是存放在服务器端的，类似于Session结构来存放用户数据。

一般情况下，服务器会在一定时间内（默认30分钟）保存这个 Session，过了时间限制，就会销毁这个Session。在销毁之前，程序员可以将用户的一些数据以Key和Value的形式暂时存放在这个 Session中。当然，也有使用数据库将这个Session序列化后保存起来的，这样的好处是没了时间的限制，坏处是随着时间的增加，这个数据 库会急速膨胀，特别是访问量增加的时候。一般还是采取前一种方式，以减轻服务器压力。

- 一些重要敏感的数据（银行卡账号，余额，验证码...），应该存储在服务器端，而不是存储在浏览器，**这种在服务器端进行状态数据保存的方案就是Session**
- **Session的使用依赖于Cookie**，如果浏览器不能保存Cookie，那么Session则失效了（也可保存至url）
- Session也是有过期时间的，如果不指定，默认两周就会过期
```
客户端实现
```
一般浏览器提供了两种方式来保存，还有一种是程序员使用html隐藏域的方式自定义实现：

- 使用Cookie来保存，这是最常见的方法，本文“记住我的登录状态”功能的实现正式基于这种方式的。服务器通过设置Cookie的方式将Session ID发送到浏览器。如果我们不设置这个过期时间，那么这个Cookie将不存放在硬盘上，当浏览器关闭的时候，Cookie就消失了，这个Session ID就丢失了。如果我们设置这个时间为若干天之后，那么这个Cookie会保存在客户端硬盘中，即使浏览器关闭，这个值仍然存在，下次访问相应网站时，同 样会发送到服务器上。
- 使用URL附加信息的方式，也就是像我们经常看到JSP网站会有`aaa.jsp?JSESSIONID=*`一样的。这种方式和第一种方式里面不设置Cookie过期时间是一样的。
- 第三种方式是在页面表单里面增加隐藏域，这种方式实际上和第二种方式一样，只不过前者通过GET方式发送数据，后者使用POST方式发送数据。但是明显后者比较麻烦。
```
- 程序操作

```python
# 传递流程
前端用户登录 ---> 后端服务器生成sesson数据保存至数据库  --->发送sesson_id=1 ---> 前端接收session_id保存到cookie中 ---> 前端后续访问携带session_id=1

# 保存内容
前端cookie
session_id:1

后端服务器
{
    session_id_1:{"user_id":123,"user_name":"python"},
    session_id_2:{"user_id":124,"user_name":"itcast"},    
}

# 保存位置
数据库
redis
文件
程序-内存(配置ngix应对跨机访问)

# 没有cookie也能实现session
方案：在url路径中保存
eg：
后端向前端返回redirect("/?session_id=1")

# 框架更改
django默认存储到后端数据库中，前端cookie保存session_id
Flask默认存储到cookie中,故需要签名认证，或使用扩展来更改保存至后端数据库中
```

Django

```python
1. 开启session功能
在django项目中，session功能默认是开启的；要禁用session功能，则可禁用session中间件：

注意事项： session默认是保存到数据库表中的，通过一张session表来保存，所以注意保存数据的session表是否已经存在了。（项目刚创建需要迁移才会有数据库表）

2. session对象操作（request.session字典）
# 保存session数据（键值对）
request.session['键']=值

#- 读取session数据
request.session.get('键',默认值)

# 清除session数据（清空值）
request.session.clear()

# 删除会话中的指定键及值，在存储中只删除某个键及对应的值。
del request.session['键']

# 设置会话的超时时间，如果没有指定过期时间则两个星期后过期
request.session.set_expiry(value)
  - 如果value是一个整数，会话将在value秒没有活动后过期。
  - 如果value为0，那么用户会话的Cookie将在用户的浏览器关闭时过期。
  - 如果value为None，那么会话永不过期。
   
3. session存储方式
- 存储在数据库中，如下设置可以写，也可以不写，这是默认存储方式。  SESSION_ENGINE='django.contrib.sessions.backends.db'

- 存储在内存中：存储在本机内存中，如果丢失则不能找回，比数据库的方式读写更快  SESSION_ENGINE='django.contrib.sessions.backends.cache'

- 混合存储：优先从本机内存中存取，如果没有则从数据库中存取。  SESSION_ENGINE='django.contrib.sessions.backends.cached_db'	

- 通过Redis存储session（后续项目中介绍）

```

Flask

```python
1.开启：
# Flask中session属于请求上下文对象

2.session对象操作
from flask import Flask, session

# flask中使用session需要设置secret_key参数，签名验证防止别人修改
app.config["SECRET_KEY"] = "asdsd@#￥uge7t7w6g@！e76r5"

@app.route('/login')
def login():
	# 设置session
	session["user_id"] = 123
	session["user_name"] = "python"
	return "login_ok"
	
@app.route('/')
def index():
	# 获取session
	name = session.get("user_name")
	return "hello!"
	
3.session存储
# Flask中默认把session经secret_key签名后存储至cookie中，
# 用扩展Flask-session可更改为存储后端数据库


# config.py
# 导入redis缓存数据库
import redis

# 配置信息(类或者文件，# 类方便继承)
class Config(object):
    """配置信息"""
 	SECRECT_KEY = "aadifsdfyugdfgytegetyfteft*@7247"

	# redis
	REDIS_HOST = "127.0.0.1"
	REDIS_PORT = 6379
	REDIS_DB = 0
	
	# 使用redis的数据库实例
	SESSION_REDIS = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=1)

    # flask_session配置信息
    # 指明session数据保存位置
    SESSION_TYPE = 'redis'
    # 指明对cookie中保存的session—id进行加密防护
    SESSION_USE_SIGNER = True
    # session数据有效期,单位秒
    PERMANENT_SESSION_LIFETIME = 3 * 24 * 60 * 60

# __init__.py
import flask import Flask
# 导入session
from flask_session import Session
# 导入配置文件
from config import config_map
# 导入redis缓存数据库
import redis

app = Flask(__name__)

# 创建redis连接实例
redis_store = redis.StrictRedis(
    host=Config.REDIS_HOST, port=Config.REDIS_PORT, db=Config.REDIS_DB)

# session需要存储在后端
# flask_session扩展 pip install flask_session
# 对扩展初始化
Session(app)   
```

#### 比较

```
http是无状态的协议,客户每次读取web页面时,服务器都打开新的会话,而且服务器也不会自动维护客户的上下文信息,那么要怎么才能实现网上商店中的购物车呢：

默认使用cookie来实现,系统会创造一个名为JSESSIONID的输出cookie，我 们叫做session cookie,以区分persistent cookie(我们通常所说的cookie)

session就是一种保存上下文信息的机制,它是针对每一个用户的,变量的值保存在服务器端,通过SessionID来区分不同的客 户,session是以cookie或URL重写 为基础的。

注意session cookie是存储于浏览器内存中的，并不是写到硬盘上的，这也就是我们刚才看到的JSESSIONID，我们通常情是看不到JSESSIONID的，但 是当我们把浏览器的cookie禁止后，web服务器会采用URL重写的方式传递Sessionid,我们就可以在地址栏看到 sessionid=KWJHUG6JJM65HS2K6之类的字符串。

明白了原理，我们就可以很容易的分辨出persistent cookie和session cookie的区别了，网上那些关于两者安全性的讨论也就一目了然了，session cookie针对某一次会话而言，会话结束session cookie也就随着消失了,而persistent cookie只是存在于客户端硬盘上的一段文本(通常是加密的)，而且可能会遭到cookie欺骗以及针对cookie的跨站脚本攻击,自然不如 session cookie安全了。

通常session cookie是不能跨窗口使用的,当你新开了一个浏览器窗口进入相同页面时,系统会赋予你一个新的sessionid,这样我们信息共享的目的就达不到了,此时我们可以先把sessionid保存在persistent cookie中,然后在新窗口中读出来,就可以得到上一个窗口SessionID了,这样通过session cookie和persistent cookie的结合我们就实现了跨窗口的session tracking()会话跟踪)
```

|               | cookie                                                      | session                            |
| ------------- | ----------------------------------------------------------- | ---------------------------------- |
| 存储位置      | 存储在客户端                                                | 存储在服务器端                     |
| 类型/实现方式 | 有生命周期，无生命周期                                      | 依赖于cookie，url重写              |
| 是否共享      | 父路径不能访问子路径的cookie,子路径可访问父路径设置的cookie | 同一个session的窗口共享一个session |
| 典型应用      | 3个月不用再登录；购物车                                     | 用户登录；购物车                   |
| 安全          | 不可靠                                                      | 可靠                               |
| 大小          | 客户端限制3K                                                |                                    |

