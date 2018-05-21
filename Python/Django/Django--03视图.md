# 视图

```
- 视图：即一个python函数，可以叫 视图函数，或者简称 视图，定义在 应用/views.py 文件中。

- 作用：接收并处理请求，调用M和T，响应请求（返回HttpResponse或其子类）

- 每一个请求的url地址，都对应着一个视图，由视图处理请求后，再返回html页面内容给浏览器显示。
```

## 使用过程

- 在"应用/views.py"中定义视图

```python
# 在booktest/views.py中定义视图函数index
def index(request):
    return HttpResponse("视图函数index")
```

- 配置URLconf，将视图函数和url对应起来

```python
# 项目下的urls.py
from django.conf.urls import include, url
from django.contrib import admin
import apps.booketest.urls  # 采用元组时需要

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    # 包含应用下的urls.py文件
    # 方法一：字符串，不需要导包，自动识别
    # url(r'^', include('booktest.urls')),
    # 方法二：元组，需要导包
	url(r'^', include(booktest.urls， namespace='users')),
]


# 应用下的urls.py
from django.conf.urls import url 
from booktest import views 

urlpatterns = [
    # 函数视图引用
    url(r'^$', views.index， name='index'), 
    # 类视图引用
    # url(r'^$', views.IndexView.as_view()， name='index')
]
```

## URLconf

用户通过在浏览器的地址栏中输入网址请求网站，对于Django开发的网站，由哪一个视图进行处理请求，是由url匹配找到的

### 配置

- 在test3/settings.py中通过ROOT_URLCONF指定url配置，默认已经有此配置。

```
root_utlconf = 'test3.urls'
```

- 打开test3/urls.py可以看到默认配置

```python
from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
]
```

### 语法

url()对象，被定义在django.conf.urls包中，有两种语法结构：

- 语法一：包含，一般在自定义应用中创建一个urls.py来定义url。

这种语法用于test3/urls.py中，目的是将应用的urls配置到应用内部，数据更清晰并且易于维护。

```
url(正则,include('应用.urls'))
```

- **语法二**：定义，指定URL和视图函数的对应关系。

在应用内部创建urls.py文件，指定请求地址与视图的对应关系。

```
url(正则,'视图函数名称')
```

### 命名

- 在使用include函数定义路由时，可以使用namespace参数定义路由的命名空间，如

```python
url(r'^users/', include('users.urls', namespace='users')),
```

命名空间表示，凡是users.urls中定义的路由，均属于namespace指明的users名下。

**命名空间的作用：避免不同应用中的路由使用了相同的名字发生冲突，使用命名空间区别开。**

- 在定义普通路由时，可以使用name参数指明路由的名字，如

```python
urlpatterns = [
    url(r'^index/$', views.index, name='index'),
    url(r'^say', views.say, name='say'),
]
```

### 反解析

使用reverse函数，可以根据路由名称，返回具体的路径，如：

```python
from django.core.urlresolvers import reverse  # 注意导包路径

def index(request):
    return HttpResponse("hello the world!")

def say(request):
    url = reverse('users:index')  # 返回 /users/index/
    print(url)
    return HttpResponse('say')
```

对于未指明namespace的，reverse(路由name)

对于指明namespace的，reverse(命名空间namespace:路由name)

### 匹配

```python
# url匹配流程
(1)浏览器发出请求
(2)去除域名、端口号、参数、/，剩余部分与项目中的url匹配
(3)去除在项目中url匹配成功的部分，剩下部分与应用里面的url匹配
(4)若匹配成功，则调用对应的视图函数，若失败，则返回相应信息5)

# url配置规则 （针对应用下的url配置）
正则表达式应使用 ^ 和 /$ 严格匹配请求url的开头和结尾，以便匹配用户以 / 结尾的url请求地址。
了解：django中的 APPEND_SLASH参数：默认会让浏览器在请求的url末尾添加 /，所以浏览器请求时url末尾不添加 / 也没问题

# url匹配小结：
域名、端口、参数不参与匹配
先到项目下的urls.py进行匹配，再到应用的urls.py匹配
自上而下的匹配
匹配成功的url部分会去掉，剩下的部分继续作匹配
匹配不成功提示404错误
```

## 请求Request

利用HTTP协议向服务器传参有几种途径？

- 提取URL的特定部分，如/weather/beijing/2018，可以在服务器端的路由中用正则表达式截取；
- 查询字符串（query string)，形如key1=value1&key2=value2；
- 请求体（body）中发送的数据，比如表单数据、json、xml；
- 在http报文的头（header）中。

### URL路径参数

```
把url中的需要获取的值部分，设置为正则表达式的一个组。 django在进行url匹配时，就会自动把匹配成功的内容，作为参数传递给视图函数。

位置参数： url中的正则表达式组，和视图函数中的参数一一对应，函数中的参数名可以随意指定。

关键字参数： 在位置参数的基础上，对正则表达式分组进行命名：?P<组名>
视图函数中参数名，必须和正则表达式分组名一致。
```

实例：

```python
# /show_news/新闻类别/页码/
http://127.0.0.1:8000/show_news/1/2/

# 位置参数：新闻查看/新闻类别/第几页
url(r'^show_news/(\d+)/(\d+)/$', views.show_news),
# 视图函数:views.py
def show_news(request, a, b):
	"""显示新闻界面"""
	return HttpResponse("新闻界面：%s %s" % (a, b))
	
# 关键字参数：新闻查看/新闻类别/第几页
url(r'^show_news2/(?P<category>\d+)/(?P<page_no>\d+)/$', views.show_news2),
# 视图函数:views.py
def show_news2(request, category, page_no):
	"""显示新闻界面2"""
	return HttpResponse("新闻界面：%s %s" % (category, page_no))
```

### QueryDict对象

定义在django.http.QueryDict

HttpRequest对象的属性GET、POST都是QueryDict类型的对象

与python字典不同，QueryDict类型的对象用来处理同一个键带有多个值的情况

- 方法get()：根据键获取值

  如果一个键同时拥有多个值将获取最后一个值

  如果键不存在则返回None值，可以设置默认值进行后续处理

  ```
  dict.get('键',默认值)
  可简写为
  dict['键']
  ```

- 方法getlist()：根据键获取值，值以列表返回，可以获取指定键的所有值

  如果键不存在则返回空列表[]，可以设置默认值进行后续处理

  ```
  dict.getlist('键',默认值)
  ```

### Query String

获取请求路径中的查询字符串参数（形如?k1=v1&k2=v2），可以通过request.GET属性获取，返回QueryDict对象。

```
# /qs/?a=1&b=2&a=3

def qs(request):
    a = request.GET.get('a')
    b = request.GET.get('b')
    alist = request.GET.getlist('a')
    print(a)  # 3
    print(b)  # 2
    print(alist)  # ['1', '3']
    return HttpResponse('OK')
```

**重要：查询字符串不区分请求方式，即假使客户端进行POST方式的请求，依然可以通过request.GET获取请求中的查询字符串数据。**

### 请求体

请求体数据格式不固定，可以是表单类型字符串，可以是JSON字符串，可以是XML字符串，应区别对待。

可以发送请求体数据的请求方式有**POST**、**PUT**、**PATCH**、**DELETE**。

**Django默认开启了CSRF防护**，会对上述请求方式进行CSRF防护验证，在测试时可以关闭CSRF防护机制，方法为在settings.py文件中注释掉CSRF中间件

- 表单类型

前端发送的表单类型的请求体数据，可以通过request.POST属性获取，返回QueryDict对象。

```
def get_body(request):
    a = request.POST.get('a')
    b = request.POST.get('b')
    alist = request.POST.getlist('a')
    print(a)
    print(b)
    print(alist)
    return HttpResponse('OK')
```

**重要：只要请求体的数据是表单类型，无论是哪种请求方式（POST、PUT、PATCH、DELETE），都是使用request.POST来获取请求体的表单数据。**

- 非表单

非表单类型的请求体数据，Django无法自动解析，可以通过**request.body**属性获取最原始的请求体数据，自己按照请求体格式（JSON、XML等）进行解析。**request.body返回bytes类型。**

例如要获取请求体中的如下JSON数据

```
{"a": 1, "b": 2}
```

可以进行如下方法操作：

```
import json

def get_body_json(request):
    json_str = request.body
    json_str = json_str.decode()  # python3.6 无需执行此步
    req_data = json.loads(json_str)
    print(req_data['a'])
    print(req_data['b'])
    return HttpResponse('OK')
```

### 请求头

可以通过**request.META**属性获取请求头headers中的数据，**request.META为字典类型**。

常见的请求头如：

- `CONTENT_LENGTH` – The length of the request body (as a string).
- `CONTENT_TYPE` – The MIME type of the request body.
- `HTTP_ACCEPT` – Acceptable content types for the response.
- `HTTP_ACCEPT_ENCODING` – Acceptable encodings for the response.
- `HTTP_ACCEPT_LANGUAGE` – Acceptable languages for the response.
- `HTTP_HOST` – The HTTP Host header sent by the client.
- `HTTP_REFERER` – The referring page, if any.
- `HTTP_USER_AGENT` – The client’s user-agent string.
- `QUERY_STRING` – The query string, as a single (unparsed) string.
- `REMOTE_ADDR` – The IP address of the client.
- `REMOTE_HOST` – The hostname of the client.
- `REMOTE_USER` – The user authenticated by the Web server, if any.
- `REQUEST_METHOD` – A string such as `"GET"` or `"POST"`.
- `SERVER_NAME` – The hostname of the server.
- `SERVER_PORT` – The port of the server (as a string).

具体使用如:

```
def get_headers(request):
    print(request.META['CONTENT_TYPE'])
    return HttpResponse('OK')
```

## HttpRequest对象

```python
请求一个页面时，Django会把请求数据包装成一个HttpRequest对象，然后调用对应的视图函数，把这个HttpRequest对象作为第一个参数传给视图函数。
```

| Attribute | Description                                                  |
| --------- | ------------------------------------------------------------ |
| path      | 请求页面的全路径，不包括域名端口参数。例如： "/music/bands/beatles/" |
| method    | 一个全大写的字符串，表示请求中使用的HTTP方法。常用值：‘GET’, 'POST'。以下三种会为Get请求：<br/><li>`form表单默认提交（或者method指定为get）`<li>`在浏览器中输入地址直接请求`<li>`网页中的超链接（a标签）`<br/>form表单中指定method为post，则为post请求 |
| encoding  | 一个字符串，表示提交的数据的编码方式（如果为 None 则表示使用 DEFAULT_CHARSET 的设置，默认为 'utf-8'） |
| GET       | 类似字典的QueryDict对象，包含get请求的所有参数               |
| POST      | 类似字典的QueryDict对象，包含post请求的所有参数<br><li>`服务器收到空的POST请求的情况也是有可能发生的` <li> `因此，不能使用语句if request.POST来判断是否使用HTTP POST方法`, <br>`应该使用if request.method == "POST" ` |
| COOKIES   | 一个标准的python字典，包含所有的cookies, 键和值都是字符串    |
| session   | 可读可写的类似字典的对象(`django.contrib.sessions.backends.db.SessionStore`)。django提供了session模块，默认就会开启用来保存session数据 |
| FILES     | 类似字典的对象，包含所有的上传文件<br>                       |

[官方文档：request和reponse对象](http://python.usyiyi.cn/documents/django_182/ref/request-response.html)

### path/encoding

1）打开booktest/views.py文件，代码如下：

```python
def index(request):
    str='%s,%s'%(request.path,request.encoding)
    return render(request, 'booktest/index.html', {'str':str})
```

2）在templates/booktest/下创建index.html文件，代码如下：

```html
<html>
<head>
    <title>首页</title>
</head>
<body>
1. request对象的path,encoding属性：<br/>
{{ str }}
<br/>
</body>
</html>
```

### method

1）打开booktest/views.py文件，编写视图method_show，代码如下：

```python
def method_show(request):
    return HttpResponse(request.method)
```

2）打开booktest/urls.py文件，新增配置如下：

```python
url(r'^method_show/$', views.method_show),
```

3）修改templates/booktest/下创建index.html文件，添加代码如下：

```html
<html>
<head>
    <title>首页</title>
</head>
<body>
...
...
2.request对象的method属性：<br/>
<a href='/method_show/'>get方式</a><br/>
<form method="post" action="/method_show/">
    <input type="submit" value="post方式">
</form>
<br/>
</body>
</html>
```


### GET属性

- 关于GET请求：
请求格式：

```
在请求地址结尾使用?，之后以"键=值"的格式拼接，多个键值对之间以&连接。
```

例：网址如下

```
http://www.itcast.cn/?a=10&b=20&c=python
```

其中的请求参数为：

```
a=10&b=20&c=python
```

分析

```
分析请求参数，键为'a'、'b'、'c'，值为'10'、'20'、'python'。

在Django中可以使用HttpRequest对象的GET属性获得get方方式请求的参数。

GET属性是一个QueryDict类型的对象，键和值都是字符串类型。

键是开发人员在编写代码时确定下来的。

值是根据数据生成的。
```

- 实现参考：

  ```
  1. 配置url
  # app01/urls.py文件
  urlpatterns = [
  	...
      # 获取get请求的参数
      url(r'^get/$', views.get),
  ]

  2. 定义视图函数
  def get(request):
   """获取get请求的参数"""
      a = request.GET.get('a')
      b = request.GET.getlist('b')
      text = ('a = %s <br/>b = %s' % (a, b))
      return HttpResponse(text)
      
   3. 在浏览器中测试：
  对于超连接方式的get请求，上面的视图函数同样能获取到对应的参数：
  <html>
  <head>
      <title>GET属性</title>
  </head>
  <body>
  <a href="/get/?a=1&b=2&b=3"> 获取get请求的参数 </a>
  </body>
  </html>
  ```

### POST属性

- POST请求

  ```
  - 用于向服务器提交数据，会修改服务器中的数据
  - 请求参数会通过请求体（request body）传递给服务器
  - 提交的参数没有大小限制
  - 安全性相对get较高（HTTPS）
  ```

- 案例： 

   需求：从POST属性中获取请求参数
   	html表单提交界面参考：`templates/app01/post.html` 
   ```
   <html>
   	<head>
   	    <title>POST属性</title>
   	</head>
   	<body>
   	<form method="post" action="/do_post/">
   	    用户名：<input type="text" name="username"/><br>
   	    密码： <input type="password" name="password"/><br>
   	    性别： <input type="radio" name="gender" value="0"/>男
   	    <input type="radio" name="gender" value="1"/>女<br>
   	    爱好： <input type="checkbox" name="hobby" value="胸口碎大石"/>胸口碎大石
   	    <input type="checkbox" name="hobby" value="脚踩电灯炮"/>脚踩电灯炮
   	    <input type="checkbox" name="hobby" value="口吐火"/>口吐火<br>
   	    <input type="submit" value="提交"/>
   	</form>
   	</body>
   	</html>
   ```

## HttpResponse对象

可以使用**django.http.HttpResponse**来构造响应对象。

```
HttpResponse(content=响应体, content_type=响应体数据类型, status=状态码)
```

响应头可以直接将HttpResponse对象当做字典进行响应头键值对的设置：

```
response = HttpResponse()
response['Itcast'] = 'Python'  # 自定义响应头Itcast, 值为Python
```

#### 属性

- content：表示返回的内容。
- charset：表示response采用的编码字符集，默认为utf-8。
- status_code：返回的HTTP响应状态码。
- content-type：指定返回数据的的MIME类型，默认为'text/html'。

#### 方法

- `__init__`：创建HttpResponse对象后完成返回内容的初始化。

- set_cookie：设置Cookie信息。

  ```python
  set_cookie(key, value='', max_age=None, expires=None)
  # max_age是一个整数，表示在指定秒数后过期。
  # expires是一个datetime或timedelta对象，会话将在这个指定的日期/时间过期。
  # 如果不指定过期时间，在关闭浏览器时cookie会过期
  ```

- delete_cookie(key)：删除指定的key的Cookie，如果key不存在则什么也不发生。

- write：向响应体中写数据

```
视图函数处理完逻辑后，必须返回一个HttpResponse对象或子对象作为响应：

HttpResponse对象 (render 函数)
HttpResponseRedirect对象 (redirect 函数)
JsonResponse对象
```

### HttpResponse

HttpResponse():参数为字符串

若要返回浏览器html文件，可以有以下三种：

> 直接传入html编码，难以维护，代码混乱
>
> 传入读取好的html，难以处理动态数据
>
> 调用Django模板，可处理动态数据，便于维护

- 直接传入

```
1）打开booktest/views.py文件，定义视图index2如下：
def index2(request):
    str='<h1>hello world</h1>'
    return HttpResponse(str)
    
2）打开booktest/urls.py文件，配置url。
url(r'^index2/$',views.index2),

3）运行服务器，在浏览器中打开如下网址。
http://127.0.0.1:8000/index2/
```

- 使用模板

```
1）打开booktest/views.py文件，定义视图index3如下：
# 视图函数完整写法
from django.template import RequestContext, loader
...
def index3(request):
    # 1.加载模板
    t1=loader.get_template('booktest/index3.html')
    # 2.构造上下文
    context=RequestContext(request,{'h1':'hello'})
    # 3.使用上下文渲染模板，生成字符串后返回响应对象
    return HttpResponse(t1.render(context))

2）打开booktest/urls.py文件，配置url。
    url(r'^index3/$',views.index3),
    
3）在templates/booktest/目录下创建index3.html，代码如下：
<html>
<head>
    <title>使用模板</title>
</head>
<body>
<h1>{{h1}}</h1>
</body>
</html>

4）运行服务器，在浏览器中打开如下网址。
http://127.0.0.1:8000/index3/
```

视图函数简写

```python
from django.shortcuts import render
...
def index3(request):
    # 参数1：请求对象
    # 参数2：html文件
    # 参数3：字典，表示向模板中传递的上下文数据
    return render(request, 'booktest/index3.html', {'h1': 'hello'})
```

### HttpResponseRedirect

```
HttpResponseRedirect,可以重定向到某个界面：
它可以是一个完整的URL，例如： 'http://search.yahoo.com/'
或者不包括域名的相对路径，例如:'/search/'
注意它返回HTTP 状态码 302。

redict(),重定向到某个界面
参数与HttpResponseRedirect类似
```

- 示例

```
1）在booktest/views.py文件中定义视图red1，代码如下：
from django.http import HttpResponseRedirect
...
# 定义重定义向视图，转向首页
def red1(request):
    return HttpResponseRedirect('/')
    
2）在booktest/urls.py文件中配置url。
    url(r'^red1/$', views.red1),
    
3）在地址栏中输入网址如下：
http://127.0.0.1:8000/red1/
```

- 简写

```
1）修改booktest/views.py文件中red1视图，代码如下：
from django.shortcuts import redirect
...
def red1(request):
    return redirect('/')
```

### JsonResponse

```python
JsonReponse： 给客户端请求返回json格式的数据

应用场景：网页的局部刷新(ajax技术)

作用：
帮助我们将数据转换为json字符串
设置响应头Content-Type为 application/json
```

- 示例

```python
1）在booktest/views.py文件中定义视图json1、json2，代码如下：
from django.http import JsonResponse
...
def json1(request):
    return render(request,'booktest/json1.html')
def json2(request):
    return JsonResponse({'h1':'hello','h2':'world'})
    
2）在booktest/urls.py文件中配置url。
    url(r'^json1/$', views.json1),
    url(r'^json2/$', views.json2),
    
3）创建目录static/js/，把jquery文件拷贝到这个目录下。

4）打开test3/settings.py文件，在文件最底部，配置静态文件查找路径，并且要求开启调试
DEBUG = True
...
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

5）在templates/booktest/目录下创建json1.html，代码如下：
<html>
<head>
    <title>json</title>
    <script src="/static/js/jquery-1.12.4.min.js"></script>
    <script>
        $(function () {
            $('#btnJson').click(function () {
                $.get('/json2/',function (data) {
                    ul=$('#jsonList');
                    ul.append('<li>'+data['h1']+'</li>')
                    ul.append('<li>'+data['h2']+'</li>')
                })
            });
        });
    </script>
</head>
<body>
<input type="button" id="btnJson" value="获取json数据">
<ul id="jsonList"></ul>
</body>
</html>

6）运行服务器，在浏览器中输入如下地址。
http://127.0.0.1:8000/json1/
```

实现步骤： 

```
1. 显示入口界面
   1. 配置请求的url地址 
   2. 创建视图函数
   3. 创建html界面
2. 服务器提供被调用的接口
   1. 配置被调用的url
   2. 定义被调用的视图函数： 返回JsonResponse对象
   3. 测试视图函数
3. 配置静态文件目录
1). 图片，css文件，js文件都是静态文件，案例中会使用jquery库实现ajax异步请求, 需要先把它导进来.
2). 在项目中创建static目录，在static下创建js目录，把jquery-1.12.4.min.js库复制到该目录下
3). 在项目下的setting.py中，指定静态文件所在的目录：
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
4. 在html界面中，使用ajax发请求，获取服务器的json数据并显示
```

## 错误视图

```
Django提供了一系列HttpResponse的子类，可以快速设置状态码
HttpResponseRedirect 301
HttpResponsePermanentRedirect 302
HttpResponseNotModified 304
HttpResponseBadRequest 400
HttpResponseNotFound 404
HttpResponseForbidden 403
HttpResponseNotAllowed 405
HttpResponseGone 410
HttpResponseServerError 500


Django内置了处理HTTP错误的视图（在django.views.defaults包下），主要错误及视图包括：

404错误：
找不到界面，url匹配失败后，django会调用内置的page_not_found 视图,调用404.html
500错误：
服务器内部错误，若是在执行视图函数时出现运行时错误，Django会默认会调用server_error 视图,调用500.html
403误误：
权限拒绝，permission_denied视图，调用403.html

自定义显示的界面：
在项目文件夹下的templates创建相应的html文件，可优先被调用

# 产看返回的错误日志：
查看 Exception Type 以及 Exception Value
查看 Traceback中的出错行

# 关闭调试模式
DEBUG = False
# 表示允许所有的
ALLOWED_HOSTS = ['*']
```

## 类视图

以函数的方式定义的视图称为**函数视图**，函数视图便于理解。但是遇到一个视图对应的路径提供了多种不同HTTP请求方式的支持时，便需要在一个函数中编写不同的业务逻辑，代码可读性与复用性都不佳。

```
 def register(request):
    """处理注册"""

    # 获取请求方法，判断是GET/POST请求
    if request.method == 'GET':
        # 处理GET请求，返回注册页面
        return render(request, 'register.html')
    else:
        # 处理POST请求，实现注册逻辑
        return HttpResponse('这里实现注册逻辑')
```

在Django中也可以使用类来定义一个视图，称为**类视图**。

使用类视图可以将视图对应的不同请求方式以类中的不同方法来区别定义。如下所示

```
from django.views.generic import View

class RegisterView(View):
    """类视图：处理注册"""

    def get(self, request):
        """处理GET请求，返回注册页面"""
        return render(request, 'register.html')

    def post(self, request):
        """处理POST请求，实现注册逻辑"""
        return HttpResponse('这里实现注册逻辑')
```

类视图的好处：

- **代码可读性好**
- **类视图相对于函数视图有更高的复用性**， 如果其他地方需要用到某个类视图的某个特定逻辑，直接继承该类视图即可

### 类视图使用

定义类视图需要继承自Django提供的父类**View**，可使用`from django.views.generic import View`或者`from django.views.generic.base import View` 导入，定义方式如上所示。

**配置路由时，使用类视图的as_view()方法来添加**。

```
urlpatterns = [
    # 视图函数：注册
    # url(r'^register/$', views.register, name='register'),
    # 类视图：注册
    url(r'^register/$', views.RegisterView.as_view(), name='register'),
]
```

### 类视图原理

```
    @classonlymethod
    def as_view(cls, **initkwargs):
        """
        Main entry point for a request-response process.
        """
        ...省略代码...

        def view(request, *args, **kwargs):
            self = cls(**initkwargs)
            if hasattr(self, 'get') and not hasattr(self, 'head'):
                self.head = self.get
            self.request = request
            self.args = args
            self.kwargs = kwargs
            # 调用dispatch方法，按照不同请求方式调用不同请求方法
            return self.dispatch(request, *args, **kwargs)

        ...省略代码...

        # 返回真正的函数视图
        return view


    def dispatch(self, request, *args, **kwargs):
        # Try to dispatch to the right method; if a method doesn't exist,
        # defer to the error handler. Also defer to the error handler if the
        # request method isn't on the approved list.
        if request.method.lower() in self.http_method_names:
            handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
        else:
            handler = self.http_method_not_allowed
        return handler(request, *args, **kwargs)
```

### 类视图使用装饰器

为类视图添加装饰器，可以使用三种方法。

为了理解方便，我们先来定义一个**为函数视图准备的装饰器**（在设计装饰器时基本都以函数视图作为考虑的被装饰对象），及一个要被装饰的类视图。

```
def my_decorator(func):
    def wrapper(request, *args, **kwargs):
        print('自定义装饰器被调用了')
        print('请求路径%s' % request.path)
        return func(request, *args, **kwargs)
    return wrapper

class DemoView(View):
    def get(self, request):
        print('get方法')
        return HttpResponse('ok')

    def post(self, request):
        print('post方法')
        return HttpResponse('ok')
```

- 在URL配置中装饰

```
urlpatterns = [
    url(r'^demo/$', my_decorate(DemoView.as_view()))
]
```

此种方式最简单，但因装饰行为被放置到了url配置中，单看视图的时候无法知道此视图还被添加了装饰器，不利于代码的完整性，不建议使用。

**此种方式会为类视图中的所有请求方法都加上装饰器行为**（因为是在视图入口处，分发请求方式前）。

- 在类视图中装饰

在类视图中使用为函数视图准备的装饰器时，不能直接添加装饰器，需要使用**method_decorator**将其转换为适用于类视图方法的装饰器。

```
from django.utils.decorators import method_decorator

# 为全部请求方法添加装饰器
class DemoView(View):

    @method_decorator(my_decorator)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    def get(self, request):
        print('get方法')
        return HttpResponse('ok')

    def post(self, request):
        print('post方法')
        return HttpResponse('ok')


# 为特定请求方法添加装饰器
class DemoView(View):

    @method_decorator(my_decorator)
    def get(self, request):
        print('get方法')
        return HttpResponse('ok')

    def post(self, request):
        print('post方法')
        return HttpResponse('ok')
```

**method_decorator装饰器还支持使用name参数指明被装饰的方法**

```
# 为全部请求方法添加装饰器
@method_decorator(my_decorator, name='dispatch')
class DemoView(View):
    def get(self, request):
        print('get方法')
        return HttpResponse('ok')

    def post(self, request):
        print('post方法')
        return HttpResponse('ok')


# 为特定请求方法添加装饰器
@method_decorator(my_decorator, name='get')
class DemoView(View):
    def get(self, request):
        print('get方法')
        return HttpResponse('ok')

    def post(self, request):
        print('post方法')
        return HttpResponse('ok')
```

为什么需要使用method_decorator???

为函数视图准备的装饰器，其被调用时，第一个参数用于接收request对象

```
def my_decorate(func):
    def wrapper(request, *args, **kwargs):  # 第一个参数request对象
        ...代码省略...
        return func(request, *args, **kwargs)
    return wrapper
```

而类视图中请求方法被调用时，传入的第一个参数不是request对象，而是self 视图对象本身，第二个位置参数才是request对象

```
class DemoView(View):
    def dispatch(self, request, *args, **kwargs):
        ...代码省略...

    def get(self, request):
        ...代码省略...
```

所以如果直接将用于函数视图的装饰器装饰类视图方法，会导致参数传递出现问题。

**method_decorator的作用是为函数视图装饰器补充第一个self参数，以适配类视图方法。**

如果将装饰器本身改为可以适配类视图方法的，类似如下，则无需再使用method_decorator。

```
def my_decorator(func):
    def wrapper(self, request, *args, **kwargs):  # 此处增加了self
        print('自定义装饰器被调用了')
        print('请求路径%s' % request.path)
        return func(self, request, *args, **kwargs)  # 此处增加了self
    return wrapper
```

- 构造Mixin扩展类

使用面向对象多继承的特性。

```
class MyDecoratorMixin(object):
    @classmethod
    def as_view(cls, *args, **kwargs):
        view = super().as_view(*args, **kwargs)
        view = my_decorator(view)
        return view

class DemoView(MyDecoratorMixin, View):
    def get(self, request):
        print('get方法')
        return HttpResponse('ok')

    def post(self, request):
        print('post方法')
        return HttpResponse('ok')
```

**使用Mixin扩展类，也会为类视图的所有请求方法都添加装饰行为。**