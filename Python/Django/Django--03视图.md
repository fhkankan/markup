# 视图

```
- 视图：即一个python函数，可以叫 视图函数，或者简称 视图，定义在 应用/views.py 文件中。

- 作用：接收并处理请求，调用M和T，响应请求（返回HttpResponse或其子类）

- 每一个请求的url地址，都对应着一个视图，由视图处理请求后，再返回html页面内容给浏览器显示。
```

## 创建示例项目

- 创建项目test

```
django-admin startproject test
```

- 进入项目目录，创建应用booktest

```
cd test
python manage.py startapp booktest
```

- 在test3/settings.py中INSTALLED_APPS项安装应用

```
INSTALLED_APPS = (
	'django.contrib.admin',
	'django.contrib.auth',
	'django.contrib.contenttypes',
	'django.contrib.sessions',
	'django.contrib.messages',
	'django.contrib.staticfiles',
	'booktest',
)
```

- 在test/settings.py中DATABASES项配置使用MySQL数据库test2，数据库在第二部分已经创建。

```
DATABASES = {
    'default':{
        'ENGINE':'django.db.backends.mysql',
        'NAME':'test',
        'HOST':'localhost',
        'PORT':'3306',
        'USER':'root',
        'PASSWORD':'mysql',
    }
}
```

- 在test3/settings.py中TEMPLATES项配置模板查找路径

```
TEMPLATES=[
{
'DIRS': 
[os.path.join(BASE_DIR, 'templates')]
},
]
```

- 创建模板目录结构

```
test/templates/booktest
```

## 使用视图的过程

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

## 捕获URL中的值

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

## 错误视图

```
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

- path/encoding

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

- method

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

### QueryDict对象

- 定义在django.http.QueryDict
- HttpRequest对象的 **属性 GET、POST 都是QueryDict类型的对象**
- 与python字典不同，**使用QueryDict类型的对象，同一个键可以有多个值**
- get()方法：根据键获取值，如果一个键同时拥有多个值将获取最后一个值，键不存在则返回None，也可以指定默认值：

```
dict.get('键',默认值)
	可简写为
	dict['键']
```

注意：通过 dict['键'] 访问，如果键不存在会报错

- 方法getlist()：根据键获取值，值以列表返回，可以获取指定键的所有值，如果键不存在则返回空列表[]，可以设置默认值进行后续处理

```
dict.getlist('键',默认值)
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

#### 属性

- content：表示返回的内容。
- charset：表示response采用的编码字符集，默认为utf-8。
- status_code：返回的HTTP响应状态码。
- content-type：指定返回数据的的MIME类型，默认为'text/html'。

#### 方法

- `__init__`：创建HttpResponse对象后完成返回内容的初始化。

- set_cookie：设置Cookie信息。

  ```
  set_cookie(key, value='', max_age=None, expires=None)
  ```

- cookie是网站以键值对格式存储在浏览器中的一段纯文本信息，用于实现用户跟踪。

  - max_age是一个整数，表示在指定秒数后过期。
  - expires是一个datetime或timedelta对象，会话将在这个指定的日期/时间过期。
  - max_age与expires二选一。
  - 如果不指定过期时间，在关闭浏览器时cookie会过期。

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
类JsonResponse继承自HttpResponse，被定义在django.http模块中
接收字典作为参数
JsonResponse对象的 content-type 为 application/json
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

## 会话状态保存

- 浏览器请求服务器是无状态的： 无状态指一次用户请求时，浏览器、服务器无法知道之前这个用户做过什么，每次请求都是一次新的请求。无状态的应用层面的原因是：浏览器和服务器之间的通信都遵守HTTP协议。根本原因是：浏览器与服务器是使用Socket套接字进行通信的，服务器将请求结果返回给浏览器之后，会关闭当前的Socket连接，而且服务器也会在处理页面完毕之后销毁页面对象。


- 但是： 有时需要保存用户浏览的状态，比如： 用户是否登录过，浏览过哪些商品等

- 解决：cookie和session

### Cookie

#### 介绍

- **Cookie是由服务器生成的，存储在浏览器端的少量数据(键值对)**
- 服务器生成Cookie后，会在响应请求时发送Cookie数据给浏览器，浏览器接收到后会自动保存
- **浏览器再次请求服务器时，会自动上传该服务器生成的所有的Cookie**
- **Cookie是有过期时间的，默认关闭浏览器之后Cookie就会过期** 
- 每个域名下保存的Cookie的个数是有限制的，不同浏览器保存的个数不一样；
- 每个Cookie保存的数据大小是有限制的，不同的浏览器保存的数据大小不一样；

- Cookie是基于域名安全的： 
  - Cookie的存储是以域名的方式进行区分的； 
  - 每个网站只能读取自己生成的Cookie，而无法读取其它网站生成的Cookie； 
  - 浏览器请求某个网站时，会自动携带该网站所有的Cookie数据给服务器，但不会携带其它网站生成的Cookie数据。


#### 操作

```python
# 读取数据
request.COOKIE['键名']
或者：
request.COOKIES.get('键名')

# 保存数据
response.set_cookie('键名', count，max_age, expires)

- max_age是一个整数，表示在指定秒数后过期
- expires是一个datetime或timedelta对象，会话将在这个指定的日期/时间过期
- max_age与expires二选一
- 如果不指定过期时间，在关闭浏览器时cookie会过期
```

### Session

#### 介绍

- 一些重要敏感的数据（银行卡账号，余额，验证码...），应该存储在服务器端，而不是存储在浏览器，**这种在服务器端进行状态数据保存的方案就是Session**

- **Session的使用依赖于Cookie**，如果浏览器不能保存Cookie，那么Session则失效了
- django项目有session模块，默认开启session功能，会自动存储session数据到数据库表中

- Session也是有过期时间的，如果不指定，默认两周就会过期


#### 启动

```
在django项目中，session功能默认是开启的；要禁用session功能，则可禁用session中间件：
```

#### 存储

Session数据可以存储在数据库、内存、Redis等，可以通过在项目的setting.py中设置SESSION_ENGINE项，指定Session数据存储的方式。

```python
# 存储在数据库中，如下设置可以写，也可以不写，这是默认存储方式。
SESSION_ENGINE='django.contrib.sessions.backends.db'

# 存储在缓存中：存储在本机内存中，如果丢失则不能找回，比数据库的方式读写更快。
SESSION_ENGINE='django.contrib.sessions.backends.cache'

# 混合存储：优先从本机内存中存取，如果没有则从数据库中存取。
SESSION_ENGINE='django.contrib.sessions.backends.cached_db'

# 注意：如果存储在数据库中，需要在项INSTALLED_APPS中安装Session应用。
INSTALLED_APPS = (
	'django.contrib.sessions',
)
```

#### 操作

```python
# 保存session数据（键值对）
request.session['键']=值

#- 读取session数据
request.session.get('键',默认值)

# 清除session数据（清空值）
request.session.clear()

# 清除session数据(在存储中删除session的整条数据)
request.session.flush()

# 删除会话中的指定键及值，在存储中只删除某个键及对应的值。
del request.session['键']

# 设置会话的超时时间，如果没有指定过期时间则两个星期后过期
request.session.set_expiry(value)
  - 如果value是一个整数，会话将在value秒没有活动后过期。
  - 如果value为0，那么用户会话的Cookie将在用户的浏览器关闭时过期。
  - 如果value为None，那么会话永不过期。
```

