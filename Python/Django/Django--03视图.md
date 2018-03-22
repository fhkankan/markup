# 视图

```
- 视图：即一个python函数，可以叫 视图函数，或者简称 视图，定义在 应用/views.py 文件中。

- 作用：接收并处理请求，调用M和T，响应请求（返回HttpResponse或其子类）

- 每一个请求的url地址，都对应着一个视图，由视图处理请求后，再返回html页面内容给浏览器显示。
```

## URL配置及匹配

```
1、复制项目同名文件夹下的urls.py至应用下：

2、配置
在项目名下的urls中添加：
url(r'^',include('项目名.urls'))
在应用名下的urls中添加
from 应用名 import views

3、url匹配流程
(1)浏览器发出请求
(2)去除域名、端口号、参数、/，剩余部分与项目中的url匹配
(3)去除在项目中url匹配成功的部分，剩下部分与应用里面的url匹配
(4)若匹配成功，则调用对应的视图函数，若失败，则返回相应信息5)

3、url配置规则 （针对应用下的url配置）
正则表达式应使用 ^ 和 /$ 严格匹配请求url的开头和结尾，以便匹配用户以 / 结尾的url请求地址。
了解：django中的 APPEND_SLASH参数：默认会让浏览器在请求的url末尾添加 /，所以浏览器请求时url末尾不添加 / 也没问题

4、url匹配小结：
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

```
请求一个页面时，Django会把请求数据包装成一个HttpRequest对象，然后调用对应的视图函数，把这个HttpRequest对象作为第一个参数传给视图函数。
```

| Attribute | Description                              |
| --------- | ---------------------------------------- |
| path      | 请求页面的全路径，不包括域名端口参数。例如： "/music/bands/beatles/" |
| method    | 一个全大写的字符串，表示请求中使用的HTTP方法。常用值：‘GET’, 'POST'。以下三种会为Get请求：<br/><li>`form表单默认提交（或者method指定为get）`<li>`在浏览器中输入地址直接请求`<li>`网页中的超链接（a标签）`<br/>form表单中指定method为post，则为post请求 |
| encoding  | 一个字符串，表示提交的数据的编码方式（如果为 None 则表示使用 DEFAULT_CHARSET 的设置，默认为 'utf-8'） |
| GET       | 类似字典的QueryDict对象，包含get请求的所有参数            |
| POST      | 类似字典的QueryDict对象，包含post请求的所有参数<br><li>`服务器收到空的POST请求的情况也是有可能发生的` <li> `因此，不能使用语句if request.POST来判断是否使用HTTP POST方法`, <br>`应该使用if request.method == "POST" ` |
| COOKIES   | 一个标准的python字典，包含所有的cookies, 键和值都是字符串     |
| session   | 可读可写的类似字典的对象(`django.contrib.sessions.backends.db.SessionStore`)。django提供了session模块，默认就会开启用来保存session数据 |
| FILES     | 类似字典的对象，包含所有的上传文件<br>                    |

[官方文档：request和reponse对象](http://python.usyiyi.cn/documents/django_182/ref/request-response.html)

### QueryDict对象

- 定义在django.http.QueryDict
- HttpRequest对象的 **属性 GET、POST 都是QueryDict类型的对象**
- 与python字典不同，**使用QueryDict类型的对象，同一个键可以有多个值**
- get()方法：根据键获取值，如果一个键同时拥有多个值将获取最后一个值，键不存在则返回None，也可以指定默认值：

    dict.get('键',默认值)
    	可简写为
    	dict['键']

   注意：通过 dict['键'] 访问，如果键不存在会报错


### GET属性

- 关于GET请求：

  ```
  - 当需要从服务器中读取数据时使用get请求； 
  - 请求参数会跟在url末尾传递给服务器： http：//127.0.0.1:8000/login/?username=admin&password=123
  - 提交的参数有大小限制（浏览器对url长度有限制）；
  - 安全性较低
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
   ​	
   	html表单提交界面参考：templates/app01/post.html 
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

## HttpResponse对象

```
视图函数处理完逻辑后，必须返回一个HttpResponse对象或子对象作为响应：

HttpResponse对象 (render 函数)
HttpResponseRedirect对象 (redirect 函数)
JsonResponse对象
```

### HttpResponse

```
HttpResponse():参数为字符串
若要返回浏览器html文件，可以有以下三种：
直接传入html编码，难以维护，代码混乱
出入读取好的html，难以处理动态数据
调用Django模板，可处理动态数据，便于维护

在视图中调用模板分为三步：
1.加载模板
2.定义上下文
3.渲染模板

 def index(request):
     """显示index.html"""
     # 1.加载html模板文件
     template = loader.get_template('app01/index.html')
     # 2.定义上下文对象并指定模板要显示的数据
     datas = {}
     context = RequestContext(request, datas)
     # 3.模板渲染: 根据数据生成标准的html内容
     html = template.render(context)
     # 4.返回给浏览器
     return HttpResponse(html)

render( , , ),Django提供函数render对以上三步的封装
包含3个参数
第一个参数为request对象
第二个参数为模板文件路径
第三个参数为字典，表示向模板中传递的上下文数据
def index(request):
    """显示index.html"""
    # 参数1：请求对象
    # 参数2：html文件
    # 参数3：html网页中要显示的数据
    return render(request, "app01/index.html", {})
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

### JsonResponse

```
JsonReponse： 给客户端请求返回json格式的数据

应用场景：网页的局部刷新(ajax技术)
类JsonResponse继承自HttpResponse，被定义在django.http模块中
接收字典作为参数
JsonResponse对象的 content-type 为 application/json
```

#### 相关知识

##### JSON

一种轻量级的数据交换格式

```
JSON的2种结构:  
1. 对象结构  
对象结构是使用大括号“{}”括起来的，大括号内是由0个或多个用英文逗号分隔的“关键字:值”对（key:value）构成的。  

语法： 
{
    "键名1":值1,
    "键名2":值2,
	"键名n":值n
}

说明： 
对象结构是以“{”开始，到“}”结束。其中“键名”和“值”之间用英文冒号构成对，两个“键名:值”之间用英文逗号分隔。 
注意，这里的键名是字符串，但是值可以是数值、字符串、对象、数组或逻辑true和false。

2. JSON数组结构  
JSON数组结构是用中括号“[]”括起来，中括号内部由0个或多个以英文逗号“,”分隔的值列表组成。  

语法：
[
    {
        "键名1":值1,
        "键名2":值2
    },
    {
        "键名3":值3,
        "键名4":值4
    },
    ……
]

说明：
arr指的是json数组。数组结构是以“[”开始，到“]”结束，这一点跟JSON对象不同。
在JSON数组中，每一对“{}”相当于一个JSON对象。
注意，这里的键名是字符串，但是值可以是数值、字符串、对象、数组或逻辑true和false。 
```

##### AJAX

一种发送http请求与后台进行异步通讯的技术在不重载整个网页的情况下，AJAX从后台加载数据，并在网页上进行局部刷新

$.ajax方法使用：

	
	$.ajax({
	    url:'/js/data.json',
	    type:'POST', 
		dataType: json,
	    data:{name:'wang',age:25},
		async: true
	})
	.success(function(data){
	     alert(data)
	})
	.fail(function(){
		alert("出错")
	});
	
	参数说明：
	- url: 请求地址
	- type: 请求方式，默认为GET，常用的还有POST
	- dataType: 预期服务器返回的数据类型。如果不指定，jQuery 将自动根据 HTTP 包 MIME 信息来智能判断，比如 XML MIME 类型就被识别为 XML。可为：json/xml/html/script/jsonp/text
	- data： 发送给服务器的参数
	- async: 同步或者异步，默认为true，表示异步 
	- timeout: 设置请求超时时间（毫秒）,此设置将覆盖全局设置。
	- success： 请求成功之后的回调函数 
	- error： 请求失败后的回调函数  

简化的封装方法：

```
# $.get()方法: 通过 HTTP GET请求从服务器上请求数据
$.get(请求地址, 回调函数);
  
# $.post()方法： 通过 HTTP POST请求从服务器上请求数据
$.post(请求地址, 请求参数, 回调函数);
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

一、Cookie介绍

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


二、Cookie操作：

```
在Django中Cookie的读写： 

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

### Session

**一、Session介绍**

- 一些重要敏感的数据（银行卡账号，余额，验证码...），应该存储在服务器端，而不是存储在浏览器，**这种在服务器端进行状态数据保存的方案就是Session**

- **Session的使用依赖于Cookie**，如果浏览器不能保存Cookie，那么Session则失效了
- django项目有session模块，默认开启session功能，会自动存储session数据到数据库表中

- Session也是有过期时间的，如果不指定，默认两周就会过期



**二、Session的使用**

```
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
```

**三、session存储方式**

Session数据可以存储在数据库、内存、Redis等，可以通过在项目的setting.py中设置SESSION_ENGINE项，指定Session数据存储的方式。

```
- 存储在数据库中，如下设置可以写，也可以不写，这是默认存储方式。  SESSION_ENGINE='django.contrib.sessions.backends.db'

- 存储在内存中：存储在本机内存中，如果丢失则不能找回，比数据库的方式读写更快  SESSION_ENGINE='django.contrib.sessions.backends.cache'


- 混合存储：优先从本机内存中存取，如果没有则从数据库中存取。  SESSION_ENGINE='django.contrib.sessions.backends.cached_db'	

- 通过Redis存储session（后续项目中介绍）
```