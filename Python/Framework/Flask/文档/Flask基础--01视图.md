# Flask概览

**Flask官方文档：**

[中文文档](http://docs.jinkan.org/docs/flask/)

[英文文档](http://flask.pocoo.org/docs/0.11/)

## 安装

常用扩展包

```
Flask-SQLalchemy	操作数据库；
Flask-migrate		管理迁移数据库；
Flask-Mail			邮件；
Flask-WTF			表单；
Flask-script		插入脚本；
Flask-Login			认证用户状态；
Flask-RESTful		开发REST API的工具；
Flask-Bootstrap		集成前端Twitter Bootstrap框架；
Flask-Moment		本地化日期和时间；
Flask-Admin			国产管理框架
```

安装Flask

```
# 创建虚环境
mkvirtualenv flask_py2	
workon flask_py2
deactivate
rmvirtualenv flask_py2

# 指定Flask版本安装
$ pip install flask==0.10.1

# Mac系统：
$ easy_install flask==0.10.1
```

安装依赖包

```
# 安装依赖包（须在虚拟环境中）
$ pip install -r requirements.txt

# 生成依赖包（须在虚拟环境中）
$ pip freeze > requirements.txt

# 在ipython中测试安装是否成功
$ from flask import Flask
```

## 实例

新建文件hello.py

```
# 导入Flask类
from flask import Flask
# 导入配置文件
from config import Config

# 确定flask程序所在的工程目录
# __name__表示当前模块名
# flsk以模块名对应的模块所在的目录为工程目录，默认以目录中的static为静态文件目录，以tetemplates目录为模板目录
app = Flask(__name__, static_url_path="/python", static_folder="static",
template_folder="templates")

# 使用配置文件
app.config.from_object(Config)

# 装饰器的作用是将路由映射到视图函数index 
@app.route('/')
def index():
    return 'Hello World'

# Flask应用程序实例的run方法启动WEB服务器
if __name__ == '__main__':
	# 查看路由映射，默认会创建静态路由
	print app.url_map
	# run方法可以指定ip,port,debug
    app.run()  
```

# app对象

## 对象参数

```
app = Flask(参数)
# 参数
__name__: 导入路径（寻找静态目录与模板目录位置的参数）
static_url_path:默认空
static_folder: 默认‘static’
template_folder: 默认‘templates’
```

## 配置信息

```
app.config保存了flask的所有配置信息，可以当字典使用

# 设定配置参数
# 方法一：使用文件
# 创建config.cfg，在其中输入下行代码
# DEBUG = True
# 使用配置文件
app.config.from_pyfile("config.cfg")

# 方法二：使用对象
# 创建类
class Config(object):
	DEBUG = True
# 使用类
app.config.from_object(Config)

# 方式三：使用app.config字典(python)
app.config["DEBUG"] = True


# 提取配置参数
# 方法一：
app.config.get("字典的key")

# 方法二：
from flask import current_app
current_app.config.get("字典的key")
```

## 启动参数

```
# ip和端口
host：默认0.0.0.0
port:默认5000
debug:默认False
```

# 路由


## 装饰器路由的实现

```
Flask有两大核心：Werkzeug和Jinja2。Werkzeug实现路由、调试和Web服务器网关接口。Jinja2实现了模板。

Werkzeug是一个遵循WSGI协议的python函数库。其内部实现了很多Web框架底层的东西，比如request和response对象；与WSGI规范的兼容；支持Unicode；支持基本的会话管理和签名Cookie；集成URL请求路由等。

Werkzeug库的routing模块负责实现URL解析。不同的URL对应不同的视图函数，routing模块会对请求信息的URL进行解析，匹配到URL对应的视图函数，以此生成一个响应信息。

routing模块内部有Rule类（用来构造不同的URL模式的对象）、Map类（存储所有的URL规则）、MapAdapter类（负责具体URL匹配的工作）；
```
## 路由地图

```
# 查看所有路由
app.url_map
```
## 同一路由装饰多个视图函数

```
# 同一路径指向不同的视图函数
# 若请求方式相同，则前面的定义会覆盖后面的
# 若请求方式不同，则不会冲突
@app.route('/index', mehoods=["POST"])
def index1():
	pass
@app.route('/index')
def index2():
	pass
```

## 同一函数多个路由装饰器

```
@app.route('/h1')
@app.route('/h2')
def hello():
	pass
```
## 重定向redirect

```
from flask import redirect
@app.route('/')
def hello_itcast():
    return redirect('http://www.itcast.cn')
```

## url反向解析

```
# url_for()辅助函数可以使用程序URL映射中保存的信息生成URL；url_for()接收视图函数名作为参数，返回对应的URL
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/user/')
def redirect():
    return url_for('index',_external=True)
```


## 动态路由参数

```
# 路由传递的参数默认当做string处理(兼容数值)，这里指定int，尖括号中冒号后面的内容是动态的
@app.route('/user/<int:id>')
def hello_itcast(id):
    return 'hello itcast %d' %id
```
## 自定义转换器

```
# 内置了6种转换器：path(接受/)/any/str/int/float/uuid/unicode

# 自定义转换器
from flask import Flask，url_for, redirect
from werkzeug.routing import BaseConverter

# 1.以类的方式定义
class MobileConverter(BaseConverter):
    def __init__(self,url_map):
    	"""
    	url_map就是flask传递的初始化参数路由，args是正则表达式组成的元组
        """
        # 调用父类的初始化方法，将url_map传给父类
    	super(MobileConverter,self).__init__(url_map)
    	# regex用来保存正则表达式，最终被flask使用匹配提取
		self.regex = r'1[34578]\d{9}'
		
	def to_python(self, value):
		"""
		自定义重写父类，由flask调用，从路径中提取采纳数先经过这个函数的处理，函数返回的值作为视图函数的传入参数
		"""
		return '111111'
	
	def to_url(self, value):
		"""
		自定义重写父类，由flask调用，在用url_for反推路径时被调用，用来将处理后的参数添加到路径中
		"""
		return "222222"

app = Flask(__name__)

# 2.注册自定义转换器
# converters包含了flask的所有的转换器，类似字典使用方式
app.url_map.converters['mobile'] = MobileConverter

# 3.使用自定义转换器
# 根据转换器的类型名字找到转换器的类，然后实例化这个转换器对象
# 转换器对象中有一个对象属性regex，保存了用来匹配提取的正则表达式
@app.route('/send_sms/<mobile:mobile_num>')
def hello_itcast(mobile_num):
    return 'hello %s' % mobile_num
    
@app.route('/hello')
def hello():
	url = url_for(hello_itcast, mobile_num="123456")
	return redirect(url)
```

## 自定义转换器-正则URL

```
# 自定义转换器，可以限制ip访问，以及优化访问路径
from flask import Flask
from werkzeug.routing import BaseConverter

# 1.以类的方式定义
class Regex_url(BaseConverter):
    def __init__(self,url_map,*args):
    	# url_map就是flask传递的初始化参数路由，args是正则表达式组成的元组			  			
    	super(Regex_url,self).__init__(url_map)
		self.regex = args[0]

app = Flask(__name__)
# 2.注册自定义正则转换器
app.url_map.converters['re'] = Regex_url
# 3.使用自定义转换器
@app.route('/user/<re("[a-z]{3}"):id>')
def hello_itcast(id):
    return 'hello %s' %id
```

# 视图

## 前端数据传参

```
HTTp协议报文
起始行
请求头 Header
请求体 body

POST /goods/1234?a=1&b=2 HTTp/1.1
User-agent:xxx
Content-Type:xxx
Itcast: xxx
Cookie: cookie1=xxx;cookie2=xxx
\r\n
body

# 前端向后端发送参数的方式
1.从路径中使用正则传参
2.路径？传参    查询字符串 query string ?a=1&b=2  ,不限制请求方式(get,post)
3.请求头中传参
4.cookie中传参
5.请求体参数
  图片，文件，
  字符串(
  普通form格式	"c=3&d=4"
  json字符串	'{"c":3, "d":4}' 键中的引号需是双引号
  xml格式字符串 "<xml><c>3</c><d>4</d></xml>"
  )
```

## 获取请求参数

```
from flask import request
```

 Flask 中当前请求的 request 对象，request对象中保存了一次HTTP请求的一切信息。

requests常用的属性如下

| 属性    | 说明                           | 类型           |
| ------- | ------------------------------ | -------------- |
| data    | 记录请求的数据，并转换为字符串 | *              |
| form    | 记录请求中的表单数据           | MultiDict      |
| args    | 记录请求中 的查询参数          | MultiDict      |
| cookies | 记录请求中的cookie信息         | Dict           |
| headers | 记录请求中的报文头             | EnvironHeaders |
| method  | 记录请求中使用的HTTP方法       | GET/POST       |
| url     | 记录请求的URL                  | string         |
| files   | 记录请求上传的文件             | *              |

### 字符串

```
from flask import Flask, request
import json

app = Flask(__name__)

# POST /?a=***&b=***
@app.route("/", methods=["POST"])
def index():
	# 获取查询字符串的数据
	a = request.args.get("a")	# 获取同名参数的第一个值
	b = request.args.get("b")
	a_list = request.args.getlist("a")	# 获取所有同名参数
	
	# 获取请求头的数据
	content_type = request.headers.get("Content-Type")
	
	# 获取请求体数据
	# 1.普通form格式字符串
	c = request.form.get("c")
	d = request.form.get("d")
	
	# 2.json/xml格式的字符串
	# 方法一：通用
	json_str = rquest.data
	body_dict = json.loads(json_str)
	e = body_dict.get("e")
	# 方法二： Content-Type需是application/json
	request_dict = request.get_json()
	e = request_dict.get("e")
	

if __name__ = "__main__":
	app.run(debug=True)	
```

### 文件上传

```
# 前端多媒体表单
<form method="POST", enctype="multipart/form-data">
	<input type="file" name="pic">
	<input type="submit">
</form>

# 视图函数
@app.route("/upload", methods=["POST"])
def upload():
	# 通过files属性获取文件数据
	file_obj = request.files.get("pic")
	# 读取保存文件内容
	# 方法一：手动保存
	file_obj.read()	
	fie_name = file_obj.name
	with open("./file_name", "wb") as f:
		f.write(file_obj)
	# 方法二：save方法
	file_obj.save()
```

## 处理状态返回

### 自定义状态码

```
# return后面可以自主定义状态码(即使这个状态码不存在)。当客户端的请求已经处理完成，由视图函数决定返回给客户端一个状态码，告知客户端这次请求的处理结果。
@app.route('/')
def hello_itcast():
    return 'hello itcast',999
```

### 抛出异常

```
# abort()函数立即终止视图函数的执行,向前端返回一个http标准中存在的错误状态码，表示出现的错误信息。其类似于python中raise.
from flask import Flask,abort
@app.route('/')
def hello_itcast():
    abort(404)
    return 'hello itcast',999
```

### 自定义错误页面显示

```
# 通过装饰器来实现捕获异常，errorhandler()接收的参数为异常状态码
@app.errorhandler(404)
def error(e):
    return '您请求的页面不存在了，请确认后再次访问！%s'%e
```

## 响应信息

### 元组

```
@app.route("/")
def index():
	# 构造响应信息的方式
	# 方式一：元组
	# return (响应体， 状态码， 响应头)
	# return 响应体， 状态码， 响应头
	# return "index page", "403 itcast error", [("Content-Type", "application/json"), ("Itcast", "python")]
	# return "index page", 403, {"Content-Type":"application/json", "Itcast": "python"}
	# return "index page", "666 perfect", {"Content-Type":"application/json", "Itcast": "python"}
```

### make_response

```
@app.route("/")
def index():
	# 构造响应信息的方式
	# 方式二：make_response
	resp = make_response("index page 2")
	resp.status = "400 bad request"	# 响应码
	resp.headers["Itcast3"] = "python3"	# 响应头
	return resp
```
## 返回json

```
from flask import Flask,jsonify
import json

@app.route('/json')
	def resp_json():
        # my_dict = {"name": "python6", "age": 17}
        # 方法一：通用
        # json.dumps()对字典序列化
        # json_str = json.dumps(my_dict)
        # return json_str, 200, {"Content-Type": "application/json"}
        
        # 方法二：jsonify
        # 把字典数据转换为json字符串，并自动添加响应头Content-Type为application/json
        return jsonify(my_dict)
        
        # 方法三:jsonify中直接传内容
        return jsonify(name="python6", age=17)
```


## cookie

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

## session

```
# Flask中默认把session经secret_key签名后存储至cookie中，
# 用扩展Flask-session可更改为存储后端数据库

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
```

## 上下文

上下文：相当于一个容器，保存了Flask程序运行过程中的一些信息。

**区别：** 请求上下文：保存了客户端和服务器交互的数据。 应用上下文：在flask程序运行过程中，保存的一些配置信息，比如程序文件名、数据库的连接、用户信息等。

### 请求上下文

```
request和session都属于请求上下文对象。
是进程中的全局变量，在多线程的使用中，由用户请求的线程编码标记不同线程的请求内容，当做局部变量使用

request：封装了HTTP请求的内容，针对的是http请求。
user = request.args.get('user')，获取的是get请求的参数。

session：用来记录请求会话中的信息，针对的是用户信息。
session['name'] = user.id，可以记录用户信息。还可以通过session.get('name')获取用户信息。
```

### 应用上下文

```
current_app和g都属于应用上下文对象。

g:处理请求时，用于临时存储的对象，方便其他函数调用，每次请求都会重设这个变量。

urrent_app:表示当前运行程序文件的程序实例。
current_app.name		# 打印出当前应用程序实例的名字。
current_app.send_static_file(filename)	# 把文件名返回给浏览器
```

注意

```
- 当调用app = Flask(_name_)的时候，创建了程序应用对象app；
- request 在每次http请求发生时，WSGI server调用Flask.call()；然后在Flask内部创建的request对象；
- app的生命周期大于request和g，一个app存活期间，可能发生多次http请求，所以就会有多个request和g。
- 最终传入视图函数，通过return、redirect或render_template生成response对象，返回给客户端。
```

## 请求钩子hook

在客户端和服务器交互的过程中，有些准备工作或扫尾工作需要处理，比如：在请求开始时，建立数据库连接；在请求结束时，指定数据的交互格式。为了让每个视图函数避免编写重复功能的代码，Flask提供了通用设施的功能，即请求钩子。

```
# 以函数形式定义，函数名可自定义，通过装饰器的形式实现，Flask支持如下四种请求钩子：

before_first_request：在处理第一个请求前运行。
before_request：在每次请求前运行。通常情况不需要返回Response。一旦在一个before_request()中返回Response，则停止该次请求的调用链，直接将Response返回给客户端
after_request：如果没有未处理的异常抛出，在每次请求后运行。在after_request()中可以检查之前的处理函数中生成的Response,甚至可以对其修改
teardown_request：在每次请求后运行，即使有未处理的异常抛出。可以用来做异常处理

@app.before_first_request
def handle_before_first_request():
	pass
	
@app.before_request
def handle_before_request():
	pass
	
@app.after_request
def handle_after_request(response):
	return response
	
@app.teardown_request
def handle_teardown_request(response):
	return response
```

## 扩展命令行

通过使用Flask-Script扩展，我们可以在Flask服务器启动的时候，通过命令行的方式传入参数。

**安装**

```
pip install Flask-Script
```

**创建程序**

```
from flask import Flask
from flask_script import Manager

app = Flask(__name__)

manager = Manager(app)

@app.route('/')
def index():
    return '床前明月光'

if __name__ == "__main__":
	# app.run()
	manager.run()
```

**命令行启动**

```
# 来查看参数
python hello.py runserver --help

# 启动服务
python hello.py runserver -h ip地址 -p 端口号

# pycharm启动
需在Run/Debug Configurations中修改：
Script中添加启动文件manage.py
Script parameters中添加runserver
```