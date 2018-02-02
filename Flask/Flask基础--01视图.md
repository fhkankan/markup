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

# 确定flask程序所在的目录
# 可传入__name__,__main__，字符串(非模块名)
app = Flask(__name__)

# 使用配置文件
app.config.from_object(Config)

# 装饰器的作用是将路由映射到视图函数index
# 
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

# 视图

## 动态路由参数

```
# 路由传递的参数默认当做string处理(兼容数值)，这里指定int，尖括号中冒号后面的内容是动态的
@app.route('/user/<int:id>')
def hello_itcast(id):
    return 'hello itcast %d' %id
```
## 自定义转换器-正则URL

```
# 内置了6中转换器：path/any/str/int/float/uuid/unicode
# 自定义转换器，可以限制ip访问，以及优化访问路径
from flask import Flask
from werkzeug.routing import BaseConverter

class Regex_url(BaseConverter):
    def __init__(self,url_map,*args):
    	# url_map就是路由
    	# args是正则表达式组成的元组   			  						super(Regex_url,self).__init__(url_map)
        self.regex = args[0]

app = Flask(__name__)
app.url_map.converters['re'] = Regex_url

@app.route('/user/<re("[a-z]{3}"):id>')
def hello_itcast(id):
    return 'hello %s' %id
```

## 自定义状态码

```
# return后面可以自主定义状态码(即使这个状态码不存在)。当客户端的请求已经处理完成，由视图函数决定返回给客户端一个状态码，告知客户端这次请求的处理结果。
@app.route('/')
def hello_itcast():
    return 'hello itcast',999
```

## 抛出异常

```
# abort()函数立即终止视图函数的执行,向前端返回一个http标准中存在的错误状态码，表示出现的错误信息。其类似于python中raise.
from flask import Flask,abort
@app.route('/')
def hello_itcast():
    abort(404)
    return 'hello itcast',999
```

## 自定义错误页面显示

```
# 通过装饰器来实现捕获异常，errorhandler()接收的参数为异常状态码
@app.errorhandler(404)
def error(e):
    return '您请求的页面不存在了，请确认后再次访问！%s'%e
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
from flask import redirect, url_for
@app.route('/')
def hello_itcast():
    return redirect('http://www.itcast.cn')

# url_for()辅助函数可以使用程序URL映射中保存的信息生成URL；url_for()接收视图函数名作为参数，返回对应的URL
@app.route('/url')
def url_info():
    return redirect(url_for('hello_itcast')) 
```

## cookie

```
# 设置cookie
from flask import Flask,make_response
@app.route('/cookie')
def set_cookie():
    resp = make_response('this is to set cookie')
    resp.set_cookie('username', 'itcast')
    return resp
    
# 获取cookie
# request是请求上细纹对象，cookies是其对象的属性
from flask import Flask,request
@app.route('/request')
def resp_cookie():
    resp = request.cookies.get('username')
    return resp
```

## 返回json

```
from flask import Flask,jsonify

@app.route('/json')
	def resp_json():
        my_dict = {'name': 'python6', "age": 17}
        return jsonify(my_dict)
```



## 上下文

上下文：相当于一个容器，保存了Flask程序运行过程中的一些信息。

**区别：** 请求上下文：保存了客户端和服务器交互的数据。 应用上下文：在flask程序运行过程中，保存的一些配置信息，比如程序文件名、数据库的连接、用户信息等。

请求上下文

```
request和session都属于请求上下文对象。

request：封装了HTTP请求的内容，针对的是http请求。
user = request.args.get('user')，获取的是get请求的参数。

session：用来记录请求会话中的信息，针对的是用户信息。
session['name'] = user.id，可以记录用户信息。还可以通过session.get('name')获取用户信息。
```

应用上下文

```
current_app和g都属于应用上下文对象。
g:处理请求时，用于临时存储的对象，每次请求都会重设这个变量。

urrent_app:表示当前运行程序文件的程序实例。
current_app.name		# 打印出当前应用程序实例的名字。
current_app.send_static_file(filename)	# 把文件名返回给浏览器
- 当调用app = Flask(_name_)的时候，创建了程序应用对象app；
- request 在每次http请求发生时，WSGI server调用Flask.call()；然后在Flask内部创建的request对象；
- app的生命周期大于request和g，一个app存活期间，可能发生多次http请求，所以就会有多个request和g。
- 最终传入视图函数，通过return、redirect或render_template生成response对象，返回给客户端。
```

## 请求钩子

在客户端和服务器交互的过程中，有些准备工作或扫尾工作需要处理，比如：在请求开始时，建立数据库连接；在请求结束时，指定数据的交互格式。为了让每个视图函数避免编写重复功能的代码，Flask提供了通用设施的功能，即请求钩子。

```
# 请求钩子是通过装饰器的形式实现，Flask支持如下四种请求钩子：
before_first_request：在处理第一个请求前运行。
before_request：在每次请求前运行。通常情况不需要返回Response。一旦在一个before_request()中返回Response，则停止该次请求的调用链，直接将Response返回给客户端
after_request：如果没有未处理的异常抛出，在每次请求后运行。在after_request()中可以检查之前的处理函数中生成的Response,甚至可以对其修改
teardown_request：在每次请求后运行，即使有未处理的异常抛出。可以用来做异常处理
```

## 装饰器路由的实现

```
Flask有两大核心：Werkzeug和Jinja2。Werkzeug实现路由、调试和Web服务器网关接口。Jinja2实现了模板。

Werkzeug是一个遵循WSGI协议的python函数库。其内部实现了很多Web框架底层的东西，比如request和response对象；与WSGI规范的兼容；支持Unicode；支持基本的会话管理和签名Cookie；集成URL请求路由等。

Werkzeug库的routing模块负责实现URL解析。不同的URL对应不同的视图函数，routing模块会对请求信息的URL进行解析，匹配到URL对应的视图函数，以此生成一个响应信息。

routing模块内部有Rule类（用来构造不同的URL模式的对象）、Map类（存储所有的URL规则）、MapAdapter类（负责具体URL匹配的工作）；
```

## 扩展命令行

通过使用Flask-Script扩展，我们可以在Flask服务器启动的时候，通过命令行的方式传入参数。

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
    manager.run()
```

**命令行启动**

```
# 来查看参数
python hello.py runserver --help

# 启动服务
python hello.py runserver -h ip地址 -p 端口号
```

