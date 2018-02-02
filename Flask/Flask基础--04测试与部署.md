# 测试与部署

## 蓝图

蓝图：用于实现单个应用的视图、模板、静态文件的集合。

蓝图就是模块化处理的类。

简单来说，蓝图就是一个存储操作路由映射方法的容器，主要用来实现客户端请求和URL相互关联的功能。 在Flask中，使用蓝图可以帮助我们实现模块化应用的功能。

### 运行机制

```
蓝图是保存了一组将来可以在应用对象上执行的操作。

注册路由就是一种操作,当在程序实例上调用route装饰器注册路由时，这个操作将修改对象的url_map路由映射列表。

当我们在蓝图对象上调用route装饰器注册路由时，它只是在内部的一个延迟操作记录列表defered_functions中添加了一个项。

当执行应用对象的 register_blueprint() 方法时，应用对象从蓝图对象的 defered_functions 列表中取出每一项，即调用应用对象的 add_url_rule() 方法，这将会修改程序实例的路由映射列表。
```

### 蓝图的使用

**创建蓝图对象**

```
#Blueprint必须指定两个参数，admin表示蓝图的名称，__name__表示蓝图所在模块
from flask import Blueprint

admin = Blueprint('admin',__name__)
```

**创建蓝图路由**

```
@admin.route('/')
def admin_index():
    return 'admin_index'
```

**在程序实例中注册该蓝图**

```
# 注册蓝图，第一个参数logins是蓝图对象，url_prefix参数默认值是根路由，如果指定，会在蓝图注册的路由url中添加前缀。
app.register_blueprint(admin,url_prefix='/admin')
```

**蓝图的分布**

```
# 多个文件
1.主视图文件index.py
from flask import Flask, render_template,url_for,redirect

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('login.html')

# 蓝图注册
from blue import api
app.register_blueprint(api)

if __name__ == '__main__':
	print app.url_map
	app.run()



2.副文件blue.py
# 导入蓝图包
from flaks import Blueprint

# 创建蓝图对象,第一个参数为蓝图的名称，在app_url_map中显示
api = Blueprint('api', __name__)

# 导入需要从主程序中导入的模型类、属性和方法
from index import redirect, url_for

# 导入需要使用蓝图的其他所有文件
from index import app
from register import register


# 使用蓝图
@api.route('/login')
def login():
	return redirect(url_for('index'))
	

3.其他路由文件register.py
# 导入蓝图对象引用
from blue import api
# 导入需要的模型类、属性、方法
import index import render_template

# 使用蓝图
@api.route('/register')
def register():
	return render_template('register.html')





```











## 单元测试

Web程序开发过程一般包括以下几个阶段：[需求分析，设计阶段，实现阶段，测试阶段]。其中测试阶段通过人工或自动来运行测试某个系统的功能。目的是检验其是否满足需求，并得出特定的结果，以达到弄清楚预期结果和实际结果之间的差别的最终目的

**测试分类**

```
测试从软件开发过程可以分为：单元测试、集成测试、系统测试等。在众多的测试中，与程序开发人员最密切的就是单元测试，因为单元测试是由开发人员进行的，而其他测试都由专业的测试人员来完成。
```

单元测试就是开发者编写一小段代码，检验目标代码的功能是否符合预期。通常情况下，单元测试主要面向一些功能单一的模块进行。

### 断言

在Web开发过程中，单元测试实际上就是一些“断言”（assert）代码。

断言就是判断一个函数或对象的一个方法所产生的结果是否符合你期望的那个结果。 python中assert断言是声明布尔值为真的判定，如果表达式为假会发生异常。单元测试中，一般使用assert来断言结果。

**断言方法的使用**

```
a = [1,3,5,7,9]
b = 3
asert b in a
assert b not in a, 'False'
```

断言语句类似

```
if not expression:
    raise AssertionError
```

**常用的断言方法：**

```
assertEqual     如果两个值相等，则pass
assertNotEqual  如果两个值不相等，则pass
assertTrue      判断bool值为True，则pass
assertFalse     判断bool值为False，则pass
assertIsNone    不存在，则pass
assertIsNotNone 存在，则pass
```

### 单元测试的写法

**定义测试类**

```
import unittest
class TestClass(unitest.TestCase):
    pass
```

**定义测试方法**

```
import unittest
class TestClass(unittest.TestCase):

    #该方法会首先执行，方法名为固定写法
    def setUp(self):
        pass

    #该方法会在测试代码执行完后执行，方法名为固定写法
    def tearDown(self):
        pass
```

**编写测试方法**

```
import unittest
class TestClass(unittest.TestCase):

    #该方法会首先执行，相当于做测试前的准备工作
    def setUp(self):
        pass

    #该方法会在测试代码执行完后执行，相当于做测试后的扫尾工作
    def tearDown(self):
        pass
        
    #测试代码
    def test_app_exists(self):
        pass
```

### 单元测试的执行

方法一：用IDE

用Pycharm右键执行，即可得到测试结果

方法二：用终端

1. 需在代码末尾添加如下代码，

```
# 单元测试代码末尾添加
if __name__ == '__main__':
    unittest.main()
    
# 终端运行
python 文件名.py
```

2. 不更改代码，可批量运行(推荐)

```
# 只看结果
python -m unittest 文件名(去掉后缀)
# 可看详细信息
python -m unittest -v 文件名
# 查看其他特使命令
python -吗unittest -h
```

### 发邮件测试

```
#coding=utf-8
import unittest
from Flask_day04 import app
class TestCase(unittest.TestCase):
    # 创建测试环境，在测试代码执行前执行
    def setUp(self):
        self.app = app
        # 激活测试标志
        app.config['TESTING'] = True
        self.client = self.app.test_client()

    # 在测试代码执行完成后执行
    def tearDown(self):
        pass

    # 测试代码
    def test_email(self):
        resp = self.client.get('/')
        print resp.data
        self.assertEqual(resp.data,'Sent　Succeed')
```

### 数据库测试

```
#coding=utf-8
import unittest
from author_book import *

#自定义测试类，setUp方法和tearDown方法会分别在测试前后执行。以test_开头的函数就是具体的测试代码。
class DatabaseTest(unittest.TestCase):
    # 首先被执行，建立数据链接，创建表
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:mysql@localhost/test0'
        self.app = app
        db.create_all()
	# 最后被执行，清楚对数据库的改变
    def tearDown(self):
        db.session.remove()
        db.drop_all()

    #测试代码
    def test_append_data(self):
        au = Author(name='itcast')
        bk = Book(info='python')
        db.session.add_all([au,bk])
        db.session.commit()
        author = Author.query.filter_by(name='itcast').first()
        book = Book.query.filter_by(info='python').first()
        #断言数据存在
        self.assertIsNotNone(author)
        self.assertIsNotNone(book)
```

## 部署

在生产环境中，flask自带的服务器，无法满足性能要求，我们这里采用Gunicorn做wsgi容器，来部署flask程序。

Gunicorn（绿色独角兽）是一个Python WSGI的HTTP服务器。从Ruby的独角兽（Unicorn ）项目移植。该Gunicorn服务器与各种Web框架兼容，实现非常简单，轻量级的资源消耗。Gunicorn直接用命令启动，不需要编写配置文件，相对uWSGI要容易很多

```
WSGI：全称是Web Server Gateway Interface（web服务器网关接口），它是一种规范，它是web服务器和web应用程序之间的接口。它的作用就像是桥梁，连接在web服务器和web应用框架之间。

uwsgi：是一种传输协议，用于定义传输信息的类型。

uWSGI：是实现了uwsgi协议WSGI的web服务器。
```

我们的部署方式： nginx + gunicorn + flask

```
# hello.py

from flask import Flask
# Flask程序实例名app
app = Flask(__name__)
@app.route('/')
def hello():
    return '<h1>hello world</h1>'

if __name__ == '__main__':
    app.run(debug=True)
```

###使用Gunicorn

```
web开发中，部署方式大致类似。简单来说，前端代理使用Nginx主要是为了实现分流、转发、负载均衡，以及分担服务器的压力。Nginx部署简单，内存消耗少，成本低。Nginx既可以做正向代理，也可以做反向代理。

正向代理：请求经过代理服务器从局域网发出，然后到达互联网上的服务器。

特点：服务端并不知道真正的客户端是谁。

反向代理：请求从互联网发出，先进入代理服务器，再转发给局域网内的服务器。

特点：客户端并不知道真正的服务端是谁。

区别：正向代理的对象是客户端。反向代理的对象是服务端。
```

####Nginx

**安装**

```
sudo apt-get install nginx
```

**配置**

```
# 进入配置文件nginx.conf
 cd /usr/local/nginx/
 sudo vim conf/nginx.conf

# 添加配置信息
server {
    # 监听80端口
    listen 80;
    # 本机
    server_name localhost; 
    # 默认请求的url
    location / {
        #请求转发到gunicorn服务器
        proxy_pass http://127.0.0.1:5001; 
        #设置请求头，并将头信息传递给服务器端 
        proxy_set_header Host $host; 
    }
}
```

**启动**

```
# 进入安装目录
cd /usr/local/nginx/
#启动
sudo sbin/nginx
#查看
ps aux | grep nginx
#停止
sudo sbin/nginx -s stop
```


####gunicorn

```
# 安装
pip install gunicorn

# 查看命令行选项
gunicorn -h

# 直接运行，默认启动的127.0.0.1:8000
gunicorn 运行文件名称:Flask程序实例名

# 指定进程和端口号： -w: 表示进程（worker）。 -b：表示绑定ip地址和端口号（bind）
gunicorn -w 4 -b 127.0.0.1:5001 运行文件名称:Flask程序实例名
```



