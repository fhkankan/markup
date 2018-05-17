# Django

- [维基百科](https://en.wikipedia.org/wiki/Django_web_framework)
- [官方文档](https://docs.djangoproject.com/en/1.11/)  
- [Django中文文档](http://python.usyiyi.cn/translate/django_182/index.html)


## MVC模式

- **Web开发中的MVC**：

  - M： model层，负责对数据的处理，包括对数据的增删改查等操作
  - V： view层，负责显示model层的数据
  - C： controller层，负责接收和处理请求，调用model和view

## Django的MVT模式

- **Django MVT 模式**

  - M： Model, **模型** 与MVC中的M相同，负责对数据的处理  
  - V： View, **视图**  与MVC中的C类似，负责处理用户请求，调用M和T，响应请求 
  - T： Template, **模板** 与MVC中的V类似，负责如何显示数据（产生html界面） 


处理过程： **Django框架接收了用户请求和参数后，再通过正则表达式匹配URL，转发给对应视图进行处理。视图调用M处理数据，再调用T返回界面给浏览器；**

## Django核心

- 一个对象关系映射，作为数据模型和关系性数据库就按的媒介(model)
- 一个基于正则表达式的URL分发器
- 一个用于处理HTTP请求的系统，含web模板系统
- 其他
  - 一个轻量级的、独立的web服务器，只用于开发和测试
  - 一个表单序列化及验证系统，用于将HTML表单转换成适用于数据库存储的数据
  - 一个缓存框架，并且可以从几个缓存方式中选择
  - 中间件支持，能对请求处理的各个阶段进行处理
  - 内置的分发系统允许应用程序中的组件采用预定义的信号进行相互间的通信
  - 一个序列化系统，能够生成或读取采用XML或JSON表示的Django模型实例
  - 一个用于扩展末班引擎的能力的系统

# Django项目

## 虚拟环境

```shell
# 安装python包的命令： 
sudo pip3 install 包
# 包的安装路径：
/usr/local/lib/python3.5/dist-packages

# 安装虚拟环境管理工具  	
sudo pip install virtualenv
# 安装虚拟环境管理扩展包  	 
sudo pip install virtualenvwrapper 
  
# 编辑主目录下面的.bashrc文件，添加下面两行。
export WORKON_HOME=$HOME/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh

# 使用以下命令使配置立即生效
source .bashrc

# 创建虚拟环境命令(需要连网)：
# 创建python2虚拟环境：
mkvirtualenv 虚拟环境名
# 创建python3虚拟环境：
mkvirtualenv -p python3 虚拟环境名

# 进入虚拟环境工作：
workon 虚拟环境名

# 查看机器上有哪些虚拟环境：
workon

# 退出虚拟环境：
deactivate 

# 删除虚拟环境：
rmvirtualenv 虚拟环境名

# 虚拟环境下安装包：
pip install 包名
注意：不能使用sudo pip install 包名这个命令， 会把包安装到真实的主机环境上而不是安装到虚拟环境中。
# 查看虚拟环境中安装了哪些python包：
pip list
```


问题： 不同项目依赖不同版本的包的问题

使用虚拟环境

- 可以在一台机器上创建多个相互独立的python虚拟环境,
- 在不同的python虚拟环境上，安装各自需要的不同的模块（**不需要root权限**）
- 让不同的python项目，使用不同的python虚拟环境
- python虚拟环境是相互独立的，彼此之间不会有影响，也不会影响ubuntu系统上的python真实环境


## 项目创建

Django项目结构（项目和应用）： 

- 在Django中： **一个Django项目（project），包含了多个应用(app)，一个应用代表一个功能模块**，一个应用就是一个Python包

**创建项目与应用：**

```python
# 命令行
workon 虚环境名  # 进入项目虚环境
cd 安装目录  # 进入项目放置目录
django-admin startproject 项目名  # 创建项目
cd 项目名  # 进入项目根目录
mkdir apps  # 创建apps根目录
cd apps  # 进入apps根目录
python ../manage.py startapp 应用名  # 创建app


# pycharm
新建Django项目，选择虚环境的解释器，创建项目名和应用名
```

**添加文件与配置**

```python
# 添加文件：
# 静态文件：
static/
static/js/
static/css/
static/image/
# 添加模板文件夹
templates/
# 第三方文件夹
utils/
utils/__init__.py


# 配置settings.py文件信息
# 本地化
LANGUAGE_CODE='zh-hans'
TIME_ZONE='Asia/Shanghai'
# app
import sys
sys.path.insert(1, os.path.join(BASE_DIR,'apps'))
INSTALLED_APPS={
  ......,
  # 'apps.users',可引起用django自带的认证系统出错
  'users',
}
# static
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
# templates
TEMPLATES={
'DIRS': 
[os.path.join(BASE_DIR, 'templates')]
}
```

**项目目录 ** ：

```python
# 项目目录如下：
与项目同名的目录: 包含项目的配置文件
__init__.py: 空文件，说明app01是一个python包（模块）
settings.py: 项目的全局配置文件
urls.py: 项目的url配置文件
wsgi.py: web服务器和django框架交互的接口，
manage.py: 项目的管理文件，项目运行的入口，指定配置文件路径

# 应用目录如下：
__init__.py: 空文件，表示是一个python包
models.py: 编写和数据库相关的代码
views.py: 接收请求，进行处理，和M和T进行交互，返回应答
tests.py: 编写测试代码的文件
admin.py: Django后台管理页面相关的文件
```

## 项目运行

开发调试阶段，可以使用django自带的服务器。有两种方式运行服务器

```python
# 命令行
# 启动django服务器，可更改主机和端口号，默认127.0.0.1:8000
python manage.py runserver  # 启动默认服务器 
python manage.py runserver 8080  # 指定端口号 
python manage.py runserver 192.168.210.137:8001  # 指定ip和端口号
    
    
注意：增加、修改、删除python文件，服务器会自动重启，ctr+c停止服务器


# pycharm
点击工具栏或右键
```

# Django的MVT

## 模型

###模型使用

- 创建数据库

```sql
mysql –uroot –p 
show databases;
create database db_django01 charset=utf8;
```

- 数据库配置

```python
修改setting.py中的DATABASES
	# Project01/setting.py
DATABASES = {
    'default': {
        # 默认内置sqlite3数据库
        # 'ENGINE': 'django.db.backends.sqlite3',
        # 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),

        # 配置mysql数据库
        'ENGINE': 'django.db.backends.mysql',
        'NAME': "db_django01",
        'USER': "root",
        'PASSWORD': "mysql",
        'HOST': "localhost",
        'PORT': 3306,
    }
}
```

- 驱动安装

```python
# 在安装环境中安装pymysql
pip install pymysql

# 在__init__.py文件中引入
import pymysql
pymysql.install_as_MySQLdb()
```

- 定义模型类

```
在应用models.py中编写模型类, 必须继承与models.Model类。

在模型类中，定义属性，生成对应的数据库表字段：
属性名 = models.字段类型(字段选项)

字段类型(初步了解，models包下的类)：
CharField--字符串
IntegerField--整型
BooleanField--布尔
DateFiled--日期
DecimalFiled--浮点
ForeignKey--外键，建立一对多关系
不需要定义主键id，会自动生成
```

eg:

```python
# polls/models.py
from django.db import models
from django.utils.encoding import python_2_unicode_compatible

@python_2_unicode_compatible  # 如果你需要支持Python 2
class Question(models.Model):
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')
    # 在交互环境中易于识别，在自动生成的管理界面使用对象的表示
    def __str__(self):
        return self.question_text

@python_2_unicode_compatible  # 如果你需要支持Python 2
class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)
    # 在交互环境中易于识别，在自动生成的管理界面使用对象的表示
    def __str__(self):
        return self.choice_text
```

- 迁移，生成表

```shell
# 创建迁移文件
python manage.py makemigrations
# 在数据库中创建模型所对应的表
python manage.py  migrate
```

- 通过ORM实现增删改查

```python
# 用python交互环境进行数据库表增删改查,
# 方法一：
用pycharm中的python console
# 方法二：
在终端输入命令python manage.py shell

# 对模型对象增删
模型类对象.save()	# 新增或修改
模型类对象.delete()	# 删除
```



### 模型管理器

- 每个模型类默认都有 **objects** 类属性，可以把它叫 **模型管理器**。它由django自动生成，类型为 `django.db.models.manager.Manager`
- **objects模型管理器**中提供了一些查询数据的方法： 


| objects管理器中的方法   | 返回类型 | 作用                                                         |
| ----------------------- | -------- | ------------------------------------------------------------ |
| 模型类.objects.get()    | 模型对象 | **返回一个对象，且只能有一个**: <br>如果查到多条数据，则报错：MultipleObjectsReturned <br>如果查询不到数据，则报错：DoesNotExist |
| 模型类.objects.filter() | QuerySet | 返回满足条件的对象                                           |
| 模型类.objects.all()    | QuerySet | 返回所有的对象                                               |

###关联查询

假设在一对多关系中，一对应的类叫做一类，多对应的类叫做多类： 

````
由一类对象查询多类对象：
一类对象.多类名小写_set.all()

由多类对象查询一类对象：
多类对象.关联属性
````

### 综合案例

```
$ python manage.py shell
>>> from polls.models import Question, Choice

# 确认我们的 __str__()方法 正常工作.
>>> Question.objects.all()
<QuerySet [<Question: What's up?>]>

# Django 提供了丰富的数据库查询 API 通过
# 关键字参数来驱动
>>> Question.objects.filter(id=1)
<QuerySet [<Question: What's up?>]>
>>> Question.objects.filter(question_text__startswith='What')
<QuerySet [<Question: What's up?>]>

# 获取今年发布的问题
>>> from django.utils import timezone
>>> current_year = timezone.now().year
>>> Question.objects.get(pub_date__year=current_year)
<Question: What's up?>

# 请求ID不存在,这将会引发一个异常.
>>> Question.objects.get(id=2)
Traceback (most recent call last):
    ...
DoesNotExist: Question matching query does not exist.

# 通过主键查询数据是常见的情况，因此 Django 提供了精确查找主键的快捷方式。
# (与上句合并）
# 以下内容与 Question.objects.get（id = 1）相同。
>>> Question.objects.get(pk=1)
<Question: What's up?>

# 确认我们的自定义方法正常工作.
>>> q = Question.objects.get(pk=1)
>>> q.was_published_recently()
True

# 给 Question 创建几个 Choices. 创建一个新的
# Choice 对象, 使用 INSERT 语句将选项添加到可用选项的集合并返回新的“Choice”对象。
# （合并至上句） Django 创建
# 一个集合来控制通过外键关联的“另一端”。
# （例如，一个“问题”的“选项”）
>>> q = Question.objects.get(pk=1)

# 显示任何有关的选项 ——目前还没有.
>>> q.choice_set.all()
<QuerySet []>

# 创建三个choices。
>>> q.choice_set.create(choice_text='Not much', votes=0)
<Choice: Not much>
>>> q.choice_set.create(choice_text='The sky', votes=0)
<Choice: The sky>
>>> c = q.choice_set.create(choice_text='Just hacking again', votes=0)

# Choice 对象通过 API 访问与之关联的 Question 对象.
>>> c.question
<Question: What's up?>

# 反之亦然：Question对象可以访问Choice对象。
>>> q.choice_set.all()
<QuerySet [<Choice: Not much>, <Choice: The sky>, <Choice: Just hacking again>]>
>>> q.choice_set.count()
3

# AIP 根据需要自动创建关系。
# 可以使用双下划线分隔关系。
# 它的工作机制是尽可能深的创建关系，而没有限制。
# 通过 pub_date 查找今年内创建的问题的所有选项
# (再次使用了之前创建的 'current_year' 变量).
>>> Choice.objects.filter(question__pub_date__year=current_year)
<QuerySet [<Choice: Not much>, <Choice: The sky>, <Choice: Just hacking again>]>

# 让我删除一个选项. 使用 delete() 方法.
>>> c = q.choice_set.filter(choice_text__startswith='Just hacking')
>>> c.delete()
```

## 后台管理

当你的模型完成定义，Django 就会自动生成一个专业的生产级 administrative interface - 一个可以让已认证用户进行添加、更改和删除对象的 Web 站点。 你只需简单的在 admin 站点上注册你的模型即可

### 常规操作

- 管理页面本地化

```python
# 本地化
LANGUAGE_CODE='zh-hans'
TIME_ZONE='Asia/Shanghai'
```

- 创建后台的管理员	

```python
# 需要指定： 用户名，邮箱，密码
python manage.py createsuperuser
```
- 注册模型类

在应用下的admin.py中注册模型类：告诉djang框架，根据注册的模型类来生成对应表管理页面：

	# app01/admin.py:
	from django.contrib import admin
	from app01.models import Department, Employee
	
	# 注册Model类
	admin.site.register(Department)
	admin.site.register(Employee)

### 自定义显示

自定义模型管理类，作用：告诉django在生成的管理页面上显示哪些内容。


```python
# app01/admin.py:
class DepartmentAdmin(admin.ModelAdmin):
	# 指定后台网页要显示的字段
	list_display = ["id", "name", "create_date"]

class EmployeeAdmin(admin.ModelAdmin):
    # 指定后台网页要显示的字段
    list_display = ["id", "name", "age", "sex", "comment"]
    
# 注册Model类
admin.site.register(Department, DepartmentAdmin)
admin.site.register(Employee, EmployeeAdmin)
```

### 启动登录

- 开启服务器

```
python manage.py runserver
```

- 进入管理后台

```
http://127.0.0.1:8000/admin
```

## 视图

作用： 处理用户请求，调用M和T，响应请求

### 配置url

将URLs映射作为简单的正则表达式映射到Python的回调函数（视图）。 正则表达式通过圆括号来“捕获”URLs中的值。 当一个用户请求一个页面时，Django将按照顺序去匹配每一个模式，并停在第一个匹配请求的URL上。 （如果没有匹配到， Django将调用一个特殊的404视图。） 
```python
# 项目下的urls.py：
from django.conf.urls import include, url
from django.contrib import admin
import apps.users.urls

urlpatterns = [
	...
    url(r'^admin/', include(admin.site.urls)),
    # 包含应用下的urls.py文件
    # 方法一：字符串，不需要导包，自动识别
    # url(r'users/', include('apps.users.urls')),
    # 方法二：元组，需要导包
	url(r'users/', include(apps.users.urls， namespace='users')),
]


# 应用下的urls.py
from django.conf.urls import url
from . import views

urlpatterns = [
    # 函数视图引用
    # url(r'^register$', views.register, name='register'),
    # 类视图引用，使用as_view方法，将类视图转换为函数
    url(r'^register$', views.RegisterView.as_view(), name='register'),
]
```

### url匹配

- url匹配流程

```
1)浏览器发出请求
2)去除域名、端口号、参数、/，剩余部分与项目中的url匹配
3)去除在项目中url匹配成功的部分，剩下部分与应用里面的url匹配
4)若匹配成功，则调用对应的视图函数，若失败，则返回相应信息
5)返回界面内容并显示
```


- url配置规则 （针对应用下的url配置）

这些正则表达式是第一次加载URLconf模块时被编译。 它们超级快

```
1. 正则表达式 应使用 ^ 和 $ 严格匹配请求url的开头和结尾，以便匹配唯一的字符串
2. 正则表达式 应以 / 结尾，以便匹配用户以 / 结尾的url请求地址。
了解：django中的 APPEND_SLASH参数：默认会让浏览器在请求的url末尾添加 /，所以浏览器请求时url末尾不添加 / 也没问题
```


- url匹配小结：

```
域名、端口、参数不参与匹配
先到项目下的urls.py进行匹配，再到应用的urls.py匹配
自上而下的匹配
匹配成功的url部分会去掉，剩下的部分继续作匹配
匹配不成功提示404错误
```
### 视图函数
一旦正则表达式匹配，Django会调用给定的视图，这是一个Python函数。 每个视图将得到一个request对象 —— 它包含了request 的metadata(元数据) —— 和正则表达式所捕获到的值

每个视图只负责两件事中的一件：返回一个包含请求的页面内容的 HttpResponse对象， 或抛出一个异常如Http404。

```python
# 在 应用/views.py 下，定义视图函数，
from django.http import HttpResponse
  
# 函数试图，必须有一个参数request
def index(request):
"""进入首页的视图函数"""
   	......
  # 处理完请求，返回字符串内容给浏览器显示
  return HttpResponse("Hello Python")


```

## 模板

作用：用来生成html界面，返回给浏览器进行显示。


```
1、创建模板文件
项目下创建template目录，并创建app01子目录，在里面创建index.html模板文件

2、配置模板目录
在setting文件中，将TEMPLATES中设定
'DIRS': [os.path.join(BASE_DIR, 'templates')]
提示：如果使用pycharm创建的django项目，pycharm自动就创建好了templates目录，并在setting文件中作了配置。

3、在视图中调用模板
```
### 设计模板
模板文件： 即一个html文件，该html文件中有html，css，js等静态内容，还可以包含用来生成动态内容的模板语言。

```
模板变量使用：
    {{ 模板变量名 }}

模板代码段：
    {%代码段%}

for循环：
    {% for i in list %}
    {% endfor %}
```

 实现参考：

```python
1. 在视图函数中指定数据
def index(request):
"""视图函数：处理请求，调用模型和模板，返回内容给浏览器显示"""
    # 参数1：请求对象
    # 参数2：html页面
    # 参数3：字典，html页面显示的数据
    data = {"content":"hi,django","list":list(range(1,10))}
    return render(request, "app01/index.html", data)
 
    
2. HTML界面使用模板标签显示数据
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
<h1>首页</h1>
使用模板变量：<br>
{{ content }}
<br><br>
显示列表：<br>
{{ list}}
<br><br>
for循环：
<ul>
    {% for i in list %}
        <li>{{ i }}</li>
    {% endfor %}
</ul>
</body>
</html>
```
### 调用模板

- 调用模板完整写法：

```python
1.加载模板
2.定义上下文
3.渲染模板

 # 参考代码
 # app01/views.py
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
```
- 调用模板简写

```python
# Django提供了一个函数render封装了以上代码,方法render包含3个参数
def index(request):
 """显示index.html"""
    # 参数1：请求对象request
    # 参数2：html文件路径
    # 参数3：字典，表示向模板中传递的上下文数据
    return render(request, "app01/index.html", {})
```


