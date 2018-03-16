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

# Django项目

## 虚拟环境

```
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

- 一个项目会包含有多个功能模块，比如：用户管理模块，商品管理模块，订单模块等
- 在Django中： **一个Django项目（project），包含了多个应用(app)，一个应用代表一个功能模块**，一个应用就是一个Python包

**创建项目：**

```
# 命令行
cd 安装目录
workon 虚环境名
django-admin startproject 项目名
cd 项目名
python manage.py startapp 应用名

配置：
1、添加静态文件：
static/(js/,css/,image/)
2. 添加Templates文件夹
3、修改settings.py文件
# 本地化
LANGUAGE_CODE='zh-hans'
TIME_ZONE='Asia/Shanghai'
# app
INSTALLED_APPS集合增加'应用名'
# templates
TEMPLATES中设定'DIRS': 
[os.path.join(BASE_DIR, 'templates')]


# pycharm
新建Django项目，选择虚环境的解释器，创建项目名和应用名
```

**项目目录 ** ：

```
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

```
# 命令行
# 启动django服务器，可更改主机和端口号，默认127.0.0.1:8000
python manage.py runserver [192.168.210.137:8001]
注意：增加、修改、删除python文件，服务器会自动重启，ctr+c停止服务器

# pycharm
点击工具栏或右键
```

# Django的MVT

## 模型

###模型使用

1、定义模型类

```
在应用models.py中编写模型类, 必须继承与models.Model类。

在模型类中，定义属性，生成对应的数据库表字段：
属性名 = models.字段类型(字段选项)

字段类型(初步了解，models包下的类)：
CharField--字符串
IntegerField--整形
BooleanField--布尔
DateFiled--日期
DecimalFiled--浮点
ForeignKey--外键，建立一对多关系
不需要定义主键id，会自动生成
```

2、生成迁移文件(类名，属性名)

```
python manage.py makemigrations
```

3、执行迁移，生成表结构(默认使用sqllite3)

```
python manage.py  migrate
```

确认表结构(使用sqliteman工具查看表结构)

4、通过ORM实现增删改查

```
用python交互环境进行数据库表增删改查,两种方法：
1、用pycharm中的python console
2、在终端输入命令python manage.py shell

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

## 后台管理

使用Django的管理模块，需要按照如下步骤操作

````
1. 管理界面本地化  
2. 创建管理员  
3. 注册模型类  
4. 自定义管理页面  
````

操作演示：

1）本地化 (语言和时区)	

	# 修改settings.py文件。
	
	# LANGUAGE_CODE = 'en-us'
	LANGUAGE_CODE = 'zh-hans'    # 指定语言（注意不要写错，否则无法启动服务器）
	
	# TIME_ZONE = 'UTC'
	TIME_ZONE = 'Asia/Shanghai'  # 指定时间

2）创建登录后台的管理员	

	# 需要指定： 用户名，邮箱，密码
	python3 manage.py createsuperuser
3）注册模型类

在应用下的admin.py中注册模型类：告诉djang框架，根据注册的模型类来生成对应表管理页面：

	# app01/admin.py:
	from app01.models import Department, Employee
	
	# 注册Model类
	admin.site.register(Department)
	admin.site.register(Employee)

4） 启动服务器：

	python manage.py runserver

在浏览器上输入以下地址，进入管理后台，对数据库表数据进行管理:

	http://127.0.0.1:8000/admin

5）自定义数据模型显示哪些字段信息

自定义模型管理类，作用：告诉django在生成的管理页面上显示哪些内容。


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

## 视图

作用： 处理用户请求，调用M和T，响应请求

### 视图函数

```
在 应用/views.py 下，定义视图函数，示例: 
from django.http import HttpResponse
  
必须有一个参数request
def index(request):
"""进入首页的视图函数"""
   	......
  处理完请求，返回字符串内容给浏览器显示
  return HttpResponse("Hello Python")
```

### 配置url

作用：建立url地址和视图函数的对应关系，当用户请求某个url地址时，让django能找到对应的视图函数进行处理。

```
# 在应用下创建urls.py，然后在项目下的urls.py文件中包含进来： 
urlpatterns = [
	...
    # 包含应用下的urls.py文件
    url(正则表达式, include('应用名.urls'))]
    
# 在应用下的urls.py中，进行url请求的配置： 
urlpatterns = [
	# 每一个url配置项都需要调用url函数，指定两个参数
	# 参数1: 匹配url的正则表达式
	# 参数2: 匹配成功后执行的视图函数
	url(正则表达式, 视图函数名), ]
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

## 模板

作用：用来生成html界面，返回给浏览器进行显示。

### 使用模板文件

模板文件： 即一个html文件，但该html文件中除了html，css，js等静态内容，还可以包含用来生成动态内容的模板语言。

```
1、创建模板文件
项目下创建template目录，并创建app01子目录，在里面创建index.html模板文件

2、配置模板目录
在setting文件中，将TEMPLATES中设定
'DIRS': [os.path.join(BASE_DIR, 'templates')]
提示：如果使用pycharm创建的django项目，pycharm自动就创建好了templates目录，并在setting文件中作了配置。

3、在视图中调用模板
```

调用模板完整写法：

```
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

调用模板简写

```
Django提供了一个函数render封装了以上代码,方法render包含3个参数
- 第一个参数为request对象
- 第二个参数为模板文件路径
- 第三个参数为字典，表示向模板中传递的上下文数据

代码参考：
def index(request):
 """显示index.html"""
    # 参数1：请求对象
    # 参数2：html文件
    # 参数3：html网页中要显示的数据
    return render(request, "app01/index.html", {})
```

### 生成动态内容

可以在模板中，通过django中的模板语言，生成动态的html内容：

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

```
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



