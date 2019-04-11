
# 常用技术

## 静态文件

**一、静态文件的使用**

- **静态文件**：网页中使用的css，js，图片


- **静态文件的使用**： 
  1. 创建静态目录static/(css,image,js)，导入静态文件

  2. 在setting.py中配置静态目录

```python
# 通过此url来引用静态文件，可以隐藏服务器的文件的实际保存目录
STATIC_URL = '/abc/'

# 指定静态文件所在的物理目录
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
```

3. 通过setting.py中的STATIC_URL的值来引用静态文件，好处：可以隐藏服务器的文件的实际保存目录。

```
<img src="/abc/image/mm.jpg" />
<script src="/abc/js/jquery-1.12.4.min.js"></script>
```

**二、使用static标签动态引用**

上述写法是硬编码，存在维护问题，可以使用static标签动态引用

	<!DOCTYPE html>
	<html lang="en">
	
	{% load staticfiles %}
	
	<head>
	...
	</head>
	<body>
	
	动态引用：<br/>
	<img src="{% static 'image/mm.jpg' %}"/>
	
	</body>
	</html>

## 权限认证

[参考](https://www.cnblogs.com/wangwei916797941/p/7398934.html)

在Django的世界中，在权限管理中有内置的Authentication系统。用来管理帐户，组，和许可。还有基于cookie的用户session。
Django内置的权限系统包括以下三个部分：

```
1. 用户（Users）
2. 许可（Permissions）：用来定义一个用户（user）是否能够做某项任务（task）
3. 组（Groups）：一种可以批量分配许可到多个用户的通用方式
```
使用组件
```
1. 将'django.contrib.auth'和'django.contrib.contenttypes'放到settings.py中的INSTALLED_APPS中。（使用contenttypes的原因是auth中的Permission模型依赖于contenttypes）
2. 执行manage.py syncdb
3. 装豪就可以使用了
```
执行manage.py shell来启动脚本，对其中的一些API进行学习和使用。
### USer模型
User模型对应于一个用户，一个帐户，位于'django.contrib.auth.models'模块中。
User对象有两个多对多的属性分别是：groups和user_permissions：
```
>>>from django.contrib.auth.models import User
>>>es = User.objects.create_user('esperyong','esperyong@gmail.com','123456')
>>>es.groups
<django.db.models.fields.related.ManyRelatedManager at 0x10d0642d0>
>>>es.user_permissions
<django.db.models.fields.related.ManyRelatedManager at 0x10d014c50>
```
使用
```
# 直接将一个列表赋值给该属性：
es.groups = [group_list]
es.user_permissions = [permission_list]
# 使用add方法将对象加入：
es.groups.add(group, group, ...)
es.user_permissions.add(permission, permission, ...)
# 使用remove方法将对象删除：
es.groups.remove(group, group, ...)
es.user_permissions.remove(permission, permission, ...)
# 使用clear方法将所有对象删除：
es.groups.clear()
es.user_permissions.clear()
```
User对象属性
```
username:字符串类型。必填。30个字符以内。
first_name:字符串类型。可选。30个字符以内。
last_name:字符串类型。可选。30个字符以内。
email:可选。
password:明文密码的hash或者是某种元数据。该属性不应该直接赋值明文密码，而应该通过set_password()方法进行赋值，在后面有详细说明TODO。
is_staff:Boolean类型。用这个来判断是否用户可以登录进入admin site。
is_active:Boolean类型。用来判断该用户是否是可用激活状态。在删除一个帐户的时候，可以选择将这个属性置为False，而不是真正删除。这样如果应用有外键引用到这个用户，外键就不会被破坏。
is_superuser:Boolean类型。该属性用来表示该用户拥有所有的许可，而无需明确的赋予给他。
last_login:datetime类型。最近一次登陆时间。
date_joined：datetime类型。创建时间。
```

## 会话状态保存

- 浏览器请求服务器是无状态的： 无状态指一次用户请求时，浏览器、服务器无法知道之前这个用户做过什么，每次请求都是一次新的请求。无状态的应用层面的原因是：浏览器和服务器之间的通信都遵守HTTP协议。根本原因是：浏览器与服务器是使用Socket套接字进行通信的，服务器将请求结果返回给浏览器之后，会关闭当前的Socket连接，而且服务器也会在处理页面完毕之后销毁页面对象。

- 但是： 有时需要保存用户浏览的状态，比如： 用户是否登录过，浏览过哪些商品等
- 解决：cookie和session

### Cookie

- 介绍

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

- 操作

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

# 删除数据
response.delete_cookie(key)
- 删除指定的key的Cookie，如果key不存在则什么也不发生。
```

### Session

- 介绍

```
- 一些重要敏感的数据（银行卡账号，余额，验证码...），应该存储在服务器端，而不是存储在浏览器，这种在服务器端进行状态数据保存的方案就是Session
- Session的使用依赖于Cookie，如果浏览器不能保存Cookie，那么Session则失效了
- django项目有session模块，默认开启session功能，会自动存储session数据到数据库表中
- Session也是有过期时间的，如果不指定，默认两周就会过期
```

- 启动

```
在django项目中，session功能默认是开启的；要禁用session功能，则可禁用session中间件
```

- 存储

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

eg：存储的redis中

```python
# 安装
pip install django-redis

# 配置更改
# Cache
# http://django-redis-chs.readthedocs.io/zh_CN/latest/#cache-backend

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://10.211.55.5:6379/1",
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    }
}

# Session
# http://django-redis-chs.readthedocs.io/zh_CN/latest/#session-backend

SESSION_ENGINE = "django.contrib.sessions.backends.cache"
SESSION_CACHE_ALIAS = "default"
```

- 操作

```python
# 写session数据（键值对）
request.session['键']=值

# 读取session数据
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

## 中间件

MIDDLEWARE: 中间件

一、案例： 禁止指定ip地址的访问

使用request对象的META属性，可以获取用户访问的ip地址：

	request.META.get('REMOTE_ADDR')

二、中间件

中间件： django框架预留的接口，可以控制请求和响应的过程。Django在中间件中预置了6个方法，这些方法会在不同的阶段执行，对输入或输出进行干预。

- 初始化：无需任何参数，服务器响应第一个请求的时候调用一次：

```
def init():
    pass
```

- 处理请求前(url匹配前)调用： 返回None或HttpResponse对象

```
def process_request(self, request):
    pass
```

- url匹配后视图函数处理前调用： 返回None或HttpResponse对象

```
def process_view(self, request, view_func, view_args, view_kwargs):
    pass
```

- 视图函数出异常时调用：**返回一个HttpResponse对象**

```
def process_exception(self, request, exception):
    return response
```

- 视图函数处理后，模板响应处理前调用： 返回实现了render方法的响应对象

```
def process_template_response(self, request, response):
    pass
```


视图函数返回TemplateReponse时才会调用，返回HttpResponse对象不会调用

- 视图函数处理后，返回内容给浏览器前调用：**返回HttpResponse对象**

```
def process_response(self, request, response):
    return response
```

### 中间件的使用

1. 定义中间件类：

  在app01应用下创建模块：midlleware.py， 在里面创建中间件类如下：
```python
class MyMiddleware(object):
    def __init__(self):
        print('--init--')

    def process_request(self, request):
        print('--process_request--')

    def process_view(self, request, view_func, view_args, view_kwargs):
        print('--process_view--')

    def process_response(self, request, response):
        print('--process_response--')
        return respons
```

2. views.py文件的进入首页视图函数，打印日志:
```
def index(request):
    """进入首页"""
    print('=====index视图函数====')
    return HttpResponse(request, 'app01/index.html')
```

3. 在setting.py中注册中间件类：

```
# setting.py
MIDDLEWARE_CLASSES = (
    ...
    'django.middleware.security.SecurityMiddleware',
    # 注册自定义中间件
    'app01.middleware.MyMiddleware',
)
```

4. 访问首页，会输出如下结果： 

```
--init
--process_request
--process_view
--index--
--process_response
```

### 禁用ip功能

1. 在MyMiddleware的process_view方法中，新增代码如下：
```
class MyMiddleware(object):
   ...
	exclude_ips = ['127.0.0.1']
    def process_view(self, request, view_func, view_args, view_kwargs):
        print('--process_view--')

		# 禁用ip，以下代码也可以添加到process_request方法
        ip = request.META.get('REMOTE_ADDR')
        if ip in exclude_ips:
            return HttpResponse('禁止访问')
```

 注意：process_view返回了HttpResponse对象之后，视图函数就不会再执行了。

### 异常处理

异常处理： 视图函数执行出错之后，会调用中间件的process_exception方法，可以在该方法中执行异常操作	
1. 在index视图函数中，添加执行出错代码：


```
def index(request):
    """进入首页"""
    print('=====index====')

	# 添加出错代码
    aa = None
    print('aa='+ aa)
    
	return render(request, 'app01/index.html')
```

2. 在前面编写的MyMiddleware中： 添加处理异常的中间件方法，并注释前面的拦截ip的拦截： 
```python
# middleware.py
class MyMiddleware(object):
	...

    def process_view(self, request, view_func, view_args, view_kwargs):
        print('-------process_view')
        # # 禁止ip访问
        # ip = request.META.get('REMOTE_ADDR')
        # if ip in exclude_ips:
        #     return HttpResponse('禁止访问')

    def process_exception(self, request, exception):
        print('-----process_exception')
```

3. 访问首页，查看服务器，发现：处理异常的中间件方法`process_exception`执行了
4. 处理出错： 在process_exception方法中返回HttpResponse对象就可以了： 

```
# middleware.py
class MyMiddleware(object):
	...
    def process_exception(self, request, exception):
        print('-----process_exception')
		return HttpResponse('运行出错了：%s' % exception)
```

### 多个中间件的调用流程

- 视图函数之前执行的`process_request`和`process_view`方法：先注册的中间件会先执行
- 视图函数之后执行的`process_exception`和`process_response`方法：后注册的中间件先执行

代码测试： 

1. 再写一个中间件：
```python
class MyMiddleware2(object):
    def __init__(self):
        print('-------init2')

    def process_request(self, request):
        print('-------process_request2')

    def process_view(self, request, view_func, view_args, view_kwargs):
        print('-------process_view2')

    def process_response(self, request, response):
        print('-------process_response2')
        return response

    def process_exception(self, request, exception):
        print('-----process_exception2')
```

2. 在setting.py中注册中间件类：

```python
# setting.py
MIDDLEWARE_CLASSES = (
    ...
    'django.middleware.security.SecurityMiddleware',
    # 注册自定义中间件
    'app01.middleware.MyMiddleware',
    'app01.middleware.MyMiddleware2',
)
```

小结：某个中间件处理异常的方法返回HttpResponse对象后，其它中间件的处理异常的方法就不会执行了。

## 表单使用

Django提供对表单处理的支持，可以简化并自动化大部分的表单处理工作。

### 定义表单类

表单系统的核心部分是Django 的Form类。 Django 的数据库模型描述一个对象的逻辑结构、行为以及展现给我们的方式，与此类似，Form类描述一个表单并决定它如何工作和展现。

假如我们想在网页中创建一个表单，用来获取用户想保存的图书信息，可能类似的html 表单如下：

```
<form action="" method="post">
    <input type="text" name="title">
    <input type="date" name="pub_date">
    <input type="submit">
</form>
```

我们可以据此来创建一个Form类来描述这个表单。

新建一个**forms.py**文件，编写Form类。

```
from django import forms

class BookForm(forms.Form):
    title = forms.CharField(label="书名", required=True, max_length=50)
    pub_date = forms.DateField(label='出版日期', required=True)
```

注：[表单字段类型参考资料连接](https://yiyibooks.cn/xx/Django_1.11.6/ref/forms/fields.html)

### 视图中使用表单类

```
from django.shortcuts import render
from django.views.generic import View
from django.http import HttpResponse

from .forms import BookForm

class BookView(View):
    def get(self, request):
        form = BookForm()
        return render(request, 'book.html', {'form': form})

    def post(self, request):
        form = BookForm(request.POST)
        if form.is_valid():  # 验证表单数据
            print(form.cleaned_data)  # 获取验证后的表单数据
            return HttpResponse("OK")
        else:
            return render(request, 'book.html', {'form': form})
```

- form.is_valid() 验证表单数据的合法性
- form.cleaned_data 验证通过的表单数据

### 模板中使用表单类

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>书籍</title>
</head>
<body>
    <form action="" method="post">
        {% csrf_token %}
        {{ form }}
        <input type="submit">
    </form>
</body>
</html>
```

- csrf_token 用于添加CSRF防护的字段
- form 快速渲染表单字段的方法

### 模型类表单

如果表单中的数据与模型类对应，可以通过继承**forms.ModelForm**更快速的创建表单。

```
class BookForm(forms.ModelForm):
    class Meta:
        model = BookInfo
        fields = ('btitle', 'bpub_date')
```

- model 指明从属于哪个模型类
- fields 指明向表单中添加模型类的哪个字段

## Admin站点

内容发布的部分由网站的管理员负责查看、添加、修改、删除数据，开发这些重复的功能是一件单调乏味、缺乏创造力的工作，为此，Django能够根据定义的模型类自动地生成管理模块，在Django项目中默认启用Admin管理站点。 


### 后台管理准备工作

- 数据库配置：

```
1. mysql中创建数据库：db_django05

2. 在项目的__init__文件中import pymysql包
import pymysql
pymsql.instqll_as_MySQLdb()
```

- 模型操作

```
# 创建Area区域模型类
class Area(models.Model):
    """地区类"""
    title = models.CharField(max_length=50)
    # 外键： 自关联
    parent = models.ForeignKey('self', null=True, blank=True)

# 生成迁移文件，再作迁移生成数据库表； 
# 插入测试数据： source area.sql （资料中提供）
```

- admin后台管理操作

```python
# 管理页面本地化
# 在settings.py中设置语言和时区
LANGUAGE_CODE = 'zh-hans' # 使用中国语言
TIME_ZONE = 'Asia/Shanghai' # 使用中国上海时间

# 创建后台管理器账号
# 按提示填写用户名、邮箱、密码，确认密码：
python manage.py createsuperuser

# 注册模型类： 要在后台要能看到模型类表，需要在admin.py中注册模型类
from django.contrib import admin
from models import *
admin.site.register(Area)

3.登录到后台
通过http://127.0.0.1:8000/admin/访问服务器: 输入刚创建的用户名和密码，登录到后台管理界面，登录成功可以看到如下，可以对Area进行增加、修改、删除、查询的管理操作
```

### 控制管理页显示

类ModelAdmin可以控制模型在Admin界面中的展示方式，主要包括在列表页的展示方式、添加修改页的展示方式。

在booktest/admin.py中，注册模型类前定义管理类AreaAdmin。

```
class AreaAdmin(admin.ModelAdmin):
    pass
```

管理类有两种使用方式：

- 注册参数：打开booktest/admin.py文件，注册模型类代码如下：

```
admin.site.register(AreaInfo,AreaAdmin)
```

- 装饰器：打开booktest/admin.py文件，在管理类上注册模型类，代码如下：

```
@admin.register(AreaInfo)
class AreaAdmin(admin.ModelAdmin):
    pass
```

### 列表页选项

类ModelAdmin可以控制模型在Admin界面中的展示方式，主要包括在列表页的展示方式、添加修改页的展示方式。

在app01/admin.py中，注册模型类前定义管理类AreaAdmin

```
class AreaAdmin(admin.ModelAdmin):
    pass
```

打开app01/admin.py文件，注册模型类代码如下

```
admin.site.register(AreaInfo, AreaAdmin)
```

接下来介绍如何控制列表页、增加修改页展示效果

- 每页显示多少条

打开booktest/admin.py文件，修改AreaAdmin类如下：


```
class AreaAdmin(admin.ModelAdmin):
    list_per_page = 10  # 默认为100条
```

- 设置操作选项的位置

```
# app01/admin.py
class AreaAdmin(admin.ModelAdmin):
    ...
	# 显示顶部的选项
    actions_on_top = True
	# 显示底部的选项
	actions_on_bottom = True

```

- 列表中的列操作

> 定义列表中要显示哪些字段

点击列头可以进行升序或降序排列

```
# app01/admin.py
class AreaAdmin(ModelAdmin):
    # 定义列表中要显示哪些字段
    list_display = ['id', 'title']
```

> 模型类中定义的方法也可以作为列显示

无法直接访问关联对象的属性或方法，可以在模型类中封装方法，访问关联对象的成员

```
# models.py
class Area(models.Model):
    """区域显示"""
	...
    def parent_area(self):
        """返回父级区域名"""
        if self.parent is None:
            return ''
        return self.parent.title
```

注册列：

```
class AreaAdmin(ModelAdmin):
	...
	# 定义列表中要显示哪些字段(也可以指定方法名)
	list_display = ['id', 'title', 'parent_area']
```

> 修改显示的列的名字

列标题默认为属性或方法的名称，可以通过属性设置。对于模型属性，通过`verbose_name`设置，对于方法，通过`short_description`设置，如下：

```
# models.py
class Area(models.Model):
    """区域显示"""

    # 设置verbose_name属性
    title = models.CharField(verbose_name='名称', max_length=30)  # 区域名

    def parent_area(self):
        """返回父级区域名"""
        if self.parent is None:
            return ''

        return self.parent.title

    # 指定方法列显示的名称
    parent_area.short_description = '父级区域'
```

> 设置方法列排序

**方法列默认不能排序**，如果需要排序，需要为方法指定排序依据：

```
# models.py
class Area(models.Model):
    """区域显示"""
   ...
    def parent_area(self):
        """返回父级区域名"""
        if self.parent is None:
            return ''

        return self.parent.title

    # 指定方法列按id进行排序
    parent_area.admin_order_field = 'id'
    ...
```

- 右侧栏过滤器

使用`list_filter`指定过滤，只能接收字段，会将对应字段的值列出来，用于快速过滤。**一般用于有重复的字段。**

```
# admin.py
class AreaAdmin(ModelAdmin):
    ...
    # 右侧栏过滤器
    list_filter = ['title']
```

- 搜索框

使用`search_fields`属性, 对指定字段的值进行搜索，支持模糊查询

```
# admin.py
class AreaAdmin(ModelAdmin):
    ...

    # 要搜索的列的值 
    search_fields = ['title']

```

- 中文标题

打开booktest/models.py文件，修改模型类，为属性指定verbose_name参数，即第一个参数。

```
class AreaInfo(models.Model):
    atitle=models.CharField('标题',max_length=30)#名称
    ...
```

### 编辑页选项

- 显示字段顺序

```
# admin.py
class AreaAdmin(ModelAdmin):
    ...
    # 表单中字段显示的顺序
    fields = ['parent', 'title']
```

- 修改对象显示的字符串： 重写\__str\__方法

```
# models.py
class Area(models.Model):
    """区域显示"""
    ... 
    # 重写方法
    def __str__(self):
        return self.title
```

- 字段分组显示

```
# 格式如下：
fieldsets=(
    ('组1标题',{'fields':('字段1','字段2')}),
    ('组2标题',{'fields':('字段3','字段4')}),
)
注意：fieldsets和fields，只能使用其中的一个

# 修改代码：
# admin.py	
class AreaAdmin(ModelAdmin):
	...
    # 字段分组显示
    fieldsets = (
        ('基本', {'fields': ('title'，)}),
        ('高级', {'fields': ('parent',)}),
    )
```

- 编辑关联对象

在一对多的关系中，可以在一端的编辑页面中编辑多端的对象，嵌入多端对象的方式包括表格、块两种

​	类型InlineModelAdmin：表示在模型的编辑页面嵌入关联模型的编辑

​	子类TabularInline：以表格的形式嵌入

​	子类StackedInline：以块的形式嵌入

在app01/admin.py文件中添加如下代码：

```
class AreaStackedInline(admin.StackedInline):
    model = AreaInfo    # 关联子对象（多类对象）

class AreaAdmin(admin.ModelAdmin):
    ...
    inlines = [AreaStackedInline]
```

  下面再来看下表格的效果：

```
class AreaTabularInline(TabularInline):
    model = Area   # 多类的名字
    ...

class AreaAdmin(admin.ModelAdmin):
    ...
    inlines = [AreaTabularInline]
```

- 修改预留新增选项

```
class AreaTabularInline(TabularInline):
    ...
    extra = 2      # 额外预留新增选项默认为3个
```

### 调整站点信息

Admin站点的名称信息也是可以自定义的。

```python
from django.contrib import admin

admin.site.site_header = '区域选择'  # 设置网站页头
admin.site.site_title = '传智书城MIS'	# 设置页面标题
admin.site.index_title = '欢迎使用传智书城MIS'	# 设置首页标语
```

### 重写模板

1. 进入到django的admin应用的模板目录，如下：


```
/home/python/.virtualenvs/py_django/lib/python2.7/site-packages/django/contrib/admin/templates/admin
```

找到`base-site.html`文件，复制到当前项目的`templates/admin`目录下（admin目录需要自行创建出来）

2. 修改`base-site.html`内容：

  新增一行代码

```
<h1>自定义的修改界面</h1>
```

## 上传文件

### 基础知识

```
1. 安装相关的第三方模块
2. 图片上传先关模型类配置
3. 数据迁移操作
4. 上传目录配置
```

在Django中上传图片包括两种方式：

```
在管理页面admin中上传图片
自定义form表单中上传图片
```

上传图片后，将图片存储在服务器磁盘中，然后将图片的路径存储在数据库表中

### 基本环境配置

- 安装相关第三方模块

在python中进行图片操作，需要安装包PIL

```
pip install Pillow==3.4.1
```

- 模型类配置

创建包含图片的模型类，将模型类的属性定义为models.ImageField类型

```
# models.py
class PicInfo(Model):
    """上传图片"""

    # upload_to指明该字段的图片保存在MEDIA_ROOT目录中的哪个子目录
    pic_path = models.ImageField(upload_to='app01')

    # 自定义模型管理器
    objects = PicInfoManager()
```

- 生成迁移

```
python manage.py makemigrations
python manage.py migrate
```

- 上传目录配置

```
# setting.py
MEDIA_ROOT = os.path.join(BASE_DIR, 'static/media')
```

在static下创建上传图片保存的目录

````
media/app01
````

### 管理后台上传

注册模型类，以便在后台中显示出来： 

```
# app01.admin.py
from django.contrib import admin
from app01.models import PicInfo
admin.site.register(PicInfo)
```

使用创建的用户名和密码

登录进入后台，新增一条记录，进行图片上传

### 自定义表单上传

点击`http://127.0.0.1:8000/upload/`进入上传界面

- url

```python
# app01/urls.py
urlpatterns = [
    ...
    url(r'^upload/$', views.upload),                # 进入图片上传界面
    url(r'^do_upload/$', views.do_upload),          # 处理图片上传操作
]
```

- 视图函数

```python
# views.py
def upload(request):
    """进入上传文件界面"""
    return render(request, 'app01/02.upload.html')

# views.py
def do_upload(request):
    """处理文件上传操作"""
    # 获取上传的文件对象
    pic_file = request.FILES.get('pic')
    # 定义文件的保存路径
    file_path = '%s/app01/%s' % (settings.MEDIA_ROOT, pic_file.name)
    # 保存上传的文件内容到磁盘中(with as 会自动关闭文件)
    with open(file_path, 'wb') as file:
        for data in pic_file.chunks():
                file.write(data)
    # 保存上传的文件路径到数据库中
    pic_info = PicInfo()
    pic_info.pic_path = 'app01/%s' % pic_file.name
    pic_info.save()
    # 响应浏览器内容
    return HttpResponse('文件上传成功')
```

- html

```html
# templates/app01/02.upload.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>上传图片</title>
</head>
<body>
<form method="post" enctype="multipart/form-data" action="/do_upload/">
    {% csrf_token %}
    选择文件：<input type="file" name="pic"/><br/>
    <input type="submit" value="上传">
</form>
</body>
</html>
```

### 显示用户上传的图片

url

```python
urlpatterns = [
    ...     
    url(r'^show_image/$', views.show_image),   # 进入显示图片界面
]
```

view

```python
def show_image(request):
    """进入显示图片界面"""

    # 从数据库中查询出所有的图片
    pics = PicInfo.objects.all()
    data = {'pics': pics}
    return render(request, 'app01/03.show_image.html', data)
```

html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>使用上传的图片</title>
</head>

<body>
显示用户上传的图片：<br/>

{% for pic in pics %}
    <img src="/static/media/{{ pic.pic_path }}"> <br/>
{% endfor %}

</body>
</html>
```


## 分页功能

Django中的分页操作： 

- Django提供了数据分页的类，这些类被定义在django/core/paginator.py中
- 对象Paginator用于对列进行一页n条数据的分页运算
- 对象Page用于表示第m页的数据

分页对象： 

- Paginator对象

  - 方法init(列表,int)： 返回分页对象，参数为列表数据，每页数据的条数
  - 属性count： 返回对象总数
  - 属性num_pages： 返回页面总数
  - 属性page_range： 返回页码列表，从1开始，例如[1, 2, 3, 4]
  - 方法page(m)： 返回Page对象，表示第m页的数据，下标从1开始

- Page对象

  - 调用Paginator对象的page()方法返回Page对象，不需要手动构造
  - 属性object_list： 返回当前页对象的列表
  - 属性number： 返回当前是第几页，从1开始
  - 属性paginator： 当前页对应的Paginator对象
  - 方法has_next()： 如果有下一页返回True
  - 属性next_page_number： 下一页页码
  - 方法has_previous()： 如果有上一页返回True
  - 属性previous_page_number： 上一页页码
  - 方法len()： 返回当前页面对象的个数


## 案例-省市选择

- 需求： 

  可以切换选择省份或城市，查看下级区县。

- 实现步骤： 

  一、进入区域显示界面  

  	1. 配置进入界面的url地址
  	2. 定义视图函数
  	3. 定义html界面

  二、显示省份

  	1. 服务器提供获取省份服务： 定义请求省份的url地址，和获取省份的视图函数
  	2. 配置静态文件： 
  	
  		- 在项目下创建static/js目录，复制jquery到static/js目录下.
  		- 在项目下的setting.py文件中配置： 
  	
  				# 指定静态文件所在的物理目录
  				STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]
  	
  	3. 客户端请求数据： 在html界面中通过ajax请求，获取省份数据	

  三、切换省份显示城市

  	1. 服务器提供获取下级区域服务
  	2. 获取子区域的视图函数
  	3. 在html界面中，通过ajax发请求，获取城市数据

  四、切换城市显示区县

