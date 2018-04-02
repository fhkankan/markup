
# 常用技术

## 静态文件

**一、静态文件的使用**

- **静态文件**：网页中使用的css，js，图片


- **静态文件的使用**： 
  1. 创建静态目录static/(css,image,js)，导入静态文件

  2. 在setting.py中配置静态目录

```
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
```
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

异常处理： 视图函数执行出错之后，会调用中间件的process_exception方法，可以在该方法中执行异常操作。	
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
```
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
```
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

```
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

```
1. 创建后台管理器账号
python manage.py createsuperuser
按提示填写用户名、邮箱、密码，确认密码：

2.注册模型类： 要在后台要能看到模型类表，需要在admin.py中注册模型类
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

- 注册参数
- 装饰器

注册参数：打开booktest/admin.py文件，注册模型类代码如下：

```
admin.site.register(AreaInfo,AreaAdmin)
```

装饰器：打开booktest/admin.py文件，在管理类上注册模型类，代码如下：

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

1. **每页显示多少条**

  打开booktest/admin.py文件，修改AreaAdmin类如下：


```
class AreaAdmin(admin.ModelAdmin):
    list_per_page = 10  # 默认为100条
```

2. **设置操作选项的位置**

```
# app01/admin.py
class AreaAdmin(admin.ModelAdmin):
    ...
	# 显示顶部的选项
    actions_on_top = True
	# 显示底部的选项
	actions_on_bottom = True
```

3. **列表中的列操作**

- 定义列表中要显示哪些字段

```
# app01/admin.py
class AreaAdmin(ModelAdmin):
    # 定义列表中要显示哪些字段
    list_display = ['id', 'title']
```

**点击列头可以进行升序或降序排列**

- **模型类中定义的方法也可以作为列显示**（通过此方式可访问关联对象的属性）

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

- 修改显示的列的名字

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

- 设置方法列排序

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

4. 右侧栏过滤器

使用`list_filter`指定过滤，只能接收字段，会将对应字段的值列出来，用于快速过滤。**一般用于有重复的字段。**

```
# admin.py
class AreaAdmin(ModelAdmin):
    ...
    # 右侧栏过滤器
    list_filter = ['title']
```

5. 搜索框

使用`search_fields`属性, 对指定字段的值进行搜索，支持模糊查询

```
# admin.py
class AreaAdmin(ModelAdmin):
    ...

    # 要搜索的列的值 
    search_fields = ['title']

```

6.  中文标题

打开booktest/models.py文件，修改模型类，为属性指定verbose_name参数，即第一个参数。

```
class AreaInfo(models.Model):
    atitle=models.CharField('标题',max_length=30)#名称
    ...
```

### 编辑页选项

1. 显示字段顺序

```
# admin.py
class AreaAdmin(ModelAdmin):
    ...
    # 表单中字段显示的顺序
    fields = ['parent', 'title']
```

2. 修改对象显示的字符串： 重写\__str\__方法

```
# models.py
class Area(models.Model):
    """区域显示"""
    ... 
    # 重写方法
    def __str__(self):
        return self.title
```

3. 字段分组显示

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

4. 编辑关联对象

- 在一对多的关系中，可以在一端的编辑页面中编辑多端的对象，嵌入多端对象的方式包括表格、块两种
- 类型InlineModelAdmin：表示在模型的编辑页面嵌入关联模型的编辑
- 子类TabularInline：以表格的形式嵌入
- 子类StackedInline：以块的形式嵌入

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


5. 修改预留新增选项

```
class AreaTabularInline(TabularInline):
    ...
    extra = 2      # 额外预留新增选项默认为3个
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

## 上传图片

- 在python中进行图片操作，需要安装包PIL

```
pip install Pillow==3.4.1
```

- 在Django中上传图片包括两种方式：

  - 在管理页面admin中上传图片
  - 自定义form表单中上传图片
- 上传图片后，将图片存储在服务器磁盘中，然后将图片的路径存储在数据库表中

### 定义上传目录

1. 在static下创建上传图片保存的目录： `media/app01`

2. 设置上传文件的保存目录


```
# setting.py
MEDIA_ROOT = os.path.join(BASE_DIR, 'static/media')
```

### 模型类定义

1. 模型类定义
```
# models.py
class PicInfo(Model):
    """上传图片"""

    # 上传图片保存的路径(注意：相对于上面MEDIA_ROOT指定的static/media目录)
    pic_path = models.ImageField(upload_to='app01')

    # 自定义模型管理器
    objects = PicInfoManager()
```

2. 生成迁移文件，生成表。

### admin管理后台上传

1. 注册模型类，以便在后台中显示出来： 
```
# app01.admin.py
from django.contrib import admin
from app01.models import PicInfo
admin.site.register(PicInfo)
```

2. 使用创建的用户名和密码
3. 登录进入后台，新增一条记录，进行图片上传：

### 案例：自定义界面上传图片

- 需求： 自定义界面上传图片 

- 实现步骤：

  1. 在python中进行图片操作，需要安装包PIL

     pip install Pillow==3.4.1

  2. 配置url
  3. 定义视图函数
  4. 创建显示的html界面
  5. 服务器提供上传服务： 要定义上传请求的url地址，和处理上传操作的视图函数
  6. 客户端发请求实现图片上传： 在html界面，提交表单到对应的url地址，实现图片上传功能


### 案例：显示图片

- 需求： 
- 实现步骤：

  1. 配置进入界面的url地址
  2. 定义视图函数, 查询所有的图片
  3. 显示html界面


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

### 示例




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

