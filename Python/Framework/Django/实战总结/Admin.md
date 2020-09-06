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

## 