# Xadmin

## 安装

方法一：

```
pip install django-xadmin
```

方法二：

```
到 https://github.com/sshwsfc/django-xadmin 下载最新的源码包或是 clone git 库, 然后在项目目录下执行:
pip install -r requirements.txt
注解:
在执行前您可以先编辑文件 requirements.txt , 其中 xlwt 不是必选的, 如果您不需要导出 excel 的功能, 可以删除这项
```

## 使用

- 源码运行demo

如果您下载的是 Xadmin 的源码包, 您会在项目目录下找到 `demo_app` 目录, 执行一下命令可以迅速开启一个 Xadmin 的演示实例:

```
cd demo_app
python manage.py runserver
```

- django项目

> settings.py

 添加 Xadmin 的模块到 `INSTALLED_APPS` 中 (注意, 安装 Django admin 所需要的 APP 也要安装, 但是 django.admin可以不安装):

```
INSTALLED_APPS = (
    ...
    'xadmin',
    'crispy_forms',
    'reversion',
    ...
)
```

> urls.py

urls里面要添加xadmin的匹配

```
from django.conf.urls import patterns, include, url
from xadmin.plugins import xversion
import xadmin

#version模块自动注册需要版本控制的 Model
xversion.register_models()

xadmin.autodiscover()

urlpatterns = [
    ...
    url(r'xadmin/', include(xadmin.site.urls)),
]
```

> adminx.py

django自带的admin模块使用的是admin.py文件，xadmin模块的文件名则叫adminx.py。admin模块在配置时使用的参数是admin.ModelAdmin，xadmin则使用object即可。然后替换admin.site.register为xadmin.site.register。例如：

```
from django.contrib import admin
from .models import FelixProjects
import xadmin
# Register your models here.
#class FelixProjectsAdmin(admin.ModelAdmin):
class FelixProjectsAdmin(object):
    list_display = ('pj_name', 'pj_group', 'pj_category')

xadmin.site.register(FelixProjects, FelixProjectsAdmin)
```

在集成xadmin之后，admin模块其实就可以不要了，可以将原admin的代码删掉

> 收集 media 文件

```
python manage.py collectstatic 
```

# 内置插件

```
# 内置插件
Action


```

## Action

- 功能

Action 插件在数据列表页面提供了数据选择功能, 选择后的数据可以经过 Action 做特殊的处理. 默认提供的 Action 为批量删除功能.

- 使用

开发者可以设置 Model OptionClass 的 actions 属性, 该属性是一个列表, 包含您想启用的 Action 的类. 系统已经默认内置了删除数据的 Action,当然您可以自己制作 Action 来实现特定的功能, 制作 Action 的实例如下.

> 先要创建一个 Action 类, 该类需要继承 BaseActionView. BaseActionView 是 [`ModelAdminView`](https://xadmin.readthedocs.org/en/latest/views_api.html#xadmin.views.ModelAdminView) 的子类:

```
from xadmin.plugins.actions import BaseActionViewclass 
MyAction(BaseActionView):
     # 这里需要填写三个属性
     # 相当于这个 Action 的唯一标示, 尽量用比较针对性的名字
     action_name = "my_action"   
     # 描述, 出现在 Action 菜单中, 可以使用%(verbose_name_plural)s 代替 Model 的名字.
     description = _(u'Test selected %(verbose_name_plural)s') 
 	 # 该 Action 所需权限
     model_perm = 'change'    
 
     # 而后实现 do_action 方法
     def do_action(self, queryset):
         # queryset 是包含了已经选择的数据的 queryset
         for obj in queryset:
             # obj 的操作
             ...
         # 返回 HttpResponse
         return HttpResponse(...)
```

> 然后在 Model 的 OptionClass 中使用这个 Action:

```
class MyModelAdmin(object):

     actions = [MyAction, ]
```

> 这样就完成了自己的 Action

- API

```
class xadmin.plugins.actions.``ActionPlugin(admin_view)
```

## 数据过滤器

- 功能

在数据列表页面提供数据过滤功能, 包括: 模糊搜索, 数字范围搜索, 日期搜索等等

- 使用

在 Model OptionClass 中设置以下属性:

```
list_filter 属性:
该属性指定可以过滤的列的名字, 系统会自动生成搜索器

search_fields 属性:
属性指定可以通过搜索框搜索的数据列的名字, 搜索框搜索使用的是模糊查找的方式, 一般用来搜素名字等字符串字段

free_query_filter 属性:
默认为 True , 指定是否可以自由搜索. 如果开启自有搜索, 用户可以通过 url 参数来进行特定的搜索, 例如:
http://xxx.com/xadmin/auth/user/?name__contains=tony
```

eg:

```
class UserAdmin(object):
    list_filter = ('is_staff', 'is_superuser', 'is_active')
    search_fields = ('username', 'first_name', 'last_name', 'email')
```

- 制作过滤器

您也可以制作自己的过滤器, 用来进行一些特定的过滤. 过滤器需要继承 `xadmin.filters.BaseFilter` 类, 并使用`xadmin.filters.manager` 注册过滤器.

## 图表插件

- 功能

在数据列表页面, 跟列表数据生成图表. 可以指定多个数据列, 生成多个图表

- 使用

在 Model OptionClass 中设定 `data_charts` 属性, 该属性为 dict 类型, key 是图表的标示名称, value 是图表的具体设置属性. 使用示例:

```
class RecordAdmin(object):
    data_charts = {
        "user_count": {'title': u"User Report", "x-field": "date", "y-field": ("user_count", "view_count"), "order": ('date',)},
        "avg_count": {'title': u"Avg Report", "x-field": "date", "y-field": ('avg_count',), "order": ('date',)}
    }
```

图表的主要属性为:

```
title : 图表的显示名称
x-field : 图表的 X 轴数据列, 一般是日期, 时间等
y-field : 图表的 Y 轴数据列, 该项是一个 list, 可以同时设定多个列, 这样多个列的数据会在同一个图表中显示
order : 排序信息, 如果不写则使用数据列表的排序
```

- API

```
class xadmin.plugins.chart.ChartsPlugin(admin_view)
class xadmin.plugins.chart.ChartsView(request, *args, **kwargs)
```

## 书签

- 功能

记录数据列表页面特定的数据过滤, 排序等结果. 添加的书签还可以在首页仪表盘中作为小组件添加

- 使用

在 Model OptionClass 中设定如下属性:

```
show_bookmarks 属性:
设置是否开启书签功能, 默认为 True

list_bookmarks 属性:
设置默认的书签. 用户可以在列表页面添加自己的书签, 你也可以实现设定好一些书签,
使用实例如下:
class UserAdmin(object):
    list_bookmarks = [{
        'title': "Female",         # 书签的名称, 显示在书签菜单中
        'query': {'gender': True}, # 过滤参数, 是标准的 queryset 过滤
        'order': ('-age'),         # 排序参数
        'cols': ('first_name', 'age', 'phones'),  # 显示的列
        'search': 'Tom'    # 搜索参数, 指定搜索的内容
        }, {...}
    ]
```

## 数据导出

- 功能

该插件在数据列表页面提供了数据导出功能, 可以导出 Excel, CSV, XML, json 格式

- 使用

如果想要导出 Excel 数据, 需要安装 [xlwt](http://pypi.python.org/pypi/xlwt).

默认情况下, xadmin 会提供 Excel, CSV, XML, json 四种格式的数据导出. 您可以通过设置 OptionClass 的 `list_export`属性来指定使用哪些导出格式 (四种各使用分别用 `xls`, `csv`, `xml`, `json` 表示), 或是将 `list_export` 设置为 `None` 来禁用数据导出功能. 示例如下:

```
class MyModelAdmin(object):

    list_export = ('xls', xml', 'json')
```

## 列表定时刷新

- 功能

该插件在数据列表页面提供了定时刷新功能, 对于需要实时刷新列表页面查看即时数据的情况非常有用

- 使用

使用数据刷新插件非常简单, 设置 OptionClass 的 `refresh_times` 属性即可. `refresh_times` 属性是存有刷新时间的数组. xadmin 默认不开启该插件.示例如下:

```
class MyModelAdmin(object):
    
    # 这会显示一个下拉列表, 用户可以选择3秒或5秒刷新一次页面.
    refresh_times = (3, 5)
```

 ## 显示数据详情

- 功能

该插件可以在列表页中显示相关字段的详细信息, 使用 Ajax 在列表页中显示

- 使用

使用该插件主要设置 OptionClass 的 `show_detail_fields`, `show_all_rel_details` 两个属性. `show_detail_fields`属性设置哪些字段要显示详细信息, `show_all_rel_details` 属性设置时候自动显示所有关联字段的详细信息, 该属性默认为 `True`. 示例如下:

```
class MyModelAdmin(object):
    
    show_detail_fields = ['group', 'father', ...]
```

## 数据即时编辑

- 功能

该插件可以在列表页中即时编辑某字段的值, 使用 Ajax 技术, 无需提交或刷新页面即可完成数据的修改, 对于需要频繁修改的字段(如: 状态)相当有用.

- 使用

使用该插件主要设置 OptionClass 的 `list_editable` 属性. `list_editable` 属性设置哪些字段需要即时修改功能. 示例如下:

```
class MyModelAdmin(object):    
    list_editable = ['price', 'status', ...]
```

# 插件制作

