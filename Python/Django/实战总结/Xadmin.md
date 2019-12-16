# Xadmin

[参考](https://www.cnblogs.com/lyq-biu/p/9513888.html)

[参考](https://www.jianshu.com/p/49eb568c9a25)

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

### 源码运行demo

如果您下载的是 Xadmin 的源码包, 您会在项目目录下找到 `demo_app` 目录, 执行一下命令可以迅速开启一个 Xadmin 的演示实例:

```
cd demo_app
python manage.py runserver
```

### django项目

- settings.py

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

- urls.py

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

- adminx.py

django自带的admin模块使用的是admin.py文件，xadmin模块的文件名则叫adminx.py。admin模块在配置时使用的参数是admin.ModelAdmin，xadmin则使用object即可。然后替换admin.site.register为xadmin.site.register。

例如：

```python
from django.contrib import admin
from .models import FelixProjects
import xadmin
# Register your models here.
#class FelixProjectsAdmin(admin.ModelAdmin):
class FelixProjectsAdmin(object):
    list_display = ('pj_name', 'pj_group', 'pj_category')

# 将model类注册至xadmin
xadmin.site.register(FelixProjects, FelixProjectsAdmin)
```

在集成xadmin之后，admin模块其实就可以不要了，可以将原admin的代码删掉

> 收集 media 文件

```
python manage.py collectstatic 
```

## 配置

### 类中字段

| name                 | 类型      | 内置插件     | 说明                                                         |
| -------------------- | --------- | ------------ | ------------------------------------------------------------ |
| `list_display`       | 列表/元组 |              | 指定默认展示列                                               |
| `search_fileds`      | 列表/元组 | 过滤         | 搜索框，配置可搜索的字段                                     |
| `list_filter`        | 列表/元组 | 过滤         | 过滤器，配置可过滤的字段                                     |
| `list_editable`      | 列表/元组 | 数据即时编辑 | 可编辑字段                                                   |
| `show_detail_fields` | 列表/元组 | 显示数据详情 | 显示详情的字段                                               |
| `refresh_time`       | 列表/元组 | 列表定时刷新 | 可供选择的数据刷新时间，单位为秒                             |
| `list_bookmarks`     | 列表/元组 | 书签         | 书签                                                         |
| `data_charts`        | 字典      | 图表         | 图表                                                         |
| `list_export`        | 列表/元组 | 导出         | 导出类型                                                     |
| `list_export_fields` | 列表/元组 | 导出         | 导出字段                                                     |
| `model_icon`         | 字符串    |              | 配置图标                                                     |
| `list_per_page`      | 整数      |              | 每页展示个数                                                 |
| `ordering`           | 列表/元组 |              | 默认排序规则                                                 |
| `readonly_fields`    | 列表/元组 |              | 只读字段                                                     |
| `exclude`            | 列表/元组 |              | 不可见，有`readonly_fields`时，不生效                        |
| `relfield_style`     | 字符串    |              | 设置添加时可以搜索，而不是下拉框，ajax加载(外键)，如`fk_ajax` |
| `inlines`            | 列表/元组 |              | 在同一个页面添加完整数据,不可以在嵌套中嵌套，但可以同一个model注册两个管理器 |

### 全局配置

- 主题配置

注册与表注册不同，需要将类和views.BaseAdminView绑定，且顺序与表相反

```python
from xadmin import views

class BaseSetting(object):
    enable_themes=True
    use_bootswatch=True

xadmin.site.register(views.BaseAdminView, BaseSetting)
```

- 页头页脚、左侧边样式、全局图标

```python
from xadmin import views

class GlobalSetting(object):
    site_title = '悦动乐后台管理系统'  # 页头
    site_footer = '悦动乐'  # 页脚
    menu_style='accordion'  # 左侧以折叠样式展示
    # 设置models的全局图标
    global_search_models = [UserProfile, Sports]
    global_models_icon = {
        # 配置表的图标，可以在awesome官网上下载最新的font-awesome.css替换，并找寻到相应的icon书写
        UserProfile: "glyphicon glyphicon-user", Sports: "fa fa-cloud"
    }

xadmin.site.register(views.CommAdminView, GlobalSetting)
```

- app名字

修改左侧边栏上app名字

```python
# app.py
from django.apps import AppConfig

class UsersConfig(AppConfig):
    name = 'users'
    verbose_name='用户管理'
        
# __init__py
default_app_config='users.apps.UsersConfig'
```

## 内置插件

### Action

- 功能

Action 插件在数据列表页面提供了数据选择功能, 选择后的数据可以经过 Action 做特殊的处理. 默认提供的 Action 为批量删除功能.

- 使用

开发者可以设置 Model OptionClass 的 actions 属性, 该属性是一个列表, 包含您想启用的 Action 的类. 系统已经默认内置了删除数据的 Action,当然您可以自己制作 Action 来实现特定的功能, 制作 Action 的实例如下.

> 先要创建一个 Action 类, 该类需要继承 BaseActionView. BaseActionView 是 [`ModelAdminView`](https://xadmin.readthedocs.org/en/latest/views_api.html#xadmin.views.ModelAdminView) 的子类:

```python
from xadmin.plugins.actions import BaseActionViewclass 

class MyAction(BaseActionView):
     # 这里需要填写三个属性
     # 1. 相当于这个 Action 的唯一标示, 尽量用比较针对性的名字
     action_name = "my_action"   
     # 2. 描述, 出现在 Action 菜单中, 可以使用%(verbose_name_plural)s 代替 Model 的名字.
     description = _(u'Test selected %(verbose_name_plural)s') 
 	   # 3. 该 Action 所需权限
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

```python
class MyModelAdmin(object):

     actions = [MyAction, ]
```

> 这样就完成了自己的 Action

- API

```
class xadmin.plugins.actions.ActionPlugin(admin_view)
```

### 过滤

- 功能

在数据列表页面提供数据过滤功能, 包括: 模糊搜索, 数字范围搜索, 日期搜索等等

- 使用

在 Model OptionClass 中设置以下属性:

```
list_filter 属性:
该属性指定可以过滤的列的名字, 系统会自动生成搜索器。要想过滤某外键下的字段，只需xxx__yy（xxx为该表字段名，yy为外键对应表字段）

search_fields 属性:
属性指定可以通过搜索框搜索的数据列的名字, 搜索框搜索使用的是模糊查找的方式, 一般用来搜素名字等字符串字段

free_query_filter 属性:
默认为 True , 指定是否可以自由搜索. 如果开启自有搜索, 用户可以通过 url 参数来进行特定的搜索, 例如:
http://xxx.com/xadmin/auth/user/?name__contains=tony
```

eg:

```python
class UserAdmin(object):
    list_filter = ('is_staff', 'is_superuser', 'is_active')
    search_fields = ('username', 'first_name', 'last_name', 'email')
```

- 制作过滤器

您也可以制作自己的过滤器, 用来进行一些特定的过滤. 过滤器需要继承 `xadmin.filters.BaseFilter` 类, 并使用`xadmin.filters.manager` 注册过滤器.

### 图表

- 功能

在数据列表页面, 跟列表数据生成图表. 可以指定多个数据列, 生成多个图表

- 使用

在 Model OptionClass 中设定 `data_charts` 属性, 该属性为 dict 类型, key 是图表的标示名称, value 是图表的具体设置属性. 使用示例:

```python
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

### 书签

- 功能

记录数据列表页面特定的数据过滤, 排序等结果. 添加的书签还可以在首页仪表盘中作为小组件添加

- 使用

在 Model OptionClass 中设定如下属性:

```python
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

### 导出

- 功能

该插件在数据列表页面提供了数据导出功能, 可以导出 Excel, CSV, XML, json 格式

- 使用

如果想要导出 Excel 数据, 需要安装 [xlwt](http://pypi.python.org/pypi/xlwt).

默认情况下, xadmin 会提供 Excel, CSV, XML, json 四种格式的数据导出. 您可以通过设置 OptionClass 的 `list_export`属性来指定使用哪些导出格式 (四种各使用分别用 `xls`, `csv`, `xml`, `json` 表示), 或是将 `list_export` 设置为 `None` 来禁用数据导出功能. 示例如下:

```python
class MyModelAdmin(object):

    list_export = ('xls', 'xml', 'json')
    list_export_fields = ('username', 'age')
```

### 列表定时刷新

- 功能

该插件在数据列表页面提供了定时刷新功能, 对于需要实时刷新列表页面查看即时数据的情况非常有用

- 使用

使用数据刷新插件非常简单, 设置 OptionClass 的 `refresh_times` 属性即可. `refresh_times` 属性是存有刷新时间的数组. xadmin 默认不开启该插件.示例如下:

```python
class MyModelAdmin(object):
    
    # 这会显示一个下拉列表, 用户可以选择3秒或5秒刷新一次页面.
    refresh_times = (3, 5)
```

### 显示数据详情

- 功能

该插件可以在列表页中显示相关字段的详细信息, 使用 Ajax 在列表页中显示

- 使用

使用该插件主要设置 OptionClass 的 `show_detail_fields`, `show_all_rel_details` 两个属性. `show_detail_fields`属性设置哪些字段要显示详细信息, `show_all_rel_details` 属性设置时候自动显示所有关联字段的详细信息, 该属性默认为 `True`. 示例如下:

```python
class MyModelAdmin(object):
    
    show_detail_fields = ['group', 'father', ...]
```

### 数据即时编辑

- 功能

该插件可以在列表页中即时编辑某字段的值, 使用 Ajax 技术, 无需提交或刷新页面即可完成数据的修改, 对于需要频繁修改的字段(如: 状态)相当有用.

- 使用

使用该插件主要设置 OptionClass 的 `list_editable` 属性. `list_editable` 属性设置哪些字段需要即时修改功能. 示例如下:

```python
class MyModelAdmin(object):    
    list_editable = ['price', 'status', ...]
```

## 插件制作

### 插件原理

Xadmin 的插件系统架构设计一定程度上借鉴了 wordpress 的设计。  想要了解 Xadmin 的插件系统架构首先需要了解 XadminAdminView 的概念。  简单来说，就是 Xadmin 系统中每一个页面都是一个 AdminView 对象返回的 HttpResponse 结果。Xadmin 的插件系统做的事情其实就是在 AdminView运行过程中改变其执行的逻辑，  或是改变其返回的结果，起到修改或增强原有功能的效果。

### 自定义插件

自定义插件主要是为了改变系统的运行逻辑及结果

这里以chang页面的删除为例，当删除对象时，改变默认的数据库日志的message数据。

- 自定义插件类

继承`BaseAdminPlugin`

```ruby
from xadmin.views import BaseAdminPlugin, DeleteAdminView

# 自定义插件类
class LogPlugin(BaseAdminPlugin):
     # 根据返回值判断是否启动该插件
     def init_request(self, *args, **kwargs):
         object_id = self.args[0]
         model = self.model

         # 获取obj信息不能写在下面的delete_model, 下面已经删完了获取不到的,
         # 所以message也要在这里定义
         # 根据model获取到要删除的模型实例对象，添加到self中供delete_model调用
         self.obj = model.objects.filter(id=object_id)
         self.message = '删除了 %s' % list((self.obj.values()))
         return True

     # 重写xadmin自带的delete_model方法，这里主要修改了log函数的第二个参数(message)
    def delete_model(self):
        self.log('delete', self.message, self.obj)
        self.obj.delete()

# 自定义插件后，注册插件
xadmin.site.register_plugin(LogPlugin, DeleteAdminView)
```

- 插件开发

因为插件是继承 **BaseAdminPlugin** 类，而该类继承自 **BaseAdminObject**, 所以这两个类的方法都可以在插件中使用。

Xadmin 在创建插件时会自动注入以下属性到插件实例中
```
- request : Http Request
- user : 当前 User 对象
- args : View 方法的 args 参数
- kwargs : View 方法的 kwargs 参数
- admin_view : AdminView 实例
- admin_site : Xadmin 的 admin_site 对象实例
```

如果 AdminView 是 ModelAdminView 的子类，还会自动注入以下属性:
```
- model : Model 对象
- opts : Model 的 _meta 属性
```
接下来应该考虑打算制作什么功能的插件了。

不同功能的插件额能需要注册到不同的 AdminView上，Xadmin 系统中主要的 AdminView 有

```
BaseAdminView: 所有 AdminView 的基础类，注册在该 View 上的插件可以影响所有的 AdminView

CommAdminView: 用户已经登陆后显示的 View，也是所有登陆后 View 的基础类。该 View主要作用是创建了 Xadmin 的通用元素，例如：系统菜单，用户信息等。插件可以通过注册该 View 来修改这些信息。

ModelAdminView: 基于 Model 的 AdminView 的基础类，注册的插件可以影响所有基于 Model 的 View。

ListAdminView: Model 列表页面 View。
ModelFormAdminView: Model 编辑页面 View。
CreateAdminView: Model 创建页面 View。
UpdateAdminView: Model 修改页面 View。
DeleteAdminView: Model 删除页面 View。
DetailAdminView: Model 详情页面 View。
```

选择好目标 AdminView 后就要在自己的插件中编写方法来修改或增强这些 AdminView 。其中每个 AdminView 可以拦截的方法及其介绍请参看各 AdminView 的文档。http://xadmin.readthedocs.io/en/docs-chinese/views_api.html

xadmin源码中被 filter_hook() 装饰的方法都可以被插件截获或修改。

## 定制集成

### User

- 定制布局

user表默认注册到认证和授权app中，如果需要将其移到用户管理app中，且定制可以查看`extra_apps\xadmin\plugins\auth.py`，修改相应配置即可定制布局，只需如下，如果Django版本大于2.0，需修改相应文件名及相关配置才能使用：

方法一：

```python
# 导入关联用户表的Admin
from xadmin.plugins.auth import UserAdmin
from users.models import UserProfile

 class UserAdmin(UserAdmin):
     '''
    注册User到用户管理
    '''
     pass
from django.contrib.auth.models import User 
# 卸载自带的User注册
xadmin.site.unregister(User)
xadmin.site.register(UserProfile,UserAdmin)
```

方法二

在`extra_apps\xadmin\plugins\auth.py`中加入如下，如果在点击修改密码时报错，也需加入如下代码（xadmin的bug，后面可能已经修复）

```python
# 获取setting中的User
from django.contrib.auth import get_user_model

User = get_user_model()
```

- 点击用户详情修改密码报404

只需修改extra_apps\xadmin\plugins\auth.py注册时成setting.py对应的url即可：

```python
# 修改修改passwoed的url,我的是users/userprofile/(.+)/password
site.register_view(r'^users/userprofile/(.+)/password/$',
                   ChangePasswordView, name='user_change_password')
```

### DjangoUeditor

#### 安装

```
1.下载源码包
2.python setup.py install
3.pip install DjangoUeditor
```

#### 配置

- django

配置app

```python
# setting.py
INSTALLED_APPS = [
       .......   
    'DjangoUeditor',
]
```

配置URL

```python
url(r'^ueditor/',include('DjangoUeditor.urls' )),
```

修改model

```python
from DjangoUeditor.models import UEditorField
		
  	......
 		#width：宽，height：高，imagePath，filePath图片文件上传路径
    detail = UEditorField(verbose_name=u"运动详情",width=600, height=300, imagePath="courses/ueditor/", filePath="courses/ueditor/", default='')
```

- 源码

ueditor插件定制，在`\extra_apps\xadmin\plugins`添加`ueditor.py`如下

```python
import xadmin
from xadmin.views import BaseAdminPlugin, CreateAdminView, ModelFormAdminView, UpdateAdminView
from DjangoUeditor.models import UEditorField
from DjangoUeditor.widgets import UEditorWidget
from django.conf import settings


class XadminUEditorWidget(UEditorWidget):
    def __init__(self,**kwargs):
        self.ueditor_options=kwargs
        self.Media.js = None
        super(XadminUEditorWidget,self).__init__(kwargs)

class UeditorPlugin(BaseAdminPlugin):

    def get_field_style(self, attrs, db_field, style, **kwargs):
        if style == 'ueditor':
            if isinstance(db_field, UEditorField):
                widget = db_field.formfield().widget
                param = {}
                param.update(widget.ueditor_settings)
                param.update(widget.attrs)
                return {'widget': XadminUEditorWidget(**param)}
        return attrs

    def block_extrahead(self, context, nodes):
        js = '<script type="text/javascript" src="%s"></script>' % (settings.STATIC_URL + "ueditor/ueditor.config.js")         #自己的静态目录
        js += '<script type="text/javascript" src="%s"></script>' % (settings.STATIC_URL + "ueditor/ueditor.all.min.js")   #自己的静态目录
        nodes.append(js)

xadmin.site.register_plugin(UeditorPlugin, UpdateAdminView)
xadmin.site.register_plugin(UeditorPlugin, CreateAdminView)
```

plugin配置:在`extra_apps\xadmin\plugins\__init__.py`中配置

```python
PLUGINS = (
    'actions', 
    'filters', 
    'bookmark', 
    'export', 
    'layout', 
    'refresh',
    'details',
    'editable', 
    'relate', 
    'chart', 
    'ajax', 
    'relfield', 
    'inline', 
    'topnav', 
    'portal', 
    'quickform',
    'wizard', 
    'images', 
    'auth', 
    'multiselect', 
    'themes', 
    'aggregation', 
    'mobile', 
    'passwords',
    'sitemenu', 
    'language', 
    'quickfilter',
    'sortablelist',
    #ueditor配置，与uditor.py文件名一致
    'ueditor',
    #excel插件设置
    'excel']
```

#### 使用

admin样式定制

```python
class SportAdmin(object):    
     #定义样式
    style_fields={"detail":"ueditor"}
```

template取值时，防止前端转义 

```python
# 关闭转义                 
{% autoescape off%}
   {{ xxx.detail }}
{% endautoescape %}
```

### Excel

excel插件定制，在`\extra_apps\xadmin\plugins`添加`excel.py`如下

```python
# coding:utf-8

import xadmin
from xadmin.views import BaseAdminPlugin, ListAdminView
from django.template import loader


#excel 导入
class ListImportExcelPlugin(BaseAdminPlugin):
    import_excel = False

    def init_request(self, *args, **kwargs):
        return bool(self.import_excel)

    def block_top_toolbar(self, context, nodes):
#html文件
        nodes.append(loader.render_to_string('xadmin/excel/model_list.top_toolbar.import.html', context_instance=context))


xadmin.site.register_plugin(ListImportExcelPlugin, ListAdminView)
```

html文件定制，在`extra_apps\xadmin\templates\xadmin\excel\model_list.top_toolbar.import.html`

```python
function fileChange(target){
//检测上传文件的类型
            var imgName = document.all.submit_upload.value;
            var ext,idx;
            if (imgName == ''){
                document.all.submit_upload_b.disabled=true;
                alert("请选择需要上传的 xls 文件!");
                return;
            } else {
                idx = imgName.lastIndexOf(".");
                if (idx != -1){
                    ext = imgName.substr(idx+1).toUpperCase();
                    ext = ext.toLowerCase( );
{#                    alert("ext="+ext);#}
                    if (ext != 'xls' && ext != 'xlsx'){
                        document.all.submit_upload_b.disabled=true;
                        alert("只能上传 .xls 类型的文件!");

                        return;
                    }
                } else {
                    document.all.submit_upload_b.disabled=true;
                    alert("只能上传 .xls 类型的文件!");
                    return;
                }
            }

        }
```

后台逻辑

```python
......Admin(object):
#导入excel插件
     import_excel = True
     def post(self,request,*args,**kwargs):
        if 'excel' in request.FILES:
            pass
        #必须返回，不然报错（或者注释掉）
        return 
         super(CourseAdmin,self).post(request,*args,**kwargs)
```