# 视图-URL

一个干净、优雅的URL方案是高质量Web应用程序中的一个重要细节。 Django 让你随心所欲设计你的URL，不受框架束缚。

不要求有 `.php` 或 `.cgi` ，更不会要求类似`0,2097,1-1-1928,00` 这样无意义的东西。

参见万维网的发明者Berners-Lee 的[Cool URIs don’t change](https://www.w3.org/Provider/Style/URI)，里面有关于为什么URL 应该保持整洁和有意义的卓越论证。

## 概述

为了给一个应用设计URL，你需要创建一个Python模块，通常称为路由选择模块或路由解析模块**URLconf**（URL configuration）。 该模块是一个纯粹的Python模块，是 URL路径表达式 到 Python 函数（你的视图）之间的映射。

根据你的需要，这个映射可短可长。 它也可以引用其它的映射。 而且，由于它是纯粹的Python代码，因此可以动态构建它。

Django还提供了根据活动语言(active language)翻译URL的方法。 更多信息请参考[国际化文档](https://yiyibooks.cn/__trs__/qy/django2/topics/i18n/translation.html#url-internationalization)

## 处理一个请求

当用户请求Django 站点上的某个页面时，django系统用一个算法来决定执行哪段Python代码

### Django1.8

```
1. Django会使用路由解析根模块(root URLconf)来解析路由。
- 通常，该路由解析根模块的位置由settings中的ROOT_URLCONF 变量指定（该模块的默认位置在BASE_DIR所指定的目录下的主app目录下的urls.py模块）。
- 如果进来的HttpRequest 对象有urlconf 属性（该属性由中间件request processing设置），那么由ROOT_URLCONF所设置的路由解析根模块的路径则被HttpRequest对象的urlconf属性的值所替换。

2. Django 加载该路由解析模块，并寻找可用的urlpatterns。
这个urlpattens是一个Python列表，该列表的每个元素都是django.conf.urls.url()的一个实例。

3. Django 依次匹配该列表中的每个URL模式，在遇到第一个与请求的URL相匹配的模式时停下来。

4. 一旦某个正则表达式与请求的URL相匹配，则Django导入并调用给定的视图，该视图仅为一个单纯的Python函数（或者是一个基于类的视图）。同时，如下参数被传递给该视图:
		一个HttpRequest实例。
		如果所匹配的正则表达式返回的是若干个无名组，那么该正则表达式所匹配的内容将被作为位置参数提供给该视图。
		关键字参数由与正则表达式相匹配的命名组组成，并且这些关键字参数可以被django.conf.urls.url()的可选参数kwargs覆盖。

5. 如果请求的URL没有匹配到任何一个正则表达式，或者在匹配过程的任何时刻抛出了一个异常，那么Django将调用适当的错误处理视图进行处理
```

匿名正则示例

```python
# 路由选择模块
from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^articles/2003/$', views.special_case_2003),
    url(r'^articles/([0-9]{4})/$', views.year_archive),
    url(r'^articles/([0-9]{4})/([0-9]{2})/$', views.month_archive),
    url(r'^articles/([0-9]{4})/([0-9]{2})/([0-9]+)/$', views.article_detail),
]

# 请求处理
1. /articles/2005/03/ 的请求将匹配列表中的第三个模式。Django 将调用函数views.month_archive(request, '2005', '03')。
2. /articles/2005/3/ 不匹配任何URL 模式，因为列表中的第三个模式要求月份应该是两个数字。
3. /articles/2003/ 将匹配列表中的第一个模式不是第二个，因为模式按顺序匹配，第一个会首先测试是否匹配。请像这样自由插入一些特殊的情况来探测匹配的次序。
4. /articles/2003 不匹配任何一个模式，因为每个模式要求URL以一个斜线结尾。
5. /articles/2003/03/03/ 将匹配最后一个模式。Django 将调用函数views.article_detail(request, '2003', '03', '03')。
```

命名组正则示例

```python
# 在Python 正则表达式中，命名正则表达式组的语法是(?P<name>pattern)，其中name 是组的名称，pattern 是要匹配的模式。
# 路由选择模块
from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^articles/2003/$', views.special_case_2003),
    url(r'^articles/(?P<year>[0-9]{4})/$', views.year_archive),
    url(r'^articles/(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/$', views.month_archive),
    url(r'^articles/(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/(?P<day>[0-9]{2})/$', views.article_detail),
]

# 注意
1. 若要从URL 中捕获某个值，只需要在它周围放置一对圆括号。
2. 在正则表达式中不需要添加一个前导的反斜杠，因为每个URL默认都带有该符号。例如，应该写成^articles 而不是 ^/articles。
3. 每个正则表达式前面的'r' 是可选的，但是建议加上。它告诉Python 这个字符串是“原始的” —— 字符串中任何字符都不应该转义。
# 请求处理
1. /articles/2005/03/ 请求将调用views.month_archive(request, year='2005', month='03')函数，而不是views.month_archive(request, '2005', '03')。
2. /articles/2003/03/03/ 请求将调用函数views.article_detail(request, year='2003', month='03', day='03')。
```

嵌套的参数

```python
# 正则表达式允许嵌套参数，Django将解析它们并将它们传递给视图。当反查时，Django将尝试填充所有外部捕获的参数，而忽略任何嵌套的捕获参数。
from django.conf.urls import url

urlpatterns = [
    url(r'blog/(page-(\d+)/)?$', blog_articles),                  # bad
    url(r'comments/(?:page-(?P<page_number>\d+)/)?$', comments),  # good
]

# 请求
两个模式都使用嵌套的参数，其解析方式是：例如blog/page-2/ 将匹配blog_articles并带有两个位置参数page-2/ 和2。第二个comments 的模式将匹配comments/page-2/ 并带有一个值为2的关键字参数page_number。这个例子中外围参数是一个不捕获的参数(?:...)。

blog_articles 视图需要最外层捕获的参数来反查，在这个例子中是page-2/或者没有参数，而comments可以不带参数或者用一个page_number值来反查。

嵌套捕获的参数使得视图参数和URL之间存在强耦合，正如blog_articles 所示：视图接收URL的一部分（page-2/），而不只是视图参数所要的值。这种耦合在反查时更加显著，因为反查视图时我们需要传递URL 的一个片段而不只是page 的值。

通常来说，我们只捕获视图需要的参数；并且当正则需要参数但是视图忽略参数时，请使用非嵌套参数
```

匹配/分组算法

```
下面是URLconf 解析器使用的算法，针对正则表达式中的命名组和非命名组：

1. 如果有命名参数，则使用这些命名参数，忽略非命名参数。
2. 否则，它将以位置参数传递所有的非命名参数。

根据传递额外的选项给视图函数，这两种情况下，多余的关键字参数也将传递给视图。
```

捕获的参数都是字符串

```
每个捕获的参数都作为一个普通的Python 字符串传递给视图，无论正则表达式使用的是什么匹配方式。
```

### Django2.0

```
1. Django会使用根路由解析模块(root URLconf)来解析路由。 
通常，这是ROOT_URLCONF设置的值，
但是如果传入的HttpRequest对象具有urlconf属性（由中间件设置）那么ROOT_URLCONF的设置将被其替换。

2. Django加载该Python模块并查找变量urlpatterns。 
它应该是django.urls.path()或者django.urls.re_path()实例的Python列表。

3. Django按顺序遍历每个URL pattern，并在第一个匹配的请求URL被匹配时停下。

4. 一旦某个URL pattern成功匹配，Django会导入并调用给定的视图，该视图是一个简单的Python函数（或基于类的视图）。 这个视图会被传以以下参数：
			一个 HttpRequest的实例。
			如果所匹配的正则表达式返回的是若干个无名组，那么该正则表达式所匹配的内容将被作为位置参数提供给该视图。
				关键字参数是由路径表达式匹配的任何指定部件组成的，在可选的kwargs参数中指定的任何参数覆盖到django.urls.path()或django.urls.re_path()。

5. 如果请求的URL没有匹配到任何一个表达式，或者在匹配过程的任何时刻抛出了一个异常，那么Django 将调用适当的错误处理视图进行处理。
```

#### 示例

```python
from django.urls import path

from . import views

urlpatterns = [
    path('articles/2003/', views.special_case_2003),
    path('articles/<int:year>/', views.year_archive),
    path('articles/<int:year>/<int:month>/', views.month_archive),
    path('articles/<int:year>/<int:month>/<slug:slug>/', views.article_detail),
]

# 注意：
1. 要从URL捕获某个值，使用尖角括号。
2. 捕获的值可以选择包含一个转换器类型。 例如, 使用 <int:name> 来捕获一个整形参数。 如果没有包含转换器, 那么除了/ 字符外, 会匹配任意字符串。
3. 因为每个URL都有前导斜杠，所以使用时，每个URL没有必要添加一个/。 例如, path函数中URL部分应该使用 articles, 而不是/articles。
# 请求处理
1. /articles/2005/03/ 的请求将与列表中的第三个条目匹配。 Django会调用函数views.month_archive（request, year=2005, month=3）。
2. /articles/2003/ 将与列表中的第一个模式相匹配，而不是第二个，因为这些模式是按顺序匹配的，第一个模式会首先测试是否匹配。 请像这样自由插入一些特殊的情况来探测匹配的次序。 在这里，Django会调用函数views.special_case_2003（request）
3. /articles/2003 不会匹配到任何一个模式，因为每个模式都要求URL以斜杠结尾。
4. /articles/2003/03/building-a-django-site/ 会匹配到最后一个模式。 Django会调用函数views.article_detail（request, year=2003, month=3, slug ="building-a-django-site"）
```

#### 路径转换器

- 默认可用

```python
str   # 匹配除了路径分隔符'/'的任意非空字符串。如果表达式中没有包含转换器，那么这将是默认行为。
int  	# 匹配0或任意正整数。并作为 int 返回。
slug  # 匹配任意的黏接字符串(slug string)，这些黏接字符串是ASCII的字母或数字，词与词之间由连字符或下划线黏接组成。 例如, building-your-1st-django-site。
uuid  # 匹配一个格式化的UUID. 为了防止多个URL映射到同一页面，必须包含多个破折号（dash），同时字母必须小写。 例如, 075194d3-6885-417e-a8a8-6c931e272f00. 返回一个 UUID 实例。
path  # 匹配包含路径分隔符 '/'在内的任意非空字符串。 相对于str，这允许你匹配一个完整的URL路径，而不仅仅是URL路径的一部分。
```

- 注册自定义路径转换器

```python
# 一个转换器的class定义需要包含以下信息:
1. 一个字符串形式的正则表达式属性。
2. 一个to_python(self, value) 方法，负责将匹配的字符串转换成python类型的数据，处理的结果会传给视图的相关方法。当类型转换失败时，这个方法应该抛出ValueError异常。
3. 一个to_url(self, value) 方法，负责将python类型的数据转换为字符串类型，字符串用于构建URL。

# 示例
class FourDigitYearConverter:
    regex = '[0-9]{4}'

    def to_python(self, value):
        return int(value)

    def to_url(self, value):
        return '%04d' % value
  
# 使用register_converter()方法在你的URLconf配置文件中注册自定义的转换器
from django.urls import register_converter, path
from . import converters, views

register_converter(converters.FourDigitYearConverter, 'yyyy')  # 注册自定义转换器

urlpatterns = [
    path('articles/2003/', views.special_case_2003),
    path('articles/<yyyy:year>/', views.year_archive),
    ...
]
```

#### 正则表达式

如果路径和转换器语法不足以定义你的URL pattern，你还可以使用正则表达式。 为了使用正则表达式，请使用`re_path()`，而不要使用`path()`

- 命名正则

```python
# 在Python正则表达式中，命名正则表达式组的语法是 `(?P<name>pattern)`， 这里 `name` 是表达式组的名字 而`pattern` 是要匹配的模式。
from django.urls import path, re_path

from . import views

urlpatterns = [
    path('articles/2003/', views.special_case_2003),
    re_path('articles/(?P<year>[0-9]{4})/', views.year_archive),
    re_path('articles/(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/', views.month_archive),
    re_path('articles/(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/(?P<slug>[\w-_]+)/', views.article_detail),
]
```

- 匿名正则

```
除了命名的正则表达式组语法，例如(?P<year>[0-9]{4})， 你还可以使用简短的匿名正则表达式组，例如 ([0-9]{4})。

这种用法不是特别推荐的，因为它可能会意外的在你的匹配意图和视图参数之间引入错误。
```

无论哪种情况，建议在给定的正则表达式中只使用一种风格。 当两种样式混合使用时，任何匿名的正则表达式组都会被忽略，只有命名正则表达式组被传递给视图函数。

- 嵌套参数

```python
# 正则表达式允许嵌套参数，Django将解析它们并将它们传递给视图。当反查时，Django将尝试填充所有外部捕获的参数，而忽略任何嵌套的捕获参数。
from django.urls import re_path

urlpatterns = [
    re_path(r'blog/(page-(\d+)/)?$', blog_articles),                  # bad
    re_path(r'comments/(?:page-(?P<page_number>\d+)/)?$', comments),  # good
]

# 解析
这两种模式都使用嵌套参数并解析：
blog/page-2/将与blog_articles产生两个有位置顺序的参数的匹配：page-2/和2。 第二个模式comments将会匹配comments/page-2/并以page_number为关键字，设置其为2。 这个例子中的外层参数(?:...)是一个非捕获参数的用法。
blog_articles视图需要反转最外面的捕获参数，在这种情况下需要页面2/或没有参数，而comments可以使用无参数或page_number的值来反转。
嵌套捕获的参数在视图参数和URL之间创建了一个强大的耦合，如blog_articles所示：视图接收部分URL（page-2/）作为参数，而不仅仅是视图感兴趣的值。这种耦合在反转时更为明显，因为要反转视图，我们需要传递一段URL而不是页码。

作为一个经验法则，只去捕获视图所需要使用的值，并在正则表达式需要参数而视图需要忽略它时，使用非捕获参数语法。
```

## 搜索内容

URLconf把请求的URL作为普通的Python字符串来搜索。 所以在URL检索时，不包括GET或POST参数或域名。

例如，在`https://www.example.com/myapp/`的请求中，URLconf将查找`myapp/`。

在`https://www.example.com/myapp/?page=3`的请求中，URLconf将查找`myapp/`。

URLconf不关注请求的方法。 换句话说，所有的请求方法 - `POST`, `GET`, `HEAD` 等都将被路由到相同URL的同一个函数。

## 指定视图参数默认值

1.8

```python
# URLconf
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^blog/$', views.page),
    url(r'^blog/page(?P<num>[0-9]+)/$', views.page),
]

# View (in blog/views.py)
def page(request, num="1"):
    # Output the appropriate page of blog entries, according to num.
    ...
```

2.0

```python
# URLconf
from django.urls import path
from . import views

urlpatterns = [
    path('blog/', views.page),
    path('blog/page<int:num>/', views.page),
]

# View (in blog/views.py)
def page(request, num=1):
    # Output the appropriate page of blog entries, according to num.
    ...
```

## 性能

`urlpatterns` 中的每个正则表达式在第一次访问它们时被编译。这使得系统相当快。

## 变量语法

### 1.8

```python
url(regex, view, kwargs=None, name=None, prefix='')

# 参数
regex  # 参数应该是包含与Python的re兼容的正则表达式的字符串或gettext_lazy()（请参阅翻译URL模式） 模块。
view  # 参数是一个视图函数或基于类的视图的as_view()的结果。也可以是django.urls.include().
kwargs  # 参数允许您将其他参数传递给视图函数或方法
name  # 参数命名URL模式。
prefix
```

`urlpatterns` 应该是一个`url()` 实例类型的Python 列表。

```
urlpatterns = [
    url(r'^index/$', index_view, name="main-view"),
    ...
]
```

### 2.0

`urlpatterns`应该是`path()`和/或`re_path()`实例的Python列表。

- path

```python
path(route, view, kwargs=None, name=None)

# 参数
route  # 参数应该是包含URL模式的字符串或gettext_lazy()（请参阅翻译URL模式）。该字符串可能包含尖括号（如之前的<username>）用来捕获URL的一部分并作为关键字参数传给视图。尖括号可以包含转换器规范（如<int:section>中的的int部分）用于限制字符的匹配并能改变传至试图的变量的类型。 例如，<int:section>匹配一个数字字符并将其转成int类型。
view  # 参数是一个视图函数或基于类的视图的as_view()的结果。也可以是django.urls.include().
kwargs  # 参数允许您将其他参数传递给视图函数或方法。 有关示例，请参阅传递额外选项以查看函数。
name  # 参数命名URL模式
```

返回包含在`urlpatterns`中的元素

```python
from django.urls import include, path

urlpatterns = [
    path('index/', views.index, name='main-view'),
    path('bio/<username>/', views.bio, name='bio'),
    path('articles/<slug:title>/', views.article, name='article-detail'),
    path('articles/<slug:title>/<int:section>/', views.section, name='article-section'),
    path('weblog/', include('blog.urls')),
    ...
]
```

- re-path

```python
re_path(route, view, kwargs=None, name=None)

# 参数
route  # 参数应该是包含与Python的re兼容的正则表达式的字符串或gettext_lazy()（请参阅翻译URL模式） 模块。 字符串通常使用原始字符串语法（r''），以便它们可以包含像\d这样的序列，而不需要用另一个反斜杠转义反斜杠。 当有字符匹配时，正则表达式中的捕获的字符组将被传递给视图 - 如果字符组有命名则按变量名传递参数，否则按顺序传递参数。 这些值作为字符串传递，不进行任何类型转换。
其他见path()
```

返回包含在`urlpatterns`中的元素。

```python
from django.urls import include, re_path

urlpatterns = [
    re_path(r'^index/$', views.index, name='index'),
    re_path(r'^bio/(?P<username>\w+)/$', views.bio, name='bio'),
    re_path(r'^weblog/', include('blog.urls')),
    ...
]
```

## 错误处理

当Django无法为请求的URL找到匹配项或者引发异常时，Django会调用错误处理视图。

用于这些情况的视图由四个变量指定。 他们的默认值应该足以满足大多数项目，但是可以通过覆盖其默认值进一步定制。

这些值可以在你的根URLconf 中设置。在其它URLconf 中设置这些变量将不会产生效果。

它们的值必须是可调用的或者是表示视图的Python 完整导入路径的字符串，可以方便地调用它们来处理错误情况。

这些值是

````python
handler404 —— 参见django.conf.urls.handler404。
handler500 —— 参见django.conf.urls.handler500。
handler403 —— 参见django.conf.urls.handler403。
handler400 —— 参见django.conf.urls.handler400。
````

## 包含其他URLconf

在任何时候，你的`urlpatterns` 都可以包含其它URLconf 模块。这实际上将一部分URL 放置于其它URL 下面

### 包含方式

- 第一种：字符串

1.8

```python
from django.conf.urls import include, url

urlpatterns = [
    # ... snip ...
    url(r'^community/', include('django_website.aggregator.urls')),
    url(r'^contact/', include('django_website.contact.urls')),
    # ... snip ...
]

# 注意
这个例子中的正则表达式没有包含$（字符串结束匹配符），但是包含一个末尾的斜杠。每当Django 遇到include()（django.conf.urls.include()）时，它会去掉URL中匹配的部分并将剩下的字符串发送给包含的URLconf做进一步处理。
```
2.0
```python
from django.urls import include, path

urlpatterns = [
    # ... snip ...
    path('community/', include('aggregator.urls')),
    path('contact/', include('contact.urls')),
    # ... snip ...
]

# 注意
每当Django遇到include()时，它会去掉URL中匹配的部分，并将剩余的字符串发送到包含的URLconf以供进一步处理。
```

- 使用实例的列表

1.8(`url()`)

```python
from django.conf.urls import include, url

from apps.main import views as main_views
from credit import views as credit_views

extra_patterns = [
    url(r'^reports/(?P<id>[0-9]+)/$', credit_views.report),
    url(r'^charge/$', credit_views.charge),
]

urlpatterns = [
    url(r'^$', main_views.homepage),
    url(r'^help/', include('apps.help.urls')),
    url(r'^credit/', include(extra_patterns)),
]
```

2.0(`path()`)

```python
from django.urls import include, path

from apps.main import views as main_views
from credit import views as credit_views

extra_patterns = [
    path('reports/', credit_views.report),
    path('reports/<int:id>/', credit_views.report),
    path('charge/', credit_views.charge),
]

urlpatterns = [
    path('', main_views.homepage),
    path('help/', include('apps.help.urls')),
    path('credit/', include(extra_patterns)),
]
```

移除URL配置中重复部分

```python
# 1.8
# 原URL
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/history/$', views.history),
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/edit/$', views.edit),
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/discuss/$', views.discuss),
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/permissions/$', views.permissions),
]
# 改进后
from django.conf.urls import include, url
from . import views

urlpatterns = [
    url(r'^(?P<page_slug>[\w-]+)-(?P<page_id>\w+)/', include([
        url(r'^history/$', views.history),
        url(r'^edit/$', views.edit),
        url(r'^discuss/$', views.discuss),
        url(r'^permissions/$', views.permissions),
    ])),
]

# 2.0
# 原URL
from django.urls import path
from . import views

urlpatterns = [
    path('<page_slug>-<page_id>/history/', views.history),
    path('<page_slug>-<page_id>/edit/', views.edit),
    path('<page_slug>-<page_id>/discuss/', views.discuss),
    path('<page_slug>-<page_id>/permissions/', views.permissions),
]
# 改进后
from django.urls import include, path
from . import views

urlpatterns = [
    path('<page_slug>-<page_id>/', include([
        path('history/', views.history),
        path('edit/', views.edit),
        path('discuss/', views.discuss),
        path('permissions/', views.permissions),
    ])),
]
```

### 捕获参数

被包含的URLconf 会收到来自父URLconf 捕获的任何参数

```python
# 1.8
# In settings/urls/main.py
from django.conf.urls import include, url

urlpatterns = [
    url(r'^(?P<username>\w+)/blog/', include('foo.urls.blog')),  # 捕获的"username"变量将被如期传递给include()指向的URLconf。
]

# In foo/urls/blog.py
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.blog.index),
    url(r'^archive/$', views.blog.archive),
]

# 2.0
# In settings/urls/main.py
from django.urls import include, path

urlpatterns = [
    path('<username>/blog/', include('foo.urls.blog')),  # 捕获的"username"变量传递给被包含的URLconf。
]

# In foo/urls/blog.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.blog.index),
    path('archive/', views.blog.archive),
]
```

## 传递额外参数

URLconfs 具有一个钩子，让你传递一个Python 字典作为额外的参数传递给视图函数

### 视图函数

1.8

```python
# django.conf.urls.url()函数可以接收一个可选的第三个参数，它是一个字典，表示想要传递给视图函数的额外关键字参数。

from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^blog/(?P<year>[0-9]{4})/$', views.year_archive, {'foo': 'bar'}),
]

# 冲突
URL 模式捕获的命名关键字参数和在字典中传递的额外参数有可能具有相同的名称。当这种情况发生时，将使用字典中的参数而不是URL中捕获的参数。
# 解析
1. /blog/2005/请求，Django将调用views.year_archive(request, year='2005', foo='bar')。
```

2.0

```python
# path()函数可以采用可选的第三个参数，该参数应该是传递给视图函数的额外关键字参数的字典
rom django.urls import path
from . import views

urlpatterns = [
    path('blog/<int:year>/', views.year_archive, {'foo': 'bar'}),
]

# 冲突
URL 模式捕获的命名关键字参数和在字典中传递的额外参数有可能具有相同的名称。当这种情况发生时，将使用字典中的参数而不是URL中捕获的参数。
# 解析
1. /blog/2005/请求，Django将调用views.year_archive(request, year=2005, foo='bar')。
```

### include

可以将额外的选项传递给[`include()`](https://yiyibooks.cn/__trs__/qy/django2/ref/urls.html#django.urls.include)，并且包含的URLconf中的每一行都将传递额外的选项

1.8

```python
# 方法一
# main.py
from django.conf.urls import include, url

urlpatterns = [
    url(r'^blog/', include('inner'), {'blogid': 3}),
]

# inner.py
from django.conf.urls import url
from mysite import views

urlpatterns = [
    url(r'^archive/$', views.archive),
    url(r'^about/$', views.about),
]

# 方法二
# main.py
from django.conf.urls import include, url
from mysite import views

urlpatterns = [
    url(r'^blog/', include('inner')),
]

# inner.py
from django.conf.urls import url

urlpatterns = [
    url(r'^archive/$', views.archive, {'blogid': 3}),
    url(r'^about/$', views.about, {'blogid': 3}),
]
```

2.0

```python
# 方法一
# main.py
from django.urls import include, path

urlpatterns = [
    path('blog/', include('inner'), {'blog_id': 3}),
]

# inner.py
from django.urls import path
from mysite import views

urlpatterns = [
    path('archive/', views.archive),
    path('about/', views.about),
]


# 方法二
# main.py
from django.urls import include, path
from mysite import views

urlpatterns = [
    path('blog/', include('inner')),
]

# inner.py
from django.urls import path

urlpatterns = [
    path('archive/', views.archive, {'blog_id': 3}),
    path('about/', views.about, {'blog_id': 3}),
]
```

请注意，无论行的视图实际上是否接受这些选项，附加选项将始终传递到所包含的URLconf中的每一行。因此，仅当您确定所包含的URLconf中的每个视图都接受您要传递的其他选项时，此技术才有用。

## 反向解析

Django 提供了一个解决方案使得URL 映射是URL 设计唯一的储存库。你用你的URLconf填充它，然后可以双向使用它

```
- 根据用户/浏览器发起的URL请求，它调用正确的Django视图，并从URL中提取它的参数需要的值。
- 根据Django视图的标识和将要传递给它的参数的值，获取与之关联的URL。
```

在需要URL 的地方，对于不同层级，Django 提供不同的工具用于URL 反查

- 在模板中：使用url 模板标签。
- 在Python 代码中：1.8使用`django.core.urlresolvers.reverse()`，2.0使用`django.urls.reverse()`。
- 在更高层的与处理Django模型实例相关的代码中：使用`get_absolute_url()`方法

### 示例

- 1.8

URLconf

```python
from django.conf.urls import url

from . import views

urlpatterns = [
    #...
    url(r'^articles/([0-9]{4})/$', views.year_archive, name='news-year-archive'),
    #...
]
```

template

```html
<a href="{% url 'news-year-archive' 2012 %}">2012 Archive</a>

<ul>
{% for yearvar in year_list %}
<li><a href="{% url 'news-year-archive' yearvar %}">{{ yearvar }} Archive</a></li>
{% endfor %}
</ul>
```

views

```python
# python
from django.core.urlresolvers import reverse
from django.http import HttpResponseRedirect

def redirect_to_year(request):
    # ...
    year = 2006
    # ...
    return HttpResponseRedirect(reverse('news-year-archive', args=(year,)))
```

- 2.0


URLconf

```python
from django.urls import path

from . import views

urlpatterns = [
    #...
    path('articles/<int:year>/', views.year_archive, name='news-year-archive'),
    #...
]
```

template

```html
<a href="{% url 'news-year-archive' 2012 %}">2012 Archive</a>
{# Or with the year in a template context variable: #}
<ul>
{% for yearvar in year_list %}
<li><a href="{% url 'news-year-archive' yearvar %}">{{ yearvar }} Archive</a></li>
{% endfor %}
</ul>
```

views

```python
# python
from django.urls import reverse
from django.http import HttpResponseRedirect

def redirect_to_year(request):
    # ...
    year = 2006
    # ...
    return HttpResponseRedirect(reverse('news-year-archive', args=(year,)))
```

## 命名/命名空间

### 概述

- 命名

为了完成上面例子中的URL 反查，你将需要使用**命名的URL 模式**。URL 的名称使用的字符串可以包含任何你喜欢的字符。并不仅限于合法的Python 名称。

当命名你的URL 模式时，请确保使用的名称不会与其它应用中名称冲突。如果你的URL 模式叫做`comment`，而另外一个应用中也有一个同样的名称，当你在模板中使用这个名称的时候不能保证将插入哪个URL。

在URL 名称中加上一个前缀，比如应用的名称，将减少冲突的可能。我们建议使用`myapp-comment` 而不是`comment`

示例

```python
# 1.8
from django.conf.urls import url
from . import views

urlpatterns = [
    #...
    url(r'^articles/([0-9]{4})/$', views.year_archive, name='news-year-archive'),
    #...
]

# 2.0
from django.urls import path
from . import views

urlpatterns = [
    #...
    path('articles/<int:year>/', views.year_archive, name='news-year-archive'),
    #...
]
```

- 命名空间

因为一个应用的多个实例共享相同的命名URL，命名空间提供了一种区分这些命名URL 的方法

在一个站点上，正确使用URL命名空间的Django 应用可以部署多次。例如，`django.contrib.admin` 具有一个`AdminSite`类，它允许你很容易地*部署多个管理站点的实例*。在下面的例子中，我们将讨论在两个不同的地方部署教程中的polls应用，这样我们可以为两种不同的用户（作者和发布者）提供相同的功能。

一个URL命名空间有两部分，都是字符串

```python
# 应用命名空间
它表示正在部署的应用的名称。一个应用的每个实例具有相同的应用命名空间。例如，可以预见Django 的管理站点的应用命名空间是'admin'。
# 实例命名空间
它表示应用的一个特定的实例。实例的命名空间在你的全部项目中应该是唯一的。但是，一个实例的命名空间可以和应用的命名空间相同。它用于表示一个应用的默认实例。例如，Django 管理站点实例具有一个默认的实例命名空间'admin'。
```

URL 的命名空间使用`':'` 操作符指定。例如，管理站点应用的主页使用`'admin:index'`。它表示`'admin'` 的一个命名空间和`'index'` 的一个命名URL。

命名空间也可以嵌套。命名URL`'sports:polls:index'`将在命名空间`'polls'`中查找`'index'`，而poll 定义在顶层的命名空间`'sports'` 中。

### 反查带命名空间的URL

当解析一个带命名空间的URL（例如`'polls:index'`）时，Django 将切分名称为多个部分，然后按下面的步骤查找

1. 首先，Django 查找匹配的应用命名空间(在这个例子中为'polls'）。这将得到该应用实例的一个列表。

2. 如果有一个当前应用被定义，Django 将查找并返回那个实例的URL 解析器。当前应用可以通过请求上的一个属性指定。预期会具有多个部署的应用应该设置正在处理的request 的current_app 属性。
当前应用还可以通过reverse() 函数的一个参数手工设定。

3. 如果没有当前应用。Django 将查找一个默认的应用实例。默认的应用实例是实例命名空间 与应用命名空间 一致的那个实例（在这个例子中，polls 的一个叫做'polls' 的实例）。

4. 如果没有默认的应用实例，Django 将挑选该应用最后部署的实例，不管实例的名称是什么。

5. 如果提供的命名空间与第1步中的应用命名空间 不匹配，Django 将尝试直接将此命名空间作为一个实例命名空间查找。

如果有嵌套的命名空间，将为命名空间的每个部分重复调用这些步骤直至剩下视图的名称还未解析。然后该视图的名称将被解析到找到的这个命名空间中的一个URL。

- 示例

为了显示该解决方案的实际作用，请考虑本教程中民意测验应用程序的两个实例的示例：一个称为“ author-polls”，另一个称为“ publisher-polls”。假设我们已经增强了该应用程序，以便在创建和显示民意测验时考虑实例名称空间。

1.8

```python
# URLconf

# urls.py
from django.conf.urls import include, url

urlpatterns = [
    url(r'^author-polls/', include('polls.urls', namespace='author-polls', app_name='polls')),
    url(r'^publisher-polls/', include('polls.urls', namespace='publisher-polls', app_name='polls')),
]
# polls/urls.py
from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^(?P<pk>\d+)/$', views.DetailView.as_view(), name='detail'),
    ...
]
```

2.0

```python
# urls.py
from django.urls import include, path

urlpatterns = [
    path('author-polls/', include('polls.urls', namespace='author-polls')),
    path('publisher-polls/', include('polls.urls', namespace='publisher-polls')),
]
# polls/urls.py
from django.urls import path
from . import views

app_name = 'polls'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    ...
]
```
使用此设置，可以进行以下查找

1. 如果其中一个实例是当前实例 —— 如果我们正在渲染'author-polls' 实例的detail 页面 —— 'polls:index' 将解析成'author-polls' 实例的主页面；例如下面两个都将解析成"/author-polls/"。

在基于类的视图的方法

```python
reverse('polls:index', current_app=self.request.resolver_match.namespace)
```

在模板中
```
{% url 'polls:index' %}
```
> 注意，

1.8需要在模板中的反查需要添加request的current_app 属性

```python
def render_to_response(self, context, **response_kwargs):
    self.request.current_app = self.request.resolver_match.namespace
    return super(DetailView, self).render_to_response(context, **response_kwargs)
```

2. 如果没有当前实例——如果我们在站点的其它地方渲染一个页面 —— 'polls:index' 将解析到最后注册的polls的一个实例。因为没有默认的实例（命名空间为'polls'的实例），将使用注册的polls 的最后一个实例。它将是'publisher-polls'，因为它是在urlpatterns中最后一个声明的。

3. 'author-polls:index' 将永远解析到 'author-polls' 实例的主页（'publisher-polls' 类似）。

如果还有一个默认的实例——例如，一个名为'polls'的实例 —— 上面例子中唯一的变化是当没有当前实例的情况（上述第二种情况）。在这种情况下 'polls:index' 将解析到默认实例而不是urlpatterns 中最后声明的实例的主页

### URL命名空间和被包含的URLconf

被包含的URLconf 的命名空间可以通过两种方式指定

- 1.8

在构造你的URL模式时，你可以提供应用和实例的命名空间给`include()`作为参数。这将包含polls.urls中定义的URL 到应用命名空间 `polls`中，其实例命名空间`author-polls`

```python
# 方法一
url(r'^polls/', include('polls.urls', namespace='author-polls', app_name='polls')),
```

可以include一个包含嵌套命名空间数据的对象,如果你include()一个url() 实例的列表，那么该对象中包含的URL 将添加到全局命名空间。然而，你还可以include() 一个3个元素的元组，这会包含命名的URL模式进入到给定的应用和实例命名空间中

```python
# 方法二
polls_patterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^(?P<pk>\d+)/$', views.DetailView.as_view(), name='detail'),
]
url(r'^polls/', include((polls_patterns, 'polls', 'author-polls'))),  # (<list of url() instances>, <application namespace>, <instance namespace>)
```

- 2.0

首先，您可以在包含的URLconf模块中将app_name属性设置为与urlpatterns属性相同的级别。您必须将实际模块或对该模块的字符串引用传递给`include()`，而不是urlpatterns本身的列表。

```python
# 方法一
# polls/urls.py
from django.urls import path
from . import views

app_name = 'polls'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
    ...
]

# urls.py
from django.urls import include, path

urlpatterns = [
    path('polls/', include('polls.urls')),
]
```

`polls.urls`中定义的URL将具有应用程序名称空间`polls`。

其次，您可以包括一个包含嵌入式名称空间数据的对象。如果你`include()`一系列`path()`或`re_path()`实例，则该对象中包含的URL将被添加到全局名称空间中。但是，您还可以`include()`一个2元组，其中包含
```python
(<list of path()/re_path() instances>, <application namespace>)
```
例如
```python
# 方法二
from django.urls import include, path
from . import views

polls_patterns = ([
    path('', views.IndexView.as_view(), name='index'),
    path('<int:pk>/', views.DetailView.as_view(), name='detail'),
], 'polls')

urlpatterns = [
    path('polls/', include(polls_patterns)),
]
```

这会将提名的URL模式包括到给定的应用程序名称空间中。

可以使用`include()`的namespace参数指定实例名称空间。如果未指定实例名称空间，它将默认为包含的URLconf的应用程序名称空间。这意味着它将也是该名称空间的默认实例。