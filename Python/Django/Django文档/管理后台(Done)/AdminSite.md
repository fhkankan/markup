## AdminSite

```
class AdminSite(name='admin')
```

Django管理站点由`django.contrib.admin.sites.AdminSite`的实例表示；默认情况下，此类的实例将创建为`django.contrib.admin.site`，您可以使用它注册模型和`ModelAdmin`实例。

当构造`AdminSite` 的实例时，你可以使用`name` 参数给构造函数提供一个唯一的实例名称。 这个实例名称用于标识实例，尤其是[reversing admin URLs](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#admin-reverse-urls) 的时候。 如果没有提供实例的名称，将使用默认的实例名称`admin`。 有关自定义[`AdminSite`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite) 类的示例，请参见[Customizing the AdminSite class](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#customizing-adminsite`

### AdmiSite属性

如[Overriding admin templates](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#admin-overriding-templates)中所述，模板可以覆盖或扩展基础的Admin 模板。

`AdminSite.site_header`

每个Admin 页面顶部的文本，形式为`<h1>`（字符串）。 默认为 “Django administration”。

`AdminSite.site_title`

每个Admin 页面底部的文本，形式为`<title>`（字符串）。 默认为“Django site admin”。

`AdminSite.site_url`

每个Admin 页面顶部"View site" 链接的URL。 默认情况下，`site_url` 为`/`。 设置为`None` 可以删除这个链接。对于在子路径上运行的站点，[`each_context()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite.each_context)方法会检查当前请求是否具有`request.META['SCRIPT_NAME']`设置并使用该值，如果`site_url`未设置为`/`以外的其他内容。**在Django更改1.10：**上一段描述的`SCRIPT_NAME`支持已添加。

`AdminSite.index_title`

Admin 主页顶部的文本（一个字符串）。 默认为 “Site administration”。

`AdminSite.index_template`

Admin 站点主页的视图使用的自定义模板的路径。

`AdminSite.app_index_template`

Admin 站点app index 的视图使用的自定义模板的路径。

`AdminSite.empty_value_display`

用于在管理站点更改列表中显示空值的字符串。 默认为破折号。 通过在字段上设置`empty_value_display`属性，也可以在每个`ModelAdmin`以及`ModelAdmin`中的自定义字段上覆盖该值。 有关示例，请参见[`ModelAdmin.empty_value_display`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.empty_value_display)。

`AdminSite.login_template`

Admin 站点登录视图使用的自定义模板的路径。

`AdminSite.login_form`

Admin 站点登录视图使用的[`AuthenticationForm`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.forms.AuthenticationForm) 的子类。

`AdminSite.logout_template`

Admin 站点登出视图使用的自定义模板的路径。

`AdminSite.password_change_template`

Admin 站点密码修改视图使用的自定义模板的路径。

`AdminSite.password_change_done_template`

Admin 站点密码修改完成视图使用的自定义模板的路径。

### AdminSite方法

`AdminSite.each_context(request)`

返回一个字典，包含将放置在Admin 站点每个页面的模板上下文中的变量。包含以下变量和默认值：`site_header`：[`AdminSite.site_header`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite.site_header)`site_title`：[`AdminSite.site_title`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite.site_title)`site_url`：[`AdminSite.site_url`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite.site_url)`has_permission`：[`AdminSite.has_permission()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite.has_permission)`available_apps`：从当前用户可用的[application registry](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/applications.html)中的应用程序列表。 列表中的每个条目都是表示具有以下密钥的应用程序的dict：`app_label`：应用程序标签`app_url`：管理员中的应用程序索引的URL`has_module_perms`：一个布尔值，表示当前用户是否允许显示和访问模块的索引页面`models`：应用程序中可用的模型列表每个模型都是具有以下键的dict：`object_name`：模型的类名`name`：复数名称的模型`perms`：a `dict` tracking `add`，`change`和`delete` permissions`admin_url`：admin changelist模型的URL`add_url`：添加新模型实例的admin URL

`AdminSite.has_permission(request)`

对于给定的`True`，如果用户有权查看Admin 网站中的至少一个页面，则返回 `HttpRequest`。 默认要求[`User.is_active`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_active) 和[`User.is_staff`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/auth.html#django.contrib.auth.models.User.is_staff) 都为`True`。

`AdminSite.register(model_or_iterable, admin_class=None, **options)`

使用给定的`admin_class`注册给定的模型类（或模型类组成的可迭代对象）。 `admin_class`默认为[`ModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin)（默认的管理后台选项）。 如果给出了关键字参数 — 例如`list_display` — 它们将作为选项应用于admin_class。如果模型是抽象的，则引发[`ImproperlyConfigured`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.core.exceptions.ImproperlyConfigured)。 如果模型已经注册则引发`django.contrib.admin.sites.AlreadyRegistered`。

### 将`AdminSite`的实例挂接到URLconf中

设置Django管理后台的最后一步是放置你的`AdminSite`到你的URLconf中。 将一个给定的URL指向`AdminSite.urls`方法就可以做到。 没有必要使用[`include()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/urls.html#django.conf.urls.include)。

在下面的示例中，我们注册默认的`AdminSite`实例`django.contrib.admin.site`到URL `/admin/`。

```python
# urls.py

// 1.11
from django.conf.urls import url
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
]

// 2.0
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
```

### 定制AdminSite类

如果你想要建立你自己的具有自定义行为Admin 站点，你可以自由地子类化`AdminSite` 并重写或添加任何你喜欢的东西。 你只需创建`AdminSite` 子类的实例（方式与你会实例化任何其它Python 类相同） 并注册你的模型和`ModelAdmin` 子类与它而不是默认的站点。 最后，更新`myproject/urls.py` 来引用你的[`AdminSite`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite) 子类。

```python
# myapp/ admin.py
from django.contrib.admin import AdminSite

from .models import MyModel

class MyAdminSite(AdminSite):
    site_header = 'Monty Python administration'

admin_site = MyAdminSite(name='myadmin')
admin_site.register(MyModel)
```

```python
# MyProject/urls.py
// 1.11
from django.conf.urls import url

from myapp.admin import admin_site

urlpatterns = [
    url(r'^myadmin/', admin_site.urls),
]

//2.0
from django.urls import path
urlpatterns = [
    path('myadmin/', admin_site.urls),
]
```

注意，当使用你自己的`admin` 实例时，你可能不希望自动发现`AdminSite` 模块，因为这将导入`admin` 模块到你的每个`myproject.admin` 模块中 。 这时，你需要将`'django.contrib.admin'` 而不是`'django.contrib.admin.apps.SimpleAdminConfig'` 放置在你的[`INSTALLED_APPS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-INSTALLED_APPS) 设置中。

### 相同的URLconf 中的多个管理站点

在同一个Django供电的网站上创建管理站点的多个实例很容易。 只需要创建`AdminSite` 的多个实例并将每个实例放置在不同的URL 下。

在下面的示例中，`AdminSite` 和`/advanced-admin/` 分别使用`/basic-admin/` 的`myproject.admin.basic_site` 实例和`myproject.admin.advanced_site` 实例表示不同版本的Admin 站点：

```python
# urls.py
// 1.11
from django.conf.urls import url
from myproject.admin import basic_site, advanced_site

urlpatterns = [
    url(r'^basic-admin/', basic_site.urls),
    url(r'^advanced-admin/', advanced_site.urls),
]

// 2.0
from django.urls import path

urlpatterns = [
    path('basic-admin/', basic_site.urls),
    path('advanced-admin/', advanced_site.urls),
]
```

`AdminSite` 实例的构造函数中接受一个单一参数用做它们的名字，可以是任何你喜欢的东西。 此参数将成为[reversing them](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#admin-reverse-urls) 时URL 名称的前缀。 只有在你使用多个`AdminSite` 时它才是必要的。

### 将视图添加到管理站点

与[`ModelAdmin`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin)一样，[`AdminSite`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.AdminSite)提供了一个[`get_urls()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_urls)方法，可以重写该方法以定义网站的其他视图。 要向您的管理网站添加新视图，请扩展基本[`get_urls()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/admin/index.html#django.contrib.admin.ModelAdmin.get_urls)方法，为新视图添加模式。

> 注
您呈现的任何使用管理模板的视图或扩展基本管理模板，应在渲染模板之前设置`request.current_app`。 It should be set to either `self.name` if your view is on an `AdminSite` or `self.admin_site.name` if your view is on a `ModelAdmin`.

### 添加密码重置功能

您可以通过在URLconf中添加几行来将密码重置功能添加到管理站点。 具体操作就是加入下面四个正则规则。

```python
from django.contrib.auth import views as auth_views

// 1.11
url(
    r'^admin/password_reset/$',
    auth_views.PasswordResetView.as_view(),
    name='admin_password_reset',
),
url(
    r'^admin/password_reset/done/$',
    auth_views.PasswordResetDoneView.as_view(),
    name='password_reset_done',
),
url(
    r'^reset/(?P<uidb64>[0-9A-Za-z_\-]+)/(?P<token>.+)/$',
    auth_views.PasswordResetConfirmView.as_view(),
    name='password_reset_confirm',
),
url(
    r'^reset/done/$',
    auth_views.PasswordResetCompleteView.as_view(),
    name='password_reset_complete',
),

// 2.0
path(
    'admin/password_reset/',
    auth_views.PasswordResetView.as_view(),
    name='admin_password_reset',
),
path(
    'admin/password_reset/done/',
    auth_views.PasswordResetDoneView.as_view(),
    name='password_reset_done',
),
path(
    'reset/<uidb64>/<token>/',
    auth_views.PasswordResetConfirmView.as_view(),
    name='password_reset_confirm',
),
path(
    'reset/done/',
    auth_views.PasswordResetCompleteView.as_view(),
    name='password_reset_complete',
),
```

（假设您已在`admin/`添加了管理员，并要求您在包含管理应用程序的行之前将`^admin/`开头的网址）。

如果存在`admin_password_reset`命名的URL，则会在密码框下的默认管理登录页面上显示“忘记了您的密码？”链接。

