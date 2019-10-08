# “站点”框架

Django 原生带有一个可选的“sites”框架。 这是将对象和功能与特定网站相关联的一个钩子，它是Django动力网站的域名和“详细”名称的保留位置。

如果你的Django程序 不只为一个站点提供支持，而且你需要区分这些不同的站点，你就可以使用它。

Sites 框架主要依据一个简单的模型：

- *class* `models.``Site`

  用于存储网站的`domain`和`name`属性的模型。`domain`与网站相关联的完全限定域名。 例如，`www.example.com`。`name`该网站是人类可读的“详细”名称。

[`SITE_ID`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-SITE_ID) 设置指定与特定的设置文件关联的[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 对象在数据库中ID。 如果省略该设置，[`get_current_site()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.shortcuts.get_current_site) 函数将会通过比较[`domain`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site.domain) 与[`request.get_host()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest.get_host) 方法中得到的主机名，来得到当前的Site。

怎样使用取决于你，但是django自动的在几个方面通过一些简单的约定使用它。



## Example usage

为什么要使用Sites 框架？ 这点通过实例来理解的效果最好



### 将内容与多个站点相关联

通过Django开发的站点[LJWorld.com](http://www.ljworld.com/) 和[Lawrence.com](http://www.lawrence.com/) 是位于Lawrence, Kansas 的同一家机构Lawrence Journal-World newspaper 运营的。 LJWorld.com 关注新闻，而Lawrence.com 关注当地的环境问题。 但是有时编辑需要发布同一篇文章到*两个*站点。

解决问题的天真的方法是要求现场生产者发布两次相同的故事：一次是LJWorld.com，另一次是Lawrence.com。 但这是很低效的行为，而且在数据库中必须存储同一内容很多次（多副本存储，浪费资源）。

最好的解决方法很简单：两个站点用相同的文章数据库，一篇文章可以关联一个或者多个站点。 用Django 模型的术语，它通过`Article` 模型的一个[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField)表示：

```
from django.db import models
from django.contrib.sites.models import Site

class Article(models.Model):
    headline = models.CharField(max_length=200)
    # ...
    sites = models.ManyToManyField(Site)
```

这很快很好的完成了几件事：

- 它使得站点编辑者利用一个接口(Django admin)编辑多站点上的所有内容。

- 这意味着同一个故事不必在数据库中发布两次；它在数据库中只有一个记录。

- 对于两个站点，开发者可以使用相同的Django 视图代码。 显示内容的视图代码需要检查，以确保请求的内容属于当前的站点。 就像下面一样:

  ```
  from django.contrib.sites.shortcuts import get_current_site
  
  def article_detail(request, article_id):
      try:
          a = Article.objects.get(id=article_id, sites__id=get_current_site(request).id)
      except Article.DoesNotExist:
          raise Http404("Article does not exist on this site")
      # ...
  ```



### 将内容与单个站点关联

类似地，你可以用[`ForeignKey`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey) 关联一个模型到[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 模型实现多对一关系。

例如，一篇文章只允许在一个单独的站点，你应该像这样用模型：

```
from django.db import models
from django.contrib.sites.models import Site

class Article(models.Model):
    headline = models.CharField(max_length=200)
    # ...
    site = models.ForeignKey(Site, on_delete=models.CASCADE)
```

这个好处和上节描述的好处是相同的。



### 从视图钩入当前站点

你可以在Django 视图中使用Sites 框架基于正在调用的视图所在的Site 实现特定的功能。 像这样：

```
from django.conf import settings

def my_view(request):
    if settings.SITE_ID == 3:
        # Do something.
        pass
    else:
        # Do something else.
        pass
```

当然，这样硬编码Site ID 比较丑陋。 这种硬编码是你最需要尽快修复的。 完成这件事情的更清洁的方法是检查当前站点的域名：

```
from django.contrib.sites.shortcuts import get_current_site

def my_view(request):
    current_site = get_current_site(request)
    if current_site.domain == 'foo.com':
        # Do something
        pass
    else:
        # Do something else.
        pass
```

它还有一个优点是检查Sites 框架是否安装，如果没有安装将返回一个 [`RequestSite`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.requests.RequestSite) 实例。

如果你不能访问request 对象，你可以使用[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 模型管理器的`get_current()` 方法。 此时，你需要确保你的设置文件包含[`SITE_ID`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-SITE_ID) 设置。 下面的示例与前面的示例等同：

```
from django.contrib.sites.models import Site

def my_function_without_request():
    current_site = Site.objects.get_current()
    if current_site.domain == 'foo.com':
        # Do something
        pass
    else:
        # Do something else.
        pass
```



### 获取当前域显示

LJWorld.com 和Lawrence.com 都具有邮件通知功能，它让读者注册以在新闻发生时获得通知。 这很简单：读者通过网页表单注册，然后立即收到一封邮件说 “感谢您的订阅”。

将这个注册过程的代码实现两次是低效而冗余的，所以这两个站点在后台使用相同的代码。 但是每个Site 的“感谢您的订阅”的通知需要不同。 通过使用[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 对象，我们可以抽象这个通知并利用当前Site 的[`name`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site.name) 和[`domain`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site.domain) 的值。

下面是该表单处理视图的一个例子：

```
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import send_mail

def register_for_newsletter(request):
    # Check form values, etc., and subscribe the user.
    # ...

    current_site = get_current_site(request)
    send_mail(
        'Thanks for subscribing to %s alerts' % current_site.name,
        'Thanks for your subscription. We appreciate it.\n\n-The %s team.' % (
            current_site.name,
        ),
        'editor@%s' % current_site.domain,
        [user.email],
    )

    # ...
```

在Lawrence.com上，这封电子邮件的主题是“感谢订阅lawrence.com警报”。在LJWorld.com上，电子邮件的主题是“感谢订阅LJWorld.com警报”。同样的电子邮件消息体。

注意，更加灵活（但是更沉重）的方法是使用Django 的模板系统。 假设Lawrence.com 和LJWorld.com 具有不同的模板目录（[`DIRS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-TEMPLATES-DIRS)），你可以很容易地根据模板系统写出：

```
from django.core.mail import send_mail
from django.template import loader, Context

def register_for_newsletter(request):
    # Check form values, etc., and subscribe the user.
    # ...

    subject = loader.get_template('alerts/subject.txt').render(Context({}))
    message = loader.get_template('alerts/message.txt').render(Context({}))
    send_mail(subject, message, 'editor@ljworld.com', [user.email])

    # ...
```

在这种情况下，你必须为LJWorld.com 和Lawrence.com 模板目录都创建`subject.txt` 和`message.txt` 模板文件。 它更灵活，但是也更复杂。

尽可能地发掘[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 对象的用法以删除不需要的复杂性和冗余是个不错的主意。



### 获取当前域的完整URL 

Django 的`get_absolute_url()` 可以很方便地获得对象不带域名的URL，但是某些情况下，你可能想显示完整的URL，带有`http://`和域名以及其它部分。 要实现这点，你可以使用Sites 框架。 一个简单的示例：

```
>>> from django.contrib.sites.models import Site
>>> obj = MyModel.objects.get(id=3)
>>> obj.get_absolute_url()
'/mymodel/objects/3/'
>>> Site.objects.get_current().domain
'example.com'
>>> 'https://%s%s' % (Site.objects.get_current().domain, obj.get_absolute_url())
'https://example.com/mymodel/objects/3/'
```



## 启用站点框架

按照以下步骤启用Sites 框架：

1. 添加`'django.contrib.sites'` 到你的[`INSTALLED_APPS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-INSTALLED_APPS) 设置中。

2. 定义[`SITE_ID`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-SITE_ID) 设置：

   ```
   SITE_ID = 1
   ```

3. 运行[`migrate`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/django-admin.html#django-admin-migrate)。

`example.com` 注册一个[`post_migrate`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/signals.html#django.db.models.signals.post_migrate) 信号处理器，它创建一个默认的Site`django.contrib.sites`，其域名为`example.com`。 在Django 创建测试数据库之后，也会创建该Site。 你可以使用[data migration](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/migrations.html#data-migrations)来为你的项目设置正确的name 和domain。

为了在线上环境中启用多个Site，你应该为每个`SITE_ID` 创建一个单独的设置文件（可以从一个共同的设置文件导入，以避免重复共享的配置），然后为每个Site 指定合适的[`DJANGO_SETTINGS_MODULE`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/settings.html#envvar-DJANGO_SETTINGS_MODULE)。



## 缓存当前的`Site`对象

因为当前站点储存在数据库,每一次调用 `Site.objects.get_current()`都会导致数据库查询。 但是Django还是比这个聪明滴, 当前站点被放在缓存当中了, 所以后续的调用返回的都是缓存的数据而不是直接查询数据库。

如果出于一些原因你想要强制用数据库查询, 你可以告诉Django清除缓存，用下面这个方法 `Site.objects.clear_cache()`:

```
# First call; current site fetched from database.
current_site = Site.objects.get_current()
# ...

# Second call; current site fetched from cache.
current_site = Site.objects.get_current()
# ...

# Force a database query for the third call.
Site.objects.clear_cache()
current_site = Site.objects.get_current()
```



## `CurrentSiteManager` 

- *类* `经理。`` CurrentSiteManager T0> `

  

如果 [`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 在你的应用中非常的关键， 你可以考虑用 [`CurrentSiteManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.managers.CurrentSiteManager) 在你的模型中(s). 它是一个 model[manager](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/managers.html)用来自动过滤，留下只与当前站点有关的数据查询 [`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site).

必须[`SITE_ID`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-SITE_ID)

`CurrentSiteManager` 只有在你定义了[`SITE_ID`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-SITE_ID) 在setting 中才起作用。

使用 [`CurrentSiteManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.managers.CurrentSiteManager) ，你只要直接把他添加到你的model 中。 像这样：

```
from django.db import models
from django.contrib.sites.models import Site
from django.contrib.sites.managers import CurrentSiteManager

class Photo(models.Model):
    photo = models.FileField(upload_to='photos')
    photographer_name = models.CharField(max_length=100)
    pub_date = models.DateField()
    site = models.ForeignKey(Site, on_delete=models.CASCADE)
    objects = models.Manager()
    on_site = CurrentSiteManager()
```

通过这个model, `Photo.on_site.all()` 将会返回所有在数据库中的 `Photo`对象，但是 `Photo.objects.all()`只会返回 与当前site相关的`Photo`对象, 这是根据 [`SITE_ID`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-SITE_ID) 在setting的设置。

换句话说，这两种表达方式是等价的:

```
Photo.objects.filter(site=settings.SITE_ID)
Photo.on_site.all()
```

[`CurrentSiteManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.managers.CurrentSiteManager)是如何知道哪个`Photo`字段是 [`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site)的? 通常来说， [`CurrentSiteManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.managers.CurrentSiteManager)查找一个 [`ForeignKey`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey) 它的名字叫`sites` 或者是一个 [`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField)字段 ，叫做 `site`来筛选出. 如果你用名字不叫`sites` or `site`的字段来表示一个与[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site)对象相关联,，那么你就需要在你的model中显示得传递自定义的字段名给[`CurrentSiteManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.managers.CurrentSiteManager)。 下面的model, 它有一个字段叫做 `publish_on`, 说明了这个问题：

```
from django.db import models
from django.contrib.sites.models import Site
from django.contrib.sites.managers import CurrentSiteManager

class Photo(models.Model):
    photo = models.FileField(upload_to='photos')
    photographer_name = models.CharField(max_length=100)
    pub_date = models.DateField()
    publish_on = models.ForeignKey(Site, on_delete=models.CASCADE)
    objects = models.Manager()
    on_site = CurrentSiteManager('publish_on')
```

如果你尝试使用[`CurrentSiteManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.managers.CurrentSiteManager) 并且传递了一个并不存在的字段名称给他, Django 就会引发一个 `ValueError`.

最后, 注意你可能会想要保持一个正常的 (non-site-specific) `Manager` 在你的model, 虽然你使用了 [`CurrentSiteManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.managers.CurrentSiteManager). 如[manager documentation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/managers.html)中所述，如果手动定义管理员，则Django将不会创建自动`对象 = models.Manager()`经理。 还要注意，Django的某些部分 - 即Django管理站点和通用视图 - 使用模型中的*第一个*定义的管理员，因此如果您希望管理员站点访问所有对象在您定义之前，在模型中放置`对象 = models.Manager()` [`CurrentSiteManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.managers.CurrentSiteManager)



## 站点中间件

如果你经常使用这个模式：

```
from django.contrib.sites.models import Site

def my_view(request):
    site = Site.objects.get_current()
    ...
```

这里有些方法可以防止这种重复调用。 将[`django.contrib.sites.middleware.CurrentSiteMiddleware`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/middleware.html#django.contrib.sites.middleware.CurrentSiteMiddleware)添加到[`MIDDLEWARE`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-MIDDLEWARE)。 中间件设置 `site` 属性给每一次request对象, 所以你可以用 `request.site` 来获取当前site。



## Django如何使用站点框架

虽然不强制要求你的网站使用site框架，但是我们鼓励你使用它，因为在一些地方Django利用它。 即使你的Django只在支持单个站点, 你也应该花两秒时间来给你的站点对象创建`name` 和`domain`,并且设置它的ID在你的 [`SITE_ID`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-SITE_ID) setting中。

下面是Django 如何使用sites framework:

- 在 [`redirects framework`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/redirects.html#module-django.contrib.redirects),每一个redirect都和特定的站点相关联。 当Django查找一个 redirect, 它就考虑在当前的站点中查找。
- 在 [`flatpages framework`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/flatpages.html#module-django.contrib.flatpages), 每一个flatpage 都被关联到特定的站点。 当一个 flatpage 被创建， 你指定它的 [`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site),并且[`FlatpageFallbackMiddleware`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/flatpages.html#django.contrib.flatpages.middleware.FlatpageFallbackMiddleware) 在返回flatpages 中检查当前站以显示。
- 在 [`syndication framework`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/syndication.html#module-django.contrib.syndication)中, 模板的 `title` and `description` 自动访问变量`{{ site }}`, 这个 [`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 代表当前站点的站点对象。. 此外，挂钩提供项URL将使用当前 [`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site)对象的`domain`，如果你不指定一个完全合格的域名。
- In the [`authentication framework`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/index.html#module-django.contrib.auth), [`django.contrib.auth.views.LoginView`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/auth/default.html#django.contrib.auth.views.LoginView) passes the current [`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) name to the template as `{{ site_name }}`.
- 快捷视图 (`django.contrib.contenttypes.views.shortcut`) 使用当前[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site)对象的的域 计算对象的URL。
- 在管理框架, “view on site” 链接使用当前 [`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 算出将重定向的域名.



## `RequestSite`对象

一些 [django.contrib](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/index.html)应用有利用到 sites framework 但是它们的架构不会*require* sites framework必须安装在你的数据库中。 有些人不想, 或者不能安装site framework所要求的*able*在他们的数据库中。) 出于这种情况，framework 提供了一个 [`django.contrib.sites.requests.RequestSite`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.requests.RequestSite)类，当你数据支持的站点框架不可用的时候做一个回退

- *类* `要求。`` RequestSite T0> `

  一个共享[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site)（即，它具有`domain`和`name`属性）的主接口，但从Django [`HttpRequest`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest)对象，而不是从数据库。`__初始化__ T0>（*请求 T1>）*`将`domain`和`name`属性设置为[`get_host()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest.get_host)的值。

除了其[`__init__()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.requests.RequestSite.__init__)方法采用[`HttpRequest`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest)对象，[`RequestSite`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.requests.RequestSite)对象具有与正常[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 它可以通过查看请求的域来推断`domain`和`name`。 它具有`delete()`和`save()`方法来匹配[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site)的接口，但是方法产生[`NotImplementedError`](https://docs.python.org/3/library/exceptions.html#NotImplementedError) 。



## `get_current_site`快捷方式

最后,为了避免重复的回退代码，site framework 提供了一个 [`django.contrib.sites.shortcuts.get_current_site()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.shortcuts.get_current_site) 功能。

- `快捷键。`` get_current_site T0>（*请求 T1>）*`

  这是函数是用来检查`django.contrib.sites` 是否安装并且返回一个基于request的[`Site`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.models.Site) 对象或者一个[`RequestSite`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.requests.RequestSite) 对象。 如果没有定义[`SITE_ID`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-SITE_ID)设置，它会根据[`request.get_host()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest.get_host)查找当前站点。当主机头有明确指定的端口时，域和端口可以由[`request.get_host()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html#django.http.HttpRequest.get_host)返回。 `example.com:80` 在这种情况下，如果查找失败，因为主机与数据库中的记录不匹配，则将剥离该端口，并仅使用域部分重试查找。 这不适用于[`RequestSite`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#django.contrib.sites.requests.RequestSite)，它将始终使用未修改的主机。

### [目录](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/contents.html)

- “网站”框架
  - 示例
    - [关联内容到多个站点](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#associating-content-with-multiple-sites)
    - [关联内容到单独的站点](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#associating-content-with-a-single-site)
    - [在视图中获得当前的Site](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#hooking-into-the-current-site-from-views)
    - [显示当前的域名](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#getting-the-current-domain-for-display)
    - [获取当前域名的url全路径](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#getting-the-current-domain-for-full-urls)
  - [启用Sites 框架](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#enabling-the-sites-framework)
  - [缓存当前的`Site`对象](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#caching-the-current-site-object)
  - [`CurrentSiteManager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#the-currentsitemanager)
  - [网站中间件](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#site-middleware)
  - [Django是如何使用的站点框架](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#how-django-uses-the-sites-framework)
  - [`RequestSite`对象](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#requestsite-objects)
  - [`get_current_site`快捷方式](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sites.html#get-current-site-shortcut)

### 浏览

- 上一页：[网站架构](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/sitemaps.html)
- 下一个：[`staticfiles` app](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/staticfiles.html)

### 你在这里：

- Django 1.11.6 文档
  - API参考
    - `contrib`包
      - “网站”框架

### 这一页

- [显示源](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_sources/ref/contrib/sites.txt)

### 快速搜索





### 最后更新：

2017年9月6日