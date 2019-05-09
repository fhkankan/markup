# 视图-视图函数

一个视图函数，简称*视图*，是一个简单的Python 函数，它接受Web请求并且返回Web响应。响应可以是一张网页的HTML内容，一个重定向，一个404错误，一个XML文档，或者一张图片. . . 是任何东西都可以。无论视图本身包含什么逻辑，都要返回响应。代码写在哪里也无所谓，只要它在你的Python目录下面。

## 简单视图

```python
from django.http import HttpResponse
import datetime

def current_datetime(request):
    now = datetime.datetime.now()  # 注意时区在settings中TIME_ZONE设置，默认America/Chicago
    html = "<html><body>It is now %s.</body></html>" % now
    return HttpResponse(html)
```

## 内置视图

###  开发环境中的文件服务器

```
static.serve(request, path, document_root, show_indexes=False)
```

在本地的开发环境中，除了你的项目中的静态文件，可能还有一些文件，出于方便，你希望让Django 来作为服务器。`serve()`视图可以用来作为任意目录的服务器。（该视图**不**能用于生产环境，应该只用于开发时辅助使用；在生产环境中你应该使用一个真实的前端Web 服务器来服务这些文件）

如：用户上传文档到`MEDIA_ROOT`中。`django.contrib.staticfiles` 用于静态文件且没有对用户上传的文件做处理，但是你可以通过在URLconf 中添加一些内容来让Django 作为`MEDIA_ROOT`的服务器：

```python
from django.conf import settings
from django.views.static import serve

# ... the rest of your URLconf goes here ...

if settings.DEBUG:
    urlpatterns += [
        url(r'^media/(?P<path>.*)$', serve, {
            'document_root': settings.MEDIA_ROOT,
        }),
   ]
    
# 注意
这里的代码片段假设你的MEDIA_URL的值为'/media/'。它将调用serve() 视图，传递来自URLconf 的路径和（必选的）document_root 参数。

因为定义这个URL 模式显得有些笨拙，Django 提供一个小巧的URL 辅助函数static()，它接收MEDIA_URL这样的参数作为前缀和视图的路径如'django.views.static.serve'。其它任何函数参数都将透明地传递给视图。
```

### 错误视图

- 404(page not found)

```
defaults.page_not_found(request, exception, template_name='404.html')
```

当你在一个视图中引发`Http404`时，Django 将加载一个专门的视图用于处理404 错误。默认为`django.views.defaults.page_not_found()` 视图，它产生一个非常简单的“Not Found” 消息或者渲染`404.html`模板，如果你在根模板目录下创建了它的话。

默认的404 视图将传递一个变量给模板：`request_path`，它是导致错误的URL。

需要注意

```python
- 如果Django在检测URLconf中的每个正则表达式后没有找到匹配的内容也将调用404视图。
- 404 视图会被传递一个RequestContext并且可以访问模板上下文处理器提供的变量（例如MEDIA_URL）。
- 如果DEBUG设置为True（在你的settings 模块中）,那么将永远不会调用404视图，而是显示你的URLconf 并带有 一些调试信息。
```

- 500(server error)

```
defaults.server_error(request, template_name='500.html')
```

在视图代码中出现运行时错误，Django 将执行特殊情况下的行为。如果一个视图导致异常，Django 默认情况下将调用`django.views.defaults.server_error` 视图，它产生一个非常简单的“Server Error” 消息或者渲染`500.html`，如果你在你的根模板目录下定义了它的话。

默认的500 视图不会传递变量给`500.html` 模板，且使用一个空`Context` 来渲染以减少再次出现错误的可能性。

如果`DEBUG`设置为`True`（在你的settings 模块中），那么将永远不会调用500 视图，而是显示回溯并带有一些调试信息。

- 403(HTTP Forbidden)

```
defaults.permission_denied(request, exception, template_name='403.html')
```

如果一个视图导致一个403 视图，那么Django 将默认调用`django.views.defaults.permission_denied`视图。

该视图加载并渲染你的根模板目录下的`403.html`，如果这个文件不存在则根据RFC 2616（HTTP 1.1 Specification）返回“403 Forbidden”文本。

```python
# django.views.defaults.permission_denied 通过PermissionDenied 异常触发
from django.core.exceptions import PermissionDenied

def edit(request, pk):
    if not request.user.is_staff:
        raise PermissionDenied  # 要拒绝访问一个视图
    # ...
```

- 400(bad request)

```
defaults.bad_request(request, exception, template_name='400.html')
```

当Django 中引发一个`SuspiciousOperation`时，它可能通过Django 的一个组件处理（例如重设会话的数据）。如果没有特殊处理，Django 将认为当前的请求是一个'bad request' 而不是一个server error。

`django.views.defaults.bad_request` 和`server_error`视图非常相似，除了返回400 状态码来表示错误来自客户端的操作。

`bad_request` 视图同样只是在`DEBUG` 为`False` 时使用。

## 返回错误

### 普通`HttpResponse`

```python
from django.http import HttpResponse

def my_view(request):
    # ...
    return HttpResponse('<h1>Page was found</h1>')  # 状态码200，返回html页面
    return HttpResponse(status=201)  # 直接返回状态码201
```

###  `HttpResponse`子类

| name                            | desc                                                         |
| ------------------------------- | ------------------------------------------------------------ |
| `HttpResponseRedirect`          | 重定向，返回一个found重定向，状态码302                       |
| `HttpResponsePermanentRedirect` | 重定向，返回一个永久重定向，状态码301                        |
| `HttpResponseNotModified`       | 构造函数不会有任何的参数，并且不应该向这个响应（response）中加入内容（content）。使用此选项可指定自用户上次请求以来尚未修改页面，状态代码304 |
| `HttpResponseBadRequest`        | 与`HttpResponse`的行为类似，但是状态码400                    |
| `HttpResponseNotFound`          | 与`HttpResponse`的行为类似，但是状态码404                    |
| `HttpResponseForbidden`         | 与`HttpResponse`的行为类似，但是状态码403                    |
| `HttpResponseNotAllowed`        | 与`HttpResponse`的行为类似，构造函数的第一个参数是必须的：一个允许使用的方法构成的列表（例如，`['GET', 'POST']`），但是状态码405 |
| `HttpResponseGone`              | 与`HttpResponse`的行为类似，但是状态码410                    |
| `HttpResponseServerError`       | 与`HttpResponse`的行为类似，但是状态码500                    |

示例

```python
from django.http import HttpResponseNotFound

def my_view(request):
    # ...
    return HttpResponseNotFound('<h1>Page not found</h1>')
```

### 内置异常视图

| name      | desc |
| --------- | ---- |
| `Http404` |      |

示例

```python
from django.http import Http404
from django.shortcuts import render_to_response
from polls.models import Poll

def detail(request, poll_id):
    try:
        p = Poll.objects.get(pk=poll_id)
    except Poll.DoesNotExist:
        raise Http404("Poll does not exist")  
    return render_to_response('polls/detail.html', {'poll': p})
  
# Django提供了Http404异常。如果你在视图函数中的任何地方抛出Http404异常，Django都会捕获它，并且带上HTTP404错误码返回你应用的标准错误页面
# 如果你在抛出Http404异常时提供了一条消息，当DEBUG为True时它会出现在标准404模板的展示中。你可以将这些消息用于调试；但他们通常不适用于404模板本身。
```

### 自定义错误视图

如果你需要任何自定义行为，重写它很容易。只要在你的URLconf中指定下面的处理器（在其他任何地方设置它们不会有效）。

```python
# handler404覆盖page_not_found()视图
handler404 = 'mysite.views.my_custom_page_not_found_view'
# handler500覆盖server_error()视图
handler500 = 'mysite.views.my_custom_error_view'
# handler403覆盖permission_denied()视图
handler403 = 'mysite.views.my_custom_permission_denied_view'
# handler404覆盖bad_request()视图
handler400 = 'mysite.views.my_custom_bad_request_view'
```

## 快捷函数

`django.shortcuts` 收集了“跨越” 多层MVC 的辅助函数和类。 换句话讲，这些函数/类为了方便，引入了可控的耦合。

| name                 | desc                                                         |
| -------------------- | ------------------------------------------------------------ |
| `render`             | 结合一个给定的模板和一个给定的上下文字典，并返回渲染后的 `HttpResponse ` |
| `render_to_response` | 根据一个给定的上下文字典渲染一个给定的目标，并返回渲染后的`HttpResponse` |
| `redirect`           | 为传递进来的参数返回`HttpResponseRedirect`给正确的URL        |
| `get_object_or_404`  | 在一个给定的模型管理器上调用`get()`，但是引发`Http404` 而不是模型的`DoesNotExist`异常 |
| `get_list_or_404`    | 返回一个给定模型管理器上`filter()`的结果，并将结果映射为一个列表，如果结果为空则返回`Http404` |

### render

```python
render(request, template_name, context=None, content_type=None, status=None, using=None)
# 结合一个给定的模板和一个给定的上下文字典，并返回一个渲染后的 HttpResponse对象。
# render() 与以一个强制使用RequestContext的context_instance 参数调用render_to_response() 相同。
# Django 不提供返回TemplateResponse 的快捷函数，因为TemplateResponse 的构造与render() 提供的便利是一个层次的。
```

参数

| name            | option | desc                                                         |
| --------------- | ------ | ------------------------------------------------------------ |
| `request`       | 必选   | 用于成response                                               |
| `template_name` | 必选   | 要使用的模板的完整名称或者模板名称的一个序列                 |
| `context`       | 可选   | 要添加到模板上下文的值的字典。默认情况下，这是一个空字典。如果字典中的值是可调用的，则视图将在呈现模板之前调用它。 |
| `content_type`  | 可选   | 生成的文档要使用的MIME 类型。默认为`DEFAULT_CONTENT_TYPE`设置的值。 |
| `status`        | 可选   | 响应的状态码。默认为`200`                                    |
| `using`         | 可选   | 用于加载模板使用的模板引擎的`名称`                           |

示例

```python
# 渲染模板myapp/index.html，MIIME 类型为application/xhtml+xml
from django.shortcuts import render

def my_view(request):
    # View code here...
    return render(request, 'myapp/index.html', {"foo": "bar"},
        content_type="application/xhtml+xml")
  
# 非快捷等价形式
# 1.8
from django.http import HttpResponse
from django.template import RequestContext, loader

def my_view(request):
    # View code here...
    t = loader.get_template('myapp/index.html')
    c = RequestContext(request, {'foo': 'bar'})
    return HttpResponse(t.render(c),
        content_type="application/xhtml+xml")
# 2.0
from django.http import HttpResponse
from django.template import loader

def my_view(request):
    # View code here...
    t = loader.get_template('myapp/index.html')
    c = {'foo': 'bar'}
    return HttpResponse(t.render(c, request), content_type='application/xhtml+xml')
```

### render_to_response

```python
render_to_response(template_name, context=None, content_type=None, status=None, using=None)

# 根据一个给定的上下文字典渲染一个给定的目标，并返回渲染后的HttpResponse
# 参数含义类似render
# 2.0中不推荐使用
```

示例

```python
# 渲染模板myapp/index.html，MIIME 类型为application/xhtml+xml
from django.shortcuts import render_to_response

def my_view(request):
    # View code here...
    return render_to_response('myapp/index.html', {"foo": "bar"}, content_type="application/xhtml+xml")
  
# 非快捷等价形式
from django.http import HttpResponse
from django.template import Context, loader

def my_view(request):
    # View code here...
    t = loader.get_template('myapp/index.html')
    c = Context({'foo': 'bar'})
    return HttpResponse(t.render(c), content_type="application/xhtml+xml")
```

### redirect

```python
redirect(to, permanent=False, *args, **kwargs)

# 为传递进来的参数返回`HttpResponseRedirect`给正确的URL

# 参数可以是
一个模型：将调用模型的get_absolute_url() 函数
一个视图，可以带有参数：将使用urlresolvers.reverse 来反向解析名称
一个绝对的或相对的URL，将原封不动的作为重定向的位置。
# 默认返回一个临时的重定向；传递permanent=True 可以返回一个永久的重定向。
```

示例

```python
# 通过传递一个对象；将调用get_absolute_url()方法来获取重定向的URL
from django.shortcuts import redirect

def my_view(request):
    ...
    object = MyModel.objects.get(...)
    return redirect(object)  # 临时重定向
  	return redirect(object, permanent=True)  # 永久重定向
  
# 通过传递一个视图的名称，可以带有位置参数和关键字参数；将使用reverse()方法反向解析URL
return redirect('some-view-name', foo='bar')
 
# 传递要重定向的一个硬编码的URL
return redirect('/some/url/')
return redirect('http://example.com/')  # 完整URL
```

### get_object_or_404

```python
get_object_or_404(klass, *args, **kwargs)

# 在一个给定的模型管理器上调用get()，但是引发`Http404` 而不是模型的`DoesNotExist`异常
# 参数
klass  		# 获取该对象的一个Model类，Manager或QuerySet实例。
**kwargs	# 查询的参数，格式应该可以被get()和filter()接受
```

示例

```python
# 从MyModel中使用主键1来获取对象
from django.shortcuts import get_object_or_404

def my_view(request):
    my_object = get_object_or_404(MyModel, pk=1)
# 等价
from django.http import Http404

def my_view(request):
    try:
        my_object = MyModel.objects.get(pk=1)
    except MyModel.DoesNotExist:
        raise Http404("No MyModel matches the given query.")

# QuerySet
queryset = Book.objects.filter(title__startswith='M')
get_object_or_404(queryset, pk=1)
# 等价
get_object_or_404(Book, title__startswith='M', pk=1)

# Manager
get_object_or_404(Book.dahl_objects, title='Matilda')   # 自定义管理器尤其有用
author = Author.objects.get(name='Roald Dahl')					# 使用关联的管理器
get_object_or_404(author.book_set, title='Matilda')  
```

### get_list_or_404

```python
get_list_or_404(klass, *args, **kwargs)

# 返回一个给定模型管理器上filter()的结果，并将结果映射为一个列表，如果结果为空则返回`Http404`
# 参数
klass  		# 获取该对象的一个Model类，Manager或QuerySet实例。
**kwargs	# 查询的参数，格式应该可以被get()和filter()接受
```

示例

```python
# 从MyModel 中获取所有发布出来的对象
from django.shortcuts import get_list_or_404

def my_view(request):
    my_objects = get_list_or_404(MyModel, published=True)
# 等价
from django.http import Http404

def my_view(request):
    my_objects = list(MyModel.objects.filter(published=True))
    if not my_objects:
        raise Http404("No MyModel matches the given query.")
```

## 视图装饰器

Django为视图提供了数个装饰器，用以支持相关的HTTP服务

### 允许的HTTP方法

`django.views.decorators.http`包里的装饰器可以基于请求的方法来限制对视图的访问。若条件不满足会返回 `django.http.HttpResponseNotAllowed`。

| name                                        | desc                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| `require_http_methods(request_method_list)` | 限制视图只能服务规定的http方法                               |
| `require_GET()`                             | 只允许视图接受GET方法的装饰器                                |
| `require_POST()`                            | 只允许视图接受POST方法的装饰器                               |
| `require_safe()`                            | 只允许视图接受 GET和HEAD方法的装饰器。 这些方法通常被认为是安全的，因为方法不该有请求资源以外的目的 |

注意

```
Django会自动清除对HEAD请求的响应中的内容而只保留头部，所以在你的视图中你处理HEAD 请求的方式可以完全与GET 请求一致。因为某些软件，例如链接检查器，依赖于HEAD 请求，所以你可能应该使用require_safe 而不是require_GET
```

示例

```python
from django.views.decorators.http import require_http_methods

@require_http_methods(["GET", "POST"])  # 方法名必须大写
def my_view(request):
    # I can assume now that only GET or POST requests make it this far
    # ...
    pass
```

### 可控制的视图处理

`django.views.decorators.http`中的以下装饰器可以用来控制特定视图的缓存行为

| name                                                 | desc                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| `condition(etag_func=None, last_modified_func=None)` | 这个装饰器使用两个函数（如果你不能既快又容易得计算出来，你只需要提供一个）来弄清楚是否HTTP请求中的协议头匹配那些资源。如果它们不匹配，会生成资源的一份新的副本，并调用你的普通视图。 |
| `etag(etag_func)`                                    |                                                              |
| `last_modified(last_modified_func)`                  |                                                              |

这些装饰器可以用于生成`ETag` 和`Last-Modified` 头部

### GZip压缩

`django.views.decorators.gzip` 里的装饰器基于每个视图控制其内容压缩

| name          | desc                                                         |
| ------------- | ------------------------------------------------------------ |
| `gzip_page()` | 如果浏览器允许gzip 压缩，这个装饰器将对内容进行压缩。它设置相应的`Vary`头部，以使得缓存根据`Accept-Encoding`头来存储信息。 |

### Vary头部

`django.views.decorators.vary` 可以用来基于特定的请求头部来控制缓存

| name                       | desc                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `vary_on_cookie(func)`     |                                                              |
| `vary_on_header(*headers)` | 到当构建缓存的键时，`Vary` 头部定义一个缓存机制应该考虑的请求头 |

### cache

`django.views.decorators.cache`控制服务器和客户端缓存

| name                      | desc                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `cache_control(**kwargs)` | 此装饰器通过向其添加所有关键字参数来修补响应的Cache-Control标头 |
| `never_cache(view_func)`  | 此装饰器将一个`Cache-Control:max-age=0,no-cache,no-store,must-revalidate`头添加到响应中，以指示永远不应缓存页面。 |

