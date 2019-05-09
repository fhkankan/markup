# 视图-Request/Response

Django 使用Request 对象和Response 对象在系统间传递状态， 定义在`django.http`模块中。

当请求一个页面时，Django会建立一个包含请求元数据的`HttpRequest`对象。 当Django 加载对应的视图时，`HttpRequest`对象将作为视图函数的第一个参数。每个视图会返回一个`HttpResponse`对象。

## HttpRequest

```
class HttpRequest
```

### 属性

下面除非特别说明，所有属性都认为是只读的

| name             | desc                                                         |
| ---------------- | ------------------------------------------------------------ |
| `scheme`         | 一个字符串，表示请求的方案（通常是`http` 或`https`）         |
| `body`           | 一个字节字符串，表示原始HTTP 请求的正文。                    |
| `path`           | 一个字符串，表示请求的页面的完整路径，不包含域名。如`"/music/bands/"` |
| `path_info`      | 在某些Web 服务器配置下，主机名后的URL 部分被分成脚本前缀部分和路径信息部分。`path_info` 属性将始终包含路径信息部分，不论使用的Web 服务器是什么。使用它代替`path` 可以让代码在测试和开发环境中更容易地切换 |
| `method`         | 一个字符串，表示请求使用的HTTP 方法。必须使用大写            |
| `encoding`       | 一个字符串，表示提交的数据的编码方式（如果为`None` 则表示使用`DEFAULT_CHARSET`设置） |
| `content_type`   | 表示请求的MIME类型的字符串，从`CONTENT_TYPE`头解析。         |
| `content_params` | `CONTENT_TYPE`头中包含的键/值参数字典。                      |
| `GET`            | 一个类似于字典的对象，包含HTTP GET 的所有参数。详情请参考下面的`QueryDict` |
| `POST`           | 一个包含所有给定的HTTP POST参数的类字典对象，提供了包含表单数据的请求。 |
| `COOKIES`        | 一个标准的Python 字典，包含所有的cookie。键和值都为字符串。  |
| `FILES`          | 一个类似于字典的对象，包含所有的上传文件。                   |
| `META`           | 一个标准的Python 字典，包含所有的HTTP 头部                   |
| `resolver_match` | 一个`ResolverMatch`]的实例，表示解析后的URL。                |

详细

```python
# body
它对于处理非HTML 形式的数据非常有用：二进制图像、XML等。 如果要处理常规的表单数据，应该使用`HttpRequest.POST`。你也可以使用”类文件“形式的接口从HttpRequest 中读取数据。参见`HttpRequest.read()`

# path_info
例如，如果应用的WSGIScriptAlias设置为"/minfo"，那么当path 是"/minfo/music/bands/the_beatles/" 时path_info 将是"/music/bands/the_beatles/"

# method
if request.method == 'GET':
    do_something()
elif request.method == 'POST':
    do_something_else()

# encoding    
这个属性是可写的，你可以修改它来修改访问表单数据使用的编码。接下来对属性的任何访问（例如从`GET` 或 `POST` 中读取数据）将使用新的`encoding`值。如果你知道表单数据的编码不在`DEFAULT_CHARSET`中，则使用它。

# POST
详情请参考下面的QueryDict 文档。如果需要访问请求中的原始或非表单数据，可以使用HttpRequest.body 属性。

POST 请求可以带有空的POST字典 —— 如果通过HTTP POST方法请求一个表单但是没有包含表单数据的话。因此，不应该使用if request.POST 来检查使用的是否是POST方法；应该使用if request.method == "POST"（参见上文）。

注意：POST 不包含上传的文件信息。参见FILES。

# FILES
`FILES` 中的每个键为`<input type="file" name="" />` 中的`name`
注意，FILES 只有在请求的方法为POST 且提交的<form> 带有enctype="multipart/form-data" 的情况下才会包含数据。否则，FILES 将为一个空的类似于字典的对象。

# META
具体的头部信息取决于客户端和服务器，下面是一些示例：
CONTENT_LENGTH —— 请求的正文的长度（是一个字符串）。
CONTENT_TYPE —— 请求的正文的MIME 类型。
HTTP_ACCEPT —— 响应可接收的Content-Type。
HTTP_ACCEPT_ENCODING —— 响应可接收的编码。
HTTP_ACCEPT_LANGUAGE —— 响应可接收的语言。
HTTP_HOST —— 客服端发送的HTTP Host 头部。
HTTP_REFERER —— Referring 页面。
HTTP_USER_AGENT —— 客户端的user-agent 字符串。
QUERY_STRING —— 单个字符串形式的查询字符串（未解析过的形式）。
REMOTE_ADDR —— 客户端的IP地址。
REMOTE_HOST —— 客户端的主机名。
REMOTE_USER —— 服务器认证后的用户。
REQUEST_METHOD —— 一个字符串，例如"GET" 或"POST"。
SERVER_NAME —— 服务器的主机名。
SERVER_PORT —— 服务器的端口（是一个字符串）。
从上面可以看到，除CONTENT_LENGTH 和CONTENT_TYPE 之外，请求中的任何HTTP 头部转换为META 的键时，都会将所有字母大写并将连接符替换为下划线最后加上HTTP_ 前缀。所以，一个叫做X-Bender 的头部将转换成META 中的HTTP_X_BENDER 键。

# resolver_match
这个属性只有在URL解析方法之后才设置，这意味着它在所有的视图中可以访问，但是在在URL 解析发生之前执行的中间件方法中不可以访问（比如process_request，但你可以使用process_view 代替）
```

### 被应用设置的属性

Django本身不设置这些属性，但如果由应用程序设置，则使用它们。

| name          | desc                                                    |
| ------------- | ------------------------------------------------------- |
| `current_app` | url模板标记将使用其值作为`reverse()`的`current_app`参数 |
| `urlconf`     | 这将用作当前请求的根URLconf，覆盖ROOT_URLCONF设置       |

### 被中间件设置的属性

Django的contrib应用程序中包含的一些中间件在请求中设置了属性。如果在请求中没有看到该属性，请确保MIDDLEWARE中列出了相应的中间件类。

| name      | desc                                                         |
| --------- | ------------------------------------------------------------ |
| `session` | 来自`SessionMiddleware`，一个可读写的类字典对象，代表当前会话 |
| `site`    | 来自`CurrentSiteMiddleware`，由`get_current_site()`返回的`Site`或`RequestSite`实例，表示当前站点 |
| `user`    | 来自`AuthenticationMiddleware`，表示当前登录用户的`AUTH_USER_MODEL`实例。如果用户当前未登录，则将用户设置为`AnonymousUser`的实例。 |

示例

```python
# 您可以使用is_authenticated区分是否登陆
if request.user.is_authenticated:
    ... # Do something for logged-in users.
else:
    ... # Do something for anonymous users.
```

### 方法

| name                                                         | desc                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `get_host()`                                                 | 根据从`HTTP_X_FORWARDED_HOST`（如果打开`USE_X_FORWARDED_HOST`）和`HTTP_HOST` 头部信息返回请求的原始主机。如果这两个头部没有提供相应的值，则使用`SERVER_NAME` 和`SERVER_PORT` |
| `get_port()`                                                 | 使用来自`HTTP_X_FORWARDED_PORT`(如果启用了`USE_X_FORWARDED_PORT`)和`SERVER_PORT META`变量的信息，依次返回请求的原始端口。 |
| `get_full_path()`                                            | 返回`path`，如果可以将加上查询字符串                         |
| `build_absolute_uri(location)`                               | 返回`location` 的绝对URI。如果location 没有提供，则设置为`request.get_full_path()`。如果URI 已经是一个绝对的URI，将不会修改。否则，使用请求中的服务器相关的变量构建绝对URI。 |
| `get_signed_cookie(key, default=RAISE_ERROR, salt='', max_age=None)` | 返回签名过的Cookie 对应的值，如果签名不再合法则返回`django.core.signing.BadSignature`。 |
| `is_secure()`                                                | 如果请求时是安全的，则返回`True`；即请求是通过HTTPS 发起的   |
| `is_ajax()`                                                  | 如果请求是通过`XMLHttpRequest` 发起的，则返回`True`          |
| `read(size=None)`                                            | 读取HttpRequest实例                                          |
| `readline()`                                                 | 读取HttpRequest实例                                          |
| `readlines()`                                                | 读取HttpRequest实例                                          |
| `__iter__()`                                                 | 读取HttpRequest实例                                          |

`get_host`

```python
# 当主机位于多个代理的后面，get_host()方法将会失败。有一个解决办法是使用中间件重写代理的头部
# 这个中间件应该放置在所有依赖于get_host()的中间件之前，如CommonMiddleware 和CsrfViewMiddleware。
class MultipleProxyMiddleware(object):
    FORWARDED_FOR_FIELDS = [
        'HTTP_X_FORWARDED_FOR',
        'HTTP_X_FORWARDED_HOST',
        'HTTP_X_FORWARDED_SERVER',
    ]

    def process_request(self, request):
        """
        Rewrites the proxy headers so that only the most
        recent proxy is used.
        """
        for field in self.FORWARDED_FOR_FIELDS:
            if field in request.META:
                if ',' in request.META[field]:
                    parts = request.META[field].split(',')
                    request.META[field] = parts[-1].strip()
```

`get_signed_cookie`

```python
# 参数
default  # 如果提供参数，将不会引发异常并返回default 的值。
salt  # 可选参数，可以用来对安全密钥强力攻击提供额外的保护。
max_age  # 用于检查Cookie对应的时间戳以确保Cookie 的时间不会超过max_age秒

# 示例
>>> request.get_signed_cookie('name')
'Tony'
>>> request.get_signed_cookie('name', salt='name-salt')
'Tony' # assuming cookie was set using the same salt
>>> request.get_signed_cookie('non-existing-cookie')
...
KeyError: 'non-existing-cookie'
>>> request.get_signed_cookie('non-existing-cookie', False)
False
>>> request.get_signed_cookie('cookie-that-was-tampered-with')
...
BadSignature: ...
>>> request.get_signed_cookie('name', max_age=60)
...
SignatureExpired: Signature age 1677.3839159 > 60 seconds
>>> request.get_signed_cookie('name', False, max_age=60)
False
```

`is_ajax`

```
方法是检查HTTP_X_REQUESTED_WITH头部是否是字符串'XMLHttpRequest'。大部分现代的JavaScript库都会发送这个头部。如果你编写自己的XMLHttpRequest调用(在浏览器端)，你必须手工设置这个值来让is_ajax()可以工作。

如果一个响应需要根据请求是否是通过AJAX发起的，并且你正在使用某种形式的缓存例如Django的cache middleware，你应该使用vary_on_headers('HTTP_X_REQUESTED_WITH') 装饰你的视图以让响应能够正确地缓存。
```

`read/readline/readlines/xreadlines/__iter__`

```python
# 这几个方法实现类文件的接口用于读取HttpRequest实例。这使得可以用流的方式读取进来的请求。一个常见的用例是使用迭代解析器处理大型XML有效载荷，而不在内存中构造一个完整的XML树。
# 根据这个标准的接口，一个HttpRequest 实例可以直接传递给XML 解析器，例如ElementTree：

import xml.etree.ElementTree as ET
for element in ET.iterparse(request):
    process(element)
```

## QueryDict

```
class QueryDict
```



### 方法

## HttpResponse

### 使用



### 属性



### 方法



### 子类

## JsonResponse

### 序列化非字典对象

### 改变默认JSON编码

## StreamingHttpResponse

### 属性



## FileResponse

