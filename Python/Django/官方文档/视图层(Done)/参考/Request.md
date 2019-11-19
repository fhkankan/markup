# 视图-Request

Django 使用Request 对象和Response 对象在系统间传递状态， 定义在`django.http`模块中。

当请求一个页面时，Django会建立一个包含请求元数据的`HttpRequest`对象。 当Django 加载对应的视图时，`HttpRequest`对象将作为视图函数的第一个参数。每个视图会返回一个`HttpResponse`对象。

## HttpRequest

```
class HttpRequest
```

### 属性

下面除非特别说明，所有属性都认为是只读的

- `scheme`

一个字符串，表示请求的方案（通常是`http` 或`https`）

- `body`

一个字节字符串，表示原始HTTP 请求的正文。 它对于处理非HTML 形式的数据非常有用：二进制图像、XML等。 如果要处理常规的表单数据，应该使用`HttpRequest.POST`。

你也可以使用”类文件“形式的接口从HttpRequest 中读取数据。参见`HttpRequest.read()`

- `path`

一个字符串，表示请求的页面的完整路径，不包含域名。如`"/music/bands/"` 

- `path_info`

在某些Web 服务器配置下，主机名后的URL 部分被分成脚本前缀部分和路径信息部分。`path_info` 属性将始终包含路径信息部分，不论使用的Web 服务器是什么。使用它代替`path` 可以让代码在测试和开发环境中更容易地切换 

例如，如果应用的WSGIScriptAlias设置为`"/minfo"`，那么当path 是`"/minfo/music/bands/the_beatles/"` 时path_info 将是`"/music/bands/the_beatles/"`

- `method` 

一个字符串，表示请求使用的HTTP 方法。必须使用大写 

```python
if request.method == 'GET':
    do_something()
elif request.method == 'POST':
    do_something_else()
```

- `encoding` 

一个字符串，表示提交的数据的编码方式（如果为`None` 则表示使用`DEFAULT_CHARSET`设置） 

这个属性是可写的，你可以修改它来修改访问表单数据使用的编码。接下来对属性的任何访问（例如从`GET` 或 `POST` 中读取数据）将使用新的`encoding`值。如果你知道表单数据的编码不在`DEFAULT_CHARSET`中，则使用它。

- `content_type` 

表示请求的MIME类型的字符串，从`CONTENT_TYPE`头解析。

- `content_params` 

`CONTENT_TYPE`头中包含的键/值参数字典。 

- `GET` 

一个类似于字典的对象，包含HTTP GET 的所有参数。详情请参考下面的`QueryDict` 

- `POST` 

一个包含所有给定的HTTP POST参数的类字典对象，提供了包含表单数据的请求。详情请参考下面的QueryDict 文档。如果需要访问请求中的原始或非表单数据，可以使用HttpRequest.body 属性。

POST 请求可以带有空的POST字典 —— 如果通过HTTP POST方法请求一个表单但是没有包含表单数据的话。因此，不应该使用if request.POST 来检查使用的是否是POST方法；应该使用if request.method == "POST"（参见上文）。

注意：POST 不包含上传的文件信息。参见FILES。

- `COOKIES`     

一个标准的Python 字典，包含所有的cookie。键和值都为字符串。

- `FILES`          

一个类似于字典的对象，包含所有的上传文件。 

`FILES` 中的每个键为`<input type="file" name="" />` 中的`name`
注意，FILES 只有在请求的方法为POST 且提交的`<form>` 带有`content_type="multipart/form-data"`的情况下才会包含数据。否则，FILES 将为一个空的类似于字典的对象。

- `META`           

一个标准的Python 字典，包含所有的HTTP 头部 

具体的头部信息取决于客户端和服务器，下面是一些示例：
```
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
```
从上面可以看到，除`CONTENT_LENGTH` 和`CONTENT_TYPE` 之外，请求中的任何HTTP 头部转换为META 的键时，都会将所有字母大写并将连接符替换为下划线最后加上`HTTP_` 前缀。所以，一个叫做X-Bender 的头部将转换成META 中的HTTP_X_BENDER 键。

请注意，runserver会删除名称中带有下划线的所有标头，因此您不会在META中看到它们。这样可以防止基于下划线和破折号之间的歧义的标头欺骗在WSGI环境变量中均被标准化为下划线。它与Nginx和Apache 2.4+等Web服务器的行为匹配。

- `resolver_match` 

一个`ResolverMatch`]的实例，表示解析后的URL。

这个属性只有在URL解析方法之后才设置，这意味着它在所有的视图中可以访问，但是在在URL 解析发生之前执行的中间件方法中不可以访问（比如process_request，但你可以使用process_view 代替）

### 被应用设置的属性

Django本身不设置这些属性，但如果由应用程序设置，则使用它们。

- `HttpRequest.current_app`

url模板标记将使用其值作为`reverse()`的`current_app`参数

- `HttpRequest.urlconf`

这将用作当前请求的根URLconf，覆盖ROOT_URLCONF设置，细节详见 [How Django processes a request](https://yiyibooks.cn/__trs__/qy/django2/topics/http/urls.html#how-django-processes-a-request) 

可以将urlconf设置为None，以还原以前的中间件所做的任何更改，并返回使用ROOT_URLCONF。

### 被中间件设置的属性

Django的contrib应用程序中包含的一些中间件在请求中设置了属性。如果在请求中没有看到该属性，请确保MIDDLEWARE中列出了相应的中间件类。

| name | desc |
| ---- | ---- |
|      |      |
|      |      |
|      |      |

- `session`

来自`SessionMiddleware`，一个可读写的类字典对象，代表当前会话

- `site`

来自`CurrentSiteMiddleware`，由`get_current_site()`返回的`Site`或`RequestSite`实例，表示当前站点

- `user`

来自`AuthenticationMiddleware`，表示当前登录用户的`AUTH_USER_MODEL`实例。如果用户当前未登录，则将用户设置为`AnonymousUser`的实例。您可以使用`is_authenticated`区分它们，如下所示

```python
# 您可以使用is_authenticated区分是否登陆
if request.user.is_authenticated:
    ... # Do something for logged-in users.
else:
    ... # Do something for anonymous users.
```

### 方法

- `get_host`

根据从`HTTP_X_FORWARDED_HOST`（如果打开`USE_X_FORWARDED_HOST`）和`HTTP_HOST` 头部信息返回请求的原始主机。如果这两个头部没有提供相应的值，则使用`SERVER_NAME` 和`SERVER_PORT`的组合

> 注

当主机位于多个代理的后面，get_host()方法将会失败。有一个解决办法是使用中间件重写代理的头部

```python
from django.utils.deprecation import MiddlewareMixin

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

这个中间件应该放置在所有依赖于get_host()的中间件之前，如CommonMiddleware 和CsrfViewMiddleware。

- `get_port()` 

使用来自`HTTP_X_FORWARDED_PORT`(如果启用了`USE_X_FORWARDED_PORT`)和`SERVER_PORT META`变量的信息，依次返回请求的原始端口。

- `get_full_path()`

返回`path`，如果可以将加上查询字符串

例如`"/music/bands/the_beatles/?print=true"`

- `build_absolute_uri(location)` 

返回`location` 的绝对URI。如果location 没有提供，则设置为`request.get_full_path()`。

如果URI 已经是一个绝对的URI，将不会修改。否则，使用请求中的服务器相关的变量构建绝对URI。

例如：`"https://example.com/music/bands/the_beatles/?print=true"`

> 注
不建议在同一站点上混合使用HTTP和HTTPS，因此`build_absolute_uri()`将始终以与当前请求相同的方案生成绝对URI。如果您需要将用户重定向到HTTPS，最好让您的网络服务器将所有HTTP流量重定向到HTTPS。

- `get_signed_cookie(key, default=RAISE_ERROR, salt='', max_age=None)`

返回签名的cookie的cookie值，或者如果签名不再有效，则引发django.core.signing.BadSignature异常。如果提供默认参数，则将抑制该异常，并将改为返回该默认值。

可选的salt参数可用于提供额外的保护，以防止对密钥的暴力攻击。如果提供了该参数，将对照该cookie值所附的签名时间戳检查max_age参数，以确保该cookie不早于max_age秒。 如果提供default 参数，将不会引发异常并返回default 的值。

示例

```python
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

- `is_secure()`

如果请求时是安全的，则返回`True`；即请求是通过HTTPS 发起的 

- `is_ajax()`

如果请求是通过`XMLHttpRequest` 发起的，则返回`True` 。方法是检查HTTP_X_REQUESTED_WITH头部是否是字符串'XMLHttpRequest'。大部分现代的JavaScript库都会发送这个头部。如果你编写自己的XMLHttpRequest调用(在浏览器端)，你必须手工设置这个值来让is_ajax()可以工作。

如果一个响应需要根据请求是否是通过AJAX发起的，并且你正在使用某种形式的缓存例如Django的cache middleware，你应该使用vary_on_headers('HTTP_X_REQUESTED_WITH') 装饰你的视图以让响应能够正确地缓存。

- `read(size=None)`
- `readline()`
- `readlines()`
- `__iter__()`

实现类文件的接口用于读取HttpRequest实例。这使得可以用流的方式读取进来的请求。一个常见的用例是使用迭代解析器处理大型XML有效载荷，而不在内存中构造一个完整的XML树。

根据这个标准的接口，一个HttpRequest 实例可以直接传递给XML 解析器，例如ElementTree：

```python
import xml.etree.ElementTree as ET
for element in ET.iterparse(request):
    process(element)
```

## QueryDict

```
class QueryDict
```

在`HttpRequest`对象中，`GET` 和`POST` 属性是`django.http.QueryDict` 的实例，它是一个自定义的类似字典的类，用来处理同一个键带有多个值。这个类的需求来自某些HTML 表单元素传递多个值给同一个键，`<selectmultiple>` 是一个显著的例子。

`request.POST` 和`request.GET` 的`QueryDict` 在一个正常的请求/响应循环中是不可变的。若要获得可变的版本，需要使用`.copy()`。

### 方法

`QueryDict`实现了字典的所有标准方法，因为它是字典的子类。

####特有方法

- `__init__(query_string=None, mutable=False, encoding=None)`

基于`query_string` 实例化`QueryDict` 一个对象 

```python
>>> QueryDict('a=1&a=2&c=3')
<QueryDict: {'a': ['1', '2'], 'c': ['3']}>
```

若query_string 没被传入,QueryDict 的结果是空的(将没有键和值).

你所遇到的大部分对象都是不可修改的，例如request.POST和request.GET。如果需要实例化你自己的可以修改的对象，通过往它的`__init__()`方法来传递参数 mutable=True 可以实现。

设置键和值的字符串都将从encoding 转换为unicode。如果没有指定编码,默认设置为DEFAULT_CHARSET
```

- `classmethod QUeryDict.fromkeys(iterable, value='', mutable=false, encoding=None)`

Django 1.11新增

使用可迭代的键创建一个新的QueryDict，每个值都等于value。例如：
​```python
>>> QueryDict.fromkeys(['a', 'a', 'b'], value='val')
<QueryDict: {'a': ['val', 'val'], 'b': ['val']}>
```

- `__getitem__(key)`

返回给定键的值。如果键有多个值，则返回最后一个值。如果键不存在，则引发`django.utils.datastructures.MultiValueDictKeyError`。（这是Python标准KeyError的子类，因此您可以继续捕捉KeyError。）

- `__setitem__(key,value)`  

设置给出的key 的值为`value`（一个Python 列表，它具有唯一一个元素`value`）。注意，这和其它具有副作用的字典函数一样，只能在可变的`QueryDict` 上调用（如通过`copy()` 创建的字典）。

- `__contains__(key)`

如果给出的key 已经设置，则返回`True`。它让你可以做`if "foo" in request.GET`这样的操作。

- `get(key, default=None)`

使用与上面`__getitem__()` 相同的逻辑，但是当key 不存在时返回一个默认值 

- `setdefault(key, default=None)`

类似标准字典的`setdefault()` 方法，只是它在内部使用的是`__setitem__()` 

- `update(other_dict)`

接收一个`QueryDict` 或标准字典。类似标准字典的`update()` 方法，但是它附加到当前字典项的后面，而不是替换掉它们

```python
>>> q = QueryDict('a=1', mutable=True)
>>> q.update({'a': '2'})
>>> q.getlist('a')
['1', '2']
>>> q['a'] # returns the last
['2']
```

- `items()`

类似标准字典的`items()` 方法，但是它使用的是和`__getitem__` 一样返回最新的值的逻辑。

```python
>>> q = QueryDict('a=1&a=2&a=3')
>>> q.items()
[('a', '3')]
```

- `values()`

类似标准字典的`values()` 方法，但是它使用的是和`__getitem__` 一样返回最新的值的逻辑 

```python
>>> q = QueryDict('a=1&a=2&a=3')
>>> q.values()
['3']
```

#### 字典方法

此外，QueryDict具有以下方法：

- `copy()`                                                    

返回对象的副本，使用Python 标准库中的`copy.deepcopy()`。此副本是可变的即使原始对象是不可变的

- `getlist(key, default=None)`                              

以Python 列表形式返回所请求的键的数据。如果键不存在并且没有提供默认值，则返回空列表。它保证返回的是某种类型的列表，除非默认值不是列表 

- `setlist(key, list_)`                                   

设置给定的键为`list_`（与`__setitem__()` 不同)               

- `appendlist(key, item)`                                 

将项追加到内部与键相关联的列表中                             
`setlist(key, item)`                                                                                                                              
- `setlistdefault(key, default_list=None)`                

类似`setdefault`，除了它接受一个列表而不是单个值             

- `lists()`                                               

类似`items`，只是它将字典中的每个成员作为列表                

```shell
>>> q = QueryDict('a=1&a=2&a=3')
>>> q.lists()
[('a', ['1', '2', '3'])]
```

- `pop(key)`                                              

返回给定键的值的列表，并从字典中移除它们。如果键不存在，将引发`KeyError` 

```shell
>>> q = QueryDict('a=1&a=2&a=3', mutable=True)
>>> q.pop('a')
['1', '2', '3']
```

- `popitem()`                                             

删除字典任意一个成员（因为没有顺序的概念），并返回二值元组，包含键和键的所有值的列表。在一个空的字典上调用时将引发`KeyError` 

```shell
>>> q = QueryDict('a=1&a=2&a=3', mutable=True)
>>> q.popitem()
('a', ['1', '2', '3'])
```

- `dict()`                                                

返回`QueryDict` 的`dict` 表示形式。对于`QueryDict` 中的每个(key, list)对 ，`dict`将有(key, item) 对，其中item 是列表中的一个元素，使用与`QueryDict.__getitem__()`相同的逻辑 

```shell
>>> q = QueryDict('a=1&a=3&a=5')
>>> q.dict()
{'a': '5'}
```

- `urlencode(safe=None)`                                  

在查询字符串格式返回数据的字符串形式                         

```shell
以查询字符串格式返回数据的字符串
>>> q = QueryDict('a=2&b=3&b=5')
>>> q.urlencode()
'a=2&b=3&b=5'
```

使用safe参数传递不需要编码的字符

```shell
>>> q = QueryDict(mutable=True)
>>> q['next'] = '/a&b/'
>>> q.urlencode(safe='/')
'next=/a%26b/'
```