# 视图-Response

Django 使用Request 对象和Response 对象在系统间传递状态， 定义在`django.http`模块中。

当请求一个页面时，Django会建立一个包含请求元数据的`HttpRequest`对象。 当Django 加载对应的视图时，`HttpRequest`对象将作为视图函数的第一个参数。每个视图会返回一个`HttpResponse`对象。

## HttpResponse

与由Django自动创建的`HttpRequest` 对象相比，`HttpResponse` 对象由程序员创建. 您编写的每个视图都负责实例化，填充和返回一个`HttpResponse` 类是在`django.http`模块中定义的。

### 使用

####  传递字符串

典型的应用是传递一个字符串作为页面的内容到`HttpResponse` 构造函数:

```shell
>>> from django.http import HttpResponse
>>> response = HttpResponse("Here's the text of the Web page.")
>>> response = HttpResponse("Text only, please.", content_type="text/plain")
```

如果你想增量增加内容，你可以将`response` 看做一个类文件对象

```shell
>>> response = HttpResponse()
>>> response.write("<p>Here's the text of the Web page.</p>")
>>> response.write("<p>Here's another paragraph.</p>")
```

#### 传递迭代器

最后，您可以传递给`HttpResponse`一个迭代器而不是字符串。`HttpResponse`将立即使用迭代器，将其内容存储为字符串，然后丢弃它。具有close（）方法的对象（如文件和生成器）会立即关闭。

如果需要将响应从迭代器流式传输到客户端，则必须使用`StreamingHttpResponse`类。

#### 设置头字段

把它当作一个类似字典的结构，从你的response中设置和移除一个header field。

```shell
>>> response = HttpResponse()
>>> response['Age'] = 120
>>> del response['Age']
```

注意！与字典不同的是，如果要删除的header field不存在，`del`不会抛出`KeyError`异常。

对于设置`Cache-Control`和`Vary`头字段，建议使用`django.utils.cache`中的`patch_cache_control()`和`patch_vary_headers()`方法，因为这些字段可以有多个逗号分隔的值。 “补丁”方法确保其他值，例如，通过中间件添加的，不会删除。

HTTP header fields 不能包含换行。 当我们尝试让header field包含一个换行符（CR 或者 LF），那么将会抛出一个`BadHeaderError`异常。

#### 浏览器将响应视为文件附件

让浏览器以文件附件的形式处理响应, 需要声明 `content_type` 类型 和设置`Content-Disposition` 头信息. 例如，下面是 如何给浏览器返回一个微软电子表格：

```shell
>>> response = HttpResponse(my_data, content_type='application/vnd.ms-excel')
>>> response['Content-Disposition'] = 'attachment; filename="foo.xls"'
```

没有关于`Content-Disposition`头的Django特定的，但是很容易忘记语法，所以我们在这里包括它。

### 属性

- `HttpResponse.content`        

一个用来代替content的字节字符串，如果必要，则从一个Unicode对象编码而来 

- `HttpResponse.charset` 

一个字符串，用来表示response将会被编码的字符集。 如果没有在`HttpResponse`实例化的时候给定这个字符集，那么将会从`content_type` 中解析出来。并且当这种解析成功的时候，`DEFAULT_CHARSET`选项将会被使用。 

- `HttpResponse.status_code` 

响应的 HTTP status code。除非明确设置了`reason_phrase`，否则在构造函数之外修改`status_code`的值也会修改`reason_phrase`的值。 

- `HttpResponse.reason_phrase` 

响应的HTTP原因短语。 它使用HTTP standard’s默认原因短语。除非明确设置，否则`reason_phrase`由`status_code`的值决定 

- `HttpResponse.streaming`      

这个选项总是`False`。由于这个属性的存在，使得中间件（middleware）能够区别对待流式response和常规response。 

- `HttpResponse.closed` 

`True`如果响应已关闭。 

### 方法

- `HttpResponse.__init__(content='', content_type=None, status=200,reason=None,charset=None)` 

使用页面的内容（content）和content-type来实例化一个`HttpResponse`对象             

参数

```
content  # 应该是一个迭代器或者字符串。 如果它是一个迭代器，那么它应该返回一串字符串，并且这些字符串连接起来形成response的内容。 如果不是迭代器或者字符串，那么在其被接收的时候将转换成字符串。
content_type  # 是可选地通过字符集编码完成的MIME类型，并且用于填充HTTP Content-Type头部。 如果未指定，它由默认情况下由 DEFAULT_CONTENT_TYPE 和 DEFAULT_CHARSET 设置组成，默认为“text/html; charset=utf-8”。
status  # 是响应的 HTTP status code。
reason  # 是HTTP响应短语 如果没有指定, 则使用默认响应短语.
charset  # 在response中被编码的字符集。 如果没有给定，将会从 content_type中提取, 如果提取不成功, 那么 DEFAULT_CHARSET 的设定将被使用.
```

- `HttpResponse.__setitem__(header, value)`

由给定的首部名称和值设定相应的报文首部。 `header` 和 `value` 都应该是字符串类型 

- `HttpResponse.__delitem__(header)`

根据给定的首部名称来删除报文中的首部。 如果对应的首部不存在将沉默地（不引发异常）失败。 不区分大小写。                                                                                                                   
- `HttpResponse.__getitem__(header)`                                       

根据首部名称返回其值。 不区分大小写                                             

- `HttpResponse.has_header(header)`                                                                                                     
通过检查首部中是否有给定的首部名称（不区分大小写），来返回`True` 或 `False` 。       

- `HttpResponse.setdefault(header, value)`                                                                                              
设置一个首部，除非该首部 header 已经存在了。                                                                                                               
- `HttpResponse.set_cookie(key, value='', max_age=None, expires=None,path='/', domain=None, secure=None, httponly=False)`         

设置一个Cookie。 参数与Python 标准库中的`Morsel`Cookie 对象相同                   

参数
```
max_age  # 以秒为单位，如果Cookie 只应该持续客户端浏览器的会话时长则应该为None（默认值）。 如果没有指定expires，则会通过计算得到。
expires  # 应该是一个 UTC "Wdy, DD-Mon-YY HH:MM:SS GMT" 格式的字符串，或者一个datetime.datetime 对象。 如果expires 是一个datetime 对象，则max_age 会通过计算得到。
domain   # 如果你想设置一个跨域的Cookie，请使用domain 参数。 例如，domain=".lawrence.com" 将设置一个www.lawrence.com、blogs.lawrence.com 和calendars.lawrence.com 都可读的Cookie。 否则，Cookie 将只能被设置它的域读取。

httponly  # 如果你想阻止客服端的JavaScript 访问Cookie，可以设置httponly=True。
```

- `HttpResponse.set_signed_cookie(key, value, salt='', max_age=None, expires=None, path='/', domain=None, secure=None, httponly=True)`

与set_cookie() 类似，但是在设置之前将cryptographic signing。 通常与HttpRequest.get_signed_cookie() 一起使用。 你可以使用可选的salt 参考来增加密钥强度，但需要记住将它传递给对应的HttpRequest.get_signed_cookie() 调用。 

- `HttpResponse.delete_cookie(key, path='/', domain=None)`               

删除指定的key 的Cookie。 如果key 不存在则什么也不发生。

由于Cookie 的工作方式，path 和domain 应该与set_cookie() 中使用的值相同 —— 否则Cookie 不会删掉。

- `HttpResponse.write(content) `                                         

此方法使HttpResponse实例是一个类似文件的对象                                   

- `HttpResponse.flush()`                                                 

此方法使HttpResponse实例是一个类似文件的对象                                   

- `HttpResponse.tell()`                                                   

此方法使HttpResponse实例是一个类似文件的对象                                                                                     
- `HttpResponse.getvalue() `                                               

返回HttpResponse.content的值。 此方法使HttpResponse实例是一个类似流的对象                                                                                 
- `HttpResponse.readable() `                                               

始终False。 此方法使HttpResponse实例是一个类似流的对象                                                                                                    
- `HttpResponse.seekable() `

始终False。 此方法使HttpResponse实例是一个类似流的对象                                                                                                    
- `HttpResponse.writable() `

始终为True。 此方法使HttpResponse实例是一个类似流的对象                                                                                                   
- `HttpResponse.writelines(lines) `

将一个包含行的列表写入响应。 不添加行分隔符。 此方法使HttpResponse实例是一个类似流的对象。                                                                     

### 子类

Django包含了一系列的HttpResponse衍生类（子类），用来处理不同类型的HTTP 响应（response）。 与 HttpResponse相同, 这些衍生类（子类）存在于django.http之中。

```
class HttpResponseRedirect 
```

构造函数的第一个参数是必要的 — 用来重定向的地址。 该参数可以是一个完整的URL地址(如'https://www.yahoo.com/search/')，或者是一个相对于项目的绝对路径(如 '/search/')，还可以是一个相对路径(如 'search/')。 在最后一种情况下，客户端浏览器将根据当前路径重建完整的URL本身。 关于构造函数的其他参数，可以参见 HttpResponse。 注意！这个响应会返回一个302的HTTP状态码。 

```
class HttpResponsePermanentRedirect
```

与HttpResponseRedirect一样，但是它会返回一个永久的重定向（HTTP状态码301）而不是一个“found”重定向（状态码302） 

```
class HttpResponseNotModified       
```

构造函数不会有任何的参数，并且不应该向这个响应（response）中加入内容（content）。 使用此选项可指定自用户上次请求（状态代码304）以来尚未修改页面。 

```
class HttpResponseBadRequest        
```

HttpResponse类似，使用了一个400的状态码。

```
class HttpResponseNotFound          
```

HttpResponse类似，使用的404状态码

```
class HttpResponseForbidden         
```

HttpResponse类似，使用403状态代码

```
class HttpResponseNotAllowed        
```
HttpResponse类似，使用405状态码。 构造函数的第一个参数是必须的：一个允许使用的方法构成的列表（例如，`['GET', 'POST']`） 

```
class HttpResponseGone          
```

HttpResponse类似，使用410状态码                                        

```
class HttpResponseServerError      
```

HttpResponse类似，使用500状态代码 

> 注意
如果HttpResponse的自定义子类实现了render方法，Django会将其视为模拟SimpleTemplateResponse，且render方法必须自己返回一个有效的响应对象。

## JsonResponse

```
class JsonResponse(data, encoder=DjangoJSONEncoder, safe=True, json_dumps_params=None, **kwargs)
```

HttpResponse 的一个子类，帮助用户创建JSON 编码的响应。 它从父类继承大部分行为，并具有以下不同点：
1. 它的默认Content-Type 头部设置为application/json。
2. 它的第一个参数data，应该为一个dict 实例。 如果safe 参数设置为False，它可以是任何可JSON 序列化的对象。
3. encoder，默认为 django.core.serializers.json.DjangoJSONEncoder，用于序列化data。 关于这个序列化的更多信息参见JSON serialization。
4. 布尔参数safe 默认为True。 如果设置为False，可以传递任何对象进行序列化（否则，只允许dict 实例）。 如果safe 为True，而第一个参数传递的不是dict 对象，将抛出一个TypeError。
5. json_dumps_params参数是传递给用于生成响应的json.dumps()调用的关键字参数的字典。

###  用法

```shell
>> from django.http import JsonResponse
>>> response = JsonResponse({'foo': 'bar'})
>>> response.content
b'{"foo": "bar"}'
```

- 序列化非字典对象

若要序列化非dict 对象，你必须设置safe 参数为False：
```shell
>>> response = JsonResponse([1, 2, 3], safe=False)
```

如果不传递safe=False，将抛出一个TypeError。

> 警告
>
> 在第五版ECMAScript之前，可能会中毒JavaScript Array构造函数。因此，默认情况下，Django不允许将非字典对象传递给JsonResponse构造函数。但是，大多数现代浏览器都实现了EcmaScript 5，它删除了此攻击媒介。因此可以禁用此安全预防措施。

- 改变默认JSON编码

如果你需要使用不同的JSON 编码器类，你可以传递encoder 参数给构造函数：

```shell
>>> response = JsonResponse(data, encoder=MyJSONEncoder)
```
## StreamingHttpResponse

StreamingHttpResponse类被用来从Django流式化一个响应（response）到浏览器。 如果生成响应太长或者是有使用太多的内存，你可能会想要这样做。 例如，它对于generating large CSV files非常有用。

> 基于性能的考虑
Django是为了那些短期的请求（request）设计的。 流式响应将会为整个响应期协同工作进程。 这可能导致性能变差。
总的来说，你需要将代价高的任务移除请求—响应的循环，而不是求助于流式响应。

StreamingHttpResponse 不是 HttpResponse的衍生类（子类），因为它实现了完全不同的应用程序接口（API）。 尽管如此，除了以下的几个明显不同的地方，其他几乎完全相同：

1. 应该提供一个迭代器给它，这个迭代器生成字符串来构成内容（content）
2. 你不能直接访问它的内容（content），除非迭代它自己的响应对象。 这只在响应被返回到客户端的时候发生。
3. 它没有 content 属性。 取而代之的是，它有一个 streaming_content 属性。
4. 你不能使用类似文件对象的tell()或者 write() 方法。 那么做会抛出一个异常。

StreamingHttpResponse应仅在绝对需要在将数据传输到客户端之前不重复整个内容的情况下使用。 由于无法访问内容，因此许多中间件无法正常运行。 例如，不能为流响应生成ETag和Content-Length头。

### 属性

- `StreamingHttpResponse.streaming_content`

一个表示内容（content）的字符串的迭代器                                         

- `StreamingHttpResponse.status_code`      

响应的 HTTP status code。

除非明确设置了reason_phrase，否则在构造函数之外修改status_code的值也会修改reason_phrase的值。 

- `StreamingHttpResponse.reason_phrase`     

响应的HTTP原因短语。 它使用 HTTP standard’s默认原因短语。除非明确设置，否则reason_phrase由status_code的值决定          

- `StreamingHttpResponse.streaming`         

这个选项总是 True                                                                                                

## FileResponse

```
class FileResponse
```

FileResponse是StreamingHttpResponse的衍生类（子类），为二进制文件做了优化。 如果 wsgi server 来提供，则使用了 wsgi.file_wrapper ，否则将会流式化一个文件为一些小块。

FileResponse 需要通过二进制模式打开文件，如下:
```shell
>>> from django.http import FileResponse
>>> response = FileResponse(open('myfile.png', 'rb'))
```
