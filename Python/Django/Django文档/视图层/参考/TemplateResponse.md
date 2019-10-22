# 视图-TemplateResponse

标准的HttpResponse 对象是静态的结构。 在构造的时候提供给它们一个渲染好的内容，但是当内容改变时它们却不能很容易地完成相应的改变。

然而，有时候允许装饰器或者中间件在响应构造之后修改它是很有用的。 例如，你可能想改变使用的模板，或者添加额外的数据到上下文中。

TemplateResponse 提供了实现这一点的方法。 与基本的HttpResponse 对象不同，TemplateResponse 对象会记住视图提供的模板和上下文的详细信息来计算响应。 响应的最终结果在后来的响应处理过程中直到需要时才计算。

## SimpleTemplateResponse 
`class SimpleTemplateResponse`

### 属性
`SimpleTemplateResponse.template_name`
```
渲染的模板的名称。 接收一个与后端有关的模板对象（例如get_template() 返回的对象）、模板的名称或者一个模板名称列表。

例如：['foo.html', 'path/to/bar.html']
```
`SimpleTemplateResponse.context_data`
```
渲染模板时用到的上下文数据。 它必须是一个dict。

例如： {'foo': 123}
```
`SimpleTemplateResponse.rendered_content`
```
使用当前的模板和上下文数据渲染出来的响应内容。
```
`SimpleTemplateResponse.is_rendered`
```
一个布尔值，表示响应内容是否已经渲染。
```
#### 方法
`SimpleTemplateResponse.__init__(template, context=None, content_type=None, status=None, charset=None, using=None)`
```python
使用给定的模板、上下文、Content-Type、HTTP 状态和字符集初始化一个SimpleTemplateResponse 对象。
# 参数
template  # 一个与后端有关的模板对象（例如get_template() 返回的对象）、模板的名称或者一个模板名称列表。
context  # 一个dict，包含要添加到模板上下文中的值。 它默认是一个空的字典。
content_type  # HTTP Content-Type 头部包含的值，包含MIME 类型和字符集的编码。 如果指定content_type，则使用它的值。 否则，使用DEFAULT_CONTENT_TYPE。
status  # 响应的HTTP 状态码。
charset  # 响应编码使用的字符集。 如果没有给出则从content_type中提取，如果提取不成功则使用 DEFAULT_CHARSET 设置。
using  # 加载模板使用的模板引擎的NAME。
```
`SimpleTemplateResponse.resolve_context(context)`
```
预处理上下文数据（context data），这个上下文数据将会被用来渲染的模版。 接受包含上下文数据的一个dict。 默认返回同一个dict。

若要自定义上下文，请覆盖这个方法。
```
`SimpleTemplateResponse.resolve_template(template)`
```
解析渲染用到的模板实例。 接收一个与后端有关的模板对象（例如get_template() 返回的对象）、模板的名称或者一个模板名称列表。
返回将被渲染的模板对象。
若要自定义模板的加载，请覆盖这个方法。
```
`SimpleTemplateResponse.add_post_render_callback()`
```
添加一个渲染之后调用的回调函数。 这个钩子可以用来延迟某些特定的处理操作（例如缓存）到渲染之后。

如果SimpleTemplateResponse 已经渲染，那么回调函数将立即执行。

调用时，将只传递给回调函数一个参数 —— 渲染后的 SimpleTemplateResponse 实例。

如果回调函数返回非None 值，它将用作响应并替换原始的响应对象（以及传递给下一个渲染之后的回调函数，以此类推）。
```
`SimpleTemplateResponse.render()`
```
设置response.content 为SimpleTemplateResponse.rendered_content 的结果，执行所有渲染之后的回调函数，最后返回生成的响应对象。

render() 只在第一次调用它时其作用。 以后的调用将返回第一次调用的结果。
```
## TemplateResponse
```
class TemplateResponse
TemplateResponse 是 SimpleTemplateResponse 的子类，而且能知道当前的 HttpRequest。
```
### 方法
`TemplateResponse.__init__(request, template, context=None, content_type=None, status=None, charset=None, using=None)`
```python
使用给定的模板、上下文、Content-Type、HTTP 状态和字符集实例化一个TemplateResponse 对象。
# 参数
request
一个HttpRequest实例。
template
一个与后端有关的模板对象（例如get_template() 返回的对象）、模板的名称或者一个模板名称列表。
context
一个dict，包含要添加到模板上下文中的值。 它默认是一个空的字典。
content_type
HTTP Content-Type 头部包含的值，包含MIME 类型和字符集的编码。 如果指定content_type，则使用它的值。 否则，使用DEFAULT_CONTENT_TYPE。
status
响应的HTTP 状态码。
charset
响应编码使用的字符集。 如果没有给出则从content_type中提取，如果提取不成功则使用 DEFAULT_CHARSET 设置。
using
加载模板使用的模板引擎的NAME。
```
###  渲染过程

在TemplateResponse 实例返回给客户端之前，它必须被渲染。 渲染的过程采用模板和上下文变量的中间表示形式，并最终将它转换为可以发送给客户端的字节流。

有三种情况会渲染TemplateResponse：
```
1. 使用SimpleTemplateResponse.render() 方法显式渲染TemplateResponse 实例的时候。
2. 通过给response.content 赋值显式设置响应内容的时候。
3. 传递给模板响应中间件之后，响应中间件之前。
```
TemplateResponse 只能渲染一次。 第一次调用SimpleTemplateResponse.render()设置响应的内容；后续呈现调用不会更改响应内容。

然而，当显式给response.content 赋值时，修改会始终生效。 如果你想强制重新渲染内容，你可以重新计算渲染的内容并手工赋值给响应的内容：
```shell
# Set up a rendered TemplateResponse
>>> from django.template.response import TemplateResponse
>>> t = TemplateResponse(request, 'original.html', {})
>>> t.render()
>>> print(t.content)
Original content

# Re-rendering doesn't change content
>>> t.template_name = 'new.html'
>>> t.render()
>>> print(t.content)
Original content

# Assigning content does change, no render() call required
>>> t.content = t.rendered_content
>>> print(t.content)
New content
```
### 后渲染回调
某些操作 —— 例如缓存 —— 不可以在没有渲染的模板上执行。 它们必须在完整的渲染后的模板上执行。

如果你正在使用中间件，解决办法很容易。 中间件提供多种在从视图退出时处理响应的机会。 如果您将行为放在响应中间件中，则可以保证在模板呈现发生后执行。

然而，如果正在使用装饰器，就不会有这样的机会。 装饰器中定义的行为会立即执行。

为了补偿这一点（以及其它类似的使用情形）TemplateResponse 允许你注册在渲染完成时调用的回调函数。 使用这个回调函数，你可以延迟某些关键的处理直到你可以保证渲染后的内容是可以访问的。

要定义渲染后的回调函数，只需定义一个接收一个响应作为参数的函数并将这个函数注册到模板响应中：
```
from django.template.response import TemplateResponse

def my_render_callback(response):
    # Do content-sensitive processing
    do_post_processing()

def my_view(request):
    # Create a response
    response = TemplateResponse(request, 'mytemplate.html', {})
    # Register the callback
    response.add_post_render_callback(my_render_callback)
    # Return the response
    return response
```
my_render_callback() 将在mytemplate.html 渲染之后调用，并将被传递一个TemplateResponse 实例作为参数。

如果模板已经渲染，回调函数将立即执行。

## 使用
TemplateResponse 对象和普通的django.http.HttpResponse 一样可以用于任何地方。 它也可以用作调用render()的替代方法。

例如，下面这个简单的视图使用一个简单模板和包含查询集的上下文返回一个TemplateResponse：
```python

from django.template.response import TemplateResponse

def blog_index(request):
    return TemplateResponse(request, 'entry_list.html', {'entries': Entry.objects.all()})
```