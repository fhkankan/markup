# drf请求与回复

## Request对象

REST框架引入了一个Request对象，该对象扩展了常规的HttpRequest，并提供了更灵活的请求解析。Request对象的核心功能是request.data属性，它类似于request.POST，但对于使用Web API更有用。

```python
request.POST  
# Only handles form data.  Only works for 'POST' method.
request.data  
# Handles arbitrary data.  Works for 'POST', 'PUT' and 'PATCH' methods.
```

## Response对象

REST框架还引入了一个Response对象，它是一种TemplateResponse，它接收未呈现的内容并使用内容协商来确定返回给客户端的正确内容类型。

```python
return Response(data)  
# Renders to content type as requested by the client.
```

## 状态码

在视图中使用数字HTTP状态代码并不总能显示出明显的读数，如果错误代码错误，很容易注意到。REST框架为每个状态代码提供更明确的标识符，例如状态模块中的HTTP_400_BAD_REQUEST。最好始终使用这些而不是使用数字标识符。

## 封装API视图

REST框架提供了两个可用于编写API视图的包装器。

```python
@api_view
# 用于处理基于函数的视图的@api_view装饰器。

APIView
# 用于处理基于类的视图的APIView类。
```

这些包装器提供了一些功能，例如确保在视图中接收Request实例，以及向Response对象添加上下文，以便可以执行内容协商。

包装器还提供行为，例如在适当时返回405 Method Not Allowed响应，以及处理访问带有格式错误输入的request.data时发生的任何ParseError异常。

## 视图

我们不再需要`views.py`中的`JSONResponse`类了，所以继续删除它。一旦完成，我们可以开始略微重构我们的视图。

我们的实例视图是对前一个示例的改进。它更简洁一些，现在代码与我们使用Forms API非常相似。我们还使用命名状态代码，这使得响应意义更加明显。

```python
# views.py
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from snippets.models import Snippet
from snippets.serializers import SnippetSerializer


@api_view(['GET', 'POST'])
def snippet_list(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        snippets = Snippet.objects.all()
        serializer = SnippetSerializer(snippets, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = SnippetSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

以下是views.py模块中单个代码段的视图。

```python
@api_view(['GET', 'PUT', 'DELETE'])
def snippet_detail(request, pk):
    """
    Retrieve, update or delete a code snippet.
    """
    try:
        snippet = Snippet.objects.get(pk=pk)
    except Snippet.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = SnippetSerializer(snippet)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = SnippetSerializer(snippet, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        snippet.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
```

请注意，我们不再明确地将我们的请求或响应绑定到给定的内容类型。`request.data`可以处理传入的json请求，但它也可以处理其他格式。类似地，我们返回带有数据的响应对象，但允许REST框架将响应呈现给我们正确的内容类型。

## RULs

为了利用我们的响应不再硬连接到单个内容类型这一事实，我们将API格式后缀添加到API端点。使用格式后缀为我们提供了明确引用给定格式的URL，这意味着我们的API将能够处理诸如http://example.com/api/items/4.json之类的URL。

视图函数变更

```python
def snippet_list(request, format=None):
def snippet_detail(request, pk, format=None):
```

路由变更

```python
# snippets/urls.py 
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from snippets import views

urlpatterns = [
    path('snippets/', views.snippet_list),
    path('snippets/<int:pk>', views.snippet_detail),
]

urlpatterns = format_suffix_patterns(urlpatterns)
```

我们不一定需要添加这些额外的url模式，但它为我们提供了一种简单，干净的方式来引用特定的格式。

## 测试

获取所有snippets列表

```shell
http http://127.0.0.1:8000/snippets/

HTTP/1.1 200 OK
...
[
  {
    "id": 1,
    "title": "",
    "code": "foo = \"bar\"\n",
    "linenos": false,
    "language": "python",
    "style": "friendly"
  },
  {
    "id": 2,
    "title": "",
    "code": "print(\"hello, world\")\n",
    "linenos": false,
    "language": "python",
    "style": "friendly"
  }
]
```

我们可以控制我们返回的响应的格式

```shell
# 通过使用Accept标头来
http http://127.0.0.1:8000/snippets/ Accept:application/json  # Request JSON
http http://127.0.0.1:8000/snippets/ Accept:text/html         # Request HTML

# 使用添加格式后缀
http http://127.0.0.1:8000/snippets.json  # JSON suffix
http http://127.0.0.1:8000/snippets.api   # Browsable API suffix
```

可以控制我们发送的数据格式

```python
# 使用Content-Type标头
# POST using form data
http --form POST http://127.0.0.1:8000/snippets/ code="print(123)"

{
  "id": 3,
  "title": "",
  "code": "print(123)",
  "linenos": false,
  "language": "python",
  "style": "friendly"
}

# POST using JSON
http --json POST http://127.0.0.1:8000/snippets/ code="print(456)"

{
    "id": 4,
    "title": "",
    "code": "print(456)",
    "linenos": false,
    "language": "python",
    "style": "friendly"
}
```

如果您将--debug开关添加到上面的http请求，您将能够在请求标头中看到请求类型。

