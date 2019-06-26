# drf类视图

我们还可以使用基于类的视图而不是基于函数的视图来编写API视图。我们将看到这是一个强大的模式，允许我们重用通用功能，并帮助我们保持代码DRY。

## 使用类视图重构

我们首先将根视图重写为基于类的视图。所有这些都涉及到`views.py`的一些重构。

```python
# veiws.py
from snippets.models import Snippet
from snippets.serializers import SnippetSerializer
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status


class SnippetList(APIView):
    """
    List all snippets, or create a new snippet.
    """
    def get(self, request, format=None):
        snippets = Snippet.objects.all()
        serializer = SnippetSerializer(snippets, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = SnippetSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

到现在为止还挺好。它看起来与前一种情况非常相似，但我们在不同的HTTP方法之间有了更好的分离。我们还需要更新views.py中的实例视图。

```python
class SnippetDetail(APIView):
    """
    Retrieve, update or delete a snippet instance.
    """
    def get_object(self, pk):
        try:
            return Snippet.objects.get(pk=pk)
        except Snippet.DoesNotExist:
            raise Http404

    def get(self, request, pk, format=None):
        snippet = self.get_object(pk)
        serializer = SnippetSerializer(snippet)
        return Response(serializer.data)

    def put(self, request, pk, format=None):
        snippet = self.get_object(pk)
        serializer = SnippetSerializer(snippet, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk, format=None):
        snippet = self.get_object(pk)
        snippet.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
```

## URLs

我们还需要稍微重构我们的路由，因为我们正在使用基于类的视图。

```python
# snippets/urls.py
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from snippets import views

urlpatterns = [
    path('snippets/', views.SnippetList.as_view()),
    path('snippets/<int:pk>/', views.SnippetDetail.as_view()),
]

urlpatterns = format_suffix_patterns(urlpatterns)
```

## 使用mixins

使用基于类的视图的一大胜利是它允许我们轻松地编写可重用的行为。

到目前为止，我们一直使用的创建/检索/更新/删除操作对于我们创建的任何模型支持的API视图都非常相似。这些常见行为在REST框架的mixin类中实现。

让我们看一下如何使用mixin类组合视图。这是我们的views.py模块。

```python
from snippets.models import Snippet
from snippets.serializers import SnippetSerializer
from rest_framework import mixins
from rest_framework import generics


class SnippetList(mixins.ListModelMixin,
                  mixins.CreateModelMixin,
                  generics.GenericAPIView):
    queryset = Snippet.objects.all()
    serializer_class = SnippetSerializer

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)
      

class SnippetDetail(mixins.RetrieveModelMixin,
                    mixins.UpdateModelMixin,
                    mixins.DestroyModelMixin,
                    generics.GenericAPIView):
    queryset = Snippet.objects.all()
    serializer_class = SnippetSerializer

    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)
```

说明

```
GenericAPIView			构建视图，提供核心功能
ListModelMixin			提供list()
CreateModelMixin		提供create()
RetrieveModelMixin	提供retrive()
UpdateModelMixin		提供update()
DestroyModelMixin		提供destroy()

get/post/put/delete方法显式绑定到适当的操作。
```

## 使用通用类视图

使用mixin类，我们重写了视图，使用的代码比以前略少，但我们可以更进一步。REST框架提供了一组已经混合的通用视图，我们可以使用它来减少我们的`views.py`模块。

```python
from snippets.models import Snippet
from snippets.serializers import SnippetSerializer
from rest_framework import generics


class SnippetList(generics.ListCreateAPIView):
    queryset = Snippet.objects.all()
    serializer_class = SnippetSerializer


class SnippetDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Snippet.objects.all()
    serializer_class = SnippetSerializer
```



