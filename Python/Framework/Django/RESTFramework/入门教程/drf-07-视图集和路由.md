# drf视图集和路由

REST框架包括用于处理`ViewSets`的抽象，允许开发人员专注于建模API的状态和交互，并使URL构造基于通用约定自动处理。

`ViewSet`类与`View`类几乎相同，只是它们提供诸如`read`或`update`之类的操作，而不是诸如`get`或`put`之类的方法处理程序。

`ViewSet`类在最后一刻只绑定到一组方法处理程序，当它被实例化为一组视图时，通常使用一个处理为您定义URL conf的复杂性的`Router`类。

## 重构以使用ViewSet

首先，让我们将`UserList`和`UserDetail`视图重构为单个`UserViewSet`。我们可以删除这两个视图，并用一个类替换它们：

```python
from rest_framework import viewsets


# 使用ReadOnlyModelViewSet类自动提供默认的“只读”操作。
# 仍然像使用常规视图时那样设置queryset和serializer_class属性，但我们不再需要向两个单独的类提供相同的信息。
class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    This viewset automatically provides `list` and `detail` actions.
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer
```

接下来，我们将替换`SnippetList`，`SnippetDetail`和`SnippetHighlight`视图类。我们可以删除这三个视图，并再次用一个类替换它们。

```python
from rest_framework.decorators import action
from rest_framework.response import Response

# 使用ModelViewSet类来获取完整的默认读写操作集。
class SnippetViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.
    Additionally we also provide an extra `highlight` action.
    """
    queryset = Snippet.objects.all()
    serializer_class = SnippetSerializer
    permission_classes = (permissions.IsAuthenticatedOrReadOnly,
                          IsOwnerOrReadOnly,)

    # 使用了@action装饰器来创建一个名为highlight的自定义操作。此装饰器可用于添加任何不适合标准创建/更新/删除样式的自定义端点。
    # 默认情况下，使用@action装饰器的自定义操作将响应GET请求。如果我们想要一个响应POST请求的动作，我们可以使用methods参数。
    # 默认情况下，自定义操作的URL取决于方法名称本身。如果要更改url的构造方式，可以将url_path包含为decorator关键字参数。
    @action(detail=True, renderer_classes=[renderers.StaticHTMLRenderer])
    def highlight(self, request, *args, **kwargs):
        snippet = self.get_object()
        return Response(snippet.highlighted)

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)
```

## 将ViewSets显式绑定到URL

处理程序方法仅在定义URLConf时绑定到操作。要了解幕后发生了什么，我们首先要从ViewSets中明确创建一组视图。

在`snippets/urls.py`文件中，我们将`ViewSet`类绑定到一组具体视图中。

```python
# snippets/urls.py
from snippets.views import SnippetViewSet, UserViewSet, api_root
from rest_framework import renderers

snippet_list = SnippetViewSet.as_view({
    'get': 'list',
    'post': 'create'
})
snippet_detail = SnippetViewSet.as_view({
    'get': 'retrieve',
    'put': 'update',
    'patch': 'partial_update',
    'delete': 'destroy'
})
snippet_highlight = SnippetViewSet.as_view({
    'get': 'highlight'
}, renderer_classes=[renderers.StaticHTMLRenderer])
user_list = UserViewSet.as_view({
    'get': 'list'
})
user_detail = UserViewSet.as_view({
    'get': 'retrieve'
})
```

请注意我们如何通过将http方法绑定到每个视图所需的操作来从每个ViewSet类创建多个视图。

现在我们已将资源绑定到具体视图中，我们可以像往常一样使用URL conf注册视图。

```python
urlpatterns = format_suffix_patterns([
    path('', api_root),
    path('snippets/', snippet_list, name='snippet-list'),
    path('snippets/<int:pk>/', snippet_detail, name='snippet-detail'),
    path('snippets/<int:pk>/highlight/', snippet_highlight, name='snippet-highlight'),
    path('users/', user_list, name='user-list'),
    path('users/<int:pk>/', user_detail, name='user-detail')
])
```

## 使用路由器

因为我们使用的是`ViewSet`类而不是`View`类，所以我们实际上不需要自己设计URL。可以使用Router类自动处理将资源连接到视图和URL的约定。我们需要做的就是用路由器注册适当的视图集，然后让它完成剩下的工作。

```python
# snippets/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from snippets import views

# Create a router and register our viewsets with it.
router = DefaultRouter()
router.register(r'snippets', views.SnippetViewSet)
router.register(r'users', views.UserViewSet)

# The API URLs are now determined automatically by the router.
urlpatterns = [
    path('', include(router.urls)),
]
```

向路由器注册视图集与提供urlpattern类似。我们包括两个参数 - 视图的URL前缀和视图集本身。

我们正在使用的`DefaultRouter`类也会自动为我们创建API根视图，因此我们现在可以从`views`模块中删除`api_root`方法。

## 视图与视图集之间的权衡

使用视图集可能是一个非常有用的抽象。它有助于确保URL约定在您的API中保持一致，最大限度地减少您需要编写的代码量，并使您可以专注于API提供的交互和表示，而不是URL conf的细节。

这并不意味着它始终是正确的方法。当使用基于类的视图而不是基于函数的视图时，需要考虑类似的一组权衡。使用视图集不如单独构建视图那么明确。

```
ViewSet
		ViewSetMixin
		ViewSet
		GenericViewSet
		ReadOnlyModelViewSet
		ModelViewSet

APIView
		GenericAPIView
		CreateAPIView
		ListAPIView
		RetrieveAPIView
		DestroyAPIView
		UpdateAPIView
		ListCreateAPIView
		RetrieveUpdateAPIView
		RetrieveDestroyAPIView
		RetrieveUpdateDestroyAPIView						
						
Mixin
		CreateModelMixin
		ListModeMixin
		UpdateModelMixin
		RetrieveModeMixin
		DestoryModelMixin
				
APIView

View
```

![视图和视图集](/Users/henry/Markup/Python/Django/视图和视图集.png)

