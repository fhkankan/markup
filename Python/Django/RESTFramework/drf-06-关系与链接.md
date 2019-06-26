# drf关系与链接

目前我们的API中的关系是使用主键表示的。在本教程的这一部分中，我们将通过使用超链接建立关系来提高API的内聚性和可发现性。

## 为API创建根端点

现在我们有`snippets`和`users`的端点，但我们的API没有单一的入口点。要创建一个，我们将使用基于函数的常规视图和我们之前介绍的`@api_view`装饰器。

```python
# snippets/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse


@api_view(['GET'])
def api_root(request, format=None):
    return Response({
        'users': reverse('user-list', request=request, format=format),
        'snippets': reverse('snippet-list', request=request, format=format)
    })
```

这里应该注意两件事。首先，我们使用REST框架的`reverse`函数来返回完全限定的URL;第二，URL模式由便利名称标识，稍后我们将在我们的`snippets / urls.py`中声明。

## 为高亮的snippets创建断点

我们的pastebin API中仍然缺少的另一个显而易见的事情是代码突出显示端点。

与我们所有其他API端点不同，我们不想使用JSON，而只是呈现HTML表示。REST框架提供了两种HTML呈现器样式，一种用于处理使用模板呈现的HTML，另一种用于处理预呈现的HTML。第二个渲染器是我们要用于此端点的渲染器。

- Views

在创建代码突出显示视图时，我们需要考虑的另一件事是，没有现有的具体通用视图可供我们使用。我们不返回对象实例，而是返回对象实例的属性。

我们将使用基类来表示实例，而不是使用具体的通用视图，并创建我们自己的`.get()`方法。

```python
# snippets/views.py
from rest_framework import renderers
from rest_framework.response import Response

class SnippetHighlight(generics.GenericAPIView):
    queryset = Snippet.objects.all()
    renderer_classes = (renderers.StaticHTMLRenderer,)

    def get(self, request, *args, **kwargs):
        snippet = self.get_object()
        return Response(snippet.highlighted)
```

- URLs

像往常一样，我们需要将我们创建的新视图添加到URLconf中。

```python
# snippets/urls.py

# 为新API根添加url模式
path('', views.api_root),

# 为片段突出显示添加url模式
path('snippets/<int:pk>/highlight/', views.SnippetHighlight.as_view()),
```

## 超链接API

处理实体之间的关系是Web API设计中更具挑战性的方面之一。我们可以选择多种不同的方式来表示关系

```
使用主键。
在实体之间使用超链接。
在相关实体上使用唯一标识表示字段。
使用相关实体的默认字符串表示形式。
将相关实体嵌套在父表示中。
一些其他自定义表示。
```

REST框架支持所有这些样式，并且可以跨正向或反向关系应用它们，或者将它们应用于自定义管理器（如通用外键）。

在这种情况下，我们想在实体之间使用超链接样式。为此，我们将修改我们的序列化程序以扩展`HyperlinkedModelSerializer`而不是现有的`ModelSerializer`。

`HyperlinkedModelSerializer`与`ModelSerializer`有如下不同

```python
1. 默认情况下，它不包含id字段。
2. 它包含一个url字段，使用HyperlinkedIdentityField。
3. 关系使用HyperlinkedRelatedField而不是PrimaryKeyRelatedField。
```

我们可以轻松地重写现有的序列化程序以使用超链接。

```python
# snippets/serializers.py
class SnippetSerializer(serializers.HyperlinkedModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')
    highlight = serializers.HyperlinkedIdentityField(view_name='snippet-highlight', format='html')

    class Meta:
        model = Snippet
        fields = ('url', 'id', 'highlight', 'owner',
                  'title', 'code', 'linenos', 'language', 'style')


class UserSerializer(serializers.HyperlinkedModelSerializer):
    snippets = serializers.HyperlinkedRelatedField(many=True, view_name='snippet-detail', read_only=True)

    class Meta:
        model = User
        fields = ('url', 'id', 'username', 'snippets')
```

请注意，我们还添加了一个新的`highlight`字段。此字段与`url`字段的类型相同，只是它指向`snippet-highlight`url模式，而不是`snippet-detail`url模式。

因为我们已经包含格式后缀的URL，例如`'.json'`，我们还需要在`highlight`字段上指出它返回的任何格式后缀超链接应该使用`'.html'`后缀。

## 命名URL模式

如果我们要使用超链接API，我们需要确保命名我们的URL模式。我们来看看我们需要命名的URL模式。

```
1. API的根引用 'user-list'和'snippet-list'.
2. snippet序列化器包含一个引用'snippet-highlight'的字段.
3. user序列化器包含一个引用'snippet-detail'的字段.
4. snippet和user序列化器包含'url'字段， 默认情况下会引用'{model_name}-detail', 本例中为'snippet-detail'和'user-detail'.
```

将所有这些名称添加到我们的URLconf后，我们的`snippets/urls.py`文件应该看起来像

```python
# snippets/urls.py
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from snippets import views

# API endpoints
urlpatterns = format_suffix_patterns([
    path('', views.api_root),
    path('snippets/',
        views.SnippetList.as_view(),
        name='snippet-list'),
    path('snippets/<int:pk>/',
        views.SnippetDetail.as_view(),
        name='snippet-detail'),
    path('snippets/<int:pk>/highlight/',
        views.SnippetHighlight.as_view(),
        name='snippet-highlight'),
    path('users/',
        views.UserList.as_view(),
        name='user-list'),
    path('users/<int:pk>/',
        views.UserDetail.as_view(),
        name='user-detail')
])
```

## 添加分页功能

用户和代码片段的列表视图最终可能会返回很多实例，因此我们确实希望确保对结果进行分页，并允许API客户端逐步浏览每个页面。

我们可以通过稍微修改`tutorial/settings.py`文件来更改默认列表样式以使用分页。添加以下设置：

```python
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10
}
```

请注意，REST框架中的设置都被命名为单个字典设置，名为`REST_FRAMEWORK`，这有助于将它们与其他项目设置完全分开。如果我们需要，我们也可以自定义分页样式，但在这种情况下，我们只会坚持使用默认值。