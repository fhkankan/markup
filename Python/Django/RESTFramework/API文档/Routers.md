# Routers

> 资源路由允许你快速声明给定的有足够控制器的所有公共路由。而不是为你的index...声明单独的路由，一个强大的路由能在一行代码中声明它们。
>
> — [Ruby on Rails 文档][cite]

某些Web框架（如Rails）提供了自动确定应用程序的URL应如何映射到处理传入请求的逻辑的功能。

REST框架添加了对自动URL路由到Django的支持，并为你提供了一种简单、快速和一致的方式来将视图逻辑连接到一组URL。

## 用法

这里有一个简单的URL conf的例子，它使用 `SimpleRouter`。

```python
from rest_framework import routers

router = routers.SimpleRouter()
router.register(r'users', UserViewSet)
router.register(r'accounts', AccountViewSet)
urlpatterns = router.urls
```

`register()` 方法有两个强制参数：

- `prefix` - 用于此组路由的URL前缀。
- `viewset` - 处理请求的viewset类。

还可以指定一个附加参数（可选）：

- `base_name` - 用于创建的URL名称的基本名称。如果不设置该参数，将根据视图集的`queryset`属性（如果有）来自动生成基本名称。注意，如果视图集不包括`queryset`属性，那么在注册视图集时必须设置`base_name`。

上面的示例将生成以下URL模式：

- URL pattern: `^users/$` Name: `'user-list'`
- URL pattern: `^users/{pk}/$` Name: `'user-detail'`
- URL pattern: `^accounts/$` Name: `'account-list'`
- URL pattern: `^accounts/{pk}/$` Name: `'account-detail'`

------

**注意**: `base_name` 参数用于指定视图名称模式的初始部分。在上面的例子中就是指 `user` 或 `account` 部分。

通常，你*不需要*指定`base_name`参数，但是如果你有自定义`get_queryset`方法的视图集，那么那个视图集可能没有设置`.queryset`属性。当你注册这个视图集的时候，你就有可能会看到类似如下的错误：

```python
'base_name' argument not specified, and could not automatically determine the name from the viewset, as it does not have a '.queryset' attribute.
```

这意味着你需要在注册视图集时显式设置`base_name`参数，因为无法从model名自动确定。

------

### 在路由中使用 `include`

路由器实例上的`.urls`属性只是一个URL模式的标准列表。对于如何添加这些URL，有很多不同的写法。

例如，你可以将`router.urls`附加到现有视图的列表中...

```python
router = routers.SimpleRouter()
router.register(r'users', UserViewSet)
router.register(r'accounts', AccountViewSet)

urlpatterns = [
    url(r'^forgot-password/$', ForgotPasswordFormView.as_view()),
]

urlpatterns += router.urls
```

或者，你可以使用Django的`include`函数，像这样...

```python
urlpatterns = [
    url(r'^forgot-password/$', ForgotPasswordFormView.as_view()),
    url(r'^', include(router.urls)),
]
```

您可以将`include`与应用程序名称空间一起使用：

```python
urlpatterns = [
    url(r'^forgot-password/$', ForgotPasswordFormView.as_view()),
    url(r'^api/', include((router.urls, 'app_name'))),
]
```

或应用程序和实例名称空间：

```python
urlpatterns = [
    url(r'^forgot-password/$', ForgotPasswordFormView.as_view()),
    url(r'^api/', include((router.urls, 'app_name'), namespace='instance_name')),
]
```

有关更多详细信息，请参见Django的 [URL namespaces docs](https://docs.djangoproject.com/en/1.11/topics/http/urls/#url-namespaces) 和 [`include` API reference](https://docs.djangoproject.com/en/2.0/ref/urls/#include)。

> 注
>
> 如果使用带超链接序列化器的命名空间，你还需要确保序列化器上的任何`view_name`参数正确地反映命名空间。在上面的示例中，你需要让超链接到用户详细信息视图的序列化器字段包含一个参数，例如`view_name ='api：user-detail'`。
>
> 自动的`view_name`生成使用`％(model_name)-detail`之类的模式。除非您的模型名称实际发生冲突，否则在使用超链接序列化程序时最好不要为Django REST Framework视图命名。


### 额外链接和操作

视图集可以通过使用`@action`装饰器装饰方法来标记用于路由的其他操作。这些额外的动作将包含在生成的路线中。例如，给定`UserViewSet`类的`set_password`方法：

```python
from myapp.permissions import IsAdminOrIsSelf
from rest_framework.decorators import action

class UserViewSet(ModelViewSet):
    ...

    @action(methods=['post'], detail=True, permission_classes=[IsAdminOrIsSelf])
    def set_password(self, request, pk=None):
        ...
```

将生成以下URL模式：

- URL pattern: `^users/{pk}/set_password/$` 
- URL Name: `'user-set-password'`

默认情况下，URL模式基于方法名称，并且URL名称是`ViewSet.basename`和带连字符的方法名称的组合。如果您不想为这些值中的任何一个使用默认值，则可以向`@action`装饰器提供`url_path`和`url_name`参数。

例如，如果你要将自定义操作的URL更改为`^users/{pk}/change-password/$`, 你可以这样写：

```python
from myapp.permissions import IsAdminOrIsSelf
from rest_framework.decorators import action

class UserViewSet(ModelViewSet):
    ...

    @action(methods=['post'], detail=True, permission_classes=[IsAdminOrIsSelf],
            url_path='change-password', url_name='change_password')
    def set_password(self, request, pk=None):
        ...
```

以上示例将生成以下URL格式：

- URL pattern: `^users/{pk}/change-password/$` 
- URL Name: `'user-change-password'`

# API 向导

## SimpleRouter

该路由器包括标准集合`list`, `create`, `retrieve`, `update`, `partial_update` 和 `destroy`动作的路由。视图集还可以使用`@action`装饰器标记要路由的其他方法。

| URL 样式                        | HTTP 方法                    | 动作                              | URL 名                  |
| ------------------------------- | ---------------------------- | --------------------------------- | ----------------------- |
| `{prefix}/`                     | GET                          | list                              | `{basename}-list`       |
| `{prefix}/`                     | POST                         | create                            | `{basename}-list`       |
| `{prefix}/{url_path}/`          | GET, 或专门制定`methods`变量 | `@action(detail=False)`装饰器方法 | `{basename}-{url_name}` |
| `{prefix}/{lookup}/`            | GET                          | retrieve                          | `{basename}-detail`     |
| `{prefix}/{lookup}/`            | PUT                          | update                            | `{basename}-detail`     |
| `{prefix}/{lookup}/`            | PATCH                        | partial_update                    | `{basename}-detail`     |
| `{prefix}/{lookup}/`            | DELETE                       | destroy                           | `{basename}-detail`     |
| `{prefix}/{lookup}/{url_path}/` | GET, 或专门制定`methods`变量 | `@action(detail=False)`装饰器方法 | `{basename}-{url_name}` |

默认情况下，由`SimpleRouter`创建的URL将附加尾部斜杠。 在实例化路由器时，可以通过将`trailing_slash`参数设置为`False'来修改此行为。比如：

```python
router = SimpleRouter(trailing_slash=False)
```

尾部斜杠在Django中是常见的，但是在其他一些框架（如Rails）中默认不使用。你选择使用哪种风格在很大程度上是你个人偏好问题，虽然一些javascript框架可能需要一个特定的路由风格。

路由器将匹配包含除斜杠和句点字符以外的任何字符的查找值。对于更严格（或更宽松）的查找模式，请在视图集上设置`lookup_value_regex`属性。例如，你可以将查找限制为有效的UUID：

```python
class MyModelViewSet(mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    lookup_field = 'my_model_id'
    lookup_value_regex = '[0-9a-f]{32}'
```

## 默认路由器

这个路由器类似于上面的`SimpleRouter`，但是还包括一个默认返回所有列表视图的超链接的API根视图。它还生成可选的`.json`样式格式后缀的路由。

| URL 样式                                   | HTTP 方法                                  | 动作                                     | URL 名称              |
| ------------------------------------------ | ------------------------------------------ | ---------------------------------------- | --------------------- |
| `[.format]`                                | GET                                        | automatically generated root view        | api-root              |
| `{prefix}/[.format]`                       | GET                                        | list                                     | {basename}-list       |
| `{prefix}/[.format]`                       | POST                                       | create                                   | {basename}-list       |
| `{prefix}/{methodname}/[.format]`          | GET, or as specified by `methods` argument | `@action(detail=False)` decorated method | {basename}-{url_name} |
| `{prefix}/{lookup}/[.format]`              | GET                                        | retrieve                                 | {basename}-detail     |
| `{prefix}/{lookup}/[.format]`              | PUT                                        | update                                   | {basename}-detail     |
| `{prefix}/{lookup}/[.format]`              | PATCH                                      | partial_update                           | {basename}-detail     |
| `{prefix}/{lookup}/[.format]`              | DELETE                                     | destroy                                  | {basename}-detail     |
| `{prefix}/{lookup}/{methodname}/[.format]` | GET, or as specified by `methods` argument | `@action(detail=False)` decorated method | {basename}-{url_name} |

与`SimpleRouter`一样，在实例化路由器时，可以通过将`trailing_slash`参数设置为`False`来删除URL路由的尾部斜杠。

```
router = DefaultRouter(trailing_slash=False)
```

# 自定义路由器

通常你并不需要实现自定义路由器，但如果你对API的网址结构有特定的要求，那它就十分有用了。这样做允许你以可重用的方式封装URL结构，确保你不必为每个新视图显式地编写URL模式。

实现自定义路由器的最简单的方法是继承一个现有的路由器类。`.routes`属性用于模板将被映射到每个视图集的URL模式。`.routes`属性是一个名为tuples的Route对象的列表。

`Route`命名元组的参数是：

**url**: 表示要路由的URL的字符串。可能包括以下格式字符串：

- `{prefix}` - 用于此组路由的URL前缀。
- `{lookup}` - 用于与单个实例进行匹配的查找字段。
- `{trailing_slash}` - 可以是一个'/'或一个空字符串，这取决于`trailing_slash`参数。

**mapping**: HTTP方法名称到视图方法的映射

**name**: 在`reverse`调用中使用的URL的名称。可能包括以下格式字符串：

- `{basename}` - 用于创建的URL名称的基本名称

**initkwargs**: 实例化视图时应传递的任何其他参数的字典。注意，`suffix`参数被保留用于标识视图集类型，在生成视图名称和面包屑链接时使用。

## 自定义动态路由

您还可以自定义`@action`装饰器的路由方式。在`.routes`列表中包括名为`tuple`的`DynamicRoute`，将`detail`参数设置为适合基于列表的路由和基于细节的路由。除了详细信息之外，`DynamicRoute`的参数还有：

**url**: 表示要路由的URL的字符串。可以包括与`Route`相同的格式字符串，并且另外接受`{url_path}`格式字符串。

**name**: 在`reverse`调用中使用的URL的名称。可能包括以下格式字符串：

- `{basename}`-用于创建的URL名称的基础。
- `{url_name}`-提供给`@action`的`url_name`。

**initkwargs**: 实例化视图时应传递的任何其他参数的字典。

## 示例

以下示例将只路由到`list`和`retrieve`操作，并且不使用尾部斜线约定。

```python
from rest_framework.routers import Route, DynamicDetailRoute, SimpleRouter

class CustomReadOnlyRouter(SimpleRouter):
    """
    A router for read-only APIs, which doesn't use trailing slashes.
    """
    routes = [
        Route(
            url=r'^{prefix}$',
            mapping={'get': 'list'},
            name='{basename}-list',
            initkwargs={'suffix': 'List'}
        ),
        Route(
            url=r'^{prefix}/{lookup}$',
            mapping={'get': 'retrieve'},
            name='{basename}-detail',
            initkwargs={'suffix': 'Detail'}
        ),
        DynamicDetailRoute(
            url=r'^{prefix}/{lookup}/{methodnamehyphen}$',
            name='{basename}-{methodnamehyphen}',
            initkwargs={}
        )
    ]
```

让我们来看看我们定义的`CustomReadOnlyRouter`为简单视图生成的路由。

`views.py`:

```python
class UserViewSet(viewsets.ReadOnlyModelViewSet):
    """
    A viewset that provides the standard actions
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer
    lookup_field = 'username'

    @action(detail=True)
    def group_names(self, request, pk=None):
        """
        Returns a list of all the group names that the given
        user belongs to.
        """
        user = self.get_object()
        groups = user.groups.all()
        return Response([group.name for group in groups])
```

`urls.py`:

```python
router = CustomReadOnlyRouter()
router.register('users', UserViewSet)
urlpatterns = router.urls
```

将生成以下映射...

| URL                             | HTTP 方法 | 动作        | URL 名称         |
| ------------------------------- | --------- | ----------- | ---------------- |
| `/users`                        | GET       | list        | user-list        |
| `/users/{username}`             | GET       | retrieve    | user-detail      |
| `/users/{username}/group-names` | GET       | group_names | user-group-names |

有关设置`.routes`属性的另一个示例，请参见`SimpleRouter`类的源代码。

## 高级自定义路由器

如果要提供完全自定义的行为，则可以覆盖`BaseRouter`并覆盖`get_urls(self)`方法。该方法应检查已注册的视图集并返回URL模式列表。可以通过访问`self.registry`属性来检查已注册的前缀，视图集和基本名称元组。

您可能还想覆盖`get_default_basename(self，viewset)`方法，或者在向路由器注册视图集时始终显式设置`basename`参数。

# 第三方包

以下第三方软件包也可用。

## DRF嵌套路由器

 [drf-nested-routers package](https://github.com/alanjds/drf-nested-routers) 软件包提供用于处理嵌套资源的路由器和关系字段。

## ModelRouter(wq.db.rest)

[wq.db package](https://wq.io/wq.db) 软件包提供了一个高级`ModelRouter`类（和单例实例），该类使用`register_model()`API扩展了`DefaultRouter`。就像Django的`admin.site.register`一样，`rest.router.register_model`唯一需要的参数是模型类。可以从模型和全局配置中推断出URL前缀，序列化程序和视图集的合理默认值。

```python
from wq.db import rest
from myapp.models import MyModel

rest.router.register_model(MyModel)
```

## DRF-extensions

 [`DRF-extensions` package](https://chibisov.github.io/drf-extensions/docs/提供 [routers](https://chibisov.github.io/drf-extensions/docs/#routers) for creating [nested viewsets](https://chibisov.github.io/drf-extensions/docs/#nested-routes), [collection level controllers](https://chibisov.github.io/drf-extensions/docs/#collection-level-controllers) with [customizable endpoint names](https://chibisov.github.io/drf-extensions/docs/#controller-endpoint-name).