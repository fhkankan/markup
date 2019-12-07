# ViewSets

> 在路由已经确定了用于请求的控制器之后，你的控制器负责理解请求并产生适当的输出。
>
> — [Ruby on Rails 文档](http://guides.rubyonrails.org/routing.html)

Django REST framework允许你将一组相关视图的逻辑组合在单个类（称为 `ViewSet`）中。 在其他框架中，你也可以找到概念上类似于 'Resources' 或 'Controllers'的类似实现。

`ViewSet` 只是**一种基于类的视图，它不提供任何方法处理程序**（如 `.get()`或`.post()`）,而是提供诸如 `.list()` 和 `.create()` 之类的操作。

`ViewSet` 的方法处理程序仅使用 `.as_view()` 方法绑定到完成视图的相应操作。

通常不是在 urlconf 中的视图集中显示注册视图，而是要使用路由类注册视图集，该类会自动为你确定 urlconf。

## 示例

让我们定义一个简单的视图集，可以用来列出或检索系统中的所有用户。

```python
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from myapps.serializers import UserSerializer
from rest_framework import viewsets
from rest_framework.response import Response

class UserViewSet(viewsets.ViewSet):
    """
    A simple ViewSet for listing or retrieving users.
    """
    def list(self, request):
        queryset = User.objects.all()
        serializer = UserSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = User.objects.all()
        user = get_object_or_404(queryset, pk=pk)
        serializer = UserSerializer(user)
        return Response(serializer.data)
```

如果我们需要，我们可以将这个viewset绑定到两个单独的视图，想这样：

```python
user_list = UserViewSet.as_view({'get': 'list'})
user_detail = UserViewSet.as_view({'get': 'retrieve'})
```

通常我们不会这么做，我们会用一个router来注册我们的viewset，让urlconf自动生成。

```python
from myapp.views import UserViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'users', UserViewSet)
urlpatterns = router.urls
```

不需要编写自己的视图集，你通常会想要使用提供默认行为的现有基类。例如：

```python
class UserViewSet(viewsets.ModelViewSet):
    """
    用于查看和编辑用户实例的视图集。
    """
    serializer_class = UserSerializer
    queryset = User.objects.all()
```

与使用 `View` 类相比，使用 `ViewSet` 类有两个主要优点。

- 重复的逻辑可以组合成一个类。在上面的例子中，我们只需要指定一次 `queryset`，它将在多个视图中使用。
- 通过使用 routers, 哦们不再需要自己处理URLconf。

这两者都有一个权衡。使用常规的 views 和 URL confs 更明确也能够为你提供更多的控制。ViewSets有助于快速启动和运行，或者当你有大型的API，并且希望在整个过程中执行一致的 URL 配置。

## 操作

REST framework 中包含的默任 routes 将为标准的 `create/retrieve/update/destroy `类型操作提供路由, 如下所示：

```python
class UserViewSet(viewsets.ViewSet):
    """
    示例 viewset 演示了将由路由器类处理的标准动作。

    如果你使用格式后缀，请务必为每个动作包含一个`format=None` 的关键字参数。
    """

    def list(self, request):
        pass

    def create(self, request):
        pass

    def retrieve(self, request, pk=None):
        pass

    def update(self, request, pk=None):
        pass

    def partial_update(self, request, pk=None):
        pass

    def destroy(self, request, pk=None):
        pass
```

### 内省操作

在分派期间，ViewSet上具有以下属性。

- `basename`-用于创建的URL名称的基础。

- `action`-当前动作的名称（例如`list`,`create`）。

- `detail`-布尔值，指示是否为列表视图或详细信息视图配置了当前操作。后缀-视图集类型的显示后缀-镜像`detail`属性。

- `name`-视图集的显示名称。此参数与后缀互斥。

- `description`-视图集的单个视图的显示描述。

您可以检查这些属性以根据当前操作调整行为。例如，您可以将权限限制为除了类似于以下`list`操作的所有操作：

```python
def get_permissions(self):
    """
    Instantiates and returns the list of permissions that this view requires.
    """
    if self.action == 'list':
        permission_classes = [IsAuthenticated]
    else:
        permission_classes = [IsAdmin]
    return [permission() for permission in permission_classes]
```

### 标记用于路由的其他操作

如果您具有可路由的临时方法，则可以使用`@action`装饰器将其标记为可路由的。像常规操作一样，额外的操作可能打算用于单个对象或整个集合。为了表明这一点，请将`detail`参数设置为`True`或`False`。路由器将相应地配置其URL模式。例如，`DefaultRouter`将配置详细操作以在其网址格式中包含`pk`。

例如：

```python
from django.contrib.auth.models import User
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from myapp.serializers import UserSerializer, PasswordSerializer

class UserViewSet(viewsets.ModelViewSet):
    """
    A viewset that provides the standard actions
    """
    queryset = User.objects.all()
    serializer_class = UserSerializer

    @action(detail=True, methods=['post'])
    def set_password(self, request, pk=None):
        user = self.get_object()
        serializer = PasswordSerializer(data=request.data)
        if serializer.is_valid():
            user.set_password(serializer.data['password'])
            user.save()
            return Response({'status': 'password set'})
        else:
            return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False)
    def recent_users(self, request):
        recent_users = User.objects.all().order_by('-last_login')

        page = self.paginate_queryset(recent_users)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(recent_users, many=True)
        return Response(serializer.data)
```

装饰器可以另外接受仅为路由视图设置的额外参数。例如：

```python
@action(detail=True, methods=['post'], permission_classes=[IsAdminOrIsSelf])
def set_password(self, request, pk=None):
    ...
```

这些装饰器将默认路由 `GET` 请求，但也可以通过使用 `methods` 参数接受其他 HTTP 方法。例如：

```
@action(detail=True, methods=['post', 'delete'])
def unset_password(self, request, pk=None):
   ...
```

然后，这两个新操作将在URL` ^ users / {pk} / set_password /$`和`^ users / {pk} / unset_password / $`中可用

要查看所有其他操作，请调用`.get_extra_actions()`方法。

### 路由其他HTTP方法以执行其他操作

额外的操作可以将其他HTTP方法映射到单独的`ViewSet`方法。例如，上述密码设置/取消方法可以合并为一条路由。请注意，其他映射不接受参数。

```python
@action(detail=True, methods=['put'], name='Change Password')
    def password(self, request, pk=None):
        """Update the user's password."""
        ...

    @password.mapping.delete
    def delete_password(self, request, pk=None):
        """Delete the user's password."""
        ...
```

### 反向操作URL

如果需要获取操作的URL，请使用`.reverse_action()`方法。这是对`reverse()`的方便包装，它自动传递视图的`request`对象，并在`url_name`之前添加`.basename`属性。请注意，`basename`由路由器在`ViewSet`注册期间提供。如果不使用路由器，则必须为`.as_view()`方法提供`basename`参数。

使用上一节中的示例：

```shell
>>> view.reverse_action('set-password', args=['1'])
'http://localhost:8000/api/users/1/set_password'
```

或者，您可以使用由`@action`装饰器设置的`url_name`属性。

```shell
>>> view.reverse_action(view.set_password.url_name, args=['1'])
'http://localhost:8000/api/users/1/set_password'
```

`.reverse_action()`的`url_name`参数应与`@action`装饰器匹配相同的参数。此外，此方法可用于撤消默认操作，例如`list`和`create`。

# API 参考

## ViewSet

`ViewSet` 继承自 `APIView`。你可以使用任何标准属性，如 `permission_classes`, `authentication_classes` 以便控制视图集上的 API 策略。

`ViewSet` 类不提供任何操作的实现。为了使用 `ViewSet` 类，你将重写该类并显式地定义动作实现。

## GenericViewSet

`GenericViewSet` 类继承自 `GenericAPIView`，并提供了 `get_object`， `get_queryset` 方法和其他通用视图基本行为的默认配置，但默认情况不包括任何操作。

为了使用`GenericViewSet`类，您将覆盖该类并混入所需的mixin类，或显式定义操作实现。

## ModelViewSet

`ModelViewSet`类继承自`GenericAPIView`，并通过混合各种mixin类的行为来包括各种操作的实现。

通过`ModelViewSet` 类提供的操作有： `.list()`, `.retrieve()`, `.create()`, `.update()`, `.partial_update()`, `.destroy()`.

- 示例

由于`ModelViewSet`扩展了`GenericAPIView`，因此通常需要至少提供`queryset`和`serializer_class`属性。

例如：

```python
class AccountViewSet(viewsets.ModelViewSet):
    """
    A simple ViewSet for viewing and editing accounts.
    """
    queryset = Account.objects.all()
    serializer_class = AccountSerializer
    permission_classes = [IsAccountAdminOrReadOnly]
```

请注意，您可以使用`GenericAPIView`提供的任何标准属性或方法替代。例如，要使用动态确定其应操作的查询集的`ViewSet`，可以执行以下操作：

```python
class AccountViewSet(viewsets.ModelViewSet):
    """
    A simple ViewSet for viewing and editing the accounts
    associated with the user.
    """
    serializer_class = AccountSerializer
    permission_classes = [IsAccountAdminOrReadOnly]

    def get_queryset(self):
        return self.request.user.accounts.all()
```

但是请注意，从`ViewSet`中删除`queryset`属性后，任何关联的路由器将无法自动导出`Model`的基本名称，因此您必须指定`basename`kwarg作为路由器注册中一部分。

还要注意，尽管默认情况下此类提供了完整的`create / list / retrieve / update / destroy`操作集，但是您可以使用标准权限类来限制可用的操作。

## ReadOnlyModelViewSet

`ReadOnlyModelViewSet`类也继承自`GenericAPIView`。与`ModelViewSet`一样，它也包含各种操作的实现，但与`ModelViewSet`不同的是，它仅提供“只读”操作，即`.list()`和`.retrieve()`。

- 示例

与`ModelViewSet`一样，通常至少需要提供`queryset`和`serializer_class`属性。

例如：

```python
class AccountViewSet(viewsets.ReadOnlyModelViewSet):
    """
    A simple ViewSet for viewing accounts.
    """
    queryset = Account.objects.all()
    serializer_class = AccountSerializer
```

同样，与`ModelViewSet`一样，您可以使用`GenericAPIView`可用的任何标准属性和方法替代。

# 自定义ViewSet基类

您可能需要提供不具有完整`ModelViewSet`操作集的自定义`ViewSet`类，或者以其他方式自定义行为。

- 示例

要创建提供创建，列出和检索操作，从`GenericViewSet`继承并混合所需操作的基本`viewse`t类，请执行以下操作：

```python
class CreateListRetrieveViewSet(mixins.CreateModelMixin,
                                mixins.ListModelMixin,
                                mixins.RetrieveModelMixin,
                                viewsets.GenericViewSet):
    """
    A viewset that provides `retrieve`, `create`, and `list` actions.

    To use it, override the class and set the `.queryset` and
    `.serializer_class` attributes.
    """
    pass
```

通过创建自己的基本`ViewSet`类，您可以提供可以在您的API的多个视图集中重用的常见行为。