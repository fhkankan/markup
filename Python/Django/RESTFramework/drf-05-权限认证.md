# drf权限认证

目前，我们的API对谁可以编辑或删除代码段没有任何限制。我们希望有一些更高级的行为，以确保

```
snippets始终与创建者相关联。
只有经过身份验证的用户才能创建snippets。
只有一个snippet的创建者可以更新或删除它。
未经身份验证的请求应具有完全只读访问权限。
```

## 对model添加信息

我们将对Snippet模型类进行一些更改。首先，让我们添加几个字段。其中一个字段将用于表示创建snippet的用户。另一个字段将用于存储代码的突出显示的HTML表示。

```python
# models.py中的Snippet模型
owner = models.ForeignKey('auth.User', related_name='snippets', on_delete=models.CASCADE)

highlighted = models.TextField()
```

我们还需要确保在保存模型时，使用pygments代码突出显示库填充突出显示的字段。

```python
# models.py
from pygments.lexers import get_lexer_by_name
from pygments.formatters.html import HtmlFormatter
from pygments import highlight

# 在model类中添加save方法
def save(self, *args, **kwargs):
    """
    Use the `pygments` library to create a highlighted HTML
    representation of the code snippet.
    """
    lexer = get_lexer_by_name(self.language)
    linenos = 'table' if self.linenos else False
    options = {'title': self.title} if self.title else {}
    formatter = HtmlFormatter(style=self.style, linenos=linenos,
                              full=True, **options)
    self.highlighted = highlight(self.code, lexer, formatter)
    super(Snippet, self).save(*args, **kwargs)
```

更新数据库

```python
rm -f db.sqlite3
rm -r snippets/migrations
python manage.py makemigrations snippets
python manage.py migrate
```

创建用户

```shell
python manage,py createsuperuser
```

## 对User models添加端点

- 序列化

既然我们已经有一些用户可以使用，我们最好将这些用户的表示添加到我们的API中。创建新的序列化器很简单。在`serializers.py`中添加

```python
# serializers.py
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    snippets = serializers.PrimaryKeyRelatedField(many=True, queryset=Snippet.objects.all())

    class Meta:
        model = User
        fields = ('id', 'username', 'snippets')
```

因为“snippets”在User模型上是反向关系，所以在使用`ModelSerializer`类时默认情况下不会包含它，因此我们需要为它添加显式字段。

- 视图

我们还将为`views.py`添加一些视图。我们只想对用户表示使用只读视图，因此我们将使用`ListAPIView`和`RetrieveAPIView`基于类的通用视图。

```python
# views.py
from django.contrib.auth.models import User
from snippets.serializers import UserSerializer


class UserList(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer


class UserDetail(generics.RetrieveAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
```

- Urls

```python
# snippets/urls.py
path('users/', views.UserList.as_view()),
path('users/<int:pk>/', views.UserDetail.as_view()),
```

## 对Snippets关联Users

现在，如果我们创建了代码段，则无法将创建代码段的用户与代码段实例相关联。用户不是作为序列化表示的一部分发送的，而是传入请求的属性。

我们处理的方法是在我们的代码段视图上覆盖`.perform_create()`方法，这允许我们修改实例保存的管理方式，并处理传入请求或请求的URL中隐含的任何信息。

```python
# views.py中的SnippetList视图类添加方法
def perform_create(self, serializer):
    serializer.save(owner=self.request.user)
```

我们的序列化程序的`create()`方法以及来自请求的验证数据，将传递一个额外的`'owner'`字段，

## 更新Serializer

既然片段与创建它们的用户相关联，那么让我们更新我们的SnippetSerializer以反映它。将以下字段添加到serializers.py中的序列化程序定义

```python
# serializers.py
owner = serializers.ReadOnlyField(source='owner.username')
```

注意：确保您还将`'owner'`添加到内部`Meta`类的字段列表中。

这个领域做得非常有趣。`source`参数控制用于填充字段的属性，并且可以指向序列化实例上的任何属性。它也可以采用上面显示的虚线表示法，在这种情况下，它将以与Django模板语言一样的方式遍历给定的属性。

我们添加的字段是无类型的`ReadOnlyField`类，与其他类型的字段（如`CharField，BooleanField`等）相比，无类型的`ReadOnlyField`始终是只读的，将用于序列化表示，但不会用于在反序列化时更新模型实例。我们也可以在这里使用`CharField(read_only = True)`。

## 对views增加权限许可

既然代码片段与用户相关联，我们希望确保只有经过身份验证的用户才能创建，更新和删除代码段。

REST框架包含许多权限类，我们可以使用这些权限来限制谁可以访问给定视图。在这种情况下，我们正在寻找的是`IsAuthenticatedOrReadOnly`，这将确保经过身份验证的请求获得读写访问权限，未经身份验证的请求获得只读访问权限。

首先在views模块中添加以下导入

```python
from rest_framework import permissions
```

然后，将以下属性添加到`SnippetList`和`SnippetDetail`视图类。

```python
permission_classes = (permissions.IsAuthenticatedOrReadOnly,)
```

## 对API增加login

如果您此时打开浏览器并导航到可浏览的API，您将发现您无法再创建新的代码段。为此，我们需要能够以用户身份登录。

我们可以通过在项目级`urls.py`文件中编辑URLconf来添加登录视图以与可浏览API一起使用。

```python
from django.conf.urls import include

# 添加模式以包含可浏览API的登录和注销视图。
urlpatterns += [
    # 'api-auth /'部分实际上可以是您想要使用的任何URL。
    path('api-auth/', include('rest_framework.urls')),  
]
```

现在，如果您再次打开浏览器并刷新页面，您将在页面右上角看到“登录”链接。如果您以之前创建的用户之一登录，则可以再次创建代码段。

创建几个代码段后，导航到`/ users /`端点，并注意该表示包含每个用户的`snippets`字段中与每个用户关联的snippet的ID列表。

## 对象许可级别

实际上，我们希望所有人都可以看到所有代码段，但也要确保只有创建代码段的用户才能更新或删除它。为此，我们需要创建自定义权限。

在snippets应用处，创建一个新的文件`permissions.py`

```python
# permissions.py
from rest_framework import permissions


class IsOwnerOrReadOnly(permissions.BasePermission):
    """
    Custom permission to only allow owners of an object to edit it.
    """
    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request,
        # so we'll always allow GET, HEAD or OPTIONS requests.
        if request.method in permissions.SAFE_METHODS:
            return True

        # Write permissions are only allowed to the owner of the snippet.
        return obj.owner == request.user
```

现在，我们可以通过编辑`SnippetDetail`视图类上的`permission_classes`属性，将该自定义权限添加到我们的代码段实例端点

```python
from snippets.permissions import IsOwnerOrReadOnly

permission_classes = (permissions.IsAuthenticatedOrReadOnly,
                      IsOwnerOrReadOnly,)
```

现在，如果再次打开浏览器，如果您以创建代码段的同一用户身份登录，则会发现“DELETE”和“PUT”操作仅显示在代码段实例端点上。

## 测试认证API

因为我们现在拥有API的一组权限，所以如果我们想要编辑任何代码段，我们需要对它们的请求进行身份验证。我们还没有设置任何身份验证类，因此当前应用了默认值，即`SessionAuthentication`和`BasicAuthentication`。

当我们通过Web浏览器与API交互时，我们可以登录，然后浏览器会话将为请求提供所需的身份验证。如果我们以编程方式与API交互，我们需要在每个请求上明确提供身份验证凭据。

如果我们尝试在不进行身份验证的情况下创建代码段，则会收到错误消息

```shell
http POST http://127.0.0.1:8000/snippets/ code="print(123)"

{
    "detail": "Authentication credentials were not provided."
}
```

我们可以通过包含我们之前创建的用户之一的用户名和密码来成功提出请求。

```shell
http -a admin:password123 POST http://127.0.0.1:8000/snippets/ code="print(789)"

{
    "id": 1,
    "owner": "admin",
    "title": "foo",
    "code": "print(789)",
    "linenos": false,
    "language": "python",
    "style": "friendly"
}
```

