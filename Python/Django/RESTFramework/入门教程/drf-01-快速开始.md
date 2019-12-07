# 快速开始

## 项目创建

创建一个新的django项目`tutorial`，新增一个应用`quickStart`

```shell
# Create the project directory
mkdir tutorial
cd tutorial

# Create a virtual environment to isolate our package dependencies locally
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install Django and Django REST framework into the virtual environment
pip install django
pip install djangorestframework

# Set up a new project with a single application
django-admin startproject tutorial .  # Note the trailing '.' character
cd tutorial
django-admin startapp quickstart
cd ..
```

项目目录

```
$ pwd
<some path>/tutorial
$ find .
.
./manage.py
./tutorial
./tutorial/__init__.py
./tutorial/quickstart
./tutorial/quickstart/__init__.py
./tutorial/quickstart/admin.py
./tutorial/quickstart/apps.py
./tutorial/quickstart/migrations
./tutorial/quickstart/migrations/__init__.py
./tutorial/quickstart/models.py
./tutorial/quickstart/tests.py
./tutorial/quickstart/views.py
./tutorial/settings.py
./tutorial/urls.py
./tutorial/wsgi.py
```

数据库同步

```shell
python manage.py migrate
```

初始化超级用户

```shell
python manage.py createsuperuser --email admin@example.com --username admin
```

## 序列化

首先，我们将定义一些序列化器。让我们创建一个名为`tutorial/quickstart/serializers.py`的新模块，我们将其用于数据表示。

```python
# tutorial/quickstart/serializers.py
from django.contrib.auth.models import User, Group
from rest_framework import serializers


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')
```

注意，我们在这种情况下使用超链接关系与`HyperlinkedModelSerializer`。您还可以使用主键和其他各种关系，但超链接是一种很好的RESTful设计。

## Views

```python
# tutorial/quickstart/views.py
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from tutorial.quickstart.serializers import UserSerializer, GroupSerializer


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
```

我们不是编写多个视图，而是将所有常见行为组合在一起，称为`ViewSets`。

如果需要，我们可以轻松地将它们分解为单独的视图，但使用视图集可以使视图逻辑组织良好，并且非常简洁。

## URLs

```python
# tutorial/urls.py
from django.urls import include, path
from rest_framework import routers
from tutorial.quickstart import views

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
```

因为我们使用视图集而不是视图，所以我们可以通过简单地使用路由器类注册视图集来自动为我们的API生成URL conf。

同样，如果我们需要更多地控制API URL，我们可以简单地使用常规的基于类的视图，并明确地编写URL conf。

最后，我们将包含默认登录和注销视图，以便与可浏览API一起使用。这是可选的，但如果您的API需要身份验证并且您想要使用可浏览的API，则非常有用。

## Pagination

分页允许您控制每页返回的对象数。要启用它，请将以下行添加到`tutorial/settings.py`中

```python
# tutorial/settings.py
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10
}
```

## Settings

添加 `'rest_framework'` 至 `INSTALLED_APPS`

```python
# tutorial/settings.py
INSTALLED_APPS = (
    ...
    'rest_framework',
)
```

## 测试API

启动服务

```shell
python manage.py runserver 
```

- 测试接口

方式一

```
# 浏览器地址行输入
http://127.0.0.1:8000/users/
# 点击右上角登陆
```

方式二

```shell
# 命令行curl
bash: curl -H 'Accept: application/json; indent=4' -u admin:password123 http://127.0.0.1:8000/users/
{
    "count": 2,
    "next": null,
    "previous": null,
    "results": [
        {
            "email": "admin@example.com",
            "groups": [],
            "url": "http://127.0.0.1:8000/users/1/",
            "username": "admin"
        },
        {
            "email": "tom@example.com",
            "groups": [                ],
            "url": "http://127.0.0.1:8000/users/2/",
            "username": "tom"
        }
    ]
}

```

方式三

```shell
# 命令行工具httpie
bash: http -a admin:password123 http://127.0.0.1:8000/users/

HTTP/1.1 200 OK
...
{
    "count": 2,
    "next": null,
    "previous": null,
    "results": [
        {
            "email": "admin@example.com",
            "groups": [],
            "url": "http://localhost:8000/users/1/",
            "username": "paul"
        },
        {
            "email": "tom@example.com",
            "groups": [                ],
            "url": "http://127.0.0.1:8000/users/2/",
            "username": "tom"
        }
    ]
}
```

