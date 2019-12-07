# drf序列化

## 创建环境

```shell
# 使用venv创建隔离环境
python3 -m venv env
source env/bin/activate

# 安装包
pip install django
pip install djangorestframework
pip install pygments  # We'll be using this for the code highlighting

# 创建项目
cd ~
django-admin startproject tutorial
cd tutorial
python manage.py startapp snippets

# 配置基础信息
# tutorial/settings.py
INSTALLED_APPS = (
    ...
    'rest_framework',
    'snippets.apps.SnippetsConfig',
)
```

## 创建模型

创建模型类

```python
# snippets/models.py
from django.db import models
from pygments.lexers import get_all_lexers
from pygments.styles import get_all_styles

LEXERS = [item for item in get_all_lexers() if item[1]]
LANGUAGE_CHOICES = sorted([(item[1][0], item[0]) for item in LEXERS])
STYLE_CHOICES = sorted((item, item) for item in get_all_styles())


class Snippet(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField()
    linenos = models.BooleanField(default=False)
    language = models.CharField(choices=LANGUAGE_CHOICES, default='python', max_length=100)
    style = models.CharField(choices=STYLE_CHOICES, default='friendly', max_length=100)

    class Meta:
        ordering = ('created',)
```

迁移同步数据库

```shell
python manage.py makemigrations snippets
python manage.py migrate
```

## 创建序列化类

我们需要开始使用Web API的第一件事是提供一种将模型类`Sinippet`实例序列化和反序列化为表示形式（如json）的方法。我们可以通过声明与Django表单非常相似的序列化器来完成此操作。在名为`serializers.py`的snippets目录中创建一个文件，并添加以下内容。

```python
# serializers.py
from rest_framework import serializers
from snippets.models import Snippet, LANGUAGE_CHOICES, STYLE_CHOICES


class SnippetSerializer(serializers.Serializer):
  	"""
  	定义序列化和反序列化模型类的字段
  	"""
    id = serializers.IntegerField(read_only=True)
    title = serializers.CharField(required=False, allow_blank=True, max_length=100)
    code = serializers.CharField(style={'base_template': 'textarea.html'})
    linenos = serializers.BooleanField(required=False)
    language = serializers.ChoiceField(choices=LANGUAGE_CHOICES, default='python')
    style = serializers.ChoiceField(choices=STYLE_CHOICES, default='friendly')

    def create(self, validated_data):
        """
        给予验证数据时，创建并返回一个新的Snippet类实例。在调用serializer.save()有效
        """
        return Snippet.objects.create(**validated_data)

    def update(self, instance, validated_data):
        """
        给予验证数据，更新并返回一个已经存在的Snippet类实例。在调用serializer.save()有效
        """
        instance.title = validated_data.get('title', instance.title)
        instance.code = validated_data.get('code', instance.code)
        instance.linenos = validated_data.get('linenos', instance.linenos)
        instance.language = validated_data.get('language', instance.language)
        instance.style = validated_data.get('style', instance.style)
        instance.save()
        return instance
```

序列化程序类与Django Form类非常相似，并且在各个字段中包含类似的验证标志，例如`required, max_length, default`。

字段标志还可以控制在某些情况下应该如何显示序列化程序，例如在呈现HTML时。上面的`{'base_template'：'textarea.html'}`标志等同于在Django Form类上使用`widget = widgets.Textarea`。这对于控制可浏览API的显示方式特别有用。

我们实际上也可以通过使用`ModelSerializer`类来节省一些时间，我们稍后会看到，但是现在我们将保持序列化器定义的显式。

## 使用序列化类

```python
# 进入django的shell环境中
python manage.py shell
```

创建类实例

```python
from snippets.models import Snippet
from snippets.serializers import SnippetSerializer
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser

snippet = Snippet(code='foo = "bar"\n')
snippet.save()  # 创建

snippet = Snippet(code='print("hello, world")\n')
snippet.save()  # 创建？修改？
```

查看类实例

```python
serializer = SnippetSerializer(snippet)
# python内置数据类型
serializer.data  # {'id': 2, 'title': '', 'code': 'print("hello, world")\n', 'linenos': False, 'language': 'python', 'style': 'friendly'}  

# json类型
content = JSONRenderer().render(serializer.data)
content  # b'{"id": 2, "title": "", "code": "print(\\"hello, world\\")\\n", "linenos": false, "language": "python", "style": "friendly"}'
```

反序列化

```python
# 解析为python内置数据类型
import io

stream = io.BytesIO(content)
data = JSONParser().parse(stream)

# 转换为对象实例
serializer = SnippetSerializer(data=data)
serializer.is_valid()  # True
serializer.validated_data  # OrderedDict([('title', ''), ('code', 'print("hello, world")\n'), ('linenos', False), ('language', 'python'), ('style', 'friendly')])
serializer.save()  # <Snippet: Snippet object>
```

请注意API与表单的相似程度。当我们开始编写使用我们的序列化程序的视图时，相似性应该变得更加明显。

我们还可以序列化查询集而不是模型实例。为此，我们只需在序列化程序参数中添加`many = True`标志。

```python
serializer = SnippetSerializer(Snippet.objects.all(), many=True)
serializer.data  # [OrderedDict([('id', 1), ('title', ''), ('code', 'foo = "bar"\n'), ('linenos', False), ('language', 'python'), ('style', 'friendly')]), OrderedDict([('id', 2), ('title', ''), ('code', 'print("hello, world")\n'), ('linenos', False), ('language', 'python'), ('style', 'friendly')]), OrderedDict([('id', 3), ('title', ''), ('code', 'print("hello, world")'), ('linenos', False), ('language', 'python'), ('style', 'friendly')])]
```

## 使用模型序列化

我们的SnippetSerializer类正在复制Snippet模型中包含的大量信息。如果我们能够使代码更简洁，那就太好了。

与Django同时提供Form类和ModelForm类的方式相同，REST框架包括Serializer类和ModelSerializer类。

让我们看看使用ModelSerializer类重构我们的序列化程序。再次打开文件片段/ `serializers.py`，并使用以下内容替换SnippetSerializer类。

```python
# serializers.py
class SnippetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Snippet
        fields = ('id', 'title', 'code', 'linenos', 'language', 'style')
```

序列化程序具有的一个不错的属性是，您可以通过打印其表示来检查序列化程序实例中的所有字段。用`python manage.py shell`打开Django shell，然后尝试以下操作

```python
from snippets.serializers import SnippetSerializer
serializer = SnippetSerializer()
print(repr(serializer))
# SnippetSerializer():
#    id = IntegerField(label='ID', read_only=True)
#    title = CharField(allow_blank=True, max_length=100, required=False)
#    code = CharField(style={'base_template': 'textarea.html'})
#    linenos = BooleanField(required=False)
#    language = ChoiceField(choices=[('Clipper', 'FoxPro'), ('Cucumber', 'Gherkin'), ('RobotFramework', 'RobotFramework'), ('abap', 'ABAP'), ('ada', 'Ada')...
#    style = ChoiceField(choices=[('autumn', 'autumn'), ('borland', 'borland'), ('bw', 'bw'), ('colorful', 'colorful')...
```

重要的是要记住`ModelSerialize`r类没有做任何特别神奇的事情，它们只是创建序列化程序类的快捷方式：

```
自动确定的字段集。
create()和update()方法的简单默认实现。
```

## 使用序列化类创建视图

目前我们不会使用任何REST框架的其他功能，我们只会将视图写为常规Django视图。

- 视图

```python
# snippets/views.py
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from snippets.models import Snippet
from snippets.serializers import SnippetSerializer

@csrf_exempt  # 能够从没有CSRF令牌的客户端POST到此视图
def snippet_list(request):
    """
    列出所有Snippets或创建一个新的snippet
    """
    if request.method == 'GET':
        snippets = Snippet.objects.all()
        serializer = SnippetSerializer(snippets, many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = SnippetSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        return JsonResponse(serializer.errors, status=400)
      
@csrf_exempt
def snippet_detail(request, pk):
    """
    检索，更新或删除一个snippet
    """
    try:
        snippet = Snippet.objects.get(pk=pk)
    except Snippet.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = SnippetSerializer(snippet)
        return JsonResponse(serializer.data)

    elif request.method == 'PUT':
        data = JSONParser().parse(request)
        serializer = SnippetSerializer(snippet, data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors, status=400)

    elif request.method == 'DELETE':
        snippet.delete()
        return HttpResponse(status=204)
```

- 路由

应用路由

```python
# snippets/urls.py
from django.urls import path
from snippets import views

urlpatterns = [
    path('snippets/', views.snippet_list),
    path('snippets/<int:pk>/', views.snippet_detail),
]
```

项目路由

```python
# tutorial/urls.py
from django.urls import path, include

urlpatterns = [
    path('', include('snippets.urls')),
]
```

值得注意的是，目前还有一些我们没有正确处理的边缘情况。如果我们发送格式错误的json，或者如果使用视图无法处理的方法发出请求，那么我们最终将得到500“服务器错误”响应。

## 测试API

退出shell

```
quit()
```

开启服务

```
python manage.py runserver
```

- 测试

方式一：

```python
# 浏览器
http http://127.0.0.1:8000/snippets/
http http://127.0.0.1:8000/snippets/2/
```

方法二：httpie/curl

```shell
# 安装
pip install httpie

# 测试
# 获取所有
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
# 获取部分
http http://127.0.0.1:8000/snippets/2/

HTTP/1.1 200 OK
...
{
  "id": 2,
  "title": "",
  "code": "print(\"hello, world\")\n",
  "linenos": false,
  "language": "python",
  "style": "friendly"
}
```

