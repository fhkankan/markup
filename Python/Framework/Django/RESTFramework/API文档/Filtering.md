# Filtering

> “ 由Django Manager提供的根QuerySet描述了数据库表中的所有对象。可是通常你需要的只是选择完整对象中的一个子集而已。
>
> —— Django文档 ”

REST framework列表视图的默认行为是返回一个model的全部queryset。通常你却想要你的API来限制queryset返回的数据。

最简单的过滤任意`GenericAPIView`子视图queryset的方法就是重写它的`.get_queryset()`方法。

重写这个方法允许你使用很多不同的方式来定制视图返回的queryset。

## 根据当前用户进行过滤

您可能想要过滤queryset，以确保只返回与发出请求的当前已验证用户相关的结果。

你可以通过基于request.user的值进行过滤来实现。

比如：

```python
from myapp.models import Purchase
from myapp.serializers import PurchaseSerializer
from rest_framework import generics

class PurchaseList(generics.ListAPIView):
    serializer_class = PurchaseSerializer

    def get_queryset(self):
        """
        This view should return a list of all the purchases
        for the currently authenticated user.
        """
        user = self.request.user
        return Purchase.objects.filter(purchaser=user)
```

## 根据URL进行过滤

另一种过滤方式可能包括基于URL的某些部分来限制queryset。

例如，如果你的URL配置包含一个参数如下:

```python
url('^purchases/(?P<username>.+)/$', PurchaseList.as_view()),
```

你就可以写一个view，返回基于URL中的username参数进行过滤的结果。

```python
class PurchaseList(generics.ListAPIView):
    serializer_class = PurchaseSerializer

    def get_queryset(self):
        """
        This view should return a list of all the purchases for
        the user as determined by the username portion of the URL.
        """
        username = self.kwargs['username']
        return Purchase.objects.filter(purchaser__username=username)
```

## 根据查询参数进行过滤

过滤初始查询集的最后一个示例是基于url中的查询参数确定初始查询集。

我们可以通过重写`.get_queryset()`方法来处理像`http://example.com/api/purchases?username=denvercoder9`这样的网址，并且只有在URL中包含`username`参数时，才过滤queryset：

```python
class PurchaseList(generics.ListAPIView):
    serializer_class = PurchaseSerializer

    def get_queryset(self):
        """
        Optionally restricts the returned purchases to a given user,
        by filtering against a `username` query parameter in the URL.
        """
        queryset = Purchase.objects.all()
        username = self.request.query_params.get('username', None)
        if username is not None:
            queryset = queryset.filter(purchaser__username=username)
        return queryset
```

# 通用过滤

除了能够重写默认的queryset，REST框架还包括对通用过滤后端的支持，允许你轻松构建复杂的检索器和过滤器。

通用过滤器也可以在browsable API和admin API中显示为HTML控件。

![filter-controls.png](https://q1mi.github.io/Django-REST-framework-documentation/img/filter-controls.png)

## 设置通用过滤后端

默认过滤器后端可以在全局设置中使用`DEFAULT_FILTER_BACKENDS`来配置。例如。

```python
REST_FRAMEWORK = {
    'DEFAULT_FILTER_BACKENDS': ('django_filters.rest_framework.DjangoFilterBackend',)
}
```

你还可以使用基于`GenericAPIView`类的视图在每个view或每个viewset基础上设置过滤器后端。

```python
import django_filters.rest_framework
from django.contrib.auth.models import User
from myapp.serializers import UserSerializer
from rest_framework import generics

class UserListView(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    filter_backends = (django_filters.rest_framework.DjangoFilterBackend,)
```

## 过滤和对象查找

请注意，如果为view配置过滤器后端，并且用于过滤list views，则它也将用于过滤用于返回单个对象的queryset。

例如，给定前面的示例以及一个ID为`4675`的产品，以下URL将返回相应的对象或者返回404，具体取决于给定的产品实例是否满足筛选条件。

```python
http://example.com/api/products/4675/?category=clothing&max_price=10.00
```

## 覆盖初始queryset

请注意，你可以同时重写`.get_queryset()`方法或使用通用过滤，并且一切都会按照预期生效。 例如，如果`Product`与`User`（名为`purchase`）具有多对多关系，则可能需要编写如下所示的view：

```python
class PurchasedProductsList(generics.ListAPIView):
    """
    Return a list of all the products that the authenticated
    user has ever purchased, with optional filtering.
    """
    model = Product
    serializer_class = ProductSerializer
    filter_class = ProductFilter

    def get_queryset(self):
        user = self.request.user
        return user.purchase_set.all()
```

------

# API导览

## Django过滤后端

`django-filter`库包含一个为REST framework提供高度可定制字段过滤的`DjangoFilterBackend`类。

要使用`DjangoFilterBackend`，首先要先安装`django-filter`。然后将`django_filters`添加到Django的`INSTALLED_APPS`

```
pip install django-filter
```

现在，你需要将`filter backend `添加到你django project的settings中：

```python
REST_FRAMEWORK = {
    'DEFAULT_FILTER_BACKENDS': ('django_filters.rest_framework.DjangoFilterBackend',)
}
```

或者你也可以将filter backend添加到一个单独的view或viewSet中：

```python
from django_filters.rest_framework import DjangoFilterBackend

class UserListView(generics.ListAPIView):
    ...
    filter_backends = [DjangoFilterBackend]
```

如果只需要简单的基于等式的过滤，则可以在视图或视图集上设置`filterset_fields`属性，列出要过滤的字段集。

```python
class ProductList(generics.ListAPIView):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['category', 'in_stock']
```

这将自动为给定的字段创建一个FilterSet类，并允许您发出如下请求：

```
http://example.com/api/products?category=clothing&in_stock=True
```

对于更高级的过滤要求，您可以指定视图应使用的`FilterSet`类。您可以在 [django-filter documentation](https://django-filter.readthedocs.io/en/latest/index.html)文档中阅读有关FilterSets的更多信息。还建议您阅读有关[DRF integration](https://django-filter.readthedocs.io/en/latest/guide/rest_framework.html)的部分。

## 搜索过滤

`SearchFilter`类支持基于简单单查询参数的搜索，并且基于[Django admin的搜索功能](https://docs.djangoproject.com/en/stable/ref/contrib/admin/#django.contrib.admin.ModelAdmin.search_fields)。

在使用时， browsable API将包括一个`SearchFilter`控件：![search-filter.png](https://q1mi.github.io/Django-REST-framework-documentation/img/search-filter.png)

仅当view中设置了`search_fields`属性时，才应用`SearchFilter`类。`search_fields`属性应该是model中文本类型字段的名称列表，例如`CharField`或`TextField`。

```python
from rest_framework import filters

class UserListView(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['username', 'email']
```

这将允许客户端通过进行以下查询来过滤列表中的项目：

```
http://example.com/api/users?search=russell
```

你还可以在查找API中使用双下划线符号对ForeignKey或ManyToManyField执行相关查找：

```python
search_fields = ['username', 'email', 'profile__profession']
```

默认情况下，搜索将使用不区分大小写的部分匹配。 搜索参数可以包含多个搜索项，其应该是空格和/或逗号分隔。 如果使用多个搜索术语，则仅当所有提供的术语都匹配时才在列表中返回对象。

可以通过在`search_fields`前面添加各种字符来限制搜索行为。

- '^' 以指定内容开始.
- '=' 完全匹配
- '@' 全文搜索（目前只支持Django的MySQL后端）
- '$' 正则搜索

例如：

```python
search_fields = ['=username', '=email']
```

默认情况下，搜索参数名为`'search'`，但这可以通过使用`SEARCH_PARAM`设置覆盖。

要根据请求内容动态更改搜索字段，可以对`SearchFilter`进行子类化，并覆盖`get_search_fields()`函数。例如，以下子类仅在查询参数`title_only`在请求中时才在`title`上搜索：

```python
from rest_framework import filters

class CustomSearchFilter(filters.SearchFilter):
    def get_search_fields(self, view, request):
        if request.query_params.get('title_only'):
            return ['title']
        return super(CustomSearchFilter, self).get_search_fields(view, request)
```

有关更多详细信息，请参阅[Django文档](https://docs.djangoproject.com/en/stable/ref/contrib/admin/#django.contrib.admin.ModelAdmin.search_fields)。

------

## 排序筛选

`OrderingFilter`类支持简单的查询参数控制结果排序。![ordering-filter.png](https://q1mi.github.io/Django-REST-framework-documentation/img/ordering-filter.png)

默认情况下，查询参数名为`'ordering'`，但这可以通过使用`ORDERING_PARAM`设置覆盖。

例如，按用户名排序用户：

```
http://example.com/api/users?ordering=username
```

客户端还可以通过为字段名称加上'-'来指定反向排序，如下所示：

```
http://example.com/api/users?ordering=-username
```

还可以指定多个排序：

```
http://example.com/api/users?ordering=account,username
```

## 指定支持排序的字段

建议你明确指定API应在ordering filter中允许哪些字段。您可以通过在view中设置`ordering_fields`属性来实现这一点，如下所示：

```python
class UserListView(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['username', 'email']
```

这有助于防止意外的数据泄漏，例如允许用户针对密码哈希字段或其他敏感数据进行排序。

如果不在视图上指定`ordering_fields`属性，过滤器类将默认允许用户对`serializer_class`属性指定的serializer上的任何可读字段进行过滤。

如果你确信视图正在使用的queryset不包含任何敏感数据，则还可以通过使用特殊值`'__all__'`来明确指定view应允许对任何model字段或queryset进行排序。

```python
class BookingsListView(generics.ListAPIView):
    queryset = Booking.objects.all()
    serializer_class = BookingSerializer
    filter_backends = [filters.OrderingFilter]
    ordering_fields = '__all__'
```

## 指定默认排序

如果在view中设置了`ordering`属性，则将把它用作默认排序。

通常，你可以通过在初始queryset上设置`order_by`来控制此操作，但是使用view中的`ordering`参数允许你以某种方式指定排序，然后可以将其作为上下文自动传递到呈现的模板。如果它们用于排序结果的话就能使自动渲染不同的列标题成为可能。

```python
class UserListView(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['username', 'email']
    ordering = ['username']
```

`ordering`属性可以是字符串或字符串的列表/元组。

# 自定义通用过滤

您还可以提供自己的通用过滤后端，或编写供其他开发人员使用的可安装应用。

为此，请重写`BaseFilterBackend`，并重写`.filter_queryset(self，request，queryset，view)`方法。该方法应返回一个经过过滤的新查询集。

除了允许客户端执行搜索和过滤外，通用过滤器后端对于限制对任何给定请求或用户应可见的对象也很有用。

- 示例

例如，您可能需要限制用户只能看到他们创建的对象。

```python
class IsOwnerFilterBackend(filters.BaseFilterBackend):
    """
    Filter that only allows users to see their own objects.
    """
    def filter_queryset(self, request, queryset, view):
        return queryset.filter(owner=request.user)
```

我们可以通过在视图上重写`get_queryset()`来实现相同的行为，但是使用过滤器后端可以使您更轻松地将此限制添加到多个视图，或将其应用于整个API。

## 自定义接口

通用过滤器也可以在可浏览的API中提供接口。为此，您应该实现`to_html()`方法，该方法返回过滤器的呈现的HTML表示形式。此方法应具有以下签名：

```python
to_html(self, request, queryset, view)
```

该方法应返回呈现的HTML字符串。

## 分页和模式

您还可以通过实现`get_schema_fields()`方法，使过滤器控件可用于REST框架提供的模式自动生成。此方法应具有以下签名：

```python
get_schema_fields(self, view)
```

该方法应返回`coreapi.Field`实例的列表。

# 第三方包

以下第三方软件包提供了其他过滤器实现。

## Django REST框架过滤器软件包

[django-rest-framework-filters package](https://github.com/philipn/django-rest-framework-filters)软件包与`DjangoFilterBackend`类一起使用，并允许您轻松创建跨关系的过滤器，或为给定字段创建多种过滤器查找类型。

##Django REST框架全字搜索过滤器

[djangorestframework-word-filter](https://github.com/trollknurr/django-rest-framework-word-search-filter) 开发为`filter.SearchFilter`的替代产品，它将搜索文本中的完整单词或完全匹配。

## Django URL过滤器

[django-url-filter](https://github.com/miki725/django-url-filter)提供了一种通过人类友好的URL过滤数据的安全方法。从某种意义上说，它可以嵌套，除了它们称为过滤器集和过滤器外，它的工作方式与DRF序列化器和字段非常相似。这提供了过滤相关数据的简便方法。该库也是通用的，因此可以用来过滤其他数据源，而不仅仅是Django `QuerySets`。

## drf-url-filters

[drf-url-filter](https://github.com/manjitkumar/drf-url-filters)是一个简单的Django应用程序，它以干净，简单和可配置的方式在drf `ModelViewSet`的`Queryset`上应用过滤器。它还支持对传入查询参数及其值的验证。一个漂亮的python软件包`Voluptuous`用于对传入的查询参数进行验证。关于`Voluptuous`的最好的部分是您可以根据查询参数要求定义自己的验证。

