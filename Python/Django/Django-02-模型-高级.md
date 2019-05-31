# 模型-高级

## 管理器



## 原始SQL



## 事务





## 聚合



## 自定义字段





## 自定义查找





## 多数据库

### 定义数据库

在Django中使用多个数据库的第一步是告诉Django 你将要使用的数据库服务器。这通过使用`DATABASES`设置完成。该设置映射数据库别名到一个数据库连接设置的字典，这是整个Django 中引用一个数据库的方式

你可以为数据库选择任何别名。然而，`default`这个别名具有特殊的含义，且必须定义。当没有选择其它数据库时，Django 使用具有`default` 别名的数据库。

若视图访问没有在DATABASE设置中定义的数据库，Django将抛出`django.db.utils.ConnectionDoesNotExist`异常

```python
DATABASES = {
    'default': {
        'NAME': 'app_data',
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'USER': 'postgres_user',
        'PASSWORD': 's3krit'
    },
    # 或default为空字典
    # 'default':{},
    'users': {
        'NAME': 'user_data',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'priv4te'
    }
}
```

### 同步数据库

`migrate`管理命令一次操作一个数据库。默认情况下，它在`default`数据库上操作，但可以通过提供一个`—database`参数，告诉`migrate`同步一个不同的数据库

```
./manage.py migrate  # default数据库
./manage.py migrate --database=users  # users数据库
```

若是不想每个应用都被同步到同一台数据库，可以定义一个数据库路由，它实现一个策略来控制特定模型的访问性。

### 自动路由

使用多数据库最简单的方法是建立一个数据库路由模式。默认的路由模式确保对象’粘滞‘在它们原始的数据库上（例如，从`foo` 数据库中获取的对象将保存在同一个数据库中）。默认的路由模式还确保如果没有指明数据库，所有的查询都回归到`default`数据库中。

你不需要做任何事情来激活默认的路由模式 —— 它在每个Django项目上’直接‘提供。然而，如果你想实现更有趣的数据库分配行为，你可以定义并安装你自己的数据库路由。

- 数据库路由

数据库路由是一个类，提供4个方法

```python
db_for_read(model, **hints)
# 建议model类型的对象的读操作应该使用的数据库。若是没有建议，则返回None
# 如果一个数据库操作能够提供其它额外的信息可以帮助选择一个数据库，它将在hints字典中提供
db_for_write(model, **hints)
# 建议Model 类型的对象的写操作应该使用的数据库。若是没有建议，则返回None
# 如果一个数据库操作能够提供其它额外的信息可以帮助选择一个数据库，它将在hints字典中提供。
allow_relation(obj1, obj2, **hints)
# 如果obj1和obj2之间应该允许关联则返回True，如果应该防止关联则返回False，如果路由无法判断则返回None。这是纯粹的验证操作，外键和多对多操作使用它来决定两个对象之间是否应该允许一个关联。
allow_migrate(db, app_label, model_name=None, **hints)
# 定义迁移操作是否允许在别名为db的数据库上运行。如果操作应该运行则返回True ，如果不应该运行则返回False，如果路由无法判断则返回None。
# 参数  app_label  # 正在迁移的应用的标签；hints		# 用于某些操作来传递额外的信息给路由

hints
# Hint 由数据库路由接收，用于决定哪个数据库应该接收一个给定的请求。
# 目前，唯一一个提供的hint 是instance，它是一个对象实例，与正在进行的读或者写操作关联。这可能是保存环节的实例，或者在多对多关系中添加环节的实例。在一些情况下，将不会提供hints实例。路由将检查hint实例是否存在，并决定这个hint是否应该用做路由行为的提示
```

- 使用路由

数据库路由使用`DATABASE_ROUTERS`设置安装。这个设置定义一个类名的列表，其中每个类表示一个路由，它们将被主路由（`django.db.router`）使用。

```python
# settings.py
DATABASES_ROUTERS=[]
```

Django 的数据库操作使用主路由来分配数据库的使用。每当一个查询需要知道使用哪一个数据库时，它将调用主路由，并提供一个模型和一个Hint （可选）。随后 Django 依次测试每个路由直至找到一个数据库的建议。如果找不到建议，它将尝试Hint 实例的当前`_state.db`。如果没有提供Hint 实例，或者该实例当前没有数据库状态，主路由将分配`default` 数据库

- 示例

数据库设置

```python

DATABASES = {
    'auth_db': {
        'NAME': 'auth_db',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'swordfish',
    },
    'primary': {
        'NAME': 'primary',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'spam',
    },
    'replica1': {
        'NAME': 'replica1',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'eggs',
    },
    'replica2': {
        'NAME': 'replica2',
        'ENGINE': 'django.db.backends.mysql',
        'USER': 'mysql_user',
        'PASSWORD': 'bacon',
    },
}
```

路由设置

```python
# 一个路由，知道发送auth应用的查询到auth_db
class AuthRouter(object):
    """
    A router to control all database operations on models in the
    auth application.
    """
    def db_for_read(self, model, **hints):
        """
        Attempts to read auth models go to auth_db.
        """
        if model._meta.app_label == 'auth':
            return 'auth_db'
        return None

    def db_for_write(self, model, **hints):
        """
        Attempts to write auth models go to auth_db.
        """
        if model._meta.app_label == 'auth':
            return 'auth_db'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if a model in the auth app is involved.
        """
        if obj1._meta.app_label == 'auth' or \
           obj2._meta.app_label == 'auth':
           return True
        return None

    def allow_migrate(self, db, app_label, model=None, **hints):
        """
        Make sure the auth app only appears in the 'auth_db'
        database.
        """
        if app_label == 'auth':
            return db == 'auth_db'
        return None
      
# 一个路由，它发送所有其它应用的查询到primary/replica 配置，并随机选择一个replica 来读取
import random

class PrimaryReplicaRouter(object):
    def db_for_read(self, model, **hints):
        """
        Reads go to a randomly-chosen replica.
        """
        return random.choice(['replica1', 'replica2'])

    def db_for_write(self, model, **hints):
        """
        Writes always go to primary.
        """
        return 'primary'

    def allow_relation(self, obj1, obj2, **hints):
        """
        Relations between objects are allowed if both objects are
        in the primary/replica pool.
        """
        db_list = ('primary', 'replica1', 'replica2')
        if obj1._state.db in db_list and obj2._state.db in db_list:
            return True
        return None

    def allow_migrate(self, db, app_label, model=None, **hints):
        """
        All non-auth models end up in this pool.
        """
        return True
```

设置路由配置

```python
DATABASE_ROUTERS = ['path.to.AuthRouter', 'path.to.PrimaryReplicaRouter']

# 注意：路由处理的顺序非常重要。路由的查询将按照DATABASE_ROUTERS设置中列出的顺序进行。在这个例子中，AuthRouter在PrimaryReplicaRouter之前处理，因此auth中的模型的查询处理在其它模型之前。如果DATABASE_ROUTERS设置按其它顺序列出这两个路由，PrimaryReplicaRouter.allow_migrate() 将先处理。PrimaryReplicaRouter 中实现的捕获所有的查询，这意味着所有的模型可以位于所有的数据库中。
```

### 手动选择

- QuerySet

```shell
>>> # This will run on the 'default' database.
>>> Author.objects.all()

>>> # So will this.
>>> Author.objects.using('default').all()

>>> # This will run on the 'other' database.
>>> Author.objects.using('other').all()
```

- save

```shell
>>> my_object.save(using='legacy_users')
# 若不指定，则将保存到路由分配的默认数据库中
```

将对象从一个数据库移动到另一个数据库

```shell
# 错误操作
>>> p = Person(name='Fred')
>>> p.save(using='first')  # (statement 1)  # p没有主键，django发出insert,创建主键，赋值给p
>>> p.save(using='second') # (statement 2)  # 使用p的主键，若新数据库中无此主键则ok，若有此主键，原值会被新值覆盖
# 正确操作：方法一
>>> p = Person(name='Fred')
>>> p.save(using='first')
>>> p.pk = None # Clear the primary key.
>>> p.save(using='second') # Write a completely new object.
# 正确操作：方法二
>>> p = Person(name='Fred')
>>> p.save(using='first')
>>> p.save(using='second', force_insert=True)
```

- delete

默认情况下，删除一个已存在对象的调用将在与获取对象时使用的相同数据库上执行

```shell
>>> u = User.objects.using('legacy_users').get(username='fred')
>>> u.delete() # will delete from the `legacy_users` database
```

要指定删除一个模型时使用的数据库，可以对`Model.delete()`方法使用`using` 关键字参数。这个参数的工作方式与`save()`的`using`关键字参数一样。

```shell
# 正在从legacy_users 数据库到new_users 数据库迁移一个User
>>> user_obj.save(using='new_users')
>>> user_obj.delete(using='legacy_users')
```

- 多数据库上使用管理器

在管理器上使用`db_manager()`方法来让管理器访问非默认的数据库。

你有一个自定义的管理器方法，它访问数据库时候用`User.objects.create_user()`。因为`create_user()`是一个管理器方法，不是一个`QuerySet`方法，你不可以使用`User.objects.using('new_users').create_user()`。（`create_user()` 方法只能在`User.objects`上使用，而不能在从管理器得到的`QuerySet`上使用）。解决办法是使用`db_manager()`

```python
# db_manager() 返回一个绑定在你指定的数据上的一个管理器。
User.objects.db_manager('new_users').create_user(...)
```

- 多数据库上使用get_queryset

如果你正在覆盖你的管理器上的`get_queryset()`，请确保在其父类上调用方法（使用`super()`）或者正确处理管理器上的`_db`属性（一个包含将要使用的数据库名称的字符串）

```python
# 如果你想从get_queryset 方法返回一个自定义的 QuerySet 类
class MyManager(models.Manager):
    def get_queryset(self):
        qs = CustomQuerySet(self.model)
        if self._db is not None:
            qs = qs.using(self._db)
        return qs
```

### 管理站点

Django 的管理站点没有对多数据库的任何显式的支持。如果你给数据库上某个模型提供的管理站点不想通过你的路由链指定，你将需要编写自定义的`ModelAdmin`类用来将管理站点导向一个特殊的数据库。

`ModelAdmin` 对象具有5个方法，它们需要定制以支持多数据库

```python
class MultiDBModelAdmin(admin.ModelAdmin):
    # A handy constant for the name of the alternate database.
    using = 'other'

    def save_model(self, request, obj, form, change):
        # Tell Django to save objects to the 'other' database.
        obj.save(using=self.using)

    def delete_model(self, request, obj):
        # Tell Django to delete objects from the 'other' database
        obj.delete(using=self.using)

    def get_queryset(self, request):
        # Tell Django to look for objects on the 'other' database.
        return super(MultiDBModelAdmin, self).get_queryset(request).using(self.using)

    def formfield_for_foreignkey(self, db_field, request=None, **kwargs):
        # Tell Django to populate ForeignKey widgets using a query
        # on the 'other' database.
        return super(MultiDBModelAdmin, self).formfield_for_foreignkey(db_field, request=request, using=self.using, **kwargs)

    def formfield_for_manytomany(self, db_field, request=None, **kwargs):
        # Tell Django to populate ManyToMany widgets using a query
        # on the 'other' database.
        return super(MultiDBModelAdmin, self).formfield_for_manytomany(db_field, request=request, using=self.using, **kwargs)
```

这里提供的实现实现了一个多数据库策略，其中一个给定类型的所有对象都将保存在一个特定的数据库上（例如，所有的`User`保存在`other` 数据库中）。如果你的多数据库的用法更加复杂，你的`ModelAdmin`将需要反映相应的策略。

Inlines 可以用相似的方式处理。它们需要3个自定义的方法

```python
class MultiDBTabularInline(admin.TabularInline):
    using = 'other'

    def get_queryset(self, request):
        # Tell Django to look for inline objects on the 'other' database.
        return super(MultiDBTabularInline, self).get_queryset(request).using(self.using)

    def formfield_for_foreignkey(self, db_field, request=None, **kwargs):
        # Tell Django to populate ForeignKey widgets using a query
        # on the 'other' database.
        return super(MultiDBTabularInline, self).formfield_for_foreignkey(db_field, request=request, using=self.using, **kwargs)

    def formfield_for_manytomany(self, db_field, request=None, **kwargs):
        # Tell Django to populate ManyToMany widgets using a query
        # on the 'other' database.
        return super(MultiDBTabularInline, self).formfield_for_manytomany(db_field, request=request, using=self.using, **kwargs)
```

一旦你写好你的模型管理站点的定义，它们就可以使用任何Admin实例来注册

```python
from django.contrib import admin

# Specialize the multi-db admin objects for use with specific models.
class BookInline(MultiDBTabularInline):
    model = Book

class PublisherAdmin(MultiDBModelAdmin):
    inlines = [BookInline]

admin.site.register(Author, MultiDBModelAdmin)
admin.site.register(Publisher, PublisherAdmin)

othersite = admin.AdminSite('othersite')
othersite.register(Publisher, MultiDBModelAdmin)
```

### 使用原始游标

如果你正在使用多个数据库，你可以使用`django.db.connections`来获取特定数据库的连接（和游标）：`django.db.connections`是一个类字典对象，它允许你使用别名来获取一个特定的连接

```python
from django.db import connections
cursor = connections['my_db_alias'].cursor()
```

### 局限

- 跨数据库关联

Django 目前不提供跨多个数据库的外键或多对多关系的支持。如果你使用一个路由来路由分离到不同的数据库上，这些模型定义的任何外键和多对多关联必须在单个数据库的内部。

这是因为引用完整性的原因。为了保持两个对象之间的关联，Django 需要知道关联对象的主键是合法的。如果主键存储在另外一个数据库上，判断一个主键的合法性不是很容易。

如果你正在使用Postgres、Oracle或者MySQL 的InnoDB，这是数据库完整性级别的强制要求 —— 数据库级别的主键约束防止创建不能验证合法性的关联。

然而，如果你正在使用SQLite 或MySQL的MyISAM 表，则没有强制性的引用完整性；结果是你可以‘伪造’跨数据库的外键。但是Django 官方不支持这种配置。

- Contrib应用的行为

有几个Contrib 应用包含模型，其中一些应用相互依赖。因为跨数据库的关联是不可能的，这对你如何在数据库之间划分这些模型带来一些限制：

```
- `contenttypes.ContentType`、`sessions.Session`和`sites.Site` 可以存储在分开存储在不同的数据库中，只要给出合适的路由
- `auth`模型 —— `User`、`Group`和`Permission` —— 关联在一起并与`ContentType`关联，所以它们必须与`ContentType`存储在相同的数据库中。
- `admin`依赖`auth`，所以它们的模型必须与`auth`在同一个数据库中。
- `flatpages`和`redirects`依赖`sites`，所以它们必须与`sites`在同一个数据库中。
```

另外，migrate在数据库中创建一张表后，一些对象在该表中自动创建：

```
- 一个默认的`Site`，
- 为每个模型创建一个`ContentType`（包括没有存储在同一个数据库中的模型），
- 为每个模型创建3个`Permission` （包括不是存储在同一个数据库中的模型）。
```

对于常见的多数据库架构，将这些对象放在多个数据库中没有什么用处。常见的数据库架构包括primary/replica 和连接到外部的数据库。因此，建议写一个数据库路由，它只允许同步这3个模型到一个数据中。对于不需要将表放在多个数据库中的Contrib 应用和第三方应用，可以使用同样的方法。

警告

如果你将Content Types 同步到多个数据库中，注意它们的主键在数据库之间可能不一致。这可能导致数据损坏或数据丢失。

## 数据库函数

下面记述的类为用户提供了一些方法，来在Django中使用底层数据库提供的函数用于注解、聚合或者过滤器等操作。函数也是表达式，所以可以像聚合函数一样混合使用它们。

```python
class Author(models.Model):
    name = models.CharField(max_length=50)
    age = models.PositiveIntegerField(null=True, blank=True)
    alias = models.CharField(max_length=50, null=True, blank=True)  # 不建议CharField上允许null=True,但对于Coalesce很重要
    goes_by = models.CharField(max_length=50, null=True, blank=True)
```

函数

| name                                                         | desc                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `Cast(expression, output_field)`                             | 强制转换表达式的结果类型为output_field的类型                 |
| `Coalesce(*expressions, **extra)`                            | 接受一个含有至少两个字段名称或表达式的列表，返回第一个非空的值（注意空字符串不被认为是一个空值）。每个参与都必须是相似的类型,因此混合文本和数字将导致数据库错误。 |
| `Greatest(*expressions, **extra)`                            | 接受至少两个字段名称或表达式的列表，并返回最大值。每个参数必须是类似的类型，因此混合文本和数字将导致数据库错误。在mysql中若表达式为null，则返回null |
| `Least(*expressions, **extra)`                               | 接受至少两个字段名称或表达式的列表，并返回最小值。每个参数必须是类似的类型，因此混合文本和数字将导致数据库错误。在mysql中若表达式为null，则返回null |
| `Extract(expression, lookup_name=None, txinfo=None, **extra)` | 将日期的组件提取为数字。                                     |
|                                                              |                                                              |
| `Concat(*expressions, **extra)`                              | 接受一个含有至少两个文本字段的或表达式的列表，返回连接后的文本。每个参数都必须是文本或者字符类型。 |
| `Length(expression, **extra)`                                | 接受一个文本字段或表达式，返回值的字符个数。如果表达式是null，长度也会是null。 |
| `Lower(expression, **extra)`                                 | 接受一个文本字符串或表达式，返回它的小写表示形式             |
|                                                              |                                                              |
| `Substr(expression, pos, Length=None, **extra)`              | 返回这个字段或者表达式的，以`pos`位置开始，长度为`length`的子字符串。位置从下标为1开始，所以必须大于0。如果`length`是`None`，会返回剩余的字符串 |
| `Upper(expression, **extra)`                                 | 接受一个文本字符串或表达式，返回它的大写表示形式             |
|                                                              |                                                              |

### 比对转换

Cast

```shell
>>> from django.db.models import FloatField
>>> from django.db.models.functions import Cast
>>> Value.objects.create(integer=4)
>>> value = Value.objects.annotate(as_float=Cast('integer', FloatField())).get()
>>> print(value.as_float)
4.0
```

Coalesce

```shell
# 由于类型不同，掺杂了文本和数字的列表会导致数据库错误。
>>> # Get a screen name from least to most public
>>> from django.db.models import Sum, Value as V
>>> from django.db.models.functions import Coalesce
>>> Author.objects.create(name='Margaret Smith', goes_by='Maggie')
>>> author = Author.objects.annotate(
...    screen_name=Coalesce('alias', 'goes_by', 'name')).get()
>>> print(author.screen_name)
Maggie

>>> # Prevent an aggregate Sum() from returning None
>>> aggregated = Author.objects.aggregate(
...    combined_age=Coalesce(Sum('age'), V(0)),
...    combined_age_default=Sum('age'))
>>> print(aggregated['combined_age'])
0
>>> print(aggregated['combined_age_default'])
None
```

Greatest

```python
class Blog(models.Model):
    body = models.TextField()
    modified = models.DateTimeField(auto_now=True)

class Comment(models.Model):
    body = models.TextField()
    modified = models.DateTimeField(auto_now=True)
    blog = models.ForeignKey(Blog, on_delete=models.CASCADE)

>>> from django.db.models.functions import Greatest
>>> blog = Blog.objects.create(body='Greatest is the best.')
>>> comment = Comment.objects.create(body='No, Least is better.', blog=blog)
>>> comments = Comment.objects.annotate(last_updated=Greatest('modified', 'blog__modified'))
>>> annotated_comment = comments.get()
```

### Data函数

```python
class Experiment(models.Model):
    start_datetime = models.DateTimeField()
    start_date = models.DateField(null=True, blank=True)
    start_time = models.TimeField(null=True, blank=True)
    end_datetime = models.DateTimeField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    end_time = models.TimeField(null=True, blank=True)
```

### Text函数

Concat

```python
# 如果你想把一个TextField()和一个CharField()连接， 一定要告诉Djangooutput_field应该为TextField()类型。在下面连接Value的例子中，这也是必需的。
# 这个函数不会返回null。在后端中，如果一个null参数导致了整个表达式都是null，Django会确保把每个null的部分转换成一个空字符串。
>>> # Get the display name as "name (goes_by)"
>>> from django.db.models import CharField, Value as V
>>> from django.db.models.functions import Concat
>>> Author.objects.create(name='Margaret Smith', goes_by='Maggie')
>>> author = Author.objects.annotate(
...    screen_name=Concat('name', V(' ('), 'goes_by', V(')'),
...    output_field=CharField())).get()
>>> print(author.screen_name)
Margaret Smith (Maggie)
```

Length

```shell
>>> # Get the length of the name and goes_by fields
>>> from django.db.models.functions import Length
>>> Author.objects.create(name='Margaret Smith')
>>> author = Author.objects.annotate(
...    name_length=Length('name'),
...    goes_by_length=Length('goes_by')).get()
>>> print(author.name_length, author.goes_by_length)
(14, None)

# It can also be registered as a transform.
>>> from django.db.models import CharField
>>> from django.db.models.functions import Length
>>> CharField.register_lookup(Length, 'length')
>>> # Get authors whose name is longer than 7 characters
>>> authors = Author.objects.filter(name__length__gt=7)
```

Lower

```
>>> from django.db.models.functions import Lower
>>> Author.objects.create(name='Margaret Smith')
>>> author = Author.objects.annotate(name_lower=Lower('name')).get()
>>> print(author.name_lower)
margaret smith
```

