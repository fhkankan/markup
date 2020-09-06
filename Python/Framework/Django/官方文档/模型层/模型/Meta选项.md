# 模型的`Meta`选项

这篇文档阐述所有可用的[元数据选项](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/models.html#meta-options)，你可以在你模型的`Meta类`中设置它们。

## 可用的Meta选项

### `abstract`

`Options.abstract`

如果 `abstract = True`， 就表示模型是[抽象基类](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/models.html#abstract-base-classes)。

### `app_label`

`Options.app_label`

如果该项目下有多个app，有一个model不是定义在本app下默认的model.py，而是在其他app，也即它在本app settings的[`INSTALLED_APPS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-INSTALLED_APPS)没有声明，则必须使用app_lable声明其属于哪个app：
```
app_label = 'myapp' 
```

如果要表示具有格式`app_label.object_name`或`app_label.model_name`的模型，可以使用`model._meta.label`或`model._meta.label_lower`。

### `base_manager_name`

`Options.``base_manager_name`

模型中[`_base_manager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/managers.html#django.db.models.Model._base_manager)所使用的manager的名称（模型管理器的名称）。

### `db_table`

`Options.db_table`

该模型所用的数据表的名称：
```
db_table = 'music_album' 
```

#### Table names

为了节省时间，Django 会自动的使用你的 model class 的名称和包含这个 model 的 app 名称来构建 数据库的表名称。 一个 model 的数据库表名称是通过将 “app label” – 你在 [`manage.py startapp`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/django-admin.html#django-admin-startapp) 中使用的名称 – 和 model 的类名称，加上一个下划线在他们之间来构成。

举个例子，`bookstore`应用(使用`manage.py startapp bookstore` 创建)，以`class Book`定义的模型的数据表的名称将是`bookstore_book` 。

使用` Meta`类中的 `db_table` 参数来重写数据表的名称。

如果你的数据库表名称是SQL保留字，或包含Python变量名称中不允许的字符，特别是连字符 — 没有问题。 Django在后台引用列和表名。

> 在 MySQL中使用小写字母为表命名
>
> 强烈建议你在通过`db_table` 重载数据库表名称时，使用小写字母，特别是当你在使用 MySQL 作为后台数据库时。 详见[MySQL notes](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#mysql-notes) 。

> Oracle中表名称的引号处理
>
> 为了遵从Oracle中30个字符的限制，以及一些常见的约定，Django会缩短表的名称，而且会把它全部转为大写。 如果你不想名称自动按照约定发生变化，可以在`db_table`的值外面加上引号来避免这种情况：
```
db_table = '"name_left_in_lowercase"'
```
这种带引号的名称也可以与Django所支持的其它数据库后端一起使用；但Oracle除外，引号没有效果。 详见 [Oracle notes](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#oracle-notes) 。

### `db_tablespace`

`Options.db_tablespace`

当前模型所使用的[database tablespace](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/tablespaces.html) 的名字。 默认值是项目设置中的[`DEFAULT_TABLESPACE`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-DEFAULT_TABLESPACE)，如果它存在的话。 如果后端并不支持表空间，这个选项可以忽略。

### `default_manager_name`

`Options.default_manager_name`

模型的[`_default_manager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/managers.html#django.db.models.Model._default_manager)用到的管理器的名称。

### `default_related_name`

`Options.default_related_name`

从关联的对象反向查找当前对象用到的默认名称。 默认为 `<model_name>_set`。此选项还相应地设置了[`related_query_name`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey.related_query_name)。由于一个字段的反向名称应该是唯一的，当子类化你的模型时，要格外小心。 为了规避名称冲突，名称的一部分应该含有`'%(app_label)s'`和`'%(model_name)s'`，它们会被该模型所在的应用标签的名称和模型的名称替换，二者都是小写的。 详见[抽象模型的关联名称](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/models.html#abstract-related-name)。

### `get_latest_by`

`Options.get_latest_by`

模型中某个可排序的字段的名称，比如[`DateField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.DateField)、[`DateTimeField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.DateTimeField)或者[`IntegerField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.IntegerField)。 它指定了[`Manager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/managers.html#django.db.models.Manager)的[`latest()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.latest)和[`earliest()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.earliest)中使用的默认字段。

例如

```python
# Latest by ascending order_date.
get_latest_by = "order_date"

# Latest by priority descending, order_date ascending.
get_latest_by = ['-priority', 'order_date']
```

更多详见[`latest()`](https://yiyibooks.cn/__trs__/qy/django2/ref/models/querysets.html#django.db.models.query.QuerySet.latest) 

> 在Django 2.0中进行了更改：
>
> 对字段列表的支持增加了。

### `managed`

`Options.managed`

默认为`True`，表示Django会通过[`migrate`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/django-admin.html#django-admin-migrate)创建合适的数据表，并且可通过[`flush`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/django-admin.html#django-admin-flush)管理命令移除这些数据库表。 换句话说，Django会*管理*这些数据表的生命周期。

如果是`False`，Django 就不会为当前模型创建和删除数据表。 如果当前模型表示一个已经存在的且是通过其它方法创建的者数据表或数据库视图，这会相当有用。 这是设置为`managed=False`时*唯一*的不同之处。 模型处理的其它任何方面都和平常一样。

 这包括：

- 如果你不声明它的话，会向你的模型中添加一个自增主键。 为了避免给后面的代码读者带来混乱，当你在使用未被管理的模型时，强烈推荐你指定（specify）数据表中所有的列。
- 如果一个模型设置了`managed=False`且含有[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField)，且这个多对多字段指向其他同样也是未被管理模型的，那么这两个未被管理的模型的多对多中介表也不会被创建。 但是，一个被管理模型和一个未被管理模型之间的中介表*就会*被创建。如果你需要修改这一默认行为，创建中介表作为显式的模型（也要设置`managed`），并且使用[`ManyToManyField.through`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField.through)为你的自定义模型创建关联。

如果你进行测试，测试中涉及非托管 model (`managed=False`)，那么在测试之前，你应该要确保在 测试启动时 已经创建了正确的数据表。

如果你对在Python层面修改模型类的行为感兴趣，你*可以*设置 `managed=False` ，并且为一个已经存在的模型创建一个副本。 不过在面对这种情况时还有个更好的办法就是使 用[Proxy models](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/models.html#proxy-models).

### `order_with_respect_to`

`Options.order_with_respect_to`

使此对象相对于给定字段可以排序，通常为`ForeignKey`。 这可以用于使关联的对象相对于父对象可排序。 比如，如果`Answer`和`Question`相关联，一个问题有至少一个答案，并且答案的顺序非常重要，你可以这样做：

```python
from django.db import models

class Question(models.Model):
    text = models.TextField()
    # ...

class Answer(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    # ...

    class Meta:
        order_with_respect_to = 'question'
```

设置之后，模型会提供两个额外的用于设置和获取关联对象顺序的方法：`get_RELATED_order()` 和`set_RELATED_order()`，其中`RELATED`是小写的模型名称。 例如，假设一个`Question`对象有很多相关联的`Answer`对象，返回的列表中含有与之相关联`Answer`对象的主键：

```shell
>>> question = Question.objects.get(id=1)
>>> question.get_answer_order()
[1, 2, 3]
```

与`Question`对象相关联的`Answer`对象的顺序，可以通过传入一个包含`Answer`主键的列表来设置：

```shell
>>> question.set_answer_order([3, 1, 2]) 
```

相关联的对象也有两个方法， `get_next_in_order()` 和`get_previous_in_order()`，用于按照合适的顺序访问它们。 假设`Answer`对象按照 `id`来排序：

```shell
>>> answer = Answer.objects.get(id=2)
>>> answer.get_next_in_order()
<Answer: 3>
>>> answer.get_previous_in_order()
<Answer: 1>
```
> `order_with_respect_to`隐式设置`ordering`选项
在内部，`order_with_respect_to`添加了一个名为`_order`的附加字段/数据库列，并将该模型的[`ordering`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.ordering)选项设置为此字段。 因此，`order_with_respect_to`和`ordering`不能一起使用，每当你获得此模型的对象列表的时候，将使用`order_with_respect_to`添加的排序。

> 更改`order_with_respect_to`
因为`order_with_respect_to`添加了一个新的数据库列，所以在初始化[`migrate`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/django-admin.html#django-admin-migrate)之后添加或更改`order_with_respect_to`时，请确保进行相应的迁移。

### `ordering`

- `Options.ordering`

对象默认的顺序，在获取对象的列表时使用：
```
ordering = ['-order_date'] 
```
它是一个字符串的列表或元组。 每个字符串是一个字段名，前面带有可选的“-”前缀表示倒序。 前面没有“-”的字段表示正序。 使用字符串“？”来随机排序。

例如，要按照`pub_date`字段的正序排序，这样写：
```
ordering = ['pub_date']
```

按照`pub_date`字段的倒序排序，这样写：
```
ordering = ['-pub_date'] 
```
先按照`pub_date`的倒序排序，再按照 `author` 的正序排序，这样写：
```
ordering = ['-pub_date', 'author'] 
```
您也可以使用查询表达式。要按作者升序排序并使空值最后排序，请使用以下命令：
```
from django.db.models import F

ordering = [F('author').asc(nulls_last=True)]
```
默认顺序还会影响聚合查询。

>在Django 2.0中进行了更改：
支持查询表达式。

> 警告
>
> 排序并不是没有任何代价的操作。 你向ordering属性添加的每个字段都会产生你数据库的开销。 你添加的每个外键也会隐式包含它的默认顺序。
>
> 如果查询没有指定的顺序，则会以未指定的顺序从数据库返回结果。 仅当排序的一组字段可以唯一标识结果中的每个对象时，才能保证稳定排序。 例如，如果`name`字段不唯一，则由其排序不会保证具有相同名称的对象总是以相同的顺序显示。

### `permissions`

`Options.permissions`

设置创建对象时权限表中额外的权限。 增加、删除和修改权限会自动为每个模型创建。 这个例子指定了一种额外的权限，`can_deliver_pizzas`：
```
permissions = (("can_deliver_pizzas", "Can deliver pizzas"),) 
```
它是一个包含二元组的元组或者列表，格式为 `(permission_code, human_readable_permission_name)`。

### `default_permissions`

`Options.default_permissions`

默认为`('add', 'change', 'delete')`。 你可以自定义这个列表，比如，如果你的应用不需要默认权限中的任何一项，可以把它设置成空列表。 在模型被[`migrate`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/django-admin.html#django-admin-migrate)命令创建之前，这个属性必须被指定，以防一些遗漏的属性被创建。

### `proxy`

`Options.proxy`

如果`proxy = True`, 它作为另一个模型的子类，将会作为一个[proxy model](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/models.html#proxy-models)。

### `required_db_features`

`Options.required_db_features`

当前连接应具有的数据库功能列表，以便在迁移阶段考虑该模型。 例如，如果将此列表设置为`['gis_enabled']`，则模型将仅在启用GIS的数据库上同步。 在使用多个数据库后端进行测试时，跳过某些模型也很有用。 避免与ORM无关的模型之间的关系。

### `required_db_vendor`

`Options.required_db_vendor`

此型号特定于受支持的数据库供应商的名称。 当前内置的供应商名称是：`sqlite`，`postgresql`，`mysql`，`oracle`。 如果此属性不为空，并且当前连接供应商不匹配，则该模型将不会同步。

### `select_on_save`

`Options.select_on_save`

该选项决定Django是否采用1.6之前的[`django.db.models.Model.save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save)算法。 旧的算法使用`SELECT`来判断是否存在需要更新的行。 而新的算法直接尝试使用`UPDATE`。 在某些少见的情况下，一个已存在行的`UPDATE`操作对Django不可见。 一个例子是PostgreSQL的返回`NULL`的`ON UPDATE`触发器。 这种情况下，新式的算法最终会执行`INSERT`操作，即使这一行已经在数据库中存在。

通常这个属性不需要设置。 默认为`False`。

关于旧式和新式两种算法，请参见[`django.db.models.Model.save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save)。

### `indexes`

`Options.indexes`

**Django中的新功能1.11。**
要在模型上定义的[索引](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/indexes.html)的列表：
```python
from django.db import models

class Customer(models.Model):
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)

    class Meta:
        indexes = [
            models.Index(fields=['last_name', 'first_name']),
            models.Index(fields=['first_name'], name='first_name_idx'),
        ]
```

### `unique_together`

`Options.unique_together`

用来设置的不重复的字段组合：
```
unique_together = (("driver", "restaurant"),) 
```
它是一个元组的元组，组合起来的时候必须是唯一的。 它在Django admin层面使用，在数据库层上进行数据约束(比如，在 `CREATE TABLE` 语句中包含 `UNIQUE`语句)。

为了方便起见，处理单一字段的集合时，unique_together 可以是一维的元组：
```
unique_together = ("driver", "restaurant") 
```

[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField)不能包含在unique_together中。 （不清楚它的含义是什么！） 如果你需要验证[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField)关联的唯一性，试着使用信号或者显式的[`through`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField.through)模型。

当`unique_together`的约束被违反时，模型校验期间会抛出`ValidationError`异常。

### `index_together`

`Options.index_together`
> 请改用[`indexes`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.indexes)选项。
> 较新的[`indexes`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.indexes)选项提供比`index_together`更多的功能。 `index_together`将来可能会被弃用。

用来设置带有索引的字段组合：

```
index_together = [    
  ["pub_date", "deadline"],
]
```
列表中的字段将会建立索引（例如，会在`CREATE INDEX`语句中被使用）。为了方便起见，当需要处理的字段的集合只有一个的时候（集合只有一个！），`index_together`可以只用一个中括号。也就是只用一个一维列表。
```
index_together = ["pub_date", "deadline"] 
```

### `verbose_name`

`Options.verbose_name`

对象的一个易于理解的名称，为单数：
```
verbose_name = "pizza" 
```

如果此项没有设置，Django会把类名拆分开来作为自述名，比如`CamelCase` 会变成`camel case`，

### `verbose_name_plural`

`Options.verbose_name_plural`

该对象复数形式的名称：
```
verbose_name_plural = "stories" 
```

如果此项没有设置，Django 会使用 [`verbose_name`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.verbose_name) + `"s"`。



## 只读的`Meta`属性

### `label`

`Options.label`

对象的表示，返回`app_label.object_name`，例如`'polls.Question'`。

### `label_lower`

`Options.label_lower`

模型的表示，返回`app_label.model_name`，例如`'polls.question'`。