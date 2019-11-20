# `QuerySet` API参考

本文档描述了`QuerySet` API的详细信息。 它建立在[模型](https://yiyibooks.cn/__trs__/qy/django2/topics/db/models.html)和[数据库查询](https://yiyibooks.cn/__trs__/qy/django2/topics/db/queries.html)指南中提供的材料之上，因此在阅读本文档之前，您可能希望阅读并理解这些文档。

在整个参考文献中，我们将使用[数据库查询指南](https://yiyibooks.cn/__trs__/qy/django2/topics/db/queries.html)中提供的[示例Weblog模型](https://yiyibooks.cn/__trs__/qy/django2/topics/db/queries.html#queryset-model-example)。

## `QuerySet`何时求值

本质上，可以创建、过滤、切片和传递`QuerySet`而不用真实操作数据库。 在你对查询集做求值之前，不会发生任何实际的数据库操作。

你可以使用下列方法对`QuerySet`求值：

- `Iteration`

QuerySet`是可迭代的，它在首次迭代查询集时执行实际的数据库查询。 例如， 下面的语句会将数据库中所有Entry 的headline 打印出来：

```python
for e in Entry.objects.all():
    print(e.headline)
```

注意：不要使用上面的语句来验证在数据库中是否至少存在一条记录。 使用 [`exists()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.exists)方法更高效。

- Slicing

正如在[Limiting QuerySets](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/queries.html#limiting-querysets)中解释的那样， 可以使用Python 的序列切片语法对一个`QuerySet`进行分片。 一个未求值的`QuerySet`进行切片通常返回另一个未求值的`QuerySet`，但是如果你使用切片的”step“参数，Django 将执行数据库查询并返回一个列表。 对一个已经求值的`QuerySet`进行切片将返回一个列表。

还要注意，虽然对未求值的`QuerySet`进行切片返回另一个未求值的`QuerySet`，但是却不可以进一步修改它了（例如，添加更多的Filter，或者修改排序的方式），因为这将不太好翻译成SQL而且含义也不清晰。

- Pickling/Caching

[序列化查询集](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#pickling-querysets)的细节参见下面一节。 本节提到它的目的是强调序列化将读取数据库。

- `repr()` 

当对`QuerySet`调用`repr()` 时，将对它求值。 这是为了在Python 交互式解释器中使用的方便，这样你可以在交互式使用这个API 时立即看到结果。

- `len()`

当你对`QuerySet`调用`len()` 时， 将对它求值。 正如你期望的那样，返回一个查询结果集的长度。

注意: 如果你确定只需要获取结果集的数量 (而不需要实际的查询结果), 那么在数据库级别使用SQL的`SELECT COUNT(*)`语句来处理计数会更有效率. 为此，Django 提供了 一个[`count()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.count) 方法.

- `list()`

对`QuerySet`调用`list()` 将强制对它求值。 像这样：

```
entry_list = list(Entry.objects.all())
```

- `bool()`

测试一个`QuerySet`的布尔值，例如使用`bool()`、`or`、`and` 或者`if` 语句将导致查询集的执行。 如果至少有一个记录，则`QuerySet`为`True`，否则为`False`。 像这样：

```
if Entry.objects.filter(headline="Test"):
   print("There is at least one Entry with the headline Test")
```

注：如果你需要知道是否存在至少一条记录（而不需要真实的对象），使用 [`exists()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.exists) 将更加高效。

### Pickling `QuerySet`

如果你[`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle)一个`QuerySet`，它将在Pickle 之前强制将所有的结果加载到内存中。 Pickle 通常用于缓存之前，并且当缓存的查询集重新加载时，你希望结果已经存在随时准备使用（从数据库读取耗费时间，就失去了缓存的目的）。 这意味着当你Unpickle`QuerySet`时，它包含Pickle 时的结果，而不是当前数据库中的结果。

如果此后你只想Pickle 必要的信息来从数据库重新创建`QuerySet`，可以Pickle`QuerySet`的`query` 属性。 然后你可以使用类似下面的代码重新创建原始的`QuerySet`（不用加载任何结果）：

```shell
>>> import pickle
>>> query = pickle.loads(s)     # Assuming 's' is the pickled string.
>>> qs = MyModel.objects.all()
>>> qs.query = query            # Restore the original 'query'.
```

`query` 是一个不透明的对象。 它表示查询的内部构造，不属于公开的API。 然而，这里讲到的Pickle 和Unpickle 这个属性的内容是安全的（和完全支持的）。

> 不可以在不同版本之间共享Pickle 的结果
>
> `QuerySets`的Pickle 只能用于生成它们的Django 版本中。 如果你使用Django 的版本N 生成一个Pickle，不保证这个Pickle 在Django 的版本N+1 中可以读取。 Pickle 不可用于归档的长期策略。
>
> 因为Pickle 兼容性的错误很难诊断例如产生损坏的对象，当你试图Unpickle 的查询集与Pickle 时的Django 版本不同，将引发一个`RuntimeWarning`。

## `QuerySet` API 

下面是对于`QuerySet`的正式定义：

```
class* `QuerySet(model=None, query=None, using=None)
```

通常你在使用`QuerySet`时会以[chaining filters](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/queries.html#chaining-filters) 来使用。 为了让这个能工作，大部分`QuerySet` 方法返回新的QuerySet。 这些方法在本节将详细讲述。

`QuerySet` 类具有两个公有属性用于内省：

- `ordered`

如果`QuerySet` 是排好序的则为`True` —— 例如有一个[`order_by()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.order_by) 子句或者模型有默认的排序。 否则为`False` .

- `db`

如果现在执行，则返回将使用的数据库。

>注
[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet) 存在`query` 参数是为了让具有特殊查询用途的子类如[`GeoQuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/gis/geoquerysets.html#django.contrib.gis.db.models.GeoQuerySet) 可以重新构造内部的查询状态。 这个参数的值是查询状态的不透明的表示，不是一个公开的API。 简而言之：如果你有疑问，那么你实际上不需要使用它。

### 返回新的`QuerySet`的方法

Django 提供了一系列 的`QuerySet`筛选方法，用于改变 `QuerySet` 返回的结果类型或者SQL查询执行的方式

#### `filter()`

`filte(**kwargs)`

返回一个新的`QuerySet`，它包含满足查询参数的对象。

查找的参数（`**kwargs`）应该满足下文[字段查找](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#id4)中的格式。 在底层的SQL 语句中，多个参数通过`AND` 连接。

如果你需要执行更复杂的查询（例如`OR` 语句），你可以使用[`Q objects`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.Q)。

#### `exclude()`

`exclude(**kwargs)`

返回一个新的`QuerySet`，它包含*不*满足给定的查找参数的对象。

查找的参数（`**kwargs`）应该满足下文[字段查找](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#id4)中的格式。 在底层的SQL 语句中，多个参数通过`AND` 连接，然后所有的内容放入`NOT()` 中。

下面的示例排除所有`pub_date` 晚于2005-1-3 且`headline` 为“Hello” 的记录：

```
Entry.objects.exclude(pub_date__gt=datetime.date(2005, 1, 3), headline='Hello')
```

用SQL 语句，它等同于：

```
SELECT ...
WHERE NOT (pub_date > '2005-1-3' AND headline = 'Hello')
```

下面的示例排除所有`pub_date` 晚于2005-1-3 或者headline 为“Hello” 的记录：

```
Entry.objects.exclude(pub_date__gt=datetime.date(2005, 1, 3)).exclude(headline='Hello')
```

用SQL 语句，它等同于：

```
SELECT ...
WHERE NOT pub_date > '2005-1-3'
AND NOT headline = 'Hello'
```

注意，第二个示例更严格。

如果你需要执行更复杂的查询（例如`OR` 语句），你可以使用[`Q objects`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.Q)。

#### `annotate()`

`annotate(*args, **kwargs)`

使用提供的[查询表达式](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html)Annotate`查询集`中的每个对象。 表达式可以是简单的值、模型（或任何关联模型）上的字段的引用或者聚合表达式（平均值、总和等）， 它们已经在与`QuerySet`中的对象相关的对象上进行了计算。

`annotate()`的每个参数都是一个annotation，将添加到返回的`QuerySet`中的每个对象。

Django提供的聚合函数在下文的[聚合函数](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#id5)文档中讲述。

关键字参数指定的Annotation将使用关键字作为Annotation的别名。 匿名的参数的别名将基于聚合函数的名称和模型的字段生成。 只有引用单个字段的聚合表达式才可以使用匿名参数。 其它所有形式都必须用关键字参数。

例如，如果你正在操作一个Blog列表，你可能想知道每个Blog有多少Entry：

```
>>> from django.db.models import Count
>>> q = Blog.objects.annotate(Count('entry'))
# The name of the first blog
>>> q[0].name
'Blogasaurus'
# The number of entries on the first blog
>>> q[0].entry__count
42
```

`Blog`模型本身没有定义`entry__count`属性，通过使用一个关键字参数来指定聚合函数，你可以控制Annotation的名称：

```
>>> q = Blog.objects.annotate(number_of_entries=Count('entry'))
# The number of entries on the first blog, using the name provided
>>> q[0].number_of_entries
42
```

聚合的深入讨论，参见[聚合主题的指南](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/aggregation.html)。

#### `order_by()`

`order_by(*fields)` 

默认情况下，`Meta` 根据模型`ordering` 类的`QuerySet` 选项排序。 你可以使用`QuerySet` 方法给每个`order_by` 指定特定的排序。

例如：

```
Entry.objects.filter(pub_date__year=2005).order_by('-pub_date', 'headline')
```

上面的结果将按照`pub_date` 降序排序，然后再按照`headline` 升序排序。 `"-pub_date"`前面的负号表示*降序*顺序。 升序是隐含的。 要随机订购，请使用`"?"`，如下所示：

```
Entry.objects.order_by('?')
```

注：`order_by('?')` 查询可能耗费资源且很慢，这取决于使用的数据库。

若要按照另外一个模型中的字段排序，可以使用查询关联模型时的语法。 即通过字段的名称后面跟上两个下划线（`__`），再跟上新模型中的字段的名称，直至你希望连接的模型。 像这样：

```
Entry.objects.order_by('blog__name', 'headline')
```

如果排序的字段与另外一个模型关联，Django 将使用关联的模型的默认排序，或者如果没有指定[`Meta.ordering`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.ordering) 将通过关联的模型的主键排序。 例如，因为`Blog` 模型没有指定默认的排序：

```
Entry.objects.order_by('blog')
```

...与以下相同：

```
Entry.objects.order_by('blog__id')
```

如果`Blog` 设置`ordering = ['name']`，那么第一个QuerySet 将等同于：

```
Entry.objects.order_by('blog__name')
```

你还可以通过调用表达式的`desc()` 或者`asc()`，根据[query expressions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html)排序：

```
Entry.objects.order_by(Coalesce('summary', 'headline').desc())
```

如果你还用到[`distinct()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.distinct)，在根据关联模型中的字段排序时要小心。 [`distinct()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.distinct) 中有一个备注讲述关联模型的排序如何对结果产生影响。

> 注

指定一个多值字段来排序结果（例如，一个[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField) 字段或者[`ForeignKey`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey) 字段的反向关联）。

考虑下面的情况：

```
class Event(Model):
   parent = models.ForeignKey(
       'self',
       on_delete=models.CASCADE,
       related_name='children',
   )
   date = models.DateField()

Event.objects.order_by('children__date')
```

在这里，每个`Event`可能有多个排序数据；具有多个`children`的每个`Event`将被多次返回到`order_by()`创建的新的`QuerySet`中。 换句话说, 用 `order_by()`方法对 `QuerySet`对象进行操作会返回一个扩大版的新QuerySet对象——新增的条目也许并没有什么卵用，你也用不着它们。

因此，当你使用多值字段对结果进行排序时要格外小心。 **如果**，您可以确保每个订单项只有一个订购数据，这种方法不会出现问题。 如果不确定，请确保结果是你期望的。

没有方法指定排序是否考虑大小写。 对于大小写的敏感性，Django 将根据数据库中的排序方式排序结果。

你可以通过[`Lower`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#django.db.models.functions.Lower)将一个字段转换为小写来排序，它将达到大小写一致的排序：

```
Entry.objects.order_by(Lower('headline').desc())
```

如果你不想对查询做任何排序，即使是默认的排序，可以不带参数调用[`order_by()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.order_by)。

你可以通过检查[`QuerySet.ordered`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.ordered) 属性来知道查询是否是排序的，如果`True` 有任何方式的排序它将为`QuerySet`。

每个`order_by()` 都将清除前面的任何排序。 例如，下面的查询将按照`pub_date` 排序，而不是`headline`：

```
Entry.objects.order_by('headline').order_by('pub_date')
```

> 警告
> 排序不是没有开销的操作。 添加到排序中的每个字段都将带来数据库的开销。 添加的每个外键也都将隐式包含进它的默认排序。
> 如果查询没有指定的顺序，则会以未指定的顺序从数据库返回结果。 仅当通过唯一标识结果中的每个对象的一组字段排序时，才能保证特定的排序。 例如，如果`name`字段不唯一，则由其排序不会保证具有相同名称的对象总是以相同的顺序显示。

#### `reverse()`

`reverse()` 方法反向排序QuerySet 中返回的元素。 第二次调用`reverse()` 将恢复到原有的排序。

如要获取QuerySet 中最后五个元素，你可以这样做：

```
my_queryset.reverse()[:5]
```

注意，这与Python 中从一个序列的末尾进行切片有点不一样。 上面的例子将首先返回最后一个元素，然后是倒数第二个元素，以此类推。 如果我们有一个Python 序列，当我们查看`seq[-5:]` 时，我们将一下子得到倒数五个元素。 Django 不支持这种访问模型（从末尾进行切片），因为它不可能利用SQL 高效地实现。

同时还要注意，`QuerySet` 应该只在一个已经定义排序的`reverse()` 上调用（例如，在一个定义了默认排序的模型上，或者使用[`order_by()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.order_by) 的时候）。 如果`reverse()` 没有定义排序，调用`reverse()` 将不会有任何效果（在调用`QuerySet` 之前没有定义排序，那么调用之后仍保持没有定义）。

#### `distinct()`

`distinct(*fields)` 

返回一个在SQL 查询中使用`SELECT DISTINCT` 的新`QuerySet`。 它将去除查询结果中重复的行。

默认情况下，`QuerySet` 不会去除重复的行。 在实际应用中，这一般不是个问题，因为像`Blog.objects.all()` 这样的简单查询不会引入重复的行。 但是，如果查询跨越多张表，当对`QuerySet` 求值时就可能得到重复的结果。 这时候你应该使用`distinct()`。

> 注

[`order_by()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.order_by) 调用中的任何字段都将包含在SQL 的 `SELECT` 列中。 与`distinct()` 一起使用时可能导致预计不到的结果。 如果你根据关联模型的字段排序，这些fields将添加到查询的字段中，它们可能产生本应该是唯一的重复的行。 因为多余的列没有出现在返回的结果中（它们只是为了支持排序），有时候看上去像是返回了不明确的结果。

类似地，如果您使用[`values()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.values)查询来限制所选择的列，则仍然会涉及任何[`order_by()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.order_by)（或默认模型排序）影响结果的唯一性。

这里的约束是，如果你使用的是`distinct()`，请注意相关模型的排序。 类似地，当一起使用`distinct()`和[`values()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.values)时，请注意字段在不在[`values()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.values)

在PostgreSQL上，您可以传递位置参数（`*fields`），以便指定`DISTINCT`应该应用的字段的名称。 这转换为`SELECT DISTINCT ON` SQL查询。 这里有区别。 对于正常的`distinct()`调用，数据库在确定哪些行不同时比较每行中的*每个*字段。 对于具有指定字段名称的`distinct()`调用，数据库将仅比较指定的字段名称。

> 注
> 当你指定字段名称时，*必须*在`distinct()`中提供`QuerySet`，而且`order_by()`中的字段必须以`order_by()`中的字段相同开始并且顺序相同。
> 例如，`SELECT DISTINCT ON （a）`列`a`中的每个值。 如果你没有指定一个顺序，你会得到一个任意的行。

示例（除第一个示例外，其他示例都只能在PostgreSQL 上工作）：

```
>>> Author.objects.distinct()
[...]

>>> Entry.objects.order_by('pub_date').distinct('pub_date')
[...]

>>> Entry.objects.order_by('blog').distinct('blog')
[...]

>>> Entry.objects.order_by('author', 'pub_date').distinct('author', 'pub_date')
[...]

>>> Entry.objects.order_by('blog__name', 'mod_date').distinct('blog__name', 'mod_date')
[...]

>>> Entry.objects.order_by('author', 'pub_date').distinct('author')
[...]
```

> 注

请记住，[`order_by()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.order_by)使用已定义的任何默认相关模型排序。 您可能需要通过关系`_id`或引用字段显式排序，以确保`DISTINCT ON`在`ORDER BY`子句的开头。 例如，如果`Blog`模型通过`name`定义[`ordering`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.ordering)：

```
Entry.objects.order_by('blog').distinct('blog')
```

…wouldn’t work because the query would be ordered by `blog__name` thus mismatching the `DISTINCT ON` expression. 你必须按照关系`_id`字段（在这种情况下为`blog__pk`）或引用的（`blog_id`）显式排序来确保两个表达式都匹配。

#### `values()`

`values(*fields, **expressions)`

返回一个`QuerySet`，当迭代它时，返回字典而不是模型实例。

每个字典表示一个对象，键对应于模型对象的属性名称。

下面的例子将`values()` 与普通的模型对象进行比较：

```
# This list contains a Blog object.
>>> Blog.objects.filter(name__startswith='Beatles')
<QuerySet [<Blog: Beatles Blog>]>

# This list contains a dictionary.
>>> Blog.objects.filter(name__startswith='Beatles').values()
<QuerySet [{'id': 1, 'name': 'Beatles Blog', 'tagline': 'All the latest Beatles news.'}]>
```

`SELECT` 接收可选的位置参数`*fields`，它指定`values()` 应该限制哪些字段。 如果指定字段，每个字典将只包含指定的字段的键/值。 如果没有指定字段，每个字典将包含数据库表中所有字段的键和值。

例如：

```
>>> Blog.objects.values()
<QuerySet [{'id': 1, 'name': 'Beatles Blog', 'tagline': 'All the latest Beatles news.'}]>
>>> Blog.objects.values('id', 'name')
<QuerySet [{'id': 1, 'name': 'Beatles Blog'}]>
```

`values()`方法还采用可选的关键字参数`**expressions`，这些参数传递给[`annotate()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.annotate)：

```
>>> from django.db.models.functions import Lower
>>> Blog.objects.values(lower_name=Lower('name'))
<QuerySet [{'lower_name': 'beatles blog'}]>
```

在`values()`子句中的聚合应用于相同`values()`子句中的其他参数之前。 如果您需要按另一个值分组，请将其添加到较早的`values()`子句中。 像这样：

```
>>> from django.db.models import Count
>>> Blog.objects.values('author', entries=Count('entry'))
<QuerySet [{'author': 1, 'entries': 20}, {'author': 1, 'entries': 13}]>
>>> Blog.objects.values('author').annotate(entries=Count('entry'))
<QuerySet [{'author': 1, 'entries': 33}]>
```

值得注意的几点：

- 如果你有一个字段`foo` 是一个[`ForeignKey`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey)，默认的`foo_id` 调用返回的字典将有一个叫做`foo` 的键，因为这是保存实际的值的那个隐藏的模型属性的名称（`values()` 属性引用关联的模型）。 当你调用`foo_id` 并传递字段的名称，传递`foo` 或`values()` 都可以，得到的结果是相同的（字典的键会与你传递的字段名匹配）。

  像这样：

  ```
  >>> Entry.objects.values()
  <QuerySet [{'blog_id': 1, 'headline': 'First Entry', ...}, ...]>
  
  >>> Entry.objects.values('blog')
  <QuerySet [{'blog': 1}, ...]>
  
  >>> Entry.objects.values('blog_id')
  <QuerySet [{'blog_id': 1}, ...]>
  ```

- 当`values()` 与[`distinct()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.distinct) 一起使用时，注意排序可能影响最终的结果。 详细信息参见[`distinct()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.distinct) 中的备注。

- 如果`values()` 子句位于[`extra()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.extra) 调用之后，[`extra()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.extra) 中的`select` 参数定义的字段必须显式包含在`values()` 调用中。 `values()` 调用后面的[`extra()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.extra) 调用将忽略选择的额外的字段。

- 在`values()` 之后调用[`only()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.only) 和[`defer()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.defer) 不太合理，所以将引发一个`NotImplementedError`。

当您知道您只需要少量可用字段中的值时，这是非常有用的，您将不需要模型实例对象的功能。 只选择用到的字段当然更高效。

最后，请注意，您可以调用`filter()`，`order_by()`等。 在`values()`调用之后，这意味着这两个调用是相同的：

```
Blog.objects.values().order_by('id')
Blog.objects.order_by('id').values()
```

Django 的作者喜欢将影响SQL 的方法放在前面，然后放置影响输出的方法（例如`values()`），但是实际上无所谓。 这是卖弄你个性的好机会。

你可以通过`ManyToManyField`、`ForeignKey` 和 `OneToOneField` 属性反向引用关联的模型的字段：

```
>>> Blog.objects.values('name', 'entry__headline')
<QuerySet [{'name': 'My blog', 'entry__headline': 'An entry'},
     {'name': 'My blog', 'entry__headline': 'Another entry'}, ...]>
```

> 警告
因为[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField) 字段和反向关联可能有多个关联的行，包含它们可能导致结果集的倍数放大。 如果你在`values()` 查询中包含多个这样的字段将更加明显，这种情况下将返回所有可能的组合。

**在Django更改1.11：**

支持`**expressions`。



#### values_list()``

`values_list(*fields, flat=False)`

与`values()` 类似，只是在迭代时返回的是元组而不是字典。 每个元组包含传递给`values_list()`调用的相应字段或表达式的值，因此第一个项目是第一个字段等。 像这样：

```
>>> Entry.objects.values_list('id', 'headline')
<QuerySet [(1, 'First entry'), ...]>
>>> from django.db.models.functions import Lower
>>> Entry.objects.values_list('id', Lower('headline'))
<QuerySet [(1, 'first entry'), ...]>
```

如果只传递一个字段，你还可以传递`flat` 参数。 如果为`True`，它表示返回的结果为单个值而不是元组。 一个例子会让它们的区别更加清晰：

```
>>> Entry.objects.values_list('id').order_by('id')
<QuerySet[(1,), (2,), (3,), ...]>

>>> Entry.objects.values_list('id', flat=True).order_by('id')
<QuerySet [1, 2, 3, ...]>
```

如果有多个字段，传递`flat` 将发生错误。

如果你不传递任何值给`values_list()`，它将返回模型中的所有字段，以它们在模型中定义的顺序。

常见的需求是获取某个模型实例的特定字段值。 为了实现这一点，使用`values_list()`，然后使用`get()`调用：

```
>>> Entry.objects.values_list('headline', flat=True).get(pk=1)
'First entry'
```

`values()`和`values_list()`都用于特定用例的优化：检索数据子集，而无需创建模型实例的开销。 这个比喻在处理多对多和其他多值关系（例如反向外键的一对多关系）时分歧，因为“一行一对象”的假设不成立。

例如，注意通过[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField)进行查询时的行为：

```
>>> Author.objects.values_list('name', 'entry__headline')
<QuerySet [('Noam Chomsky', 'Impressions of Gaza'),
 ('George Orwell', 'Why Socialists Do Not Believe in Fun'),
 ('George Orwell', 'In Defence of English Cooking'),
 ('Don Quixote', None)]>
```

具有多个条目的作者多次出现，没有任何条目的作者对条目标题具有`None`。

类似地，当查询反向外键时，对于没有任何作者的条目，出现`None`

```
>>> Entry.objects.values_list('authors')
<QuerySet [('Noam Chomsky',), ('George Orwell',), (None,)]>
```

**在Django更改1.11：**

添加了对`*fields`中的表达式的支持。

#### `dates()`

`dates(field, kind, order='ASC')`  

返回一个`QuerySet`，它将求值为表示`QuerySet`内容中特定类型的所有可用日期的[`datetime.date`](https://docs.python.org/3/library/datetime.html#datetime.date)对象列表。

`field`应该是您的模型的`DateField`的名称。 `kind`应为`"year"`，`"month"`或`"day"`。 结果列表中的每个`datetime.date`对象被“截断”到给定的`type`。

- `"year"` 返回对应该field的所有不同年份值的list。
- `"month"`返回字段的所有不同年/月值的列表。
- `"day"`返回字段的所有不同年/月/日值的列表。

`'ASC'`（默认为`'ASC'`）应为`order`或`'DESC'`。 它t指定如何排序结果。

例子：

```
>>> Entry.objects.dates('pub_date', 'year')
[datetime.date(2005, 1, 1)]
>>> Entry.objects.dates('pub_date', 'month')
[datetime.date(2005, 2, 1), datetime.date(2005, 3, 1)]
>>> Entry.objects.dates('pub_date', 'day')
[datetime.date(2005, 2, 20), datetime.date(2005, 3, 20)]
>>> Entry.objects.dates('pub_date', 'day', order='DESC')
[datetime.date(2005, 3, 20), datetime.date(2005, 2, 20)]
>>> Entry.objects.filter(headline__contains='Lennon').dates('pub_date', 'day')
[datetime.date(2005, 3, 20)]
```

#### `datetimes()`

`datetimes(field_name, kind, order='ASC', tzinfo=None)`

返回`QuerySet`，其计算为[`datetime.datetime`](https://docs.python.org/3/library/datetime.html#datetime.datetime)对象的列表，表示`QuerySet`内容中特定种类的所有可用日期。

`field_name`应为模型的`DateTimeField`的名称。

`kind`应为`"hour"`，`"minute"`，`"month"`，`"year"`，`"second"`或`"day"`。 结果列表中的每个`datetime.datetime`对象被“截断”到给定的`type`。

`'ASC'`（默认为`'ASC'`）应为`order`或`'DESC'`。 它t指定如何排序结果。

`tzinfo`定义在截断之前将数据时间转换到的时区。 实际上，给定的datetime具有不同的表示，这取决于使用的时区。 此参数必须是[`datetime.tzinfo`](https://docs.python.org/3/library/datetime.html#datetime.tzinfo)对象。 如果它`None`，Django使用[current time zone](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/i18n/timezones.html#default-current-time-zone)。 当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`False`时，它不起作用。

> 注

此函数直接在数据库中执行时区转换。 因此，您的数据库必须能够解释`tzinfo.tzname(None)`的值。 这转化为以下要求：

- SQLite：没有要求。 转换在Python中使用[pytz](http://pytz.sourceforge.net/)执行（安装Django时安装）。
- PostgreSQL：没有要求（见[时区](https://www.postgresql.org/docs/current/static/datatype-datetime.html#DATATYPE-TIMEZONES)）。
- Oracle：无要求（请参阅[选择时区文件](https://docs.oracle.com/database/121/NLSPG/ch4datetime.htm#NLSPG258)）。
- MySQL：使用[mysql_tzinfo_to_sql](https://dev.mysql.com/doc/refman/en/mysql-tzinfo-to-sql.html)加载时区表。

#### `none()`

`none`() 

调用none()将创建一个从不返回任何对象的查询集，并且在访问结果时不会执行任何查询。 qs.none()查询集是`EmptyQuerySet`的一个实例。

例子：

```
>>> Entry.objects.none()
<QuerySet []>
>>> from django.db.models.query import EmptyQuerySet
>>> isinstance(Entry.objects.none(), EmptyQuerySet)
True
```

#### `all()`

`all()`

返回当前`QuerySet`（或`QuerySet` 子类） 的*副本*。 它可以用于在你希望传递一个模型管理器或`QuerySet` 并对结果做进一步过滤的情况。 不管对哪一种对象调用`all()`，你都将获得一个可以工作的`QuerySet`。

当对`QuerySet`进行[evaluated](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#when-querysets-are-evaluated)时，它通常会缓存其结果。 如果数据库中的数据在`QuerySet`求值之后可能已经改变，你可以通过在以前求值过的`all()`上调用相同的`QuerySet` 查询以获得更新后的结果。

#### `union()`

`union(*other_qs, all=False)`

**Django中的新功能1.11。**

使用SQL的`UNION`运算符组合两个或更多个`QuerySet`的结果。例如：

```
>>> qs1.union(qs2, qs3)
```

默认情况下，`UNION`操作符仅选择不同的值。 要允许重复值，请使用`all=True`参数。

`union()`，`intersection()`和`difference()`返回第一个`QuerySet`即使参数是其他模型的`QuerySet`。 只要`SELECT`列表在所有`QuerySet`中是相同的，传递不同的模型就起作用（至少在类型中，名称不重要，只要在相同的顺序）。

另外，只有`LIMIT`，`OFFSET`，`COUNT(*)`和`ORDER BY `（即切片，[`count()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.count)和[`order_by()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.order_by)）在结果`QuerySet`上被允许。 此外，数据库对组合查询中允许的操作设置了限制。 例如，大多数数据库不允许组合查询中的`LIMIT`或`OFFSET`。

**在Django更改1.11.4：**

已添加`COUNT(*)`个支持。

#### `intersection()`

`intersection(*other_qs)`

**Django中的新功能1.11。**

使用SQL的`INTERSECT`运算符返回两个或更多个`QuerySet`的共享元素。例如：

```
>>> qs1.intersection(qs2, qs3)
```

有关某些限制，请参阅[`union()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.union)。

#### `difference()`

`difference(*other_qs)`

**Django中的新功能1.11。**

使用SQL的`EXCEPT`运算符只保留`QuerySet`中的元素，但不保留其他`QuerySet`中的元素。例如：

```
>>> qs1.difference(qs2, qs3)
```

有关某些限制，请参阅[`union()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.union)。

#### `select_related()`

`select_related(*fields)` 

返回一个`QuerySet`，当执行它的查询时它沿着外键关系查询关联的对象的数据。 它会生成一个复杂的查询并引起性能的损耗，但是在以后使用外键关系时将不需要数据库查询。

下面的例子解释了普通查询和`select_related()` 查询的区别。 下面是一个标准的查询：

```
# 访问数据库。
e = Entry.objects.get(id=5)

# 再次访问数据库以得到关联的Blog对象。
b = e.blog
```

下面是一个`select_related` 查询：

```
# 访问数据库。
e = Entry.objects.select_related('blog').get(id=5)

# 不会访问数据库，因为e.blog已经
# 在前面的查询中填写好。
b = e.blog
```

`select_related()`可用于objects的任何查询集：

```
from django.utils import timezone

# Find all the blogs with entries scheduled to be published in the future.
blogs = set()

for e in Entry.objects.filter(pub_date__gt=timezone.now()).select_related('blog'):
    # 没有select_related()，下面的语句将为每次循环迭代生成一个数据库查询
    # 以获得每个entry关联的blog。
    blogs.add(e.blog)
```

`filter()` 和`select_related()` 链的顺序不重要。 下面的查询集是等同的：

```
Entry.objects.filter(pub_date__gt=timezone.now()).select_related('blog')
Entry.objects.select_related('blog').filter(pub_date__gt=timezone.now())
```

你可以沿着外键查询。 如果你有以下模型：

```
from django.db import models

class City(models.Model):
    # ...
    pass

class Person(models.Model):
    # ...
    hometown = models.ForeignKey(
        City,
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
    )

class Book(models.Model):
    # ...
    author = models.ForeignKey(Person, on_delete=models.CASCADE)
```

...然后调用`Book.objects.select_related('author__hometown').get(id=4)`将缓存关联的 `Person` *和* 关联的 `City`：

```
b = Book.objects.select_related('author__hometown').get(id=4)
p = b.author         # 不会访问数据库。
c = p.hometown       # 不会访问数据库。

b = Book.objects.get(id=4) # 这个例子中没有select_related()。
p = b.author         # 访问数据库。
c = p.hometown       # 访问数据库。
```

在传递给`select_related()` 的字段中，你可以使用任何[`ForeignKey`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey) 和[`OneToOneField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.OneToOneField)。

在传递给`select_related` 的字段中，你还可以反向引用[`OneToOneField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.OneToOneField) —— 也就是说，你可以回溯到定义[`OneToOneField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.OneToOneField) 的字段。 此时，可以使用关联对象字段的[`related_name`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey.related_name)，而不要指定字段的名称。

有些情况下，你希望对很多对象调用`select_related()`，或者你不知道所有的关联关系。 在这些情况下，可以调用不带参数的`select_related()`。 它将查找能找到的所有不可为空外键 —— 可以为空的外键必须明确指定。 大部分情况下不建议这样做，因为它会使得底层的查询非常复杂并且返回的很多数据都不是真实需要的。

如果你需要清除`QuerySet`上以前的`select_related`调用添加的关联字段，可以传递一个`None`作为参数：

```
>>> without_relations = queryset.select_related(None)
```

链式调用`select_related` 的工作方式与其它方法类似 — 也就是说，`select_related('foo', 'bar')`等同于`select_related('foo').select_related('bar')`。

#### `prefetch_related()`

`prefetch_related(*lookups)`

返回`QuerySet`，它将在单个批处理中自动检索每个指定查找的相关对象。

这具有与`select_related`类似的目的，两者都被设计为阻止由访问关联对象而导致的数据库查询的泛滥，但是策略是完全不同的。

`select_related` 通过创建SQL连接并在 `SELECT` 语句中包括关联对象的字段来工作。 因此，`select_related` 在同一次数据库查询中获取关联对象。 然而，为了避免由于跨越“多个”关系而导致的大得多的结果集，`select_related` 限于单值关系（外键）和一对一关系。

相反，`prefetch_related` 为每个关系单独查找，并用 Python 进行“join”。 这允许它除 `select_related` 支持的外键和一对一关联关系之外，还可以预取不能使用 `select_related` 来完成的多对多和多对一对象。 它还支持预取 [`GenericRelation`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/contenttypes.html#django.contrib.contenttypes.fields.GenericRelation) 和 [`GenericForeignKey`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/contenttypes.html#django.contrib.contenttypes.fields.GenericForeignKey)，但是，它必须限于相同种类的结果。 例如，仅当查询被限制为一个`ContentType`时，才支持预取由`GenericForeignKey`引用的对象。

例如，假设你有这些模型：

```
from django.db import models

class Topping(models.Model):
    name = models.CharField(max_length=30)

class Pizza(models.Model):
    name = models.CharField(max_length=50)
    toppings = models.ManyToManyField(Topping)

    def __str__(self):              # 在Python 2上为__unicode__
        return "%s (%s)" % (
            self.name,
            ", ".join(topping.name for topping in self.toppings.all()),
        )
```

并运行：

```
>>> Pizza.objects.all()
["Hawaiian (ham, pineapple)", "Seafood (prawns, smoked salmon)"...
```

这里的问题是每次 `Pizza.__str__()` 请求 `Pizza.objects.all()` 时，它都必须查询数据库，因此 `Pizza.objects.all()` 将为 Pizza `QuerySet` 中的**每个**元素在 Toppings 表上运行查询。

使用`prefetch_related`，我们可以减少为只有两个查询：

```
>>> Pizza.objects.all().prefetch_related('toppings')
```

这表示每个 `Pizza` 调用一次 `self.toppings.all()`；但是现在每次调用`self.toppings.all()` 时，不必去数据库查询内容，它会在一个预取的 `QuerySet` 缓存中查找它们，这个缓存是用单个查询生成的。

也就是说，所有关联的topping都将在一个查询中获取，并用于使得`QuerySets`具有一个以关联结果预填充的缓存；然后在这些`QuerySets`用于`self.toppings.all()`调用。

`prefetch_related()`中的附加查询在`QuerySet`开始计算并且主查询已执行后执行。

如果您有一个可迭代的模型实例，则可以使用[`prefetch_related_objects()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.prefetch_related_objects)函数在这些实例上预取相关属性。

请注意，主要`QuerySet`的结果缓存和所有指定的相关对象将被完全加载到内存中。 这改变了`QuerySets`的典型行为，通常尽量避免在需要之前将所有对象加载到内存中，即使在数据库中执行了查询之后。

> 注

请记住，与`QuerySets`一样，任何后续的链接方法隐含不同的数据库查询将忽略以前缓存的结果，并使用新的数据库查询检索数据。 所以，如果你写下面的话：

```
>>> pizzas = Pizza.objects.prefetch_related('toppings')
>>> [list(pizza.toppings.filter(spicy=True)) for pizza in pizzas]
```

那么事实上，已经预取了`pizza.toppings.all()`并不能帮助你。 `prefetch_related('toppings')` 隐含 `pizza.toppings.all()`，但 `pizza.toppings.filter()` 是一个不同的新查询。 预取的缓存在这里无法帮助；实际上它会伤害性能，因为您已经完成了一个尚未使用的数据库查询。 所以使用这个功能小心！

Also, if you call the database-altering methods [`add()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/relations.html#django.db.models.fields.related.RelatedManager.add), [`remove()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/relations.html#django.db.models.fields.related.RelatedManager.remove), [`clear()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/relations.html#django.db.models.fields.related.RelatedManager.clear) or [`set()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/relations.html#django.db.models.fields.related.RelatedManager.set), on [`related managers`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/relations.html#django.db.models.fields.related.RelatedManager), any prefetched cache for the relation will be cleared.

**在Django更改1.11：**

添加了上述预取缓存的清除。

您还可以使用正常连接语法来执行相关字段的相关字段。 假设我们有一个额外的模型上面的例子：

```
class Restaurant(models.Model):
    pizzas = models.ManyToManyField(Pizza, related_name='restaurants')
    best_pizza = models.ForeignKey(Pizza, related_name='championed_by')
```

以下都是合法的：

```
>>> Restaurant.objects.prefetch_related('pizzas__toppings')
```

这将预取所有比萨饼属于餐厅，所有浇头属于那些比萨饼。 这将导致总共3个数据库查询 - 一个用于餐馆，一个用于比萨饼，一个用于浇头。

```
>>> Restaurant.objects.prefetch_related('best_pizza__toppings')
```

这将获取最好的比萨饼和每个餐厅最好的披萨的所有浇头。 这将在3个数据库查询 - 一个为餐厅，一个为“最佳比萨饼”，一个为一个为浇头。

当然，也可以使用`best_pizza`来获取`select_related`关系，以将查询计数减少为2：

```
>>> Restaurant.objects.select_related('best_pizza').prefetch_related('best_pizza__toppings')
```

由于预取在主查询（其包括`select_related`所需的连接）之后执行，因此它能够检测到`best_pizza`对象已经被提取，并且请跳过重新获取它们。

链接`prefetch_related`调用将累积预取的查找。 要清除任何`prefetch_related`行为，请传递`None`作为参数：

```
>>> non_prefetched = qs.prefetch_related(None)
```

使用`prefetch_related`时需要注意的一点是，查询创建的对象可以在它们相关的不同对象之间共享，即单个Python模型实例可以出现在树中的多个点返回的对象。 这通常会与外键关系发生。 通常这种行为不会是一个问题，并且实际上会节省内存和CPU时间。

虽然`prefetch_related`支持预取`GenericForeignKey`关系，但查询的数量将取决于数据。 由于`GenericForeignKey`可以引用多个表中的数据，因此需要对每个引用的表进行一次查询，而不是对所有项进行一次查询。 如果尚未提取相关行，则可能会对`ContentType`表执行其他查询。

`prefetch_related`在大多数情况下将使用使用“IN”运算符的SQL查询来实现。 这意味着对于一个大的`QuerySet`，可能会生成一个大的“IN”子句，根据数据库，在解析或执行SQL查询时可能会有性能问题。 始终为您的使用情况配置文件！

请注意，如果您使用`iterator()`来运行查询，则会忽略`prefetch_related()`调用，因为这两个优化并没有意义。

您可以使用[`Prefetch`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.Prefetch)对象进一步控制预取操作。

在其最简单的形式中，`Prefetch`等效于传统的基于字符串的查找：

```
>>> from django.db.models import Prefetch
>>> Restaurant.objects.prefetch_related(Prefetch('pizzas__toppings'))
```

您可以使用可选的`queryset`参数提供自定义查询集。 这可以用于更改查询集的默认顺序：

```
>>> Restaurant.objects.prefetch_related(
...     Prefetch('pizzas__toppings', queryset=Toppings.objects.order_by('name')))
```

或者在适当时调用[`select_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.select_related)以进一步减少查询数量：

```
>>> Pizza.objects.prefetch_related(
...     Prefetch('restaurants', queryset=Restaurant.objects.select_related('best_pizza')))
```

您还可以使用可选的`to_attr`参数将预取结果分配给自定义属性。 结果将直接存储在列表中。

这允许使用不同的`QuerySet`多次预取相同的关系；例如：

```
>>> vegetarian_pizzas = Pizza.objects.filter(vegetarian=True)
>>> Restaurant.objects.prefetch_related(
...     Prefetch('pizzas', to_attr='menu'),
...     Prefetch('pizzas', queryset=vegetarian_pizzas, to_attr='vegetarian_menu'))
```

使用自定义`to_attr`创建的查找仍然可以像往常一样被其他查找遍历：

```
>>> vegetarian_pizzas = Pizza.objects.filter(vegetarian=True)
>>> Restaurant.objects.prefetch_related(
...     Prefetch('pizzas', queryset=vegetarian_pizzas, to_attr='vegetarian_menu'),
...     'vegetarian_menu__toppings')
```

在过滤预取结果时，建议使用`to_attr`，因为它比在相关管理器的缓存中存储过滤的结果更不明确：

```
>>> queryset = Pizza.objects.filter(vegetarian=True)
>>>
>>> # Recommended:
>>> restaurants = Restaurant.objects.prefetch_related(
...     Prefetch('pizzas', queryset=queryset, to_attr='vegetarian_pizzas'))
>>> vegetarian_pizzas = restaurants[0].vegetarian_pizzas
>>>
>>> # Not recommended:
>>> restaurants = Restaurant.objects.prefetch_related(
...     Prefetch('pizzas', queryset=queryset))
>>> vegetarian_pizzas = restaurants[0].pizzas.all()
```

自定义预取也适用于单个相关关系，如前`ForeignKey`或`OneToOneField`。 一般来说，您希望对这些关系使用[`select_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.select_related)，但有很多情况下使用自定义`QuerySet`进行预取是有用的：

- 您想要使用在相关模型上执行进一步预取的`QuerySet`。

- 您希望仅预取相关对象的子集。

- You want to use performance optimization techniques like [`deferred fields`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.defer):

  ```
  >>> queryset = Pizza.objects.only('name')
  >>>
  >>> restaurants = Restaurant.objects.prefetch_related(
  ...     Prefetch('best_pizza', queryset=queryset))
  ```

> 注

查找的顺序很重要。

请看下面的例子：

```
>>> prefetch_related('pizzas__toppings', 'pizzas')
```

即使它是无序的，因为`'pizzas__toppings'`已经包含所有需要的信息，因此第二个参数`'pizzas'`实际上是多余的。

```
>>> prefetch_related('pizzas__toppings', Prefetch('pizzas', queryset=Pizza.objects.all()))
```

这将引发`ValueError`，因为尝试重新定义先前查看的查询的查询集。 请注意，创建了隐式查询集，以作为`'pizzas'`查找的一部分遍历`'pizzas__toppings'`。

```
>>> prefetch_related('pizza_list__toppings', Prefetch('pizzas', to_attr='pizza_list'))
```

这会触发`'pizza_list__toppings'`，因为`'pizza_list'`在处理`AttributeError`时不存在。

这种考虑不限于使用`Prefetch`对象。 一些高级技术可能要求以特定顺序执行查找以避免创建额外的查询；因此，建议始终仔细订购`prefetch_related`参数。

#### `extra()`

`extra(select=None, where=None, params=None, tables=None, order_by=None, select_params=None)`

有些情况下，Django的查询语法难以简单的表达复杂的 `WHERE` 子句， 对于这种情况, Django 提供了 `QuerySet` `QuerySet` 修改机制 — 它能在 `extra()`生成的SQL从句中注入新子句

> 使用这种方法作为最后的手段

这是一个旧的API，我们的目标是在将来的某个时候弃用。 仅当您无法使用其他查询方法表达您的查询时才使用它。如果确实需要使用它，请在您的用例中使用QuerySet.extra关键字提交票证（请先检查现有票证的列表），以便我们增强QuerySet API的功能以允许删除`extra()`。 我们不再改进或修复此方法的错误。

例如，这种使用`extra()`：

```
>>> qs.extra(
...     select={'val': "select col from sometable where othercol = %s"},
...     select_params=(someparam,),
... )
```

相当于：

```
>>> qs.annotate(val=RawSQL("select col from sometable where othercol = %s", (someparam,)))
```

使用[`RawSQL`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#django.db.models.expressions.RawSQL)的主要好处是可以根据需要设置`output_field`。 主要的缺点是，如果您在原始SQL中引用了查询器的某些表别名，那么Django可能会更改该别名（例如，当查询集用作另一个查询中的子查询）时。

>  警告

无论何时你都需要非常小心的使用`extra()`. 每次使用它时，您都应该转义用户可以使用`params`控制的任何参数，以防止SQL注入攻击。 请详细了解[SQL injection protection](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/security.html#sql-injection-protection)。

由于产品差异的原因，这些自定义的查询难以保障在不同的数据库之间兼容(因为你手写 SQL 代码的原因)，而且违背了 DRY 原则，所以如非必要，还是尽量避免写 extra。

extra可以指定一个或多个 `where`,例如 `select`, `params` or `tables`. 这些参数都不是必须的，但是你至少要使用一个

- `select`

The `select` 参数可以让你在 `SELECT` 从句中添加其他字段信息， 它应该是一个字典，存放着属性名到 SQL 从句的映射。

例如：

  ```
  Entry.objects.extra(select={'is_recent': "pub_date > '2006-01-01'"})
  ```

结果集中每个 `pub_date` 对象都有一个额外的属性`is_recent`, 它是一个布尔值，表示 Entry对象的`Entry` 是否晚于 Jan. 1, 2006.

Django 会直接在 `SELECT` 中加入对应的 SQL 片断，所以转换后的 SQL 如下：

  ```
  SELECT blog_entry.*, (pub_date > '2006-01-01') AS is_recent
  FROM blog_entry;
  ```

下一个例子是更先进的；它会执行一个子查询，为每个结果`Blog`对象提供一个`entry_count`属性，一个关联的`Entry`对象的整数：

  ```
  Blog.objects.extra(
      select={
          'entry_count': 'SELECT COUNT(*) FROM blog_entry WHERE blog_entry.blog_id = blog_blog.id'
      },
  )
  ```

在上面这个特例中，我们要了解这个事实，就是 `blog_blog` 表已经存在于`FROM`从句中

  上面例子的结果SQL将是：

  ```
  SELECT blog_blog.*, (SELECT COUNT(*) FROM blog_entry WHERE blog_entry.blog_id = blog_blog.id) AS entry_count
  FROM blog_blog;
  ```

要注意的是，大多数数据库需要在子句两端添加括号，而在 Django 的`select`从句中却无须这样。 另请注意，某些数据库后端（如某些MySQL版本）不支持子查询。

在少数情况下，您可能希望将参数传递到`extra(select=...)`中的SQL片段。 为此，请使用`select_params`参数。 由于`select_params`是一个序列，并且`select`属性是字典，因此需要注意，以便参数与额外的选择片段正确匹配。 在这种情况下，您应该使用[`collections.OrderedDict`](https://docs.python.org/3/library/collections.html#collections.OrderedDict)作为`select`值，而不仅仅是普通的Python字典。

这将工作，例如：

  ```
  Blog.objects.extra(
      select=OrderedDict([('a', '%s'), ('b', '%s')]),
      select_params=('one', 'two'))
  ```

如果您需要在选择字符串中使用文本`%s`，请使用序列`%%s`。

- `where` / `tables`

您可以使用`WHERE`定义显式SQL `where`子句 - 也许执行非显式连接。 您可以使用`FROM`手动将表添加到SQL `tables`子句。

`where`和`tables`都接受字符串列表。 所有`where`参数均为“与”任何其他搜索条件。

例如：

  ```
  Entry.objects.extra(where=["foo='a' OR bar = 'a'", "baz = 'a'"])
  ```

  ...（大致）翻译成以下SQL：

  ```
  SELECT * FROM blog_entry WHERE (foo='a' OR bar='a') AND (baz='a')
  ```

如果您要指定已在查询中使用的表，请在使用`tables`参数时小心。 当您通过`tables`参数添加额外的表时，Django假定您希望该表包含额外的时间（如果已包括）。 这会产生一个问题，因为表名将会被赋予一个别名。 如果表在SQL语句中多次出现，则第二次和后续出现必须使用别名，以便数据库可以区分它们。 如果您指的是在额外的`where`参数中添加的额外表，这将导致错误。

通常，您只需添加尚未显示在查询中的额外表。 然而，如果发生上述情况，则有几种解决方案。 首先，看看你是否可以不包括额外的表，并使用已经在查询中的一个。 如果不可能，请将`extra()`调用放在查询集结构的前面，以便您的表是该表的第一次使用。 最后，如果所有其他失败，请查看生成的查询并重写`where`添加以使用给您的额外表的别名。 每次以相同的方式构造查询集时，别名将是相同的，因此您可以依靠别名不更改。

- `order_by`

如果您需要使用通过`extra()`包含的一些新字段或表来对结果查询进行排序，请使用`order_by`参数`extra()`并传入一个字符串序列 。这些字符串应该是模型字段（如查询集上的正常[`order_by()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.order_by)方法），形式为`extra()`或您在`table_name.column_name`参数到`select`。

像这样：

  ```
  q = Entry.objects.extra(select={'is_recent': "pub_date > '2006-01-01'"})
  q = q.extra(order_by = ['-is_recent'])
  ```

这会将`False`的所有项目排序到结果集的前面（`True`在`is_recent`之前按降序排序）。

顺便说一句，你可以对`extra()`进行多次调用，它会按照你的期望（每次添加新的约束）运行。

- `params`

上述`where`参数可以使用标准Python数据库字符串占位符 - `'%s'`来指示数据库引擎应自动引用的参数。 `params`参数是要替换的任何额外参数的列表。

例如：

  ```
  Entry.objects.extra(where=['headline=%s'], params=['Lennon'])
  ```

始终使用`params`而不是将值直接嵌入`where`，因为`params`会确保根据您的特定后端正确引用值。 例如，引号将被正确转义。

Bad:

  ```
  Entry.objects.extra(where=["headline='Lennon'"])
  ```

Good:

  ```
  Entry.objects.extra(where=['headline=%s'], params=['Lennon'])
  ```

> 警告
如果您正在对MySQL执行查询，请注意，MySQL的静默类型强制可能会在混合类型时导致意外的结果。 如果在字符串类型的列上查询但具有整数值，MySQL将在执行比较之前将表中所有值的类型强制为整数。例如，如果表包含值`'def'`，`'abc'`，并查询`WHERE mycolumn = 0`，两行都将匹配。 为了防止这种情况，请在使用查询中的值之前执行正确的类型转换。

#### `defer()`

`defer(*fields)`

在一些复杂的数据建模情况下，你的模型可能包含大量字段，其中一些可能包含大量数据（例如文本字段），或者需要昂贵的处理来将它们转换为Python对象。 当你最初获取数据时不知道是否需要这些特定字段的情况下，如果你正在使用查询集的结果，你可以告诉Django不要从数据库中检索它们。

它通过传递字段名称到`defer()`实现不加载：

```
Entry.objects.defer("headline", "body")
```

具有延迟字段的查询集仍将返回模型实例。 每个延迟字段将在你访问该字段时从数据库中检索（每次只检索一个，而不是一次检索所有的延迟字段）。

你可以多次调用`defer()`。 每个调用都向延迟集添加新字段：

```
# 延迟body和headline两个字段。
Entry.objects.defer("body").filter(rating=5).defer("headline")
```

字段添加到延迟集的顺序无关紧要。 对已经延迟的字段名称再次`defer()`不会有问题（该字段仍将被延迟）。

你可以使用标准的双下划线符号来分隔关联的字段，从而推迟关联模型中的字段加载（如果关联模型通过[`select_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.select_related)加载）：

```
Blog.objects.select_related().defer("entry__headline", "entry__body")
```

如果要清除延迟字段集，请将`None`作为参数传递到`defer()`：

```
# 立即加载所有的字段。
my_queryset.defer(None)
```

模型中的某些字段不会被延迟，即使你要求它们延迟。 你永远不能延迟加载主键。 如果你使用[`select_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.select_related)检索关联模型，则不应延迟加载从主模型连接到关联模型的关联字段，否则将导致错误。

> 注

`defer()`方法（及其表兄弟，[`only()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.only)）仅适用于高级用例。 它们提供了一个优化，当你仔细分析查询并且*完全*了解需要什么信息，知道返回需要的字段与返回模型的全部字段之间的区别非常重要。

即使你认为你是在高级的情况下，**只在当你不能在查询集加载时确定是否需要额外的字段时使用defer()**。 如果你经常加载和使用特定的数据子集，最好的选择是规范化模型，并将未加载的数据放入单独的模型（和数据库表）。 如果列*必须*由于某种原因保留在一个表中，请创建一个具有`Meta.managed = false`（请参阅[`managed attribute`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.managed)文档）的模型，只包含你通常需要加载和使用否则就要调用`defer()`的字段。 这使得你的代码对读者更加明确，稍微更快一些，并且在Python进程中消耗更少的内存。

例如，这两个模型使用相同的底层数据库表：

```
class CommonlyUsedModel(models.Model):
    f1 = models.CharField(max_length=10)

    class Meta:
        managed = False
        db_table = 'app_largetable'

class ManagedModel(models.Model):
    f1 = models.CharField(max_length=10)
    f2 = models.CharField(max_length=10)

    class Meta:
        db_table = 'app_largetable'

# Two equivalent QuerySets:
CommonlyUsedModel.objects.all()
ManagedModel.objects.all().defer('f2')
```

如果许多字段需要在非托管模型中复制，最好使用共享字段创建抽象模型，然后使非托管模型和托管模型从抽象模型继承。

> 注
当对具有延迟字段的实例调用[`save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save)时，仅保存加载的字段。 有关详细信息，请参见[`save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save)。

#### `only()`

`only(*fields)`  

`only()`方法或多或少与[`defer()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.defer)相反。 你以*不*应该在检索模型时延迟的字段调用它。 如果你有一个模型几乎所有的字段需要延迟，使用`only()`指定补充的字段集可以导致更简单的代码。

假设你有一个包含字段`biography`、`age`和`name`的模型。 以下两个查询集是相同的，就延迟字段而言：

```
Person.objects.defer("age", "biography")
Person.objects.only("name")
```

每当你调用`only()`时，它将*替换*立即加载的字段集。 该方法的名称可以帮助记忆：**仅**这些字段立即加载；其余的被延迟。 因此，对`only()`的连续调用的结果是只有最后一次调用的字段被考虑：

```
# This will defer all fields except the headline.
Entry.objects.only("body", "rating").only("headline")
```

由于`defer()`以递增方式动作（向延迟列表中添加字段），因此你可以结合`only()`和`defer()`调用，它们将合乎逻辑地工作：

```
# Final result is that everything except "headline" is deferred.
Entry.objects.only("headline", "body").defer("body")

# Final result loads headline and body immediately (only() replaces any
# existing set of fields).
Entry.objects.defer("body").only("headline", "body")
```

[`defer()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.defer)文档注释中的所有注意事项也适用于`only()`。 请谨慎使用它，只有在没有其它选择之后才使用。

使用[`only()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.only)并省略使用[`select_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.select_related)请求的字段也是错误。

> 注
当对具有延迟字段的实例调用[`save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save)时，仅保存加载的字段。 有关详细信息，请参见[`save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save)。

#### `using()`

`using(alias)`  

如果你使用多个数据库，这个方法用于控制`QuerySet` 将在哪个数据库上求值。 这个方法的唯一参数是数据库的别名，定义在[`DATABASES`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-DATABASES)。

像这样：

```
# queries the database with the 'default' alias.
>>> Entry.objects.all()

# queries the database with the 'backup' alias
>>> Entry.objects.using('backup')
```

#### `select_for_update()`

`select_for_update(nowait=False, skip_locked=False, of=())`

返回一个锁住行直到事务结束的查询集，如果数据库支持，它将生成一个 `SELECT ... FOR UPDATE` 语句。

像这样：

```
entries = Entry.objects.select_for_update().filter(author=request.user)
```

所有匹配的行将被锁定，直到事务结束。这意味着可以通过锁防止数据被其它事务修改。

一般情况下如果其他事务锁定了相关行，那么本查询将被阻塞，直到锁被释放。 如果这不是你想要的行为，请使用`select_for_update(nowait=True)`. 这将使查询不阻塞。 如果其它事务持有冲突的锁, 那么查询将引发 [`DatabaseError`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.db.DatabaseError) 异常. 您也可以使用`select_for_update(skip_locked=True)`忽略锁定的行。 `nowait`和`skip_locked`是互斥的，并尝试调用`select_for_update()`启用这两个选项将导致[`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)

默认情况下，`select_for_update()`锁定查询选择的所有行。例如，除了查询集模型的行之外，在`select_related()`中指定的相关对象的行也被锁定。如果不需要，请使用与`select_related()`相同的字段语法在`select_for_update（of =（...））`中指定要锁定的相关对象。使用值“ self”来引用查询集的模型。

您不能对可为空的关系使用`select_for_update()`：

```shell
>>> Person.objects.select_related('hometown').select_for_update()
Traceback (most recent call last):
...
django.db.utils.NotSupportedError: FOR UPDATE cannot be applied to the nullable side of an outer join
```

为了避免这种限制，如果您不关心空对象，则可以排除它们：

```shell
>>> Person.objects.select_related('hometown').select_for_update().exclude(hometown=None)
<QuerySet [<Person: ...)>, ...]>
```

目前，`postgresql`，`oracle`和`mysql`数据库后端支持`select_for_update()`。 但是，MySQL不支持`nowait`和`skip_locked`参数。

使用不支持这些选项的数据库后端（如MySQL）将`nowait=True`或`skip_locked=True`转换为`select_for_update()`将导致[`DatabaseError`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.db.DatabaseError)被提出。 这可以防止代码意外阻止。

在自动提交模式下使用`select_for_update()`评估支持的后台支持的查询 `选择 ... 对于 UPDATE` 是一个[`TransactionManagementError`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.db.transaction.TransactionManagementError)错误，因为在这种情况下行不锁定。 如果允许这个调用，那么可能造成数据损坏，而且这个功能很容易在事务外被调用。

在不支持的后端使用`select_for_update()` `选择 ... 对于 UPDATE` 的后端 (例如SQLite) select_for_update() 将没有效果。 `选择 ... 对于 UPDATE` 将不会添加到查询中，并且如果在自动提交模式下使用了`select_for_update()`，则不会引发错误。

> 警告

尽管`select_for_update()`通常在自动提交模式下失败，但由于TestCase自动将每个测试包装在事务中，因此即使在`atomic()`块之外调用TestCase中的`select_for_update()`也会（可能意外地）通过而不会引发TransactionManagementError。要正确测试`select_for_update()`，应使用TransactionTestCase。

**在Django更改1.11：**

添加了`skip_locked`参数。

在Django 2.0中进行了更改：

添加了参数。

#### `raw()`

`raw(raw_query, params=None, translations=None)`

接收一个原始的SQL 查询，执行它并返回一个`django.db.models.query.RawQuerySet` 实例。 这个`RawQuerySet` 实例可以迭代以提供实例对象，就像普通的`QuerySet` 一样。

更多信息参见[Performing raw SQL queries](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/sql.html)。

> 警告
`raw()` 永远触发一个新的查询，而与之前的filter 无关。 因此，它通常应该从`Manager` 或一个全新的`QuerySet` 实例调用。

### 不返回`QuerySet`的方法

以下`QuerySet`方法评估`QuerySet`并返回*而不是* 一个 `QuerySet`。

这些方法不使用高速缓存（请参阅[Caching and QuerySets](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/queries.html#caching-and-querysets)）。 这些方法每次被调用的时候都会查询数据库。

#### `get()`

`get(**kwargs)`

返回按照查询参数匹配到的对象，参数的格式应该符合 [Field lookups](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#id4)的要求.

如果匹配到的对象个数不只一个的话，`get()` 将会触发[`MultipleObjectsReturned`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.core.exceptions.MultipleObjectsReturned) 异常. [`MultipleObjectsReturned`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.core.exceptions.MultipleObjectsReturned) 异常是模型类的属性.

如果根据给出的参数匹配不到对象的话，`get()` 将触发[`DoesNotExist`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.DoesNotExist) 异常. 这个异常是模型类的属性. 例如：

```
Entry.objects.get(id='foo') # raises Entry.DoesNotExist
```

[`DoesNotExist`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.DoesNotExist)异常从[`django.core.exceptions.ObjectDoesNotExist`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.core.exceptions.ObjectDoesNotExist)继承，因此您可以定位多个[`DoesNotExist`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.DoesNotExist)异常。 例如：

```
from django.core.exceptions import ObjectDoesNotExist
try:
    e = Entry.objects.get(id=3)
    b = Blog.objects.get(id=1)
except ObjectDoesNotExist:
    print("Either the entry or blog doesn't exist.")
```

如果您希望一个查询器返回一行，则可以使用`get()`而不使用任何参数来返回该行的对象：

```
entry = Entry.objects.filter(...).exclude(...).get()
```

#### `create()`

`create(**kwargs)`

一个在一步操作中同时创建对象并且保存的便捷方法. 因此：

```
p = Person.objects.create(first_name="Bruce", last_name="Springsteen")
```

和:

```
p = Person(first_name="Bruce", last_name="Springsteen")
p.save(force_insert=True)
```

是等同的.

参数 [force_insert](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#ref-models-force-insert) 在其他的文档中有介绍, 它意味着一个新的对象一定会被创建. 正常情况中，你不必要担心这点. 然而, 如果你的model中有一个你手动设置主键， 并且这个值已经存在于数据库中, 调用 `create()`将会失败并且触发 [`IntegrityError`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.db.IntegrityError) 因为主键必须是唯一的. 如果你手动设置了主键，做好异常处理的准备.

#### `get_or_create()`

`get_or_create(defaults=None, **kwargs)`

一个通过给出的`kwargs` 来查询对象的便捷方法（如果你的模型中的所有字段都有默认值，可以为空），需要的话创建一个对象。

返回一个由`(object, created)`组成的元组，元组中的`object` 是一个查询到的或者是被创建的对象， `created` 是一个表示是否创建了新的对象的布尔值。

这主要用作样板代码的一种快捷方式。 像这样：

```
try:
    obj = Person.objects.get(first_name='John', last_name='Lennon')
except Person.DoesNotExist:
    obj = Person(first_name='John', last_name='Lennon', birthday=date(1940, 10, 9))
    obj.save()
```

如果模型的字段数量较大的话，这种模式就变的非常不易用了。 上面的示例可以用`get_or_create()`重写 :

```
obj, created = Person.objects.get_or_create(
    first_name='John',
    last_name='Lennon',
    defaults={'birthday': date(1940, 10, 9)},
)
```

任何传递给 `get_or_create()` 的关键字参数，*除了一个可选的*`defaults`，都将传递给[`get()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.get) 调用。 如果查找到一个对象，`get_or_create()` 返回一个包含匹配到的对象以及`False` 组成的元组。 如果查找到的对象超过一个以上，`get_or_create` 将引发[`MultipleObjectsReturned`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.core.exceptions.MultipleObjectsReturned)。 *如果查找不到对象*， `get_or_create()` 将会实例化并保存一个新的对象，返回一个由新的对象以及`True` 组成的元组。 新的对象将会大概按照以下的逻辑创建:

```
params = {k: v for k, v in kwargs.items() if '__' not in k}
params.update({k: v() if callable(v) else v for k, v in defaults.items()})
obj = self.model(**params)
obj.save()
```

它表示从非`'defaults'` 且不包含双下划线的关键字参数开始（暗示这是一个不精确的查询）。 然后将`defaults` 的内容添加进来，覆盖必要的键，并使用结果作为关键字参数传递给模型类。 如果`defaults`中有任何可调用值，请对它们进行评估。 这是对用到的算法的简单描述，但它包含了所有的相关的细节。 内部实现比这更多的错误检查，并处理一些额外的边缘条件；如果您有兴趣，请阅读代码。

如果你有一个名为`'defaults__exact'`的字段，并且想在`get_or_create()` 是用它作为精确查询，只需要使用`defaults`，像这样：

```
Foo.objects.get_or_create(defaults__exact='bar', defaults={'defaults': 'baz'})
```

当你使用手动指定的主键时，`get_or_create()` 方法与[`create()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.create)方法有相似的错误行为 。 如果需要创建一个对象而该对象的主键早已存在于数据库中，[`IntegrityError`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.db.IntegrityError) 异常将会被触发。

这个方法假设正确使用原子操作，正确的数据库配置和底层数据库的正确行为。 然而，如果数据库级别没有对`get_or_create` 中用到的`kwargs` 强制要求唯一性（参见[`unique`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.Field.unique) 和 [`unique_together`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.unique_together)），这个方法容易导致竞态条件可能会仍具有相同参数的多行同时插入。

如果你正在使用MySQL，请确保使用`READ COMMITTED` 隔离级别而不是默认的`REPEATABLE READ`，否则你将会遇到`get_or_create` 引发[`IntegrityError`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/exceptions.html#django.db.IntegrityError) 但对象在接下来的[`get()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.get) 调用中并不存在的情况。

最后讲一句`get_or_create()` 在Django 视图中的使用。 请确保只在`POST` 请求中使用，除非你有充分的理由。 `GET` 请求不应该对数据有任何影响。 而`POST` 则用于对数据产生影响的请求。 有关更多信息，请参阅HTTP规范中的 [**Safe methods**](https://tools.ietf.org/html/rfc7231.html#section-4.2.1)。

> 警告

你可以通过[`ManyToManyField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ManyToManyField) 属性和反向关联使用`get_or_create()`。 在这种情况下，你应该限制查询在关联的上下文内部。 如果你不一致地使用它，将可能导致完整性问题。

根据下面的模型：

```
class Chapter(models.Model):
    title = models.CharField(max_length=255, unique=True)

class Book(models.Model):
    title = models.CharField(max_length=256)
    chapters = models.ManyToManyField(Chapter)
```

你可以通过Book 的chapters 字段使用`get_or_create()`，但是它只会获取该Book 内部的上下文：

```
>>> book = Book.objects.create(title="Ulysses")
>>> book.chapters.get_or_create(title="Telemachus")
(<Chapter: Telemachus>, True)
>>> book.chapters.get_or_create(title="Telemachus")
(<Chapter: Telemachus>, False)
>>> Chapter.objects.create(title="Chapter 1")
<Chapter: Chapter 1>
>>> book.chapters.get_or_create(title="Chapter 1")
# Raises IntegrityError
```

发生这个错误时因为它尝试通过Book “Ulysses” 获取或者创建“Chapter 1”，但是它不能：关联关系不能获取这个chapter 因为它与这个book 不关联，但因为`title` 字段是唯一的它仍然不能创建。

**在Django更改1.11：**

在`defaults`中增加了对可调用值的支持。



#### `update_or_create()`

`update_or_create(defaults=None, **kwargs)`

一个通过给出的`kwargs` 来更新对象的便捷方法， 如果需要的话创建一个新的对象。 `defaults` 是一个由 (field, value) 对组成的字典，用于更新对象。 `defaults`中的值可以是可调用的。

返回一个由 `(object, created)`组成的元组,元组中的`object` 是一个创建的或者是被更新的对象， `created` 是一个标示是否创建了新的对象的布尔值。

`update_or_create` 方法尝试通过给出的`kwargs` 去从数据库中获取匹配的对象。 如果找到匹配的对象，它将会依据`defaults` 字典给出的值更新字段。

这主要用作样板代码的一种快捷方式。 像这样：

```python
defaults = {'first_name': 'Bob'}
try:
    obj = Person.objects.get(first_name='John', last_name='Lennon')
    for key, value in defaults.items():
        setattr(obj, key, value)
    obj.save()
except Person.DoesNotExist:
    new_values = {'first_name': 'John', 'last_name': 'Lennon'}
    new_values.update(defaults)
    obj = Person(**new_values)
    obj.save()
```

如果模型的字段数量较大的话，这种模式就变的非常不易用了。 上面的示例可以用 `update_or_create()` 重写:

```
obj, created = Person.objects.update_or_create(
    first_name='John', last_name='Lennon',
    defaults={'first_name': 'Bob'},
)
```

`kwargs` 中的名称如何解析的详细描述可以参见[`get_or_create()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.get_or_create)。

和上文描述的[`get_or_create()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.get_or_create) 一样，这个方式容易导致竞态条件，如果数据库层级没有前置唯一性它会让多行同时插入。

**在Django更改1.11：**

在`defaults`中增加了对可调用值的支持。

#### `bulk_create()`

`bulk_create(objs, batch_size=None)` 

此方法以高效的方式（通常只有1个查询，无论有多少对象）将提供的对象列表插入到数据库中：

```
>>> Entry.objects.bulk_create([
...     Entry(headline='This is a test'),
...     Entry(headline='This is only a test'),
... ])
```

这有一些注意事项：

- 将不会调用模型的`save()`方法，并且不会发送`pre_save`和`post_save`信号。
- 它不适用于多表继承场景中的子模型。
- 如果模型的主键是[`AutoField`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.AutoField)，则不会像`save()`那样检索并设置主键属性，除非数据库后端支持（当前只支持PostgreSQL）。
- 它不适用于多对多关系。
- 它将objs转换为列表，如果它是生成器，则可以完全评估objs。强制转换允许检查所有对象，以便可以首先插入具有手动设置的主键的任何对象。如果您要批量插入对象而不一次评估整个生成器，则可以使用此技术，只要对象没有任何手动设置的主键即可：

```python
from itertools import islice

batch_size = 100
objs = (Entry(headling'Test %s' % i) for i in range(1000))
while True:
    batch = list(islice(objs, batch_size))
    if not batch:
        break
    Entry.objects.bulk_create(batch, batch_size)
```

`batch_size`参数控制在单个查询中创建的对象数。 默认值是在一个批处理中创建所有对象，除了SQLite，其中默认值为每个查询最多使用999个变量。

#### `count()`

`count()` 

返回在数据库中对应的 `QuerySet`.对象的个数。 `count()` 永远不会引发异常。

例如：

```
# Returns the total number of entries in the database.
Entry.objects.count()

# Returns the number of entries whose headline contains 'Lennon'
Entry.objects.filter(headline__contains='Lennon').count()
```

`count()`在后台执行`SELECT COUNT（*）` `count()`，而不是将所有的记录加载到Python对象中并在结果上调用`len()`（除非你需要将对象加载到内存中， `len()`会更快）。

根据您使用的数据库（例如PostgreSQL vs. MySQL），`count()`可能返回一个长整型而不是普通的Python整数。 这是一个潜在的实现方案，不应该引起任何真实世界的问题。

请注意，如果您想要`count()`中的项目数量，并且还要从中检索模型实例（例如，通过迭代它），使用`len(queryset)`更有效，这不会像`QuerySet`一样导致额外的数据库查询。

#### `in_bulk()`

`in_bulk(id_list=None, field_name='pk')`


获取字段值的列表（id_list）和这些值的field_name，然后返回将每个值映射到具有给定字段值的对象实例的字典。如果未提供id_list，则返回查询集中的所有对象。field_name必须是唯一字段，并且默认为主键。

例如：

```
>>> Blog.objects.in_bulk([1])
{1: <Blog: Beatles Blog>}
>>> Blog.objects.in_bulk([1, 2])
{1: <Blog: Beatles Blog>, 2: <Blog: Cheddar Talk>}
>>> Blog.objects.in_bulk([])
{}
>>> Blog.objects.in_bulk()
{1: <Blog: Beatles Blog>, 2: <Blog: Cheddar Talk>, 3: <Blog: Django Weblog>}
>>> Blog.objects.in_bulk(['beatles_blog'], field_name='slug')
{'beatles_blog': <Blog: Beatles Blog>}
```

如果你传递`in_bulk()`一个空列表，你会得到一个空的字典。

#### `iterator()`

`iterator(chunk_size=2000)` 

评估`QuerySet`（通过执行查询），并返回一个迭代器（参见 [**PEP 234**](https://www.python.org/dev/peps/pep-0234)）。 `QuerySet`通常在内部缓存其结果，以便在重复计算是不会导致额外的查询。 相反，`iterator()`将直接读取结果，而不在`QuerySet`级别执行任何缓存（内部，默认迭代器调用`iterator()`并高速缓存返回值）。 对于返回大量只需要访问一次的对象的`QuerySet`，这可以带来更好的性能和显着减少内存。

请注意，在已经求值了的`iterator()`上使用`QuerySet`会强制它再次计算，重复查询。

此外，使用`iterator()`会导致先前的`prefetch_related()`调用被忽略，因为这两个优化一起没有意义。

根据数据库后端，查询结果将一次性加载或使用服务器端游标从数据库流式传输。

- 使用服务器端光标

Oracle和[PostgreSQL](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#postgresql-server-side-cursors)使用服务器端游标从数据库中流式传输结果，而不将整个结果集加载到内存中。

Oracle数据库驱动程序始终使用服务器端游标。

在PostgreSQL上，服务器端游标仅在[`DISABLE_SERVER_SIDE_CURSORS`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-DATABASE-DISABLE_SERVER_SIDE_CURSORS)设置为`False`时使用。 如果您使用以事务池模式配置的连接池，请阅读[Transaction pooling and server-side cursors](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#transaction-pooling-server-side-cursors)。 当服务器端游标被禁用时，该行为与不支持服务器端游标的数据库相同。

- 没有服务器端游标

MySQL和SQLite不支持流结果，因此Python数据库驱动程序将整个结果集加载到内存中。 然后，使用 [**PEP 249**](https://www.python.org/dev/peps/pep-0249)中定义的`fetchmany()`方法，使用数据库适配器将结果集转换为Python行对象。

chunk_size参数控制Django从数据库驱动程序检索的批处理的大小。较大的批处理减少了与数据库驱动程序进行通信的开销，但代价是稍微增加了内存消耗。

chunk_size的默认值2000，来自对psycopg邮件列表的计算：
> 假设10-20列的行包含文本和数字数据，则2000将获取不到100KB的数据，如果循环较早退出，这似乎是传输的行数与丢弃的数据之间的一种很好的折衷。

在Django 1.11中进行了更改：
PostgreSQL支持服务器端游标。
在Django 2.0中进行了更改：
chunk_size参数已添加。

#### `latest()`

`latest(*fields)`

根据给定的字段返回表中的最新对象。

此示例根据`Entry`字段返回表中的最新`pub_date`：

```
Entry.objects.latest('pub_date')
```

您还可以根据几个字段选择最新的。例如，当两个条目具有相同的pub_date时，选择最早的expire_date的条目：
```
Entry.objects.latest('pub_date', '-expire_date')
```

`-expire_date`中的负号表示按降序对expire_date进行排序。由于`latest()`获得最后结果，因此选择了最早expire_date的条目。

如果模型的[Meta](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/models.html#meta-options)指定[`get_latest_by`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.get_latest_by)，则可以将`latest()`参数留给`earliest()`或者`field_name`。 默认情况下，Django将使用[`get_latest_by`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.get_latest_by)中指定的字段。

像[`get()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.get)，`earliest()`和`latest()` raise [`DoesNotExist`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.DoesNotExist)参数。

请注意，`earliest()`和`latest()`仅仅是为了方便和可读性。

在Django 2.0中进行了更改：

支持几个参数增加了。


> `earliest()`和`latest()`可能会返回空日期的实例。

由于订购被委派给数据库，如果使用不同的数据库，则允许空值的字段的结果可能有所不同。 例如，PostgreSQL和MySQL排序空值，就好像它们高于非空值，而SQLite则相反。

您可能需要过滤掉空值：

```
Entry.objects.filter(pub_date__isnull=False).latest('pub_date')
```

#### `earliest()`

`earliest(field_name=None)`

除非方向更改，否则像[`latest()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.latest)。

#### `first()`

`first()`

返回结果集的第一个对象, 当没有找到时返回`None`. 如果 `QuerySet` 没有设置排序,则将会自动按主键进行排序

例如：

```
p = Article.objects.order_by('title', 'pub_date').first()
```

请注意，`first()`是一种简便方法，下面的代码示例等同于上面的示例：

```
try:
    p = Article.objects.order_by('title', 'pub_date')[0]
except IndexError:
    p = None
```



#### `last()`

`last()`

工作方式类似[`first()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.first)，只是返回的是查询集中最后一个对象。


#### `aggregate()`

`aggregate(*args, **kwargs)`

返回汇总值的字典（平均值，总和等） 通过`QuerySet`进行计算。 `aggregate()` 的每个参数指定返回的字典中将要包含的值。

Django 提供的聚合函数在下文的[聚合函数](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#id5)文档中讲述。 因为聚合也是[query expressions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html)，你可以组合多个聚合以及值来创建复杂的聚合。

使用关键字参数指定的聚合将使用关键字参数的名称作为Annotation 的名称。 匿名的参数的名称将基于聚合函数的名称和模型字段生成。 复杂的聚合不可以使用匿名参数，它们必须指定一个关键字参数作为别名。

例如，当你使用Blog Entry 时，你可能想知道对Author 贡献的Blog Entry 的数目：

```
>>> from django.db.models import Count
>>> q = Blog.objects.aggregate(Count('entry'))
{'entry__count': 16}
```

通过使用关键字参数来指定聚合函数，你可以控制返回的聚合的值的名称：

```
>>> q = Blog.objects.aggregate(number_of_entries=Count('entry'))
{'number_of_entries': 16}
```

聚合的深入讨论，参见 [the topic guide on Aggregation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/aggregation.html)。

#### `exists()`

`exists()` 

如果[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet) 包含任何结果，则返回`True`，否则返回`False`。 它会试图用最简单和最快的方法完成查询，但它执行的方法与普通的*QuerySet* 查询[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet)几乎相同。

[`exists()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.exists) 用于搜寻对象是否在[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet) 中以及[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet) 是否存在任何对象，特别是[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet) 比较大的时候。

查找具有唯一性字段（例如`primary_key`）的模型是否在一个[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet) 中的最高效的方法是：

```
entry = Entry.objects.get(pk=123)
if some_queryset.filter(pk=entry.pk).exists():
    print("Entry contained in queryset")
```

它将比下面的方法快很多，这个方法要求对QuerySet 求值并迭代整个QuerySet：

```
if entry in some_queryset:
   print("Entry contained in QuerySet")
```

若要查找一个QuerySet 是否包含任何元素：

```
if some_queryset.exists():
    print("There is at least one object in some_queryset")
```

将快于：

```
if some_queryset:
    print("There is at least one object in some_queryset")
```

...但不是在很大程度上（因此需要一个大的查询来提高效率）。

另外，如果`bool(some_queryset)` 还没有求值，但你知道它将在某个时刻求值，那么使用`some_queryset.exists()` 将比简单地使用`some_queryset` 完成更多的工作（一个查询用于存在性检查，另外一个是后面的求值），后者将求值并检查是否有结果返回。

#### `update()`

`update(**kwargs)`

对指定的字段执行SQL更新查询，并返回匹配的行数（如果某些行已具有新值，则可能不等于已更新的行数）。

例如，要对2010年发布的所有博客条目启用评论，您可以执行以下操作：

```
>>> Entry.objects.filter(pub_date__year=2010).update(comments_on=False)
```

（假设您的`comments_on`模型具有字段`pub_date`和`Entry`。）

您可以更新多个字段 - 没有多少字段的限制。 例如，在这里我们更新`comments_on`和`headline`字段：

```
>>> Entry.objects.filter(pub_date__year=2010).update(comments_on=False, headline='This is old')
```

`update()`方法立即应用，对更新的[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet)的唯一限制是它只能更新模型主表中的列，而不是相关模型。 你不能这样做，例如：

```
>>> Entry.objects.update(blog__name='foo') # Won't work!
```

仍然可以根据相关字段进行过滤：

```
>>> Entry.objects.filter(blog__id=1).update(comments_on=True)
```

您不能在[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet)上调用`update()`，该查询已截取一个切片，或者无法再进行过滤。

`update()`方法返回受影响的行数：

```
>>> Entry.objects.filter(id=64).update(comments_on=True)
1

>>> Entry.objects.filter(slug='nonexistent-slug').update(comments_on=True)
0

>>> Entry.objects.filter(pub_date__year=2010).update(comments_on=False)
132
```

如果你只是更新一个记录，不需要对模型对象做任何事情，最有效的方法是调用`update()`，而不是将模型对象加载到内存中。 例如，而不是这样做：

```
e = Entry.objects.get(id=10)
e.comments_on = False
e.save()
```

…做这个：

```
Entry.objects.filter(id=10).update(comments_on=False)
```

使用`update()`还可以防止在加载对象和调用`save()`之间的短时间内数据库中某些内容可能发生更改的竞争条件。

最后，意识到`update()`在SQL级别进行更新，因此不会在模型上调用任何`save()`方法，也不会发出pre_save或post_save信号（这是调用`Model.save()`）。如果要为具有自定义`save()`方法的模型更新一堆记录，请遍历它们并调用`save()`，如下所示：

```
for e in Entry.objects.filter(pub_date__year=2010):
    e.comments_on = False
    e.save()
```

#### `delete()`

`delete()`

对[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet)中的所有行执行SQL删除查询，并返回删除的对象数和每个对象类型的删除次数的字典。

立即应用`delete()`。 您不能在[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet)上调用`delete()`，该查询已采取切片或以其他方式无法过滤。

例如，要删除特定博客中的所有条目：

```
>>> b = Blog.objects.get(pk=1)

# Delete all the entries belonging to this Blog.
>>> Entry.objects.filter(blog=b).delete()
(4, {'weblog.Entry': 2, 'weblog.Entry_authors': 2})
```

默认情况下，Django的[`ForeignKey`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey)模拟SQL约束`ON DELETE CASCADE`字，任何具有指向要删除的对象的外键的对象将与它们一起被删除。 像这样：

```
>>> blogs = Blog.objects.all()

# This will delete all Blogs and all of their Entry objects.
>>> blogs.delete()
(5, {'weblog.Blog': 1, 'weblog.Entry': 2, 'weblog.Entry_authors': 2})
```

这种级联的行为可以通过的[`ForeignKey`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey) 的[`on_delete`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey.on_delete) 参数自定义。

`delete()`方法执行批量删除，并且不会在模型上调用任何`delete()`方法。 但它会为所有已删除的对象（包括级联删除）发出[`pre_delete`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/signals.html#django.db.models.signals.pre_delete)和[`post_delete`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/signals.html#django.db.models.signals.post_delete)信号。

Django需要获取对象到内存中以发送信号和处理级联。 然而，如果没有级联和没有信号，那么Django可以采取快速路径并删除对象而不提取到内存中。 对于大型删除，这可以显着减少内存使用。 执行的查询量也可以减少。

设置为[`on_delete`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html#django.db.models.ForeignKey.on_delete) `DO_NOTHING`的外键不会阻止删除快速路径。

请注意，在对象删除中生成的查询是实施详细信息，可能会更改。

#### `as_manager()`

```
classmethod  as_manager()
```

类方法，返回[`Manager`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/managers.html#django.db.models.Manager)的实例与`QuerySet`的方法的副本。 有关详细信息，请参见[使用QuerySet方法创建一个manager](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/managers.html#create-manager-with-queryset-methods)。

### `Field`查找

字段查询是指如何指定SQL `WHERE` 子句的内容。 它们通过`QuerySet`的[`filter()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.filter), [`exclude()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.exclude) and [`get()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.get)的关键字参数指定.

查阅简介, 请参考 [模型和数据库查找文档](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/queries.html#field-lookups-intro).

Django的内置查找如下， 也可以为模型字段写入[自定义查找](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/howto/custom-lookups.html)。

为了方便当没有提供查找类型时（例如`Entry.objects.get(id=14)`），查找类型默认为[`exact`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#std:fieldlookup-exact)。

#### `exact`

精确匹配。 如果为比较提供的值为`NULL`，它将被解释为SQL `None`（有关详细信息，请参阅[`isnull`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#std:fieldlookup-isnull)）。

例子：

```
Entry.objects.get(id__exact=14)
Entry.objects.get(id__exact=None)
```

SQL等价物：

```
SELECT ... WHERE id = 14;
SELECT ... WHERE id IS NULL;
```

>  MySQL比较
在MySQL中，可通过数据库表的“排序规则”设置来决定是否执行区分大小写的`精确`匹配。 这是一个数据库设置，*而不是*一个Django设置。 可以配置MySQL表以使用区分大小写的比较，但涉及一些折衷。 有关详细信息，请参阅[数据库](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html)文档中的[整理部分](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#mysql-collation)。

#### `iexact`

不区分大小写的精确匹配 如果为比较提供的值为`NULL`，它将被解释为SQL `None`（有关详细信息，请参阅[`isnull`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#std:fieldlookup-isnull)）。

例如：

```
Blog.objects.get(name__iexact='beatles blog')
Blog.objects.get(name__iexact=None)
```

SQL等价物：

```
SELECT ... WHERE name ILIKE 'beatles blog';
SELECT ... WHERE name IS NULL;
```

请注意，第一个查询将匹配 `'Beatles Blog'`, `'beatles blog'`, `'BeAtLes BLoG'`, etc.

>  SQLite用户
当使用SQLite后端和Unicode（非ASCII）字符串时，请记住关于字符串比较的[database note](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#sqlite-string-matching)， SQLite不对Unicode字符串进行不区分大小写的匹配。

#### `contains`

大小写敏感的包含关系测试。

例如：

```
Entry.objects.get(headline__contains='Lennon')
```

SQL等效：

```
SELECT ... WHERE headline LIKE '%Lennon%';
```

请注意，这将匹配标题`'Lennon honored today'`，但不符合`'lennon honored today'`.

>  SQLite用户
在SQLite中，使用`LIKE`子句进行查询时，Like子句并不会区分大小写; 对于SQLite，使用`contains`和`icontains`得到的结果是相同的. 有关详细信息，请参阅[database note](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#sqlite-string-matching)。

#### `icontains`

测试是否包含，不区分大小写。

例如：

```
Entry.objects.get(headline__icontains='Lennon')
```

等效于SQL：

```
SELECT ... WHERE headline ILIKE '%Lennon%';
```

> SQLite用户
当使用SQLite后端和Unicode（非ASCII）字符串时，请记住关于字符串比较的[database note](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#sqlite-string-matching)。

#### `in`

在给定的列表。

例如：

```
Entry.objects.filter(id__in=[1, 3, 4])
```

SQL等效：

```
SELECT ... WHERE id IN (1, 3, 4);
```

您还可以使用查询集动态评估值列表，而不是提供文字值列表：

```
inner_qs = Blog.objects.filter(name__contains='Cheddar')
entries = Entry.objects.filter(blog__in=inner_qs)
```

此查询集将作为subselect语句求值：

```
SELECT ... WHERE blog.id IN (SELECT id FROM ... WHERE NAME LIKE '%Cheddar%')
```

如果传递从`values()`或`values_list()`得到的`QuerySet`作为`__in`查询的值，你需要确保只提取一个字段到结果中。 如下面代码所示，使用values只从查询结果中提取博客名称：

```
inner_qs = Blog.objects.filter(name__contains='Ch').values('name')
entries = Entry.objects.filter(blog__name__in=inner_qs)
```

下面这个例子将产生一个异常，由于内查询试图提取两个字段的值，但是查询语句只期望提取一个字段的值：

```
# Bad code! Will raise a TypeError.
inner_qs = Blog.objects.filter(name__contains='Ch').values('name', 'id')
entries = Entry.objects.filter(blog__name__in=inner_qs)
```

> 性能注意事项
对于使用嵌套查询和了解数据库服务器的性能特征（如果有疑问，去做基准测试）要谨慎。 一些数据库后端，最着名的是MySQL，不能很好地优化嵌套查询。 在这些情况下，提取值列表然后将其传递到第二个查询中更有效。 也就是说，执行两个查询，而不是一个：

```
values = Blog.objects.filter(
        name__contains='Cheddar').values_list('pk', flat=True)
entries = Entry.objects.filter(blog__in=list(values))
```

请注意`list()`调用Blog `QuerySet`以强制执行第一个查询。 没有它，将执行嵌套查询，因为[QuerySets are lazy](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/queries.html#querysets-are-lazy)。

#### `gt`

大于

例如：

```
Entry.objects.filter(id__gt=4)
```

SQL等效：

```
SELECT ... WHERE id > 4;
```

#### `gte`

大于或等于

#### `lt`

小于

#### `lte`

小于或等于

#### `startswith`

区分大小写，开始位置匹配

例如：

```
Entry.objects.filter(headline__startswith='Lennon')
```

SQL等效：

```
SELECT ... WHERE headline LIKE 'Lennon%';
```

SQLite不支持区分大小写的LIKE语句；startwith的行为类似于istartswith for SQLite


#### `istartswith`

不区分大小写，开始位置匹配

例如：

```
Entry.objects.filter(headline__istartswith='Lennon')
```

SQL等效：

```
SELECT ... WHERE headline ILIKE 'Lennon%';
```

> SQLite用户
当使用SQLite后端和Unicode（非ASCII）字符串时，请记住关于字符串比较的[database note](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#sqlite-string-matching)。

#### `endswith`

区分大小写。

例如：

```
Entry.objects.filter(headline__endswith='Lennon')
```

SQL等效：

```
SELECT ... WHERE headline LIKE '%Lennon';
```

> SQLite用户
>
> SQLite不支持区分大小写的LIKE语句；endsand的行为类似于SQLite的iendswith。
>
> 有关更多信息，请参阅[database note](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#sqlite-string-matching)文档。

#### `iendswith`

不区分大小写。

例如：

```
Entry.objects.filter(headline__iendswith='Lennon')
```

SQL等效：

```
SELECT ... WHERE headline ILIKE '%Lennon'
```

>SQLite用户
当使用SQLite后端和Unicode（非ASCII）字符串时，请记住关于字符串比较的[database note](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/databases.html#sqlite-string-matching)。

#### `range`

范围测试（包含于之中）。

例如：

```
import datetime
start_date = datetime.date(2005, 1, 1)
end_date = datetime.date(2005, 3, 31)
Entry.objects.filter(pub_date__range=(start_date, end_date))
```

SQL等效：

```
SELECT ... WHERE pub_date BETWEEN '2005-01-01' and '2005-03-31';
```

您可以在任何可以使用`range`的SQL中使用`BETWEEN`（对于日期，数字和偶数字符）。

> 警告
过滤具有日期的`DateTimeField`不会包含最后一天的项目，因为边界被解释为“给定日期的0am”。 如果`pub_date`是`DateTimeField`，上面的表达式将变成这个SQL：

```
SELECT ... WHERE pub_date BETWEEN '2005-01-01 00:00:00' and '2005-03-31 00:00:00';
```

一般来说，不能混合使用日期和数据时间。

#### `date`

对于datetime字段，将值作为日期转换。 允许链接附加字段查找。 获取日期值。

例如：

```
Entry.objects.filter(pub_date__date=datetime.date(2005, 1, 1))
Entry.objects.filter(pub_date__date__gt=datetime.date(2005, 1, 1))
```

（此查找不包括等效的SQL代码片段，因为相关查询的实现因不同数据库引擎而异）。

当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，字段将转换为当前时区，然后进行过滤。

#### `year`

对于日期和日期时间字段，确切的年匹配。 允许链接附加字段查找。 整数年。

例如：

```
Entry.objects.filter(pub_date__year=2005)
Entry.objects.filter(pub_date__year__gte=2005)
```

SQL等效：

```
SELECT ... WHERE pub_date BETWEEN '2005-01-01' AND '2005-12-31';
SELECT ... WHERE pub_date >= '2005-01-01';
```

（确切的SQL语法因每个数据库引擎而异）。

当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，在过滤之前，datetime字段将转换为当前时区。

#### `month`

对于日期和日期时间字段，确切的月份匹配。 允许链接附加字段查找。 取整数1（1月）至12（12月）。

例如：

```
Entry.objects.filter(pub_date__month=12)
Entry.objects.filter(pub_date__month__gte=6)
```

SQL等效：

```
SELECT ... WHERE EXTRACT('month' FROM pub_date) = '12';
SELECT ... WHERE EXTRACT('month' FROM pub_date) >= '6';
```

（确切的SQL语法因每个数据库引擎而异）。

当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，在过滤之前，datetime字段将转换为当前时区。 这需要数据库中的[time zone definitions in the database](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#database-time-zone-definitions)。

#### `day`

对于日期和日期时间字段，具体到某一天的匹配。 允许链接附加字段查找。 取一个整数的天数。

例如：

```
Entry.objects.filter(pub_date__day=3)
Entry.objects.filter(pub_date__day__gte=3)
```

SQL等效：

```
SELECT ... WHERE EXTRACT('day' FROM pub_date) = '3';
SELECT ... WHERE EXTRACT('day' FROM pub_date) >= '3';
```

（确切的SQL语法因每个数据库引擎而异）。

请注意，这将匹配每月第三天（例如1月3日，7月3日等）的任何包含pub_date的记录。

当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，在过滤之前，datetime字段将转换为当前时区。 这需要数据库中的[time zone definitions in the database](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#database-time-zone-definitions)。

#### `week`

**Django中的新功能1.11。**

对于日期和日期时间字段，请根据[ISO-8601](https://en.wikipedia.org/wiki/ISO-8601)返回周号（1-52或53），即星期一开始的星期，星期四或之前的第一周。

例如：

```
Entry.objects.filter(pub_date__week=52)
Entry.objects.filter(pub_date__week__gte=32, pub_date__week__lte=38)
```

（此查找不包括等效的SQL代码片段，因为相关查询的实现因不同数据库引擎而异）。

当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，字段将转换为当前时区，然后进行过滤。

#### `week_day`

对于日期和日期时间字段，“星期几”匹配。 允许链接附加字段查找。

取整数值，表示星期几从1（星期日）到7（星期六）。

例如：

```
Entry.objects.filter(pub_date__week_day=2)
Entry.objects.filter(pub_date__week_day__gte=2)
```

（此查找不包括等效的SQL代码片段，因为相关查询的实现因不同数据库引擎而异）。

请注意，这将匹配落在星期一（星期二）的任何记录（`pub_date`），而不管其出现的月份或年份。 周日被索引，第1天为星期天，第7天为星期六。

当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，在过滤之前，datetime字段将转换为当前时区。 这需要数据库中的[time zone definitions in the database](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#database-time-zone-definitions)。

#### `quarter`

Django 2.0的新功能。

对于日期和日期时间字段，匹配“一年的四分之一”。允许链接其他字段查找。取一个介于1到4之间的整数值，代表一年的四分之一。

在第二季度（4月1日至6月30日）检索条目的示例：

```
Entry.objects.filter(pub_date__quarter=2)
```

（此查询不包括等效的SQL代码片段，因为相关查询的实现在不同的数据库引擎之间有所不同。）

当USE_TZ为True时，日期时间字段将在过滤之前转换为当前时区。这需要数据库中的时区定义。


#### `time`

**Django中的新功能1.11。**

对于datetime字段，将值转换为时间。 允许链接附加字段查找。 获取[`datetime.time`](https://docs.python.org/3/library/datetime.html#datetime.time)值。

例如：

```
Entry.objects.filter(pub_date__time=datetime.time(14, 30))
Entry.objects.filter(pub_date__time__between=(datetime.time(8), datetime.time(17)))
```

（此查找不包括等效的SQL代码片段，因为相关查询的实现因不同数据库引擎而异）。

当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，字段将转换为当前时区，然后进行过滤。

#### `hour`

对于日期时间和时间字段，确切的时间匹配。 允许链接附加字段查找。 取0和23之间的整数。

例如：

```
Event.objects.filter(timestamp__hour=23)
Event.objects.filter(time__hour=5)
Event.objects.filter(timestamp__hour__gte=12)
```

SQL等效：

```
SELECT ... WHERE EXTRACT('hour' FROM timestamp) = '23';
SELECT ... WHERE EXTRACT('hour' FROM time) = '5';
SELECT ... WHERE EXTRACT('hour' FROM timestamp) >= '12';
```

（确切的SQL语法因每个数据库引擎而异）。

对于datetime字段，当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，值将过滤前转换为当前时区。

#### `minute`

对于日期时间和时间字段，确切的分钟匹配。 允许链接附加字段查找。 取0和59之间的整数。

例如：

```
Event.objects.filter(timestamp__minute=29)
Event.objects.filter(time__minute=46)
Event.objects.filter(timestamp__minute__gte=29)
```

SQL等效：

```
SELECT ... WHERE EXTRACT('minute' FROM timestamp) = '29';
SELECT ... WHERE EXTRACT('minute' FROM time) = '46';
SELECT ... WHERE EXTRACT('minute' FROM timestamp) >= '29';
```

（确切的SQL语法因每个数据库引擎而异）。

对于datetime字段，当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，值将被过滤前转换为当前时区。

#### `second`

对于日期时间和时间字段，确切的第二个匹配。 允许链接附加字段查找。 取0和59之间的整数。

例如：

```
Event.objects.filter(timestamp__second=31)
Event.objects.filter(time__second=2)
Event.objects.filter(timestamp__second__gte=31)
```

SQL等效：

```
SELECT ... WHERE EXTRACT('second' FROM timestamp) = '31';
SELECT ... WHERE EXTRACT('second' FROM time) = '2';
SELECT ... WHERE EXTRACT('second' FROM timestamp) >= '31';
```

（确切的SQL语法因每个数据库引擎而异）。

对于datetime字段，当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)为`True`时，值将过滤前转换为当前时区。

#### `isnull`

值为 `False` 或 `True`, 相当于 SQL语句`IS NULL`和`IS NOT NULL`.

例如：

```
Entry.objects.filter(pub_date__isnull=True)
```

SQL等效：

```
SELECT ... WHERE pub_date IS NULL;
```

#### `regex`

区分大小写的正则表达式匹配。

正则表达式语法是正在使用的数据库后端的语法。 在SQLite没有内置正则表达式支持的情况下，此功能由（Python）用户定义的REGEXP函数提供，因此正则表达式语法是Python的`re`模块。

例如：

```
Entry.objects.get(title__regex=r'^(An?|The) +')
```

SQL等价物：

```
SELECT ... WHERE title REGEXP BINARY '^(An?|The) +'; -- MySQL

SELECT ... WHERE REGEXP_LIKE(title, '^(An?|The) +', 'c'); -- Oracle

SELECT ... WHERE title ~ '^(An?|The) +'; -- PostgreSQL

SELECT ... WHERE title REGEXP '^(An?|The) +'; -- SQLite
```

建议使用原始字符串（例如，`r'foo'`而不是`'foo'`）来传递正则表达式语法。

#### `iregex`

不区分大小写的正则表达式匹配。

例如：

```
Entry.objects.get(title__iregex=r'^(an?|the) +')
```

SQL等价物：

```
SELECT ... WHERE title REGEXP '^(an?|the) +'; -- MySQL

SELECT ... WHERE REGEXP_LIKE(title, '^(an?|the) +', 'i'); -- Oracle

SELECT ... WHERE title ~* '^(an?|the) +'; -- PostgreSQL

SELECT ... WHERE title REGEXP '(?i)^(an?|the) +'; -- SQLite
```

### 聚合函数

Django 的`django.db.models` 模块提供以下聚合函数。 关于如何使用这些聚合函数的细节，参见[the topic guide on aggregation](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/aggregation.html)。 关于如何创建聚合函数，参数[`Aggregate`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#django.db.models.Aggregate) 的文档。

>  警告
SQLite 不能直接处理日期/时间字段的聚合。 这是因为SQLite 中没有原生的日期/时间字段，Django 目前使用文本字段模拟它的功能。 在SQLite 中对日期/时间字段使用聚合将引发`NotImplementedError`。

> 注
在`None` 为空时，聚合函数函数将返回`QuerySet`。 例如，如果`0` 中没有记录，`None` 聚合函数将返回`Sum` 而不是`QuerySet`。 `QuerySet` 是一个例外，如果`0` 为空，它将返回`Count`。

所有聚合函数具有以下共同的参数：

#### `expression`

引用模型字段的一个字符串，或者一个[query expression](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html)。

#### `output_field`

用来表示返回值的[model field](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/fields.html)，它是一个可选的参数。

>  注
在组合多个类型的字段时，只有在所有的字段都是相同类型的情况下，Django 才能确定`output_field`。 否则，你必须自己提供`output_field` 参数。

#### `**extra`

这些关键字参数可以给聚合函数生成的SQL 提供额外的信息。

#### `Avg`
```
class  Avg(expression, output_field=FloatField(), **extra)
```

返回给定表达式的平均值，它必须是数值，除非您指定不同的`output_field`。

默认的别名：`<field>__avg`

返回类型：`float`（或指定任何`output_field`的类型）

#### `Count`
```
class Count(expression, distinct=False, **extra)
```

返回与expression 相关的对象的个数。

默认的别名：`<field>__count`

返回类型：`int`有

一个可选的参数：

`distinct`

如果`distinct=True`，Count 将只计算唯一的实例。 它等同于`COUNT(DISTINCT <field>)` SQL 语句。 默认值为`False`。

#### `Max`
```
class Max(expression, output_field=None, **extra)
```

返回expression 的最大值。

默认的别名：`<field>__max`

返回类型：与输入字段的类型相同，如果提供则为 `output_field` 类型

#### `Min`
```
class Min(expression, output_field=None, **extra)
```

返回expression 的最小值。默认的别名：`<field>__min`返回类型：与输入字段的类型相同，如果提供则为 `output_field` 类型

#### `StdDev`
```
class StdDev(expression, sample=False, **extra)
```

返回expression 的标准差。

默认的别名：`<field>__stddev`

返回类型：`float`

有一个可选的参数：

`sample`

默认情况下，`StdDev` 返回群体的标准差。 但是，如果`sample=True`，返回的值将是样本的标准差。

> SQLite
> SQLite 没有直接提供`StdDev`。 有一个可用的实现是SQLite 的一个扩展模块。 参见[SQlite 的文档](https://www.sqlite.org/contrib) 中获取并安装这个扩展的指南。


#### `Sum`
```
class Sum(expression, output_field=None, **extra)
```

计算expression 的所有值的和。默认的别名：`<field>__sum`返回类型：与输入字段的类型相同，如果提供则为 `output_field` 类型

#### `Variance`
```
class  Variance(expression, sample=False, **extra)
```

返回expression 的方差。

默认的别名：`<field>__variance`

返回类型：`float`

有一个可选的参数：

`sample`

默认情况下，`Variance` 返回群体的方差。 但是，如果`sample=True`，返回的值将是样本的方差。

> SQLite
> SQLite 没有直接提供`Variance`。 有一个可用的实现是SQLite 的一个扩展模块。 参见[SQlite 的文档](https://www.sqlite.org/contrib) 中获取并安装这个扩展的指南。

## 查询相关工具

本节提供查询相关的工具的参考资料，它们其它地方没有文档。

### `Q()`对象
```
class  Q
```

`Q()` 对象和[`F`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#django.db.models.F) 对象类似，把一个SQL表达式封装在Python 对象中，这个对象可以用于数据库相关的操作。

通常，`Q()对象`使得定义查询条件然后重用成为可能。 它允许使用 `|`（`OR`）和` &`（`AND`）操作[构建复杂的数据库查询](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/queries.html#complex-lookups-with-q)；否则在特定情况下，在`QuerySets`使用不了`OR`。

### `Prefetch()` 对象
```
class  Prefetch(lookup, queryset=None, to_attr=None)
```

`Prefetch()`对象可用于控制[`prefetch_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.prefetch_related)的操作。

`lookup`参数描述了跟随的关系，并且工作方式与传递给[`prefetch_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.prefetch_related)的基于字符串的查找相同。 像这样：

```
>>> from django.db.models import Prefetch
>>> Question.objects.prefetch_related(Prefetch('choice_set')).get().choice_set.all()
<QuerySet [<Choice: Not much>, <Choice: The sky>, <Choice: Just hacking again>]>
# This will only execute two queries regardless of the number of Question
# and Choice objects.
>>> Question.objects.prefetch_related(Prefetch('choice_set')).all()
<QuerySet [<Question: What's up?>]>
```

`queryset`参数为给定的查找提供基本`QuerySet`。 这对于进一步过滤预取操作或从预取关系调用[`select_related()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.select_related)很有用，因此进一步减少查询数量：

```
>>> voted_choices = Choice.objects.filter(votes__gt=0)
>>> voted_choices
<QuerySet [<Choice: The sky>]>
>>> prefetch = Prefetch('choice_set', queryset=voted_choices)
>>> Question.objects.prefetch_related(prefetch).get().choice_set.all()
<QuerySet [<Choice: The sky>]>
```

`to_attr`参数将预取操作的结果设置为自定义属性：

```
>>> prefetch = Prefetch('choice_set', queryset=voted_choices, to_attr='voted_choices')
>>> Question.objects.prefetch_related(prefetch).get().voted_choices
<QuerySet [<Choice: The sky>]>
>>> Question.objects.prefetch_related(prefetch).get().choice_set.all()
<QuerySet [<Choice: Not much>, <Choice: The sky>, <Choice: Just hacking again>]>
```

> 注
当使用`to_attr`时，预取的结果存储在列表中。 这可以提供比存储在`prefetch_related`实例内的缓存结果的传统`QuerySet`调用显着的速度改进。

### `prefetch_related_objects()`

`prefetch_related_objects(model_instances, *related_lookups)`

在可迭代的模型实例上预取给定的查找。 这在接收与`QuerySet`相反的模型实例列表的代码中非常有用；例如，从缓存中获取模型或手动实例化模型。

传递一个可重复的模型实例（必须都是同一个类）和要预取的查找或[`Prefetch`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.Prefetch)对象。 像这样：

```
>>> from django.db.models import prefetch_related_objects
>>> restaurants = fetch_top_restaurants_from_cache()  # A list of Restaurants
>>> prefetch_related_objects(restaurants, 'pizzas__toppings')
```

### `FileteredRelation()`对象
django2.0新增
```
class FilteredRelation(relation_name, *, condition=Q())
```
`relation_name`

您要在其中过滤关系的字段名称。

`condition`

一个Q对象来控制过滤。

执行JOIN时，FilteredRelation与annotate（）一起使用以创建ON子句。它不会对默认关系起作用，而是对注释名称起作用（在下面的示例中为pizzas_vegetarian）。

例如，要查找名称为“ mozzarella”的素食比萨的餐厅：
```shell
>>> from django.db.models import FilteredRelation, Q
>>> Restaurant.objects.annotate(
...    pizzas_vegetarian=FilteredRelation(
...        'pizzas', condition=Q(pizzas__vegetarian=True),
...    ),
... ).filter(pizzas_vegetarian__name__icontains='mozzarella')
```
如果有大量的比萨饼，则此查询集的性能优于：
```shell
>>> Restaurant.objects.filter(
...     pizzas__vegetarian=True,
...     pizzas__name__icontains='mozzarella',
... )
```
因为第一个查询集的WHERE子句中的过滤将仅对素食比萨饼有效。

FilteredRelation不支持：

- 跨越关系字段的条件。例如：

```shell
>>> Restaurant.objects.annotate(
...    pizzas_with_toppings_startswith_n=FilteredRelation(
...        'pizzas__toppings',
...        condition=Q(pizzas__toppings__name__startswith='n'),
...    ),
... )
Traceback (most recent call last):
...
ValueError: FilteredRelation's condition doesn't support nested relations (got 'pizzas__toppings__name__startswith').
```

- `QuerySet.only()`和`prefetch_related()`。
- 从父模型继承的GenericForeignKey。