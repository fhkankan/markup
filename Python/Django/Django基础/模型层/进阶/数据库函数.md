# 数据库函数

下面记述的类为用户提供了一些方法，来在Django中使用底层数据库提供的函数用于注解、聚合或者过滤器等操作。 函数也是[expressions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html)，所以可以像[aggregate functions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#aggregation-functions)一样混合使用它们。

我们会在每个函数的实例中使用下面的模型：

```
class Author(models.Model):
    name = models.CharField(max_length=50)
    age = models.PositiveIntegerField(null=True, blank=True)
    alias = models.CharField(max_length=50, null=True, blank=True)
    goes_by = models.CharField(max_length=50, null=True, blank=True)
```

我们并不推荐在`Coalesce`上允许`CharField`，因为这会允许字段有两个“空值”，但是对于下面的`null=True`示例来说它很重要。



## `Cast`

- *class* `Cast`(*expression*, *output_field*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Cast)

  

**Django中的新功能1.10。**

强制`expression`的结果类型为`output_field`。

使用范例：

```
>>> from django.db.models import FloatField
>>> from django.db.models.functions import Cast
>>> Value.objects.create(integer=4)
>>> value = Value.objects.annotate(as_float=Cast('integer', FloatField())).get()
>>> print(value.as_float)
4.0
```



## `Coalesce`

- *class* `Coalesce`(**expressions*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Coalesce)

  

接受一个含有至少两个字段名称或表达式的列表，返回第一个非空的值（注意空字符串不被认为是一个空值）。 每个参与都必须是相似的类型，所以掺杂了文本和数字的列表会导致数据库错误。

使用范例：

```
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

警告

传递给MySQL上的`Coalesce`的Python值可能会转换为不正确的类型，除非显式转换为正确的数据库类型：

```
>>> from django.db.models import DateTimeField
>>> from django.db.models.functions import Cast, Coalesce
>>> from django.utils import timezone
>>> now = timezone.now()
>>> Coalesce('updated', Cast(now, DateTimeField()))
```



## `Concat`

- *class* `Concat`(**expressions*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Concat)

  

接受一个含有至少两个文本字段的或表达式的列表，返回连接后的文本。 每个参数都必须是文本或者字符类型。 如果你想把一个`output_field`和一个`CharField()`连接， 一定要告诉Django`TextField()`应该为`TextField()`类型。 当连接`Value`时，也需要指定`output_field`，如下例所示。

这个函数不会返回null。 在后端中，如果一个null参数导致了整个表达式都是null，Django会确保把每个null的部分转换成一个空字符串。

使用范例：

```
>>> # Get the display name as "name (goes_by)"
>>> from django.db.models import CharField, Value as V
>>> from django.db.models.functions import Concat
>>> Author.objects.create(name='Margaret Smith', goes_by='Maggie')
>>> author = Author.objects.annotate(
...     screen_name=Concat(
...         'name', V(' ('), 'goes_by', V(')'),
...         output_field=CharField()
...     )
... ).get()
>>> print(author.screen_name)
Margaret Smith (Maggie)
```



## `Greatest`

- *class* `Greatest`(**expressions*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Greatest)

  

接受至少两个字段名称或表达式的列表，并返回最大值。 每个参与都必须是相似的类型，所以掺杂了文本和数字的列表会导致数据库错误。

使用范例：

```
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

`annotated_comment.last_updated`将是最新的`blog.modified`和`comment.modified`。

警告

当一个或多个表达式可能为`null`时，`Greatest`的行为在数据库之间有所不同：

- PostgreSQL: `Greatest` will return the largest non-null expression, or `null` if all expressions are `null`.
- SQLite，Oracle和MySQL：如果任何表达式为`null`，`Greatest`将返回`null`。

如果您知道一个明智的最小值作为默认值，PostgreSQL行为可以使用`Coalesce`进行仿真。



## `Least`

- *class* `Least`(**expressions*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Least)

  

接受至少两个字段名称或表达式的列表，并返回最小值。 每个参与都必须是相似的类型，所以掺杂了文本和数字的列表会导致数据库错误。

警告

当一个或多个表达式可能为`null`时，`Least`的行为在数据库之间有所不同：

- PostgreSQL: `Least` will return the smallest non-null expression, or `null` if all expressions are `null`.
- SQLite，Oracle和MySQL：如果任何表达式为`null`，`Least`将返回`null`。

如果您知道一个明智的最大值作为默认值，PostgreSQL行为可以使用`Coalesce`进行仿真。



## `Length`

- *class* `Length`(*expression*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Length)

  

接受一个文本字段或表达式，返回值的字符个数。 如果表达式是null，长度也会是null。

使用范例：

```
>>> # Get the length of the name and goes_by fields
>>> from django.db.models.functions import Length
>>> Author.objects.create(name='Margaret Smith')
>>> author = Author.objects.annotate(
...    name_length=Length('name'),
...    goes_by_length=Length('goes_by')).get()
>>> print(author.name_length, author.goes_by_length)
(14, None)
```

它也可以注册为转换。 像这样：

```
>>> from django.db.models import CharField
>>> from django.db.models.functions import Length
>>> CharField.register_lookup(Length, 'length')
>>> # Get authors whose name is longer than 7 characters
>>> authors = Author.objects.filter(name__length__gt=7)
```



## `Lower`

- *class* `Lower`(*expression*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Lower)

  

接受一个文本字符串或表达式，返回它的小写表示形式。

它还可以注册为[`Length`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#django.db.models.functions.Length)中所述的转换。

使用范例：

```
>>> from django.db.models.functions import Lower
>>> Author.objects.create(name='Margaret Smith')
>>> author = Author.objects.annotate(name_lower=Lower('name')).get()
>>> print(author.name_lower)
margaret smith
```



## `Now`

- *class* `Now`[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Now)

  

返回数据库服务器执行查询的当前日期和时间，通常使用SQL `CURRENT_TIMESTAMP`。

使用范例：

```
>>> from django.db.models.functions import Now
>>> Article.objects.filter(published__lte=Now())
<QuerySet [<Article: How to Django>]>
```

PostgreSQL注意事项

在PostgreSQL上，SQL `CURRENT_TIMESTAMP`返回当前事务开始的时间。 因此，对于跨数据库兼容性，`Now()`替代使用`STATEMENT_TIMESTAMP`。 如果您需要交易时间戳，请使用[`django.contrib.postgres.functions.TransactionNow`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/contrib/postgres/functions.html#django.contrib.postgres.functions.TransactionNow)。



## `Substr`

- *class* `Substr`(*expression*, *pos*, *length=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Substr)

  

返回这个字段或者表达式的，以`length`位置开始，长度为`pos`的子字符串。 位置从下标为1开始，所以必须大于0。 如果`length`是`None`，会返回剩余的字符串。

使用范例：

```
>>> # Set the alias to the first 5 characters of the name as lowercase
>>> from django.db.models.functions import Substr, Lower
>>> Author.objects.create(name='Margaret Smith')
>>> Author.objects.update(alias=Lower(Substr('name', 1, 5)))
1
>>> print(Author.objects.get(name='Margaret Smith').alias)
marga
```



## `Upper`

- *class* `Upper`(*expression*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/base.html#Upper)

  

接受一个文本字符串或表达式，返回它的大写表示形式。

它还可以注册为[`Length`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#django.db.models.functions.Length)中所述的转换。

使用范例：

```
>>> from django.db.models.functions import Upper
>>> Author.objects.create(name='Margaret Smith')
>>> author = Author.objects.annotate(name_upper=Upper('name')).get()
>>> print(author.name_upper)
MARGARET SMITH
```



## 日期函数

**Django中的新功能1.10。**

我们会在每个函数的实例中使用下面的模型：

```
class Experiment(models.Model):
    start_datetime = models.DateTimeField()
    start_date = models.DateField(null=True, blank=True)
    start_time = models.TimeField(null=True, blank=True)
    end_datetime = models.DateTimeField(null=True, blank=True)
    end_date = models.DateField(null=True, blank=True)
    end_time = models.TimeField(null=True, blank=True)
```



### `Extract`

- *class* `Extract`(*expression*, *lookup_name=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#Extract)

  

将日期的组件提取为数字。

Takes an `expression` representing a `DateField` or `DateTimeField` and a `lookup_name`, and returns the part of the date referenced by `lookup_name` as an `IntegerField`. Django通常使用数据库的提取函数，因此您可以使用数据库支持的任何`lookup_name`。 可以传递通常由`pytz`提供的`tzinfo`子类，以提取特定时区中的值。

给定datetime `2015-06-15 23：30：01.000321 + 00：00`，内置`lookup_name`

- “年”：2015年
- “月”：6
- “天”：15
- “周”：25
- “week_day”：2
- “小时”：23
- “分钟”：30
- “第二”：1

如果不同的时区如`Australia/Melbourne`在Django中处于活动状态，则在提取值之前将datetime转换为时区。 上述示例中墨尔本的时区抵消是+10：00。 当该时区处于活动状态时返回的值将与上述相同，除了：

- “天”：16
- “week_day”：3
- “小时”：9

`week_day`值

计算与大多数数据库和Python标准函数不同的`week_day` `lookup_type`。 This function will return `1` for Sunday, `2` for Monday, through `7` for Saturday.

Python中的等效计算是：

```
>>> from datetime import datetime
>>> dt = datetime(2015, 6, 15)
>>> (dt.isoweekday() % 7) + 1
2
```

`week`值

`week` `lookup_type`基于[ISO-8601](https://en.wikipedia.org/wiki/ISO-8601)计算，即一个星期从星期一开始。 第一个星期是大部分时间，即星期四或之前开始的一周。 返回值在1到52或53之间。

以上每个`lookup_name`具有相应的`Extract`子类（下面列出），通常应该使用，而不是更冗长的等价物。使用`ExtractYear(...)`而不是`Extract（...， lookup_name ='year'）`。

使用范例：

```
>>> from datetime import datetime
>>> from django.db.models.functions import Extract
>>> start = datetime(2015, 6, 15)
>>> end = datetime(2015, 7, 2)
>>> Experiment.objects.create(
...    start_datetime=start, start_date=start.date(),
...    end_datetime=end, end_date=end.date())
>>> # Add the experiment start year as a field in the QuerySet.
>>> experiment = Experiment.objects.annotate(
...    start_year=Extract('start_datetime', 'year')).get()
>>> experiment.start_year
2015
>>> # How many experiments completed in the same year in which they started?
>>> Experiment.objects.filter(
...    start_datetime__year=Extract('end_datetime', 'year')).count()
1
```



#### `DateField`提取

- *class* `ExtractYear`(*expression*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#ExtractYear)

  `lookup_name = 'year'`

- *class* `ExtractMonth`(*expression*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#ExtractMonth)

  `lookup_name ='month'`

- *class* `ExtractDay`(*expression*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#ExtractDay)

  `lookup_name ='day'`

- *class* `ExtractWeekDay`(*expression*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#ExtractWeekDay)

  `lookup_name ='week_day'`

- *class* `ExtractWeek`(*expression*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#ExtractWeek)

  **Django中的新功能1.11。**`lookup_name ='week'`

这些在逻辑上等同于`Extract（'date_field'， lookup_name）`。 每个类也是在`DateField`和`DateTimeField`上注册为`__(lookup_name)`的`Transform`，例如。 `__year`

由于`DateField`没有时间分量，只能处理日期部分的`Extract`子类可以与`DateField`一起使用：

```
>>> from datetime import datetime
>>> from django.utils import timezone
>>> from django.db.models.functions import (
...     ExtractDay, ExtractMonth, ExtractWeek, ExtractWeekDay, ExtractYear,
... )
>>> start_2015 = datetime(2015, 6, 15, 23, 30, 1, tzinfo=timezone.utc)
>>> end_2015 = datetime(2015, 6, 16, 13, 11, 27, tzinfo=timezone.utc)
>>> Experiment.objects.create(
...    start_datetime=start_2015, start_date=start_2015.date(),
...    end_datetime=end_2015, end_date=end_2015.date())
>>> Experiment.objects.annotate(
...     year=ExtractYear('start_date'),
...     month=ExtractMonth('start_date'),
...     week=ExtractWeek('start_date'),
...     day=ExtractDay('start_date'),
...     weekday=ExtractWeekDay('start_date'),
... ).values('year', 'month', 'week', 'day', 'weekday').get(
...     end_date__year=ExtractYear('start_date'),
... )
{'year': 2015, 'month': 6, 'week': 25, 'day': 15, 'weekday': 2}
```



#### `DateTimeField`提取

除了以下内容之外，上述列出的`DateField`的所有提取也可以在`DateTimeField`上使用。

- *class* `ExtractHour`(*expression*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#ExtractHour)

  `lookup_name ='hour'`

- *class* `ExtractMinute`(*expression*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#ExtractMinute)

  `lookup_name ='分钟'`

- *class* `ExtractSecond`(*expression*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#ExtractSecond)

  `lookup_name ='second'`

这些在逻辑上等同于`Extract（'datetime_field'， lookup_name）`。 每个类也是在`DateTimeField`上注册为`__(lookup_name)`的`Transform`，例如。 `__minute`

`DateTimeField`示例：

```
>>> from datetime import datetime
>>> from django.utils import timezone
>>> from django.db.models.functions import (
...     ExtractDay, ExtractHour, ExtractMinute, ExtractMonth, ExtractSecond,
...     ExtractWeek, ExtractWeekDay, ExtractYear,
... )
>>> start_2015 = datetime(2015, 6, 15, 23, 30, 1, tzinfo=timezone.utc)
>>> end_2015 = datetime(2015, 6, 16, 13, 11, 27, tzinfo=timezone.utc)
>>> Experiment.objects.create(
...    start_datetime=start_2015, start_date=start_2015.date(),
...    end_datetime=end_2015, end_date=end_2015.date())
>>> Experiment.objects.annotate(
...     year=ExtractYear('start_datetime'),
...     month=ExtractMonth('start_datetime'),
...     week=ExtractWeek('start_datetime'),
...     day=ExtractDay('start_datetime'),
...     weekday=ExtractWeekDay('start_datetime'),
...     hour=ExtractHour('start_datetime'),
...     minute=ExtractMinute('start_datetime'),
...     second=ExtractSecond('start_datetime'),
... ).values(
...     'year', 'month', 'week', 'day', 'weekday', 'hour', 'minute', 'second',
... ).get(end_datetime__year=ExtractYear('start_datetime'))
{'year': 2015, 'month': 6, 'week': 25, 'day': 15, 'weekday': 2, 'hour': 23,
 'minute': 30, 'second': 1}
```

当[`USE_TZ`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/settings.html#std:setting-USE_TZ)是`True`时，数据库将以UTC存储在数据库中。 如果Django中的时区不同，则在提取值之前，将datetime转换为该时区。 以下示例转换为墨尔本时区（UTC +10：00），这会更改返回的日，工作日和小时值：

```
>>> import pytz
>>> melb = pytz.timezone('Australia/Melbourne')  # UTC+10:00
>>> with timezone.override(melb):
...    Experiment.objects.annotate(
...        day=ExtractDay('start_datetime'),
...        weekday=ExtractWeekDay('start_datetime'),
...        hour=ExtractHour('start_datetime'),
...    ).values('day', 'weekday', 'hour').get(
...        end_datetime__year=ExtractYear('start_datetime'),
...    )
{'day': 16, 'weekday': 3, 'hour': 9}
```

将时区显式传递给`Extract`函数的行为方式相同，并且优先于活动时区：

```
>>> import pytz
>>> melb = pytz.timezone('Australia/Melbourne')
>>> Experiment.objects.annotate(
...     day=ExtractDay('start_datetime', tzinfo=melb),
...     weekday=ExtractWeekDay('start_datetime', tzinfo=melb),
...     hour=ExtractHour('start_datetime', tzinfo=melb),
... ).values('day', 'weekday', 'hour').get(
...     end_datetime__year=ExtractYear('start_datetime'),
... )
{'day': 16, 'weekday': 3, 'hour': 9}
```



### `Trunc`

- *class* `Trunc`(*expression*, *kind*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#Trunc)

  

截断一个日期到一个重要的组件。

When you only care if something happened in a particular year, hour, or day, but not the exact second, then `Trunc` (and its subclasses) can be useful to filter or aggregate your data. 例如，您可以使用`Trunc`来计算每天的销售数量。

`Trunc`采用单个`expression`，表示`DateField`，`TimeField`或`DateTimeField`表示日期或时间部分的`kind`和`output_field`，它们是`DateTimeField()`，`TimeField()`或`DateField()`。 它将根据`output_field`返回日期时间，日期或时间，其中最多`kind`设置为最小值。 如果`output_field`被省略，它将默认为`expression`的`output_field`。 通常由`pytz`提供的`tzinfo`子类可以传递到截断特定时区中的值。

给定日期时间`2015-06-15 14：30：50.000321 + 00：00`，内置`kind`

- “年”：2015-01-01 00：00：00 + 00：00
- “月”：2015-06-01 00：00：00 + 00：00
- “天”：2015-06-15 00：00：00 + 00：00
- “小时”：2015-06-15 14：00：00 + 00：00
- “分钟”：2015-06-15 14：30：00 + 00：00
- “第二”：2015-06-15 14：30：50 + 00：00

如果像`Australia/Melbourne`之间的不同时区在Django中处于活动状态，那么在该值被截断之前，datetime将被转换为新的时区。 上述示例中墨尔本的时区抵消是+10：00。 当此时区处于活动状态时返回的值将为：

- “年”：2015-01-01 00：00：00 + 11：00
- “月”：2015-06-01 00：00：00 + 10：00
- “天”：2015-06-16 00：00：00 + 10：00
- “小时”：2015-06-16 00：00：00 + 10：00
- “分钟”：2015-06-16 00：30：00 + 10：00
- “第二”：2015-06-16 00：30：50 + 10：00

该年度的偏移量为+11：00，因为结果已转换为夏令时。

上述每个`kind`具有相应的`Trunc`子类（下面列出），通常应该使用而不是更冗长的等价物。使用`TruncYear(...)`而不是`Trunc（...， kind ='year'）`。

子类都被定义为转换，但是它们没有注册到任何字段，因为明显的查找名称已经由`Extract`子类保留。

使用范例：

```
>>> from datetime import datetime
>>> from django.db.models import Count, DateTimeField
>>> from django.db.models.functions import Trunc
>>> Experiment.objects.create(start_datetime=datetime(2015, 6, 15, 14, 30, 50, 321))
>>> Experiment.objects.create(start_datetime=datetime(2015, 6, 15, 14, 40, 2, 123))
>>> Experiment.objects.create(start_datetime=datetime(2015, 12, 25, 10, 5, 27, 999))
>>> experiments_per_day = Experiment.objects.annotate(
...    start_day=Trunc('start_datetime', 'day', output_field=DateTimeField())
... ).values('start_day').annotate(experiments=Count('id'))
>>> for exp in experiments_per_day:
...     print(exp['start_day'], exp['experiments'])
...
2015-06-15 00:00:00 2
2015-12-25 00:00:00 1
>>> experiments = Experiment.objects.annotate(
...    start_day=Trunc('start_datetime', 'day', output_field=DateTimeField())
... ).filter(start_day=datetime(2015, 6, 15))
>>> for exp in experiments:
...     print(exp.start_datetime)
...
2015-06-15 14:30:50.000321
2015-06-15 14:40:02.000123
```



#### `DateField` truncation 

- *class* `TruncYear`(*expression*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncYear)

  `kind ='年'`

- *class* `TruncMonth`(*expression*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncMonth)

  `kind ='month'`

这些在逻辑上等同于`Trunc（'date_field'， 种类）`。 它们截断日期的所有部分，直到`kind`，允许以较少精度进行分组或过滤日期。 `expression`可以具有`DateField`或`DateTimeField`的`output_field`。

由于`DateField`没有时间分量，只能处理日期部分的`Trunc`子类可以与`DateField`一起使用：

```
>>> from datetime import datetime
>>> from django.db.models import Count
>>> from django.db.models.functions import TruncMonth, TruncYear
>>> from django.utils import timezone
>>> start1 = datetime(2014, 6, 15, 14, 30, 50, 321, tzinfo=timezone.utc)
>>> start2 = datetime(2015, 6, 15, 14, 40, 2, 123, tzinfo=timezone.utc)
>>> start3 = datetime(2015, 12, 31, 17, 5, 27, 999, tzinfo=timezone.utc)
>>> Experiment.objects.create(start_datetime=start1, start_date=start1.date())
>>> Experiment.objects.create(start_datetime=start2, start_date=start2.date())
>>> Experiment.objects.create(start_datetime=start3, start_date=start3.date())
>>> experiments_per_year = Experiment.objects.annotate(
...    year=TruncYear('start_date')).values('year').annotate(
...    experiments=Count('id'))
>>> for exp in experiments_per_year:
...     print(exp['year'], exp['experiments'])
...
2014-01-01 1
2015-01-01 2

>>> import pytz
>>> melb = pytz.timezone('Australia/Melbourne')
>>> experiments_per_month = Experiment.objects.annotate(
...    month=TruncMonth('start_datetime', tzinfo=melb)).values('month').annotate(
...    experiments=Count('id'))
>>> for exp in experiments_per_month:
...     print(exp['month'], exp['experiments'])
...
2015-06-01 00:00:00+10:00 1
2016-01-01 00:00:00+11:00 1
2014-06-01 00:00:00+10:00 1
```



#### `TimeField` truncation 

**Django中的新功能1.11。**

- *class* `TruncHour`(*expression*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncHour)

  `kind ='hour'`

- *class* `TruncMinute`(*expression*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncMinute)

  `kind ='分钟'`

- *class* `TruncSecond`(*expression*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncSecond)

  `kind ='second'`

这些在逻辑上等同于`Trunc（'time_field'， 种类）`。 它们将时间的所有部分截断到`kind`，这样可以使精度更低的分组或过滤时间。 `expression`可以具有`TimeField`或`DateTimeField`的`output_field`。

由于`TimeField`没有日期组件，只能处理时间部分的`Trunc`子类可以与`TimeField`一起使用：

```
>>> from datetime import datetime
>>> from django.db.models import Count, TimeField
>>> from django.db.models.functions import TruncHour
>>> from django.utils import timezone
>>> start1 = datetime(2014, 6, 15, 14, 30, 50, 321, tzinfo=timezone.utc)
>>> start2 = datetime(2014, 6, 15, 14, 40, 2, 123, tzinfo=timezone.utc)
>>> start3 = datetime(2015, 12, 31, 17, 5, 27, 999, tzinfo=timezone.utc)
>>> Experiment.objects.create(start_datetime=start1, start_time=start1.time())
>>> Experiment.objects.create(start_datetime=start2, start_time=start2.time())
>>> Experiment.objects.create(start_datetime=start3, start_time=start3.time())
>>> experiments_per_hour = Experiment.objects.annotate(
...    hour=TruncHour('start_datetime', output_field=TimeField()),
... ).values('hour').annotate(experiments=Count('id'))
>>> for exp in experiments_per_hour:
...     print(exp['hour'], exp['experiments'])
...
14:00:00 2
17:00:00 1

>>> import pytz
>>> melb = pytz.timezone('Australia/Melbourne')
>>> experiments_per_hour = Experiment.objects.annotate(
...    hour=TruncHour('start_datetime', tzinfo=melb),
... ).values('hour').annotate(experiments=Count('id'))
>>> for exp in experiments_per_hour:
...     print(exp['hour'], exp['experiments'])
...
2014-06-16 00:00:00+10:00 2
2016-01-01 04:00:00+11:00 1
```



#### `DateTimeField` truncation 

- *class* `TruncDate`(*expression*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncDate)

  `lookup_name ='date'``output_field = DateField()`

`TruncDate`将`expression`转换为日期，而不是使用内置的SQL truncate函数。 它也被注册为`DateTimeField`作为`__date`的转换。

- *class* `TruncTime`(*expression*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncTime)

  

**Django中的新功能1.11：**

- `lookup_name ='time'`

  

- `output_field = TimeField()`

  

`TruncTime`将`expression`转换为一段时间，而不是使用内置的SQL truncate函数。 它也被注册为`DateTimeField`作为`__time`的转换。

- *class* `TruncDay`(*expression*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncDay)

  `kind ='day'`

- *class* `TruncHour`(*expression*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncHour)

  `kind ='hour'`

- *class* `TruncMinute`(*expression*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncMinute)

  `kind ='分钟'`

- *class* `TruncSecond`(*expression*, *output_field=None*, *tzinfo=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/functions/datetime.html#TruncSecond)

  `kind ='second'`

这些在逻辑上等同于`Trunc（'datetime_field'， 种类）`。 它们截断日期的所有部分，直到`kind`，并允许以更少的精度对数据集进行分组或过滤。 `expression`必须有`DateTimeField`的`output_field`。

使用范例：

```
>>> from datetime import date, datetime
>>> from django.db.models import Count
>>> from django.db.models.functions import (
...     TruncDate, TruncDay, TruncHour, TruncMinute, TruncSecond,
... )
>>> from django.utils import timezone
>>> import pytz
>>> start1 = datetime(2014, 6, 15, 14, 30, 50, 321, tzinfo=timezone.utc)
>>> Experiment.objects.create(start_datetime=start1, start_date=start1.date())
>>> melb = pytz.timezone('Australia/Melbourne')
>>> Experiment.objects.annotate(
...     date=TruncDate('start_datetime'),
...     day=TruncDay('start_datetime', tzinfo=melb),
...     hour=TruncHour('start_datetime', tzinfo=melb),
...     minute=TruncMinute('start_datetime'),
...     second=TruncSecond('start_datetime'),
... ).values('date', 'day', 'hour', 'minute', 'second').get()
{'date': datetime.date(2014, 6, 15),
 'day': datetime.datetime(2014, 6, 16, 0, 0, tzinfo=<DstTzInfo 'Australia/Melbourne' AEST+10:00:00 STD>),
 'hour': datetime.datetime(2014, 6, 16, 0, 0, tzinfo=<DstTzInfo 'Australia/Melbourne' AEST+10:00:00 STD>),
 'minute': 'minute': datetime.datetime(2014, 6, 15, 14, 30, tzinfo=<UTC>),
 'second': datetime.datetime(2014, 6, 15, 14, 30, 50, tzinfo=<UTC>)
}
```

### [目录](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/contents.html)

- 数据库函数
  - [`投`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#cast)
  - [`Coalesce`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#coalesce)
  - [`Concat`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#concat)
  - [`最大`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#greatest)
  - [`最小`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#least)
  - [`Length`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#length)
  - [`Lower`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#lower)
  - [`现在`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#now)
  - [`Substr`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#substr)
  - [`Upper`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#upper)
  - 日期功能
    - `Extract`
      - [`DateField`提取](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#datefield-extracts)
      - [`DateTimeField`提取](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#datetimefield-extracts)
    - `Trunc`
      - [`DateField`截断](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#datefield-truncation)
      - [`TimeField`截断](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#timefield-truncation)
      - [`DateTimeField`截断](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#datetimefield-truncation)

### 浏览

- 上一个：[条件表达式](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/conditional-expressions.html)
- 下一页：[请求和响应对象](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/request-response.html)

### 你在这里：

- Django 1.11.6 文档
  - API参考
    - 楷模
      - 数据库函数

### 这一页

- [显示源](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_sources/ref/models/database-functions.txt)

### 快速搜索





### 最后更新：

2017年9月6日