# 模型-高级

## 管理器



## 原始SQL



## 聚合



## 自定义查找



## 查询表达式



## 条件表达式



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

