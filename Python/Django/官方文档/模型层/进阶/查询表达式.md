# 查询表达式

查询表达式描述可用作更新，创建，过滤，排序，注释或聚合的一部分的值或计算。 这里（文档中）有很多内置表达式可以帮助你完成自己的查询。 表达式可以组合，甚至是嵌套，来完成更加复杂的计算



## 支持的算术

Django支持在查询表达式使用加减乘除，求模，幂运算，Python常量，变量甚至是其它表达式。



## 一些例子

```
from django.db.models import F, Count, Value
from django.db.models.functions import Length, Upper

# Find companies that have more employees than chairs.
Company.objects.filter(num_employees__gt=F('num_chairs'))

# Find companies that have at least twice as many employees
# as chairs. Both the querysets below are equivalent.
Company.objects.filter(num_employees__gt=F('num_chairs') * 2)
Company.objects.filter(
    num_employees__gt=F('num_chairs') + F('num_chairs'))

# How many chairs are needed for each company to seat all employees?
>>> company = Company.objects.filter(
...    num_employees__gt=F('num_chairs')).annotate(
...    chairs_needed=F('num_employees') - F('num_chairs')).first()
>>> company.num_employees
120
>>> company.num_chairs
50
>>> company.chairs_needed
70

# Create a new company using expressions.
>>> company = Company.objects.create(name='Google', ticker=Upper(Value('goog')))
# Be sure to refresh it if you need to access the field.
>>> company.refresh_from_db()
>>> company.ticker
'GOOG'

# Annotate models with an aggregated value. Both forms
# below are equivalent.
Company.objects.annotate(num_products=Count('products'))
Company.objects.annotate(num_products=Count(F('products')))

# Aggregates can contain complex computations also
Company.objects.annotate(num_offerings=Count(F('products') + F('services')))

# Expressions can also be used in order_by()
Company.objects.order_by(Length('name').asc())
Company.objects.order_by(Length('name').desc())
```



## 内置表达式

注

这些表达式定义在`django.db.models.aggregates` 和 `django.db.models.expressions`中, 但为了方便，通常可以直接从[`django.db.models`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/models.html#module-django.db.models)导入.



### `F()`表达式

- *class* `F`[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#F)

  

一个 `F()`对象代表了一个model的字段值或注释列。 使用它就可以直接参考model的field和执行数据库操作而不用再把它们（model field）查询出来放到python内存中。

相反，Django使用`F()`对象生成描述数据库级所需操作的SQL表达式。

这些通过一个例子可以很容易的理解。 往常，我们会这样做：

```
# Tintin filed a news story!
reporter = Reporters.objects.get(name='Tintin')
reporter.stories_filed += 1
reporter.save()
```

这里呢，我们把 `reporter.stories_filed` 的值从数据库取出放到内存中并用我们熟悉的python运算符操作它，最后再把它保存到数据库。 然而，我们还可以这样做：

```
from django.db.models import F

reporter = Reporters.objects.get(name='Tintin')
reporter.stories_filed = F('stories_filed') + 1
reporter.save()
```

虽然`reporter.stories_filed = F('stories_filed') + 1`看起来像一个正常的Python分配值赋给一个实例属性，事实上这是一个描述数据库操作的SQL概念

当Django遇到一个`F()`的实例时，它会覆盖标准的Python运算符来创建封装的SQL表达式；在这种情况下，指示数据库增加由`reporter.stories_filed`表示的数据库字段。

无论`reporter.stories_filed`的值是或曾是什么，Python一无所知--这完全是由数据库去处理的。 所有的Python，通过Django的`F()` 类，只是去创建SQL语法参考字段和描述操作。

要访问以这种方式保存的新值，必须重新加载该对象：

```
reporter = Reporters.objects.get(pk=reporter.pk)
# Or, more succinctly:
reporter.refresh_from_db()
```

除了在上述单个实例的操作中使用，`F()`可以在对象实例的`QuerySets`上使用，通过`update()` 。 这减少了我们上面使用的两个查询 - `get()`和[`save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save) - 只有一个：

```
reporter = Reporters.objects.filter(name='Tintin')
reporter.update(stories_filed=F('stories_filed') + 1)
```

我们可以使用[`update()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.update)方法批量地增加多个对象的字段值。这比先从数据库查询后，通过循环一个个增加，并一个个保存要快的很多。

```
Reporter.objects.all().update(stories_filed=F('stories_filed') + 1)
```

`F()`表达式的效率上的优点主要体现在

- 直接通过数据库操作而不是Python
- 减少数据库查询次数



#### 使用`F()` 避免竞争条件

使用 `F()` 的另一个好处是通过数据库而不是Python来更新字段值以避免*竞争条件*.

如果两个Python线程执行上面第一个例子中的代码，一个线程可能在另一个线程刚从数据库中获取完字段值后获取、增加、保存该字段值。 第二个线程保存的值将基于原始值；第一个线程的工作将会丢失。

如果让数据库对更新字段负责，这个过程将变得更稳健：它将只会在 [`save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save) 或 `update()`执行时根据数据库中该字段值来更新字段，而不是基于实例之前获取的字段值。



#### `F()`分配在`Model.save()` 

保存模型实例后，分配给模型字段的对象的`F()`将保留，并将应用于每个[`save()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/instances.html#django.db.models.Model.save)。 像这样：

```
reporter = Reporters.objects.get(name='Tintin')
reporter.stories_filed = F('stories_filed') + 1
reporter.save()

reporter.name = 'Tintin Jr.'
reporter.save()
```

在这种情况下，`stories_filed`将被更新两次。 如果最初是`1`，最终值将为`3`。



#### 在filter中使用`F()`

`F()`在`QuerySet` 过滤器中也十分有用，它使得使用条件通过字段值而不是Python值过滤一组对象变得可能。

这在 [using F() expressions in queries](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/queries.html#using-f-expressions-in-filters) 中被记录。



#### 使用`F()`注释

`F()`可用于通过将不同字段与算术相结合来在模型上创建动态字段：

```
company = Company.objects.annotate(
    chairs_needed=F('num_employees') - F('num_chairs'))
```

如果你组合的字段是不同类型，你需要告诉Django将返回什么类型的字段。 由于`output_field`不直接支持`F()`，您需要使用[`ExpressionWrapper`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#django.db.models.ExpressionWrapper)

```
from django.db.models import DateTimeField, ExpressionWrapper, F

Ticket.objects.annotate(
    expires=ExpressionWrapper(
        F('active_at') + F('duration'), output_field=DateTimeField()))
```

当引用诸如`ForeignKey`的关系字段时，`F()`返回主键值而不是模型实例：

```
>> car = Company.objects.annotate(built_by=F('manufacturer'))[0]
>> car.manufacturer
<Manufacturer: Toyota>
>> car.built_by
3
```



### `Func()`表达式

`Func()` 表达式是所有涉及数据库函数的表达式的基类，例如 `COALESCE` 和 `LOWER`, 或者 `SUM`聚合. 用下面方式可以直接使用:

```
from django.db.models import Func, F

queryset.annotate(field_lower=Func(F('field'), function='LOWER'))
```

或者它们可以用于构建数据库函数库：

```
class Lower(Func):
    function = 'LOWER'

queryset.annotate(field_lower=Lower('field'))
```

但是这两种情况都将导致查询集，其中每个模型用从以下SQL大致生成的额外属性`field_lower`注释：

```
SELECT
    ...
    LOWER("db_table"."field") as "field_lower"
```

有关内置数据库函数的列表，请参见[Database Functions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html)。

`Func` API如下：

- *class* `Func`(**expressions*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#Func)

  `function`描述将生成的函数的类属性。 具体来说，`function`将被插入为[`template`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#django.db.models.Func.template)中的`function`占位符。 默认为`None`。`template`类属性，作为格式字符串，描述为此函数生成的SQL。 默认为`'%(function)s(%(expressions)s)'`。If you’re constructing SQL like `strftime('%W', 'date')` and need a literal `%` character in the query, quadruple it (`%%%%`) in the `template` attribute because the string is interpolated twice: once during the template interpolation in `as_sql()` and once in the SQL interpolation with the query parameters in the database cursor.`arg_joiner`类属性，表示用于连接`expressions`列表的字符。 默认为`'， '`。`arity`**Django中的新功能1.10。**一个类属性，表示函数接受的参数数。 如果设置此属性并且使用不同数目的表达式调用该函数，则将引发`TypeError`。 默认为`None`。`as_sql`(*compiler*, *connection*, *function=None*, *template=None*, *arg_joiner=None*, ***extra_context*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#Func.as_sql)生成数据库功能的SQL。`as_vendor()`方法应该使用`function`，`template`，`arg_joiner`和任何其他`**extra_context`参数以根据需要自定义SQL。 像这样：django/db/models/functions.py`class ConcatPair(Func):     ...     function = 'CONCAT'     ...      def as_mysql(self, compiler, connection):         return super(ConcatPair, self).as_sql(             compiler, connection,             function='CONCAT_WS',             template="%(function)s('', %(expressions)s)",         ) `**在Django更改1.10：**添加了对`arg_joiner`和`**extra_context`参数的支持。

`*expressions`参数是函数将要应用与表达式的位置参数列表。 表达式将转换为字符串，与`rg_joiner`连接在一起，然后作为`expressions`占位符插入`template`。

位置参数可以是表达式或Python值。 字符串假定为列引用，并且将包装在`F()`表达式中，而其他值将包裹在`Value()`表达式中。

`**extra` kwargs是可以插入到`template`属性中的`key=value`对。 `function`，`template`和`arg_joiner`关键字可用于替换相同名称的属性，而无需定义自己的类。 `output_field`可用于定义预期的返回类型。



### `Aggregate()`表达式

聚合表达式是[Func() expression](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#func-expressions)的一种特殊情况，它通知查询：`GROUP BY`子句是必须的。 所有[aggregate functions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#aggregation-functions)，如`Count()`和`Sum()`，继承自`Aggregate()`。

由于`Aggregate`是表达式和换行表达式，因此您可以表示一些复杂的计算：

```
from django.db.models import Count

Company.objects.annotate(
    managers_required=(Count('num_employees') / 4) + Count('num_managers'))
```

`Aggregate` API如下：

- *class* `Aggregate`(*expression*, *output_field=None*, ***extra*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/aggregates.html#Aggregate)

  `template`类属性，作为格式字符串，描述为此聚合生成的SQL。 默认为`'％（函数）s（ ％（表达式）s ）`。`function`描述将生成的聚合函数的类属性。 具体来说，`function`将被插入为[`template`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#django.db.models.Aggregate.template)中的`function`占位符。 默认为`None`。

`expression`参数可以是模型上的字段的名称，也可以是另一个表达式。 它将转换为字符串，并用作`expressions`中的`template`占位符。

`output_field`参数需要一个模型字段实例，如`IntegerField()`或`BooleanField()`，Django将从数据库中检索值后将其装载到其中。 通常在将模型字段实例化为与数据验证有关的任何参数（`max_length`，`max_digits`等）时，无需参数。 不会对表达式的输出值执行。

注意，只有当Django无法确定结果应该是什么字段类型时，才需要`output_field`。 混合字段类型的复杂表达式应定义所需的`output_field`。 例如，将`output_field=FloatField()`和`FloatField()`添加在一起应该可以有`IntegerField()`

`**extra` kwargs是可以插入到`template`属性中的`key=value`对。



### 创建自己的聚合函数

创建自己的聚合是非常容易的。 至少，您需要定义`function`，但也可以完全自定义生成的SQL。 这里有一个简单的例子：

```
from django.db.models import Aggregate

class Count(Aggregate):
    # supports COUNT(distinct field)
    function = 'COUNT'
    template = '%(function)s(%(distinct)s%(expressions)s)'

    def __init__(self, expression, distinct=False, **extra):
        super(Count, self).__init__(
            expression,
            distinct='DISTINCT ' if distinct else '',
            output_field=IntegerField(),
            **extra
        )
```



### `Value()`表达式

- *class* `Value`(*value*, *output_field=None*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#Value)

  

`Value()`对象表示表达式的最小可能组件：简单值。 当您需要在表达式中表示整数，布尔或字符串的值时，可以在`Value()`中包装该值。

您很少需要直接使用`Value()`。 当您编写表达式`F（'field'） + 1`时，Django隐式包装`1`在`Value()`中，允许在更复杂的表达式中使用简单的值。 当您要将字符串传递给表达式时，您将需要使用`Value()`。 大多数表达式将字符串参数解释为字段的名称，如`Lower('name')`。

`True`参数描述要包括在表达式中的值，例如`1`，`value`或`None`。 Django知道如何将这些Python值转换为相应的数据库类型。

`BooleanField()`参数应为模型字段实例，如`IntegerField()`或`output_field`，Django将在检索后从数据库。 通常在将模型字段实例化为与数据验证有关的任何参数（`max_length`，`max_digits`等）时，无需参数。 不会对表达式的输出值执行。



### `ExpressionWrapper()`表达式

- *class* `ExpressionWrapper`(*expression*, *output_field*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#ExpressionWrapper)

  

`ExpressionWrapper`简单地包围另一个表达式，并提供对其他表达式可能不可用的属性（例如`output_field`）的访问。 当对[Using F() with annotations](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#using-f-with-annotations)中描述的不同类型的`F()`表达式使用算术时，必须使用`ExpressionWrapper`。



### 条件表达式

条件表达式允许您在查询中使用[`if`](https://docs.python.org/3/reference/compound_stmts.html#if) ... [`elif`](https://docs.python.org/3/reference/compound_stmts.html#elif) ... [`else`](https://docs.python.org/3/reference/compound_stmts.html#else)逻辑。 Django本地支持SQL `CASE`表达式。 有关更多详细信息，请参阅[Conditional Expressions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/conditional-expressions.html)。



### `Subquery()`表达式

- *class* `Subquery`(*queryset*, *output_field=None*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#Subquery)

  

**Django中的新功能1.11。**

您可以使用`Subquery`表达式向`QuerySet`添加显式子查询。

例如，要使用该帖子最新评论的作者的电子邮件地址来注释每篇文章：

```
>>> from django.db.models import OuterRef, Subquery
>>> newest = Comment.objects.filter(post=OuterRef('pk')).order_by('-created_at')
>>> Post.objects.annotate(newest_commenter_email=Subquery(newest.values('email')[:1]))
```

在PostgreSQL上，SQL看起来像：

```
SELECT "post"."id", (
    SELECT U0."email"
    FROM "comment" U0
    WHERE U0."post_id" = ("post"."id")
    ORDER BY U0."created_at" DESC LIMIT 1
) AS "newest_commenter_email" FROM "post"
```

注

本节中的示例旨在说明如何强制Django执行子查询。 在某些情况下，可能会编写一个更清晰或更有效的执行相同任务的等效查询。



#### 从外部查询器引用列

- *class* `OuterRef`(*field*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#OuterRef)

  

**Django中的新功能1.11。**

当`Subquery`中的查询器需要引用外部查询的字段时，请使用`OuterRef`。 它的作用就像一个[`F`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#django.db.models.F)表达式，除了在外部查询器解析之前，检查是否引用一个有效的字段是不会发生的。

`OuterRef`的实例可以与`Subquery`的嵌套实例结合使用，以引用不是直接父项的包含查询集。 例如，此查询器将需要在一个嵌套的`Subquery`实例之间进行正确解析：

```
>>> Book.objects.filter(author=OuterRef(OuterRef('pk')))
```



#### 将子查询限制为单个列

有时候，必须从`Subquery`返回单个列，例如，在查找中使用`Subquery`作为`__in` 返回在最后一天发布的帖子的所有评论：

```
>>> from datetime import timedelta
>>> from django.utils import timezone
>>> one_day_ago = timezone.now() - timedelta(days=1)
>>> posts = Post.objects.filter(published_at__gte=one_day_ago)
>>> Comment.objects.filter(post__in=Subquery(posts.values('pk')))
```

在这种情况下，子查询必须使用[`values()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.values)才能返回一个列：主键。



#### 限制子查询到一行

为了防止子查询返回多行，使用了queryset的slice（`[:1]`）：

```
>>> subquery = Subquery(newest.values('email')[:1])
>>> Post.objects.annotate(newest_commenter_email=subquery)
```

在这种情况下，子查询只能返回单列*和*单行：最近创建的注释的电子邮件地址。

（使用[`get()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.get)而不是切片将失败，因为在`Subquery`中使用查询集之前，不能解析`OuterRef`。）



#### `Exists()`子查询

- *class* `Exists`(*queryset*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#Exists)

  

**Django中的新功能1.11。**

`Exists`是使用SQL `EXISTS`语句的`Subquery`子类。 在许多情况下，它将比子查询执行得更好，因为数据库能够在找到第一个匹配行时停止对子查询的评估。

例如，要注释每个帖子是否在最后一天是否有评论：

```
>>> from django.db.models import Exists, OuterRef
>>> from datetime import timedelta
>>> from django.utils import timezone
>>> one_day_ago = timezone.now() - timedelta(days=1)
>>> recent_comments = Comment.objects.filter(
...     post=OuterRef('pk'),
...     created_at__gte=one_day_ago,
... )
>>> Post.objects.annotate(recent_comment=Exists(recent_comments))
```

在PostgreSQL上，SQL看起来像：

```
SELECT "post"."id", "post"."published_at", EXISTS(
    SELECT U0."id", U0."post_id", U0."email", U0."created_at"
    FROM "comment" U0
    WHERE (
        U0."created_at" >= YYYY-MM-DD HH:MM:SS AND
        U0."post_id" = ("post"."id")
    )
) AS "recent_comment" FROM "post"
```

不必强制`Exists`引用单个列，因为列被丢弃并返回了一个布尔结果。 类似地，由于排序在SQL `EXISTS`子查询中不重要，只会降低性能，因此会自动删除。

您可以使用`~Exists()`查询`NOT EXISTS`。



#### 在`Subquery`表达式中进行过滤

不能使用`Subquery`和`Exists`直接过滤，例如：

```
>>> Post.objects.filter(Exists(recent_comments))
...
TypeError: 'Exists' object is not iterable
```

您必须通过首先注释查询集，然后基于该注释进行过滤，以过滤子查询表达式：

```
>>> Post.objects.annotate(
...     recent_comment=Exists(recent_comments),
... ).filter(recent_comment=True)
```



#### 在`Subquery`表达式中使用聚合

聚合可以在`Subquery`中使用，但它们需要[`filter()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.filter)，[`values()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.values)和[`annotate()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.annotate)以使子查询分组正确。

假设两个模型都有`length`字段，以查找帖子长度大于所有组合注释的总长度的帖子：

```
>>> from django.db.models import OuterRef, Subquery, Sum
>>> comments = Comment.objects.filter(post=OuterRef('pk')).order_by().values('post')
>>> total_comments = comments.annotate(total=Sum('length')).values('total')
>>> Post.objects.filter(length__gt=Subquery(total_comments))
```

初始`filter(...)`将子查询限制为相关参数。 `order_by()`删除`Comment`模型上的默认[`ordering`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.ordering)（如果有）。 `values('post')`通过`Post`聚合注释。 最后，`annotate(...)`执行聚合。 这些查询方法的应用顺序很重要。 在这种情况下，由于子查询必须限于单列，因此需要`values('total')`。

This is the only way to perform an aggregation within a `Subquery`, as using [`aggregate()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.aggregate) attempts to evaluate the queryset (and if there is an `OuterRef`, this will not be possible to resolve).



### 原始SQL表达式

- *class* `RawSQL`(*sql*, *params*, *output_field=None*)[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#RawSQL)

  

有时，数据库表达式不能轻易地表达一个复杂的`WHERE`子句。 在这些边缘情况下，使用`RawSQL`表达式。 像这样：

```
>>> from django.db.models.expressions import RawSQL
>>> queryset.annotate(val=RawSQL("select col from sometable where othercol = %s", (someparam,)))
```

这些额外的查找可能无法移植到不同的数据库引擎（因为您明确地编写SQL代码）并违反了DRY原则，所以如果可能的话您应该避免它们。

警告

您应该非常小心，以避免用户可以使用`params`来控制的任何参数，以防止[SQL injection attacks](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/security.html#sql-injection-protection)。 `params`是一个必需的参数，强制您确认您没有使用用户提供的数据插入SQL。



## 技术信息

下面您将找到对图书馆作者有用的技术实施细节。 下面的技术API和示例将有助于创建可扩展Django提供的内置功能的通用查询表达式。



### 表达式API 

查询表达式实现了[query expression API](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/lookups.html#query-expression)，但也暴露了下面列出的一些额外的方法和属性。 所有查询表达式必须从`Expression()`或相关子类继承。

当查询表达式包装另一个表达式时，它负责调用包装表达式上的相应方法。

- *class* `Expression`[[source\]](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/_modules/django/db/models/expressions.html#Expression)

  `contains_aggregate`告诉Django此表达式包含聚合，并且需要将`GROUP BY`子句添加到查询中。`resolve_expression`(*query=None*, *allow_joins=True*, *reuse=None*, *summarize=False*, *for_save=False*)提供在将表达式添加到查询之前对表达式执行任何预处理或验证的机会。 还必须在任何嵌套表达式上调用`resolve_expression()`。 应该返回具有任何必要变换的`copy()`的`self`。`query`是后端查询实现。`allow_joins`是一个布尔值，允许或拒绝在查询中使用联接。`reuse`是用于多连接场景的一组可重用连接。`summarize`是一个布尔值，当`True`时，表示正在计算的查询是终端聚合查询。`get_source_expressions`()返回内部表达式的有序列表。 像这样：`>>> Sum(F('foo')).get_source_expressions() [F('foo')] ``set_source_expressions`(*expressions*)获取表达式列表并存储它们，以便`get_source_expressions()`可以返回它们。`relabeled_clone`(*change_map*)返回`self`的克隆（副本），并重新标记任何列别名。 创建子查询时，将重命名列别名。 `relabeled_clone()`也应该在任何嵌套表达式上调用并分配给克隆。`change_map`是将旧别名映射到新别名的字典。例如：`def relabeled_clone(self, change_map):     clone = copy.copy(self)     clone.expression = self.expression.relabeled_clone(change_map)     return clone ``convert_value`(*value*, *expression*, *connection*, *context*)允许表达式将`value`强制为更适当类型的钩子。`get_group_by_cols`()负责返回此表达式的列引用列表。 `get_group_by_cols()`应在任何嵌套表达式上调用。 `F()`对象，特别是保存对列的引用。`asc`(*nulls_first=False*, *nulls_last=False*)返回准备好以升序排序的表达式。`nulls_first`和`nulls_last`定义如何排序空值。**在Django更改1.11：**添加了`nulls_last`和`nulls_first`参数。`desc`(*nulls_first=False*, *nulls_last=False*)返回准备好以降序排序的表达式。`nulls_first`和`nulls_last`定义如何排序空值。**在Django更改1.11：**添加了`nulls_first`和`nulls_last`参数。`reverse_ordering`()通过在`self`调用中反转排序顺序所需的任何修改，返回`order_by`。 例如，执行`NULLS LAST`的表达式将其值更改为`NULLS FIRST`。 仅对实现类似`OrderBy`的排序顺序的表达式需要修改。 当在查询集上调用[`reverse()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.reverse)时，调用此方法。



### 编写自己的查询表达式

您可以编写自己的查询表达式类，这些类使用其他查询表达式，并可以与其集成。 让我们通过编写一个`COALESCE` SQL函数的实现，而不使用内置的[Func() expressions](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/expressions.html#func-expressions)来演示一个例子。

`COALESCE` SQL函数定义为获取列或值的列表。 它将返回不是`NULL`的第一列或值。

我们将首先定义用于生成SQL的模板，然后使用`__init__()`方法来设置一些属性：

```
import copy
from django.db.models import Expression

class Coalesce(Expression):
    template = 'COALESCE( %(expressions)s )'

    def __init__(self, expressions, output_field):
      super(Coalesce, self).__init__(output_field=output_field)
      if len(expressions) < 2:
          raise ValueError('expressions must have at least 2 elements')
      for expression in expressions:
          if not hasattr(expression, 'resolve_expression'):
              raise TypeError('%r is not an Expression' % expression)
      self.expressions = expressions
```

我们对参数进行一些基本验证，包括至少需要2列或值，并确保它们是表达式。 我们在这里需要`output_field`，以便Django知道要将最终结果分配给什么样的模型字段。

现在我们实现预处理和验证。 由于我们现在没有任何自己的验证，我们只是委托给嵌套表达式：

```
def resolve_expression(self, query=None, allow_joins=True, reuse=None, summarize=False, for_save=False):
    c = self.copy()
    c.is_summary = summarize
    for pos, expression in enumerate(self.expressions):
        c.expressions[pos] = expression.resolve_expression(query, allow_joins, reuse, summarize, for_save)
    return c
```

接下来，我们编写负责生成SQL的方法：

```
def as_sql(self, compiler, connection, template=None):
    sql_expressions, sql_params = [], []
    for expression in self.expressions:
        sql, params = compiler.compile(expression)
        sql_expressions.append(sql)
        sql_params.extend(params)
    template = template or self.template
    data = {'expressions': ','.join(sql_expressions)}
    return template % data, params

def as_oracle(self, compiler, connection):
    """
    Example of vendor specific handling (Oracle in this case).
    Let's make the function name lowercase.
    """
    return self.as_sql(compiler, connection, template='coalesce( %(expressions)s )')
```

`as_sql()`方法可以支持自定义关键字参数，允许`as_vendorname()`方法覆盖用于生成SQL字符串的数据。 使用`as_sql()`定制的关键字参数比在`as_vendorname()`方法中突变`self`更好，因为后者可能导致在不同的数据库后端 如果您的类依赖于类属性来定义数据，请考虑在`as_sql()`方法中允许覆盖。

我们使用`expressions`方法为每个`compiler.compile()`生成SQL，并用逗号连接结果。 然后使用我们的数据填充模板，并返回SQL和参数。

我们还定义了一个特定于Oracle后端的自定义实现。 如果Oracle后端正在使用，则将调用`as_oracle()`函数，而不是`as_sql()`。

最后，我们实现允许我们的查询表达式与其他查询表达式一起播放的其他方法：

```
def get_source_expressions(self):
    return self.expressions

def set_source_expressions(self, expressions):
    self.expressions = expressions
```

让我们看看它是如何工作的：

```
>>> from django.db.models import F, Value, CharField
>>> qs = Company.objects.annotate(
...    tagline=Coalesce([
...        F('motto'),
...        F('ticker_name'),
...        F('description'),
...        Value('No Tagline')
...        ], output_field=CharField()))
>>> for c in qs:
...     print("%s: %s" % (c.name, c.tagline))
...
Google: Do No Evil
Apple: AAPL
Yahoo: Internet Company
Django Software Foundation: No Tagline
```



### 在第三方数据库后端添加支持

如果您使用的数据库后端对于某个功能使用了不同的SQL语法，则可以通过在该函数的类中修补一个新方法来为其添加支持。

假设我们正在为Microsoft的SQL Server编写一个后端，它为[`Length`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/database-functions.html#django.db.models.functions.Length)函数使用SQL `LEN`而不是`LENGTH`。 我们将在`Length`类中修改一个称为`as_sqlserver()`的新方法：

```
from django.db.models.functions import Length

def sqlserver_length(self, compiler, connection):
    return self.as_sql(compiler, connection, function='LEN')

Length.as_sqlserver = sqlserver_length
```

您还可以使用`as_sql()`的`template`参数自定义SQL。

我们使用`as_sqlserver()`，因为`django.db.connection.vendor`返回后台的`sqlserver`。

第三方后端可以在后端程序包的顶级`__init__.py`文件或在导入的顶级`expressions.py`文件（或程序包）中注册其功能。从顶层`__init__.py`。

对于希望修补他们正在使用的后端的用户项目，该代码应该存在于[`AppConfig.ready()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/applications.html#django.apps.AppConfig.ready)方法中。