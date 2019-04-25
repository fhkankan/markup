[TOC]

# 模型-查询

一旦你创建了[data models](https://yiyibooks.cn/__trs__/qy/django2/topics/db/models.html)，Django就会自动为你提供一个数据库抽象API，让你可以创建，检索，更新和删除对象。本文档介绍了如何使用此API。 有关所有各种模型查找选项的完整详细信息，请参阅[数据模型参考](https://yiyibooks.cn/__trs__/qy/django2/ref/models/index.html)。

## 模型与数据

参考模型

```python
from django.db import models

class Blog(models.Model):
    name = models.CharField(max_length=100)
    tagline = models.TextField()

    def __str__(self):
        return self.name

class Author(models.Model):
    name = models.CharField(max_length=200)
    email = models.EmailField()

    def __str__(self):
        return self.name

class Entry(models.Model):
    blog = models.ForeignKey(Blog, on_delete=models.CASCADE)
    headline = models.CharField(max_length=255)
    body_text = models.TextField()
    pub_date = models.DateField()
    mod_date = models.DateField()
    authors = models.ManyToManyField(Author)
    n_comments = models.IntegerField()
    n_pingbacks = models.IntegerField()
    rating = models.IntegerField()

    def __str__(self):
        return self.headline
```

创建对象

```python
# 有两个方法
save()  	# 在执行前，Django不会访问数据库，方法没有返回值
p = Person(first_name="Bruce", last_name="Springsteen")
p.save(force_insert=True)

create()  # 一条语句创建对象
p = Person.objects.create(first_name="Bruce", last_name="Springsteen")

# 示例
>>> from blog.models import Blog
>>> b = Blog(name='Beatles Blog', tagline='All the latest Beatles news.')
>>> b.save()  # 执行SQL的insert语句
```

更新对象

```python
# 更新普通字段
>>> b5.name = 'New name'
>>> b5.save()  # 执行SQL的update语句

# 更新ForeignKey
>>> from blog.models import Entry
>>> entry = Entry.objects.get(pk=1)
>>> cheese_blog = Blog.objects.get(name="Cheddar Talk")
>>> entry.blog = cheese_blog
>>> entry.save()

# 更新ManyToManyField
>>> from blog.models import Author
>>> joe = Author.objects.create(name="Joe")
>>> entry.authors.add(joe)  # 需使用add()增加关联关系的一条记录
>>> john = Author.objects.create(name="John")
>>> paul = Author.objects.create(name="Paul")
>>> entry.authors.add(john, paul)  # 多条记录
```

## 获取对象

通过模型中的`管理器`构造一个`查询集`，来从你的数据库中获取对象。

每个模型类默认都有一个叫 `objects `的类属性，它由django自动生成，类型为： `django.db.models.manager.Manager`，可以把它叫 模型管理器。

`查询集`表示从数据库中取出来的对象的集合。它可以含有零个、一个或者多个*过滤器*。过滤器基于所给的参数限制查询的结果。 

```shell
>>> Blog.objects  # 只可以通过模型的类访问
<django.db.models.manager.Manager object at ...>
>>> b = Blog(name='Foo', tagline='Bar')
>>> b.objects  # 不可以通过模型的实例访问
Traceback:
    ...
AttributeError: "Manager isn't accessible via Blog instances."
```

### QuerySet特性

`filter(),exclude()`方法，返回QuerySet

```python
filter(**kwargs)
# 返回一个新的查询集，它包含满足查询参数的对象
exclude(**kwargs)
# 返回一个新的查询集，它包含不满足查询参数的对象

查询参数需要满足特定格式，详见“字段查询”
```

- 链式过滤

查询集的筛选结果还是查询集，所以可以将筛选语句链接在一起

```shell
>>> Entry.objects.filter(
...     headline__startswith='What'
... ).exclude(
...     pub_date__gte=datetime.date.today()
... ).filter(
...     pub_date__gte=datetime(2005, 1, 30)
... )
```

- 过滤后的查询集是独立的

每次筛选后得到的都是一个全新的独立的查询集，和之前的查询集没有任何绑定关系。可以被存储及反复使用

```shell
>>> q1 = Entry.objects.filter(headline__startswith="What")
>>> q2 = q1.exclude(pub_date__gte=datetime.date.today())
>>> q3 = q1.filter(pub_date__gte=datetime.date.today())
```

- 查询集是惰性执行的

创建查询集不会带来任何数据库的访问。只有在查询集需要求值时，Django才会真正运行这个查询

### 限制查询集

可以使用Python 的切片语法来限制`查询集`记录的数目 。它等同于SQL 的`LIMIT` 和`OFFSET` 子句。

```shell
>>> Entry.objects.all()[:5]
>>> Entry.objects.order_by('headline')[0]
```

第二条语句若没有对象，将引发`IndexError`

### 缓存和查询集

每个`查询集`都包含一个缓存来最小化对数据库的访问。在一个新创建的`查询集`中，缓存为空。首次对`查询集`进行求值 —— 同时发生数据库查询 ——Django 将保存查询的结果到`查询集`的缓存中并返回明确请求的结果（例如，如果正在迭代`查询集`，则返回下一个结果）。接下来对该`查询集`的求值将重用缓存的结果。

```python
# 相同的数据库查询执行两次，同时两个结果列表可能不相同(请求期间Entry被添或删)
>>> print([e.headline for e in Entry.objects.all()])
>>> print([e.pub_date for e in Entry.objects.all()])
# 保存查询机并重新使用
>>> queryset = Entry.objects.all()
>>> print([p.headline for p in queryset]) # Evaluate the query set.
>>> print([p.pub_date for p in queryset]) # Re-use the cache from the evaluation.
```

- 何时查询集不会被缓存

查询集不会永远缓存它们的结果。当只对查询集的*部分*进行求值时会检查缓存， 但是如果这个部分不在缓存中，那么接下来查询返回的记录都将不会被缓存。特别地，这意味着使用切片或索引来*限制查询集*将不会填充缓存。

注意：简单地打印查询集不会填充缓存。因为`__repr__()` 调用只返回全部查询集的一个切片。

```python
# 重复获取查询集对象中一个特定的索引将每次都查询数据库
>>> queryset = Entry.objects.all()
>>> print queryset[5] # Queries the database
>>> print queryset[5] # Queries the database again

# 如果已经对全部查询集求值过，则将检查缓存
>>> queryset = Entry.objects.all()
>>> [entry for entry in queryset] # Queries the database
>>> print queryset[5] # Uses cache
>>> print queryset[5] # Uses cache

# 使得全部的查询集被求值并填充到缓存中
>>> [entry for entry in queryset]
>>> bool(queryset)
>>> entry in queryset
>>> list(queryset)
```

### [查询集方法](https://yiyibooks.cn/xx/django_182/ref/models/querysets.html#queryset-api)

#### 返回新查询集

```python
filter(**kwargs)  
# 过滤，包含了与所给筛选条件相匹配的对象, 多参数时为AND关系过滤
exclude(**kwargs)  
# 排除，包含了与所给筛选条件不匹配的对象，底层SQL中多个参数通过AND连接，然后所有内容放入NOT()
annotate(*args,**kwargs)  
# 分组，使用提供的查询表达式列表注释QuerySet中的每个对象。 表达式可以是简单值，对模型（或任何相关模型）上的字段的引用，或者是通过与对象中的对象相关的对象计算的聚合表达式（平均值，总和等）
order_by(*fields)  
# 排序，隐式是升序排序，'-'前缀表示降序排序,'?'表示随机排序
reverse()  
# 对查询结果反向排序,只能在具有已定义顺序的QuerySet上调用(在model类的Meta中指定ordering或调用order_by()方法)。
distinct(*fields)  
# 从返回结果中剔除重复纪录(如果你查询跨越多个表，可能在计算QuerySet时得到重复的结果。此时可以使用distinct()，注意只有在PostgreSQL中支持按字段去重。)
values(*fields, **expressions)  
# 返回一个ValueQuerySet—一个特殊的QuerySet，迭代时得到的并不是模型实例化对象，而是一个字典序列
values_list(*fields, flat=False, named=False) 
# 它与values()非常相似，它返回的是一个元组序列，values返回的是一个字典序列 
dates(field, kind, order='ASC')
# 返回DateQuerySet - QuerySet，其计算结果为datetime.date对象列表，表示特定种类的所有可用日期QuerySet。field应为模型的DateField的名称。 kind应为"year"、"month"或"day"。隐式的是升序排序。若要随机排序，请使用"?"，order（默认为“ASC”）应为'ASC'或'DESC'
datetimes(field, kind, order='ASC', tzinfo=None)
# 返回QuerySet，其值为datetime.datetime对象的列表，表示QuerySet内容中特定类型的所有可用日期。field应为模型的DateTimeField的名称。kind应为“year”，“month”，“day”，“hour”，“minute”或“second”。结果列表中的每个datetime.datetime对象被“截断”到给定的类型。order, 默认为'ASC', 可选项为'ASC' 或者 'DESC'. 这个选项指定了返回结果的排序方式。tzinfo定义在截断之前将数据时间转换到的时区。实际上，给定的datetime具有不同的表示，这取决于使用的时区。此参数必须是datetime.tzinfo对象。如果它无，Django使用当前时区。当USE_TZ为False时，它不起作用。
none()
# 调用none()将创建一个从不返回任何对象的查询集，并且在访问结果时不会执行任何查询。qs.none()查询集是EmptyQuerySet的一个实例。
all()  
# 返回当前QuerySet（或QuerySet 子类）的副本,当对QuerySet进行求值时，它通常会缓存其结果。如果数据库中的数据在QuerySet求值之后可能已经改变，你可以通过在以前求值过的QuerySet上调用相同的all() 查询以获得更新后的结果。
union(*other_qs, all=False)
# 使用SQL的UNION来结合多个QuerySet的结果，默认查询去重后的数据，若需有重复的数据，all设置为True
intersection(*other_qs)
# 使用SQL的INTERSECT运算符返回两个或多个QuerySet的共享元素
difference(*other_qs)
# 使用SQL的EXCEPT运算符仅保留QuerySet中存在的元素，而不保留其他一些QuerySet中的元素。
select_related(fields)
# 将“跟随”外键关系，在执行查询时选择其他相关对象数据。 这是一个性能增强器，它会导致一个单个更复杂的查询，但是也意味着以后再使用外键关系时，不会重新需要数据可查询。
prefetch_related(*lookups)
# 将在一个批处理中自动检索每个指定查找的相关对象。
extra(select=None, where=None, params=None, tables=None, order_by=None, select_params=None)
# 在 QuerySet生成的SQL从句中注入新子句,实现复杂where子句。参数可选，但必须有一个
defer(*fields)
# 延迟加载，不要从数据库中检索它们
only(*fields)
# 只有这些字段立即加载，其他都被推迟
using(alias)
# 使用多个数据库，空值查询集在哪个数据库上求值
select_for_update(nowwait=False)
# 锁定相关行知道事物结束，在支持的数据库上产生一个select...for update
raw(raw_query, params=None, translation=None)
# 接收一个原始SQL查询，执行它并返回一个django.db.models.query.RawQuerySet 实例。这个RawQuerySet 实例可以迭代以提供实例对象，就像普通的QuerySet 一样。
```

- 示例

filter

```python
paper_list.filter(name__icontains=name)
```

exclude

```python
Entry.objects.exclude(pub_date__gt=datetime.date(2005, 1, 3), headline='Hello')
```

Annotate

```python
# 不指定名字,默认关键自作为Annotation的别名，只适用于单个字段
>>> from django.db.models import Count
>>> q = Blog.objects.annotate(Count('entry'))
>>> q[0].name  # 第一个blog的名字
'Blogasaurus'
>>> q[0].entry__count  # 第一个blog的entry_count的值，Entity的数量
42

# 指定名字
>>> q = Blog.objects.annotate(number_of_entries=Count('entry'))
# The number of entries on the first blog, using the name provided
>>> q[0].number_of_entries
42
```

Order_by

```python
Entry.objects.filter(pub_date__year=2005).order_by('-pub_date', 'headline')
Entry.objects.order_by('?')  # 随机
Entry.objects.order_by('blog__name', 'headline')  # 关联字段
Entry.objects.order_by('blog_id')  #  无join
Entry.objects.order_by('blog__id')  # Join
Entry.objects.order_by(Coalesce('summary', 'headline').desc())  # 表达式
Entry.objects.order_by(Lower('headline').desc())  # 大小写
Entry.objects.order_by('headline').order_by('pub_date')  # 最后一个有效
Entry.objects.order_by()  # 默认的排序
# 检查是否有任何方式的排序
QuerySet.ordered  # True/False
# 影响distinct(),values()
# order_by调用的字段会包含在SELECT中，可能会有重复行，
# 指定一个多值字段来排序结果（ManyToManyField或ForeignKey的反向关联)
class Event(Model):
   parent = models.ForeignKey('self', related_name='children')
   date = models.DateField()
Event.objects.order_by('children__date')  # 会返回扩大的新QuerySet,需注意慎用
```

reverse

```python
my_queryset.reverse()[:5]  # 后5个元素
```

distinct

```python
Author.objects.distinct()

# 如果你使用的是distinct()和order_by()，请注意相关模型的排序
# 当一起使用distinct()和values()时，请注意字段在不在values()
```

values

```shell
# 多字段
>>> Blog.objects.values()
[{'id': 1, 'name': 'Beatles Blog', 'tagline': 'All the latest Beatles news.'}],
>>> Blog.objects.values('id', 'name')
[{'id': 1, 'name': 'Beatles Blog'}]
# 表达式， 1.11支持表达式
>>> from django.db.models.functions import Lower
>>> Blog.objects.values(lower_name=Lower('name'))
<QuerySet [{'lower_name': 'beatles blog'}]>
# 聚合
>>> from django.db.models import Count
>>> Blog.objects.values('author', entries=Count('entry'))
<QuerySet [{'author': 1, 'entries': 20}, {'author': 1, 'entries': 13}]>
>>> Blog.objects.values('author').annotate(entries=Count('entry'))
<QuerySet [{'author': 1, 'entries': 33}]>
# Foreignkey
>>> Entry.objects.values()
[{'blog_id': 1, 'headline': 'First Entry', ...}, ...]
>>> Entry.objects.values('blog')
[{'blog': 1}, ...]
>>> Entry.objects.values('blog_id')
[{'blog_id': 1}, ...]
# 反向关联
>>> Blog.objects.values('name', 'entry__headline')
<QuerySet [{'name': 'My blog', 'entry__headline': 'An entry'},
     {'name': 'My blog', 'entry__headline': 'Another entry'}, ...]>
# 注意
- 当values() 与distinct() 一起使用时，注意排序可能影响最终的结果。
- 如果values() 子句位于extra() 调用之后，extra() 中的select 参数定义的字段必须显式包含在values() 调用中。values() 调用后面的extra() 调用将忽略选择的额外的字段。
- 在values() 之后调用only() 和defer() 不太合理，所以将引发一个NotImplementedError。
```

values_list

```shell
# 多字段
>>> Entry.objects.values_list()
<QuerySet [(1, 'First entry',...), ...]>
>>> Entry.objects.values_list('id', 'headline')
<QuerySet [(1, 'First entry'), ...]>
#  表达式， 1.11支持字段上加表达式
>>> from django.db.models.functions import Lower
>>> Entry.objects.values_list('id', Lower('headline'))
<QuerySet [(1, 'first entry'), ...]>

# 单字段
>>> Entry.objects.values_list('id').order_by('id')
<QuerySet[(1,), (2,), (3,), ...]>
>>> Entry.objects.values_list('id', flat=True).order_by('id')
<QuerySet [1, 2, 3, ...]>
# 关键字， 2.0支持named
>>> Entry.objects.values_list('id', 'headline', named=True)
<QuerySet [Row(id=1, headline='First entry'), ...]>

# ForeignKey
>>> Entry.objects.values_list('authors')
<QuerySet [('Noam Chomsky',), ('George Orwell',), (None,)]>

# manyToMany
>>> Author.objects.values_list('name', 'entry__headline')
<QuerySet [('Noam Chomsky', 'Impressions of Gaza'),
 ('George Orwell', 'Why Socialists Do Not Believe in Fun'),
 ('George Orwell', 'In Defence of English Cooking'),
 ('Don Quixote', None)]>
```

date

```shell
>>> Entry.objects.dates('pub_date', 'year') # 去过重的year
[datetime.date(2005, 1, 1)]
>>> Entry.objects.dates('pub_date', 'month')  # 去过重的year/month
[datetime.date(2005, 2, 1), datetime.date(2005, 3, 1)]
>>> Entry.objects.dates('pub_date', 'day')  # 去过重的year/month/day
[datetime.date(2005, 2, 20), datetime.date(2005, 3, 20)]
>>> Entry.objects.dates('pub_date', 'day', order='DESC')
[datetime.date(2005, 3, 20), datetime.date(2005, 2, 20)]
>>> Entry.objects.filter(headline__contains='Lennon').dates('pub_date', 'day')
[datetime.date(2005, 3, 20)]
```

none

```shell
>>> Entry.objects.none()
<QuerySet []>
>>> from django.db.models.query import EmptyQuerySet
>>> isinstance(Entry.objects.none(), EmptyQuerySet)
True
```

union/intersection/difference

```shell
# 1.11新增
>>> qs1.union(qs2, qs3)
>>> qs1.intersection(qs2, qs3)
>>> qs1.difference(qs2, qs3)
# union,intersection,difference会返回第一个QuerySet类型模型实例，即使变量是其他模型的QuerySet。只要所有QuerySet中的SELECT列表相同，传递不同的模型就会起作用（至少类型，只要类型相同，名称无关紧要）。在这种情况下，您必须使用应用于生成的QuerySet的QuerySet方法中的第一个QuerySet中的列名。
>>> qs1 = Author.objects.values_list('name')
>>> qs2 = Entry.objects.values_list('headline')
>>> qs1.union(qs2).order_by('name')
# 在结果QuerySet上只允许LIMIT,OFFSET,COUNT(*),ORDER BY和指定列(如slicing,count(),order_by()和values()/values_list())。此外，数据库限制组合查询中允许的操作。例如，大多数数据库在组合查询中不允许LIMIT或OFFSET。
```

Select_related

```python
# 标准查询
e = Entry.objects.get(id=5)  # Hits the database.
b = e.blog  # Hits the database again to get the related Blog object.

# select_related
e = Entry.objects.select_related('blog').get(id=5)  # Hits the database.
b = e.blog  # Doesn't hit the database, because e.blog has been prepopulated in the previous query.

# 需要清除QuerySet上过去调用select_related所添加的相关字段列表
without_relations = queryset.select_related(None)

# 多参数与链式调用类似
select_related('foo', 'bar')
select_related('foo').select_related('bar')

# 任意对象的查询集上均可用
rom django.utils import timezone

blogs = set()  # Find all the blogs with entries scheduled to be published in the future.

for e in Entry.objects.filter(pub_date__gt=timezone.now()).select_related('blog'):
    # Without select_related(), this would make a database query for each
    # loop iteration in order to fetch the related blog for each entry.
    blogs.add(e.blog)
    
# filter()和select_related()之间的链接顺序并不重要，如下等价
Entry.objects.filter(pub_date__gt=timezone.now()).select_related('blog')
Entry.objects.select_related('blog').filter(pub_date__gt=timezone.now())

# 外键查询
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

# 如下操作将会缓存与其相关的Person和 City关系:
b = Book.objects.select_related('author__hometown').get(id=4)
p = b.author         # Doesn't hit the database.
c = p.hometown       # Doesn't hit the database.

b = Book.objects.get(id=4) # No select_related() in this example.
p = b.author         # Hits the database.
c = p.hometown       # Hits the database.

# 在传递给select_related() 的字段中，你可以使用任何ForeignKey 和OneToOneField。
# 在传递给select_related 的字段中，你还可以反向引用OneToOneField —— 也就是说，你可以回溯到定义OneToOneField 的字段。此时，可以使用关联对象字段的related_name，而不要指定字段的名称。
```

Preach_related

```python
# 具有与select_related类似的目的，两者都被设计为阻止由访问相关对象而导致的数据库查询的泛滥，但是策略是完全不同的。
# select_related通过创建SQL连接并在SELECT语句中包括相关对象的字段来工作。因此，select_related在同一数据库查询中获取相关对象。然而，为了避免由于跨越“多个”关系而导致的大得多的结果集，select_related限于单值关系 - 外键和一对一关系。
# prefetch_related 独立查找每个关系，并在Python中执行“关联(joining)”。这允许它除了select_related支持的外键和一对一关系以外, 还能预取多对多和多对一对象，这正是select_related不能实现的。prefetch_related 还支持GenericRelation 和 GenericForeignKey的预取.
```

extra

```python
# select
# 在 SELECT 从句中添加其他字段信息，它应该是一个字典，存放着属性名到 SQL 从句的映射。
Entry.objects.extra(select={'is_recent': "pub_date > '2006-01-01'"})
Blog.objects.extra(
    select={
        'entry_count': 'SELECT COUNT(*) FROM blog_entry WHERE blog_entry.blog_id = blog_blog.id'
    },
)
Blog.objects.extra(
    select=OrderedDict([('a', '%s'), ('b', '%s')]),
    select_params=('one', 'two'))

# where/tables
# 可以使用where定义显式SQL WHERE子句 - 也许执行非显式连接。您可以使用tables手动将表添加到SQL FROM子句。where和tables都接受字符串列表。所有where参数均为“与”任何其他搜索条件
Entry.objects.extra(where=["foo='a' OR bar = 'a'", "baz = 'a'"])
Entry.objects.extra(where=['headline=%s'], params=['Lennon'])

# order_by
# 如需要使用通过extra()包含的一些新字段或表来对结果查询进行排序，请使用order_by参数extra()并传入一个字符串序列。这些字符串应该是模型字段（如查询集上的正常order_by()方法），形式为table_name.column_name或您在select参数到extra()
q = Entry.objects.extra(select={'is_recent': "pub_date > '2006-01-01'"})
q = q.extra(order_by = ['-is_recent'])
```

defer

```python
# 传递字段名不加载到defer()
Entry.objects.defer("headline", "body")
# 多次调用，均会添加新字段，顺序无关
Entry.objects.defer("body").filter(rating=5).defer("headline")
# 关联模型
Blog.objects.select_related().defer("entry__headline", "entry__body")
# 清空延迟字段
my_queryset.defer(None)
```

only

```python
# 就延迟而言，如下等价
Person.objects.defer("age", "biography")
Person.objects.only("name")

# 连续调用，只有最后有效
Entry.objects.only("body", "rating").only("headline")

# defer与only连用
Entry.objects.only("headline", "body").defer("body")  # Final result is that everything except "headline" is deferred.
Entry.objects.defer("body").only("headline", "body")  # Final result loads headline and body immediately (only() replaces any existing set of fields).
```

using

```shell
# queries the database with the 'default' alias.
>>> Entry.objects.all()

# queries the database with the 'backup' alias
>>> Entry.objects.using('backup')
```

select_for_update

```python
entries = Entry.objects.select_for_update().filter(author=request.user)  # 所有匹配的行将被锁定，直到事务结束。这意味着可以通过锁防止数据被其它事务修改

# 解决阻塞
一般情况下如果其他事务锁定了相关行，那么本查询将被阻塞，直到锁被释放。如果这不是你想要的行为，可以使用如下方法之一：
1. 请使用select_for_update(nowait=True). 这将使查询不阻塞。如果其它事务持有冲突的锁, 那么查询将引发 DatabaseError 异常
2.可以使用select_for_update(skip_locked=True)来忽略行锁定。nowait和skip_locked是互斥的，并且尝试在启用两个选项的情况下调用select_for_update（）将导致ValueError。

# nullable关系不能使用
>>> Person.objects.select_related('hometown').select_for_update()
Traceback (most recent call last):
...
django.db.utils.NotSupportedError: FOR UPDATE cannot be applied to the nullable side of an outer join
# 若是不关注null对象，可排除
>>> Person.objects.select_related('hometown').select_for_update().exclude(hometown=None)
<QuerySet [<Person: ...)>, ...]>
```

#### 不返回查询集

```python
get(**kwargs)   
# 返回与所给筛选条件相匹配的对象，返回结果有且只有一个，如果符合筛选条件的对象超过一个或者没有都会抛出错误。
create(**kwargs)
# 一个在一步操作中同时创建对象并且保存的便捷方法. 
get_or_create(defaults=None, **kwargs)
# 一个通过给出的kwargs 来查询对象的便捷方法（如果你的模型中的所有字段都有默认值，可以为空），需要的话创建一个对象。返回一个由(object, created)组成的元组，元组中的object 是一个查询到的或者是被创建的对象， created 是一个表示是否创建了新的对象的布尔值。
update_or_create(defaults=None, **kwargs)
# 一个通过给出的kwargs 来更新对象的便捷方法， 如果需要的话创建一个新的对象。defaults 是一个由 (field, value) 对组成的字典，用于更新对象。返回一个由 (object, created)组成的元组,元组中的object 是一个创建的或者是被更新的对象， created 是一个标示是否创建了新的对象的布尔值。尝试通过给出的kwargs 去从数据库中获取匹配的对象。如果找到匹配的对象，它将会依据defaults 字典给出的值更新字段。
bulk_create(objs, batch_size=None)
# 此方法以有效的方式（通常只有1个查询，无论有多少对象）将提供的对象列表插入到数据库中。batch_size参数控制在单个查询中创建的对象数。默认值是在一个批处理中创建所有对象，除了SQLite，其中默认值为每个查询最多使用999个变量。
count()
# 返回数据库中匹配查询(QuerySet)的对象数量。永远不会引发异常
in_bulk(id_list=None, field_name='pk')
# 获取主键值的列表和字段名，并返回将每个主键值映射到具有给定ID的对象的实例的字典。若主键列表缺省，返回所有，field_name必须是唯一的，默认主键
iterator(chunk_size=2000)
# 评估QuerySet（通过执行查询），并返回一个迭代器。
latest(*fields)
# 使用作为日期字段提供的field_name，按日期返回表中的最新对象。
earliest(*fields)
# 除非方向更改，类似latest()
first()
# 返回结果集的第一个对象, 当没有找到时返回None.如果 QuerySet 没有设置排序,则将会自动按主键进行排序
last()
# 返回最后一条记录对象 ,当没有找到时返回None.如果 QuerySet 没有设置排序,则将会自动按主键进行排序
aggregate(*args, **kwargs)
# 聚合，返回一个字典，包含根据QuerySet计算得到的聚合值（平均数、和等等）。aggregate() 的每个参数指定返回的字典中将要包含的值
exists()
# 如果QuerySet包含数据，就返回True，否则返回False
update(**kwargs)
# 对指定的字段执行SQL更新查询，并返回匹配的行数（如果某些行已具有新值，则可能不等于已更新的行数）。不能在已采取切片或以其他方式无法过滤的QuerySet上调用
delete()
# 对QuerySet中的所有行执行SQL删除查询。立即应用delete()。不能在已采取切片或以其他方式无法过滤的QuerySet上调用delete()
as_manager()
# 类方法返回一个复制了QueSet方法的Manager对象的实例
```

- 示例

get

```python
Entry.objects.get(id='foo') # raises Entry.DoesNotExist
```

create

```python
p = Person.objects.create(first_name="Bruce", last_name="Springsteen")

p = Person(first_name="Bruce", last_name="Springsteen")
p.save(force_insert=True)
```

get_or_create

```python
# 1.11支持defaults中设置可调用值
try:
    obj = Person.objects.get(first_name='John', last_name='Lennon')
except Person.DoesNotExist:
    obj = Person(first_name='John', last_name='Lennon', birthday=date(1940, 10, 9))
    obj.save()
    
# 改写为
obj, created = Person.objects.get_or_create(first_name='John', last_name='Lennon', defaults={'birthday': date(1940, 10, 9)})

# 创建逻辑
params = {k: v for k, v in kwargs.items() if '__' not in k}
params.update(defaults)
obj = self.model(**params)
obj.save()

# 有名为defaults的字段，做精确查询
Foo.objects.get_or_create(defaults__exact='bar', defaults={'defaults': 'baz'})

# 反向关联
class Chapter(models.Model):
    title = models.CharField(max_length=255, unique=True)

class Book(models.Model):
    title = models.CharField(max_length=256)
    chapters = models.ManyToManyField(Chapter)

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

Update_create

```python
# 在1.11支持defaults中使用可调用值
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
    
# uodate_create简写
obj, created = Person.objects.update_or_create(
    first_name='John', last_name='Lennon',
    defaults={'first_name': 'Bob'},
)
```

Bulk_create

```shell
>>> Entry.objects.bulk_create([
...     Entry(headline="Django 1.0 Released"),
...     Entry(headline="Django 1.1 Announced"),
...     Entry(headline="Breaking: Django is awesome")
... ])

# 注意：
- 将不会调用模型的save()方法，并且不会发送pre_save和post_save信号。
- 它不适用于多表继承场景中的子模型。
- 如果模型的主键是AutoField，它不会像save()那样检索和设置主键属性。
- 它不适用于多对多关系
- 它将objs转换为一个列表，如果它是一个生成器，它会完全评估objs。强制转换允许检查所有对象，以便可以首先插入具有手动设置主键的任何对象。如果要在不评估整个生成器的情况下立即批量插入对象，只要对象没有任何手动设置的主键，就可以使用此技术：
from itertools import islice

batch_size = 100
objs = (Entry(headling'Test %s' % i) for i in range(1000))
while True:
    batch = list(islice(objs, batch_size))
    if not batch:
        break
    Entry.objects.bulk_create(batch, batch_size)
```

count

```python
# Returns the total number of entries in the database.
Entry.objects.count()

# Returns the number of entries whose headline contains 'Lennon'
Entry.objects.filter(headline__contains='Lennon').count()

# count()在后台执行SELECT COUNT（*） count()，而不是将所有的记录加载到Python对象中并在结果上调用len()（除非你需要将对象加载到内存中， len()会更快）。
# 如果您想要QuerySet中的项目数量，并且还要从中检索模型实例（例如，通过迭代它），使用len（查询集）更有效，这不会像count()一样导致额外的数据库查询。
```

is_bulk

```shell
>>> Blog.objects.in_bulk([1])
{1: <Blog: Beatles Blog>}
>>> Blog.objects.in_bulk([1, 2])
{1: <Blog: Beatles Blog>, 2: <Blog: Cheddar Talk>}
>>> Blog.objects.in_bulk([])  #  传空列表会得到空字典
{}
>>> Blog.objects.in_bulk()
{1: <Blog: Beatles Blog>, 2: <Blog: Cheddar Talk>, 3: <Blog: Django Weblog>}
# field_name在2.0新增
>>> Blog.objects.in_bulk(['beatles_blog'], field_name='slug')
{'beatles_blog': <Blog: Beatles Blog>}
```

iterator

```python
# chunk_size参数在2.0添加

# QuerySet通常在内部缓存其结果，以便在重复计算是不会导致额外的查询。相反，iterator()将直接读取结果，而不在QuerySet级别执行任何缓存（内部，默认迭代器调用iterator()并高速缓存返回值）。对于返回大量只需要访问一次的对象的QuerySet，这可以带来更好的性能和显着减少内存。

# 请注意，在已经求值了的QuerySet上使用iterator()会强制它再次计算，重复查询。

# 此外，使用iterator()会导致先前的prefetch_related()调用被忽略，因为这两个优化一起没有意义。
```

latest/earliest

```python
Entry.objects.latest('pub_date')
# 2.0支持多变量
Entry.objects.latest('pub_date', '-expire_date')

# get,latest,earliest对于查询无对象时，抛出DoesNotExist错误
```

first/last

```python
p = Article.objects.order_by('title', 'pub_date').first()
# 等价于如下
try:
    p = Article.objects.order_by('title', 'pub_date')[0]
except IndexError:
    p = None
```

aggregate

```python
# 使用关键字参数指定的聚合将使用关键字参数的名称作为Annotation 的名称。匿名的参数的名称将基于聚合函数的名称和模型字段生成。复杂的聚合不可以使用匿名参数，它们必须指定一个关键字参数作为别名。
# 默认
>>> from django.db.models import Count
>>> q = Blog.objects.aggregate(Count('entry'))
{'entry__count': 16}
# 指定名称
>>> q = Blog.objects.aggregate(number_of_entries=Count('entry'))
{'number_of_entries': 16}
```

exists

```python
entry = Entry.objects.get(pk=123)
# 此方法优于下面
if some_queryset.filter(pk=entry.pk).exists():
    print("Entry contained in queryset")

if entry in some_queryset:
   print("Entry contained in QuerySet")
```

update

```shell
>>> Entry.objects.filter(pub_date__year=2010).update(comments_on=False)
# 更新多个字段
>>> Entry.objects.filter(pub_date__year=2010).update(comments_on=False, headline='This is old')
# 法立即应用，对更新的QuerySet的唯一限制是它只能更新模型主表中的列，而不是相关模型
>>> Entry.objects.update(blog__name='foo') # Won't work!
# 使用关联字段查询时可以的
>>> Entry.objects.filter(blog__id=1).update(comments_on=True)

# 如果你只是更新一个记录，不需要对模型对象做任何事情，最有效的方法是调用update()，而不是将模型对象加载到内存中
# not do this
e = Entry.objects.get(id=10)
e.comments_on = False
e.save()
# to do this
Entry.objects.filter(id=10).update(comments_on=False)

# update不执行save()，也不传输pre_save和post_save信号，若是需要调用自定义的save()方法
for e in Entry.objects.filter(pub_date__year=2010):
    e.comments_on = False
    e.save()
```

delete

```shell
>>> b = Blog.objects.get(pk=1)
# Delete all the entries belonging to this Blog.
>>> Entry.objects.filter(blog=b).delete()  # 批量删除
(4, {'weblog.Entry': 2, 'weblog.Entry_authors': 2})
>>> b.delete()  # 单独删除


# ForeignKey
# 默认情况下，Django的ForeignKey模拟SQL约束ON DELETE CASCADE字，任何具有指向要删除的对象的外键的对象将与它们一起被删除。
>>> blogs = Blog.objects.all()
# This will delete all Blogs and all of their Entry objects.
>>> blogs.delete()
(5, {'weblog.Blog': 1, 'weblog.Entry': 2, 'weblog.Entry_authors': 2})

# 此级联行为可通过ForeignKey的on_delete参数自定义。

# delete()方法执行批量删除，并且不会在模型上调用任何delete()方法。但它会为所有已删除的对象（包括级联删除）发出pre_delete和post_delete信号。若要使用自定义的delete()方法
for e in Entry.objects.filter(pub_date__year=2010):
    e.save()

# Django需要获取对象到内存中以发送信号和处理级联。然而，如果没有级联和没有信号，那么Django可以采取快速路径并删除对象而不提取到内存中。对于大型删除，这可以显着减少内存使用。执行的查询量也可以减少。

# 设置为on_delete DO_NOTHING的外键不会阻止删除快速路径。
```

### 字段查询

字段查询是指如何指定SQL `WHERE` 子句的内容。它们通过查询集方法的关键字参数指定。

基本形式

```
field__lookuptype=value
```

字段名

```
查询条件中指定的字段必须是模型字段的名称

ForeignKey在字段名加上`_id`后缀时，该参数的值应该是外键的原始值
```

字段查询参数

```python
exact  # 精确等于,若参数为None，则按照NULL进行SQL，SQL中=
iexact  # 不区分大小写的精确匹配,若参数为None，则按照NULL进行SQL，SQL中ilike
contains  # 区分大小写的包含，SQL中like '%...%'
icontains  # 不区分大小写的包含, SQL中ilike '%...%'
in  # 在给定的列表, SQL中in(...,...)
exclude  # 
gt  # 大于,SQL中>
gte  # 大于等于
lt  # 小于
lte  # 小于等于
startswith  # 区分大小写，开始位置匹配, SQL中like '...%'
istartswith  # 不区分大小写，开始位置匹配,SQL中ilike '...%'
endswith  # 以…结尾, 区分大小写，SQL中like '%...'
iendswith # 以…结尾，忽略大小写,SQL中ilike '%...'
rang	# 在…范围内, SQL中between...with...
year  # 对于日期和日期时间字段，确切的年匹配。整数年
month  # 对于日期和日期时间字段，确切的月份匹配。取整数1~12
day  # 对于日期和日期时间字段，具体到某一天的匹配。取一个整数的天数
week  # 对于日期和日期时间字段，取周数（1-52或53），即周一开始周数，第一周开始于周四或之前。
week_day # 对于日期和日期时间字段，“星期几”匹配。取整数值，表示星期几从1(星期日)到7(星期六)
quarter  # 对于日期和日期时间字段，“一年中的四分之一”匹配。允许链接其他字段查找。取1到4之间的整数值，表示一年中的四分之一。
time  # 对于datetime字段，将值转换为时间。允许链接其他字段查找。采用datetime.time值。
hour  # 对于日期时间字段，精确的小时匹配。取0和23之间的整数
minute  # 对于日期时间字段，精确的分钟匹配。取0和59之间的整数
second  # 对于datetime字段，精确的第二个匹配。取0和59之间的整数。
isnull  # 值为 True 或 False, 相当于 SQL语句IS NULL和IS NOT NULL.
search  # 一个Boolean类型的全文搜索，以全文搜索的优势。这个很像 contains ，但是由于全文索引的优势，以使它更显著的快
regex  # 区分大小写的正则表达式匹配
iregex  # 不区分大小写的正则表达式匹配
# 正则表达式语法是正在使用的数据库后端的语法。在SQLite没有内置正则表达式支持的情况下，此功能由（Python）用户定义的REGEXP函数提供，因此正则表达式语法是Python的re模块。
```

示例

```python
# exact/iexact
Entry.objects.get(id__exact=14)
Entry.objects.get(id__exact=None)  # SQL中使用is NULL
Blog.objects.get(name__iexact='beatles blog')
Blog.objects.get(name__iexact=None)  # SQL中使用is NULL
# contains/icontains
Entry.objects.get(headline__contains='Lennon')
Entry.objects.get(headline__icontains='Lennon')
# in
Entry.objects.filter(id__in=[1, 3, 4])
inner_qs = Blog.objects.filter(name__contains='Cheddar')  # 动态查询
entries = Entry.objects.filter(blog__in=inner_qs)
inner_qs = Blog.objects.filter(name__contains='Ch').values('name')  # 嵌套查询
entries = Entry.objects.filter(blog__name__in=inner_qs)
values = Blog.objects.filter(name__contains='Cheddar').values_list('pk', flat=True)  # 分步查询
entries = Entry.objects.filter(blog__in=list(values))
# gt,gte,lt,lte
Entry.objects.filter(id__gt=4)
# startwith,istartwith,endwith,iendwith
Entry.objects.filter(headline__startswith='Will')
Entry.objects.filter(headline__istartswith='will')
Entry.objects.filter(headline__endswith='cats')
Entry.objects.filter(headline__iendswith='will')
# range
import datetime
start_date = datetime.date(2005, 1, 1)
end_date = datetime.date(2005, 3, 31)
Entry.objects.filter(pub_date__range=(start_date, end_date))  # 过滤具有日期的DateTimeField不会包含最后一天的项目，因为边界被解释为“给定日期的0am”
# year,month,day,week,week_day,quarter,time,hour,minute,second
Entry.objects.filter(pub_date__year=2005)
SELECT ... WHERE pub_date BETWEEN '2005-01-01' AND '2005-12-31';  # 等价SQL
Entry.objects.filter(pub_date__month=12)
SELECT ... WHERE EXTRACT('month' FROM pub_date) = '12';  # 等价SQL
Entry.objects.filter(pub_date__day=3)  
SELECT ... WHERE EXTRACT('day' FROM pub_date) = '3';  # 等价SQL
Entry.objects.filter(pub_date__week__gte=32, pub_date__week__lte=38)
Entry.objects.filter(pub_date__week_day=2)
Entry.objects.filter(pub_date__quarter=2)
Entry.objects.filter(pub_date__time__between=(datetime.time(8), datetime.time(17)))
Event.objects.filter(timestamp__hour=23)
SELECT ... WHERE EXTRACT('hour' FROM timestamp) = '23'; # 等价SQL
Event.objects.filter(timestamp__minute=29)
SELECT ... WHERE EXTRACT('minute' FROM timestamp) = '29';  # 等价SQL
Event.objects.filter(timestamp__second=31)
SELECT ... WHERE EXTRACT('second' FROM timestamp) = '31';  # 等价SQL
# isnull
Entry.objects.filter(pub_date__isnull=True)
# search
Entry.objects.filter(headline__search="+Django -jazz Python")
SELECT ... WHERE MATCH(tablename, headline) AGAINST (+Django -jazz Python IN BOOLEAN MODE);  # 等价SQL
# regex/iregex
Entry.objects.get(title__regex=r'^(An?|The) +')
SELECT ... WHERE title REGEXP BINARY '^(An?|The) +'; -- MySQL  # 等价SQL
Entry.objects.get(title__iregex=r'^(an?|the) +')
```

### 聚合函数

```python
Avg(expression, output_field=FloatField(), filter=None, **extra)
# 返回给定expression 的平均值，其中expression 必须为数值。
# 默认的别名：<field>__avg
# 返回类型：float
Count(expression, distinct=False, filter=None, **extra)
# 返回与expression 相关的对象的个数。distinct默认False，若为True，则将只计算唯一的实例
# 默认的别名：<field>__count
# 返回类型：int
Max(expression, output_field=None, filter=None, **extra)
# 返回expression 的最大值。
# 默认的别名：<field>__max
# 返回类型：与输入字段的类型相同，如果提供则为 output_field 类型
Min(expression, output_field=None, filter=None, **extra)
# 返回expression 的最小值
# 默认的别名：<field>__min
# 返回的类型：与输入字段的类型相同，如果提供则为 output_field 类型
StdDev(expression, sample=False, filter=None, **extra)
# 返回expression 的标准差。默认情况下，StdDev 返回群体的标准差。但是，如果sample=True，返回的值将是样本的标准差。
# 默认的别名：<field>__stddev
# 返回类型：float
Sum(expression, output_field=None, filter=None, **extra)
# 计算expression 的所有值的和。
# 默认的别名：<field>__sum
# 返回类型：与输入的字段相同，如果提供则为output_field 的类型
Variance(expression, sample=False, filter=None, **extra)
# 返回expression 的方差。默认情况下，Variance 返回群体的方差。但是，如果sample=True，返回的值将是样本的方差。
# 默认的别名：<field>__variance
# 返回的类型：float

# 参数
expression  
# 引用模型字段的一个字符串，或者一个查询表达式。
output_field  
# 用来表示返回值的模型字段，它是一个可选的参数。在组合多个类型的字段时，只有在所有的字段都是相同类型的情况下，Django 才能确定output_field。否则，你必须自己提供output_field 参数。
filter
# 2.0新增，一个可选的Q对象，用于过滤聚合的行
**extra
# 这些关键字参数可以给聚合函数生成的SQL 提供额外的信息。
```

### 查询pk

为了方便，Django 提供一个查询快捷方式`pk` ，它表示“primary key” 的意思

```shell
# 精确查询
>>> Blog.objects.get(id__exact=14) # Explicit form
>>> Blog.objects.get(id=14) # __exact is implied
>>> Blog.objects.get(pk=14) # pk implies id__exact
# 与其他类型结合
# Get blogs entries with id 1, 4 and 7
>>> Blog.objects.filter(pk__in=[1,4,7])
# Get all blog entries with id > 14
>>> Blog.objects.filter(pk__gt=14)
# 在join中工作
>>> Entry.objects.filter(blog__id__exact=3) # Explicit form
>>> Entry.objects.filter(blog__id=3)        # __exact is implied
>>> Entry.objects.filter(blog__pk=3)        # __pk implies __id__exact
```

### 转义like语句中的`%,_`

与`LIKE` SQL 语句等同的字段查询（`iexact`、`contains`、`icontains`、`startswith`、`istartswith`、`endswith` 和`iendswith`）将自动转义在`LIKE` 语句中使用的两个特殊的字符 —— 百分号和下划线。（在`LIKE` 语句中，百分号通配符表示多个字符，下划线通配符表示单个字符）。

```python
# 查询
>>> Entry.objects.filter(headline__contains='%')
# django自动转义为类似如下的SQL
SELECT ... WHERE headline LIKE '%\%%';
```



### 查询表达式

#### F

`F()` 返回的实例用作查询内部对模型字段的引用。这些引用可以用于查询的filter 中来比较相同模型实例上不同字段之间值的比较。

```shell
>>> from django.db.models import F
>>> Entry.objects.filter(n_comments__gt=F('n_pingbacks'))
```

Django 支持对`F()` 对象使用加法、减法、乘法、除法、取模以及幂计算等算术操作，两个操作数可以都是常数和其它`F()` 对象。

```shell
>>> Entry.objects.filter(n_comments__gt=F('n_pingbacks') * 2)
>>> Entry.objects.filter(rating__lt=F('n_comments') + F('n_pingbacks'))
>>> from datetime import timedelta
>>> Entry.objects.filter(mod_date__gt=F('pub_date') + timedelta(days=3))
```

可以在`F()` 对象中使用双下划线标记来跨越关联关系。带有双下划线的`F()` 对象将引入任何需要的join 操作以访问关联的对象

```shell
>>> Entry.objects.filter(authors__name=F('blog__name'))
```

`F()` 对象支持`.bitand()` 和`.bitor()` 两种位操作

```shell
>>> F('somefield').bitand(16)
```

###  查询相关的类

#### Q

`Q()` 对象用于封装一组关键字参数。这些关键字参数就是“字段查询” 中所提及的那些。

`Q` 对象可以使用`&` 求和，使用`|` 操作符组求或，使用`~` 操作符取反，允许组合操作。当一个操作符在两个`Q` 对象上使用时，它产生一个新的`Q` 对象

```python
from django.db.models import Q
# 或
list = BookInfo.objects.filter(Q(bread__gt=20) | Q(pk__lt=3))
# 非
list = BookInfo.objects.filter(~Q(pk=3))
# 与
BookInfo.objects.filter(bread_gt=20,id_lt=3)
BookInfo.objects.filter(bread_gt=20).filter(id_lt=3)
BookInfo.objects.filter(Q(bread_gt=20)&(id_lt=3))
# 多个Q对象参数，逻辑关系为AND
Poll.objects.get(Q(question__startswith='Who'),Q(pub_date=date(2005, 5, 2)) | Q(pub_date=date(2005, 5, 6)))
# 混合关键字和Q对象，但是Q对象必须置前
Poll.objects.get(Q(pub_date=date(2005, 5, 2)) | Q(pub_date=date(2005, 5, 6)), question__startswith='Who')
```

#### Prefetch

通常，Prefetch() 对象能够用于控制prefetch_related( )的操作.

```python
class Prefetch(lookup, queryset=None, to_attr=None)

# lookup参数描述了跟随的关系，并且工作方式与传递给prefetch_related()的基于字符串的查找相同。
# queryset参数为给定的查找提供基本QuerySet。这对于进一步过滤预取操作或从预取关系调用select_related()很有用，因此进一步减少查询数量
# to_attr参数将预取操作的结果设置为自定义属性。当使用to_attr时，预取的结果存储在列表中。这可以提供比存储在QuerySet实例内的缓存结果的传统prefetch_related调用显着的速度改进。
```

示例

```shell
# lookup
>>> Question.objects.prefetch_related(Prefetch('choice_set')).get().choice_set.all()
[<Choice: Not much>, <Choice: The sky>, <Choice: Just hacking again>]
# This will only execute two queries regardless of the number of Question
# and Choice objects.
>>> Question.objects.prefetch_related(Prefetch('choice_set')).all()
[<Question: Question object>]

# queryset
>>> voted_choices = Choice.objects.filter(votes__gt=0)
>>> voted_choices
[<Choice: The sky>]
>>> prefetch = Prefetch('choice_set', queryset=voted_choices)
>>> Question.objects.prefetch_related(prefetch).get().choice_set.all()
[<Choice: The sky>]

# to_attr
>>> prefetch = Prefetch('choice_set', queryset=voted_choices, to_attr='voted_choices')
>>> Question.objects.prefetch_related(prefetch).get().voted_choices
[<Choice: The sky>]
>>> Question.objects.prefetch_related(prefetch).get().choice_set.all()
[<Choice: Not much>, <Choice: The sky>, <Choice: Just hacking again>]
```

#### prefetch_related_objects

在可迭代的模型实例上预取给定的查找。这在接收模型实例列表而不是QuerySet的代码中很有用;例如，从缓存中获取模型或手动实例化它们时。2.0中新增

```python
prefetch_related_objects(model_instances, *related_lookups)

# 传递一个可迭代的模型实例（必须都是同一个类）以及要预取的查找或预取对象
```

示例

```shell
>>> from django.db.models import prefetch_related_objects
>>> restaurants = fetch_top_restaurants_from_cache()  # A list of Restaurants
>>> prefetch_related_objects(restaurants, 'pizzas__toppings')
```

#### FilteredRelation

FilteredRelation与`annotate（）`一起使用，以在执行JOIN时创建ON子句。它不作用于默认关系，而是作用于注释名称（下面的示例中为pizzas_vegetarian）。2.0新增

```python
FilteredRelation(relation_name, *, condition=Q())[source]

# relation_name参数：要过滤关系的字段的名称。
# condition参数：用于控制过滤的Q对象。
```

不支持如下场景

```python
# 1.跨越关系段的条件
>>> Restaurant.objects.annotate(
...    pizzas_with_toppings_startswith_n=FilteredRelation(
...        'pizzas__toppings',
...        condition=Q(pizzas__toppings__name__startswith='n'),
...    ),
... )
Traceback (most recent call last):
...
ValueError: FilteredRelation's condition doesn't support nested relations (got 'pizzas__toppings__name__startswith').
# 2. QuerySet.only() and prefetch_related()
# 3. 从父模型继承的GenericForeignKey
```

示例

```shell
>>> from django.db.models import FilteredRelation, Q
>>> Restaurant.objects.annotate(
...    pizzas_vegetarian=FilteredRelation(
...        'pizzas', condition=Q(pizzas__vegetarian=True),
...    ),
... ).filter(pizzas_vegetarian__name__icontains='mozzarella')

# 大量数据，以下更优
>>> Restaurant.objects.filter(
...     pizzas__vegetarian=True,
...     pizzas__name__icontains='mozzarella',
... )
```





### 跨关联关系查询

查询中的关联关系，Django会在后台自动帮你处理`JOIN`。若要跨越关联关系，只需使用关联的模型字段的名称，并使用双下划线分隔，直至你想要的字段。这种跨越可以是任意的深度

```shell
>>> Entry.objects.filter(blog__name='Beatles Blog')
```

若要引用一个“反向”的关系，只需要使用该模型的小写的名称。

```shell
>>> Blog.objects.filter(entry__headline__contains='Lennon')
```

如果你在多个关联关系直接过滤而且其中某个中介模型没有满足过滤条件的值，Django 将把它当做一个空的（所有的值都为`NULL`）但是合法的对象。这意味着不会有错误引发

```python
Blog.objects.filter(entry__authors__name='Lennon')
```

isnull使用

```python
Blog.objects.filter(entry__authors__name__isnull=True)
# 返回包括author的name为空,以及author的name不为空但entry的author为空的的Blog对象。
Blog.objects.filter(entry__authors__isnull=False, entry__authors__name__isnull=True)
# 返回author的name为空，entry的author不为空的的Blog对象
```

### 跨越多值的查询

当你基于`ManyToManyField` 或反向的`ForeignKey`来过滤一个对象时，有两种不同种类的过滤器。考虑`Blog`/`Entry` 关联关系（`Blog` 和 `Entry` 是一对多的关系）。我们可能想找出headline为*“Lennon”* 并且pub_date为'2008'年的Entry。或者我们可能想查询headline为*“Lennon”* 的Entry或者pub_date为'2008'的Entry。因为实际上有和单个`Blog` 相关联的多个Entry，所以这两个查询在某些场景下都是有可能并有意义的。

`ManyToManyField`有类似的情况。例如，如果`Entry`有一个`ManyToManyField`叫做 `tags`，我们可能想找到tag 叫做*“music”* 和*“bands”* 的Entry，或者我们想找一个tag 名为*“music”* 且状态为*“public”*的Entry。

对于这两种情况，Django 有种一致的方法来处理`filter()`调用。一个`filter()` 调用中的所有参数会同时应用以过滤出满足所有要求的记录。接下来的`filter()`调用进一步限制对象集，但是对于多值关系，它们应用到与主模型关联的对象，而不是应用到前一个`filter()`调用选择出来的对象。

```python
# 假设这里有一个blog拥有一条包含'Lennon'的entries条目和一条来自2008的entries条目,但是没有一条来自2008并且包含"Lennon"的entries条目。
Blog.objects.filter(entry__headline__contains='Lennon', entry__pub_date__year=2008)  # 且，return None
Blog.objects.filter(entry__headline__contains='Lennon').filter(entry__pub_date__year=2008)  # 或，有一个blog
```

exclude与filter的实现并不同。单个`exclude()`调用中的条件不必引用同一个记录

```python
# 排除headline 中包含“Lennon”的Entry和在2008 年发布的Entry,但不是排除同时满足两个条件
Blog.objects.exclude(
    entry__headline__contains='Lennon',
    entry__pub_date__year=2008,
)
# 排除同时满足两个条件
Blog.objects.exclude(
    entry=Entry.objects.filter(
        headline__contains='Lennon',
        pub_date__year=2008,
    ),
)
```



## 对查询集求值

- 迭代

`queryset`是可迭代的，它在首次迭代查询集时执行实际的数据库查询

```python
for e in Entry.objects.all():
    print(e.headline)
```

如果您只想确定是否存在至少一个结果， 使用`exists()`更有效

- 切片

可以使用Python的数组切片语法对`QuerySet`进行切片。 对一个未求值的QuerySet进行切片操作通常返回另一个未求值的Queryset，但是如果你在切片操作时使用了“step”参数，那么Django就会执行数据库的查询，结果是返回一个列表。 切片已经求值过的`QuerySet`也会返回一个列表。

还要注意的是，即使切割的未计算的`查询集`返回另一个未计算的`查询集`，进一步修改它（例如，添加更多的过滤器，或修改排序）是不允许的，因为这不很好地转换为SQL，它也没有明确的含义。

- 序列化/缓存

如果你`Pickle`一个`查询集`，它将在Pickle 之前强制将所有的结果加载到内存中。Pickle 通常用于缓存之前，并且当缓存的查询集重新加载时，你希望结果已经存在随时准备使用（从数据库读取耗费时间，就失去了缓存的目的）。这意味着当你Unpickle`查询集`时，它包含Pickle 时的结果，而不是当前数据库中的结果。

如果此后你只想Pickle 必要的信息来从数据库重新创建`查询集`，可以Pickle`查询集`的`query` 属性。

然后你可以使用类似下面的代码重新创建原始的`查询集`（不用加载任何结果）

```python
>>> import pickle
>>> query = pickle.loads(s)     # Assuming 's' is the pickled string.
>>> qs = MyModel.objects.all()
>>> qs.query = query            # Restore the original 'query'.
```

注意：不能在不同版本的Django中使用pickles。不可用于归档的长期策略

- repr

一个`QuerySet`就等价于你使用`repr()`时的效果。 这会为你在python的交互式编译下提供方便，所以在你使用API交互的时候就会立马看到你的结果。

- len

你对`查询集`调用`len()` 时， 将对它求值。正如你期望的那样，返回一个查询结果集的长度。

注意：当求数量时，使用`count()`方法更合适

- list

对`查询集`调用`list()` 将强制对它求值

```python
entry_list = list(Entry.objects.all())
entry_id_list = list(Entry.objects.values_list('id', flat=True).order_by('id'))
```

- bool

测试一个`查询集`的布尔值，例如使用`bool()`、`or`、`and`或者`if` 语句将导致查询集的执行。如果至少有一个记录，则`查询集`为`True`，否则为`False`。

```python
if Entry.objects.filter(headline="Test"):
   print("There is at least one Entry with the headline Test")
```

注意：如果你需要知道是否存在至少一条记录（而不需要真实的对象），使用 `exists()`将更加高效。

## 比较对象

为了比较两个模型实例，只需要使用标准的Python 比较操作符，即双等于符号：`==`。在后台，它会比较两个模型主键的值。

```python
>>> some_entry == other_entry
>>> some_entry.id == other_entry.id
# 主键名无关
>>> some_obj == other_obj
>>> some_obj.name == other_obj.name
```





## ORM查询

[参考](https://blog.csdn.net/qq_34755081/article/details/82779489)

### 概述

每个模型类默认都有一个叫 objects 的类属性，它由django自动生成，类型为： `django.db.models.manager.Manager`，可以把它叫 模型管理器

查询集表示从数据库中获取的对象集合，在管理器上调用某些过滤器方法会返回查询集，查询集可以含有零个、一个或多个过滤器。

- 常用过滤器

```python
all():                 # 查询所有结果 
filter(**kwargs):      # 它包含了与所给筛选条件相匹配的对象, 多参数时为AND关系
get(**kwargs):         # 返回与所给筛选条件相匹配的对象，返回结果有且只有一个，如果符合筛选条件的对象超过一个或者没有都会抛出错误。
exclude(**kwargs):     # 它包含了与所给筛选条件不匹配的对象
values(*field):        # 返回一个ValueQuerySet——一个特殊的QuerySet，运行后得到的并不是一系列model的实例化对象，而是一个可迭代的字典序列
values_list(*field, flat=False):   # 它与values()非常相似，它返回的是一个元组序列，values返回的是一个字典序列 
order_by(*field):      # 对查询结果排序,默认升序，若是在字段前加'-',则降序
reverse():             # 对查询结果反向排序，请注意reverse()通常只能在具有已定义顺序的QuerySet上调用(在model类的Meta中指定ordering或调用order_by()方法)。
distinct(*field):            # 从返回结果中剔除重复纪录(如果你查询跨越多个表，可能在计算QuerySet时得到重复的结果。此时可以使用distinct()，注意只有在PostgreSQL中支持按字段去重。)
count():               # 返回数据库中匹配查询(QuerySet)的对象数量。
first():               # 返回第一条记录
last():                # 返回最后一条记录 
exists():              # 如果QuerySet包含数据，就返回True，否则返回False
```

获取多对象

```python
all()   		# 返回所有数据。
filter()    # 返回满足条件的数据。
exclude() 	# 返回满足条件之外的数据，相当于sql语句中where部分的not关键字。
order_by()	# 对结果进行排序。
reverse()		# 对查询结果反向排序
distinct()  # 从返回结果中剔除重复纪录
```

获取单对象

```python
get()					# 返回单个满足条件的对象
first()				# 获得第一条记录对象
last()				# 获得最后一条记录对象
count()				# 返回当前查询结果的总条数。
aggregate()		# 聚合，返回一个字典。
```

获取具体对象属性值的过滤器

```python
values()  			# 返回所有查询对象指定属性的值(字典格式)
values_list()		# 返回所有查询对象指定属性的值(元组格式)
values_list('id', flat=True)  # 返回值的列表
```

获取布尔值

```python
exists()  # 判断查询集中是否有数据，如果有则返回True，没有则返回False。
```

获取数字

```python
count()  # 返回数据库中匹配查询(QuerySet)的对象数量
```

- QuerySet方法

多级调用

```python
# 调用模型管理器的all, filter, exclude, order_by方法会产生一个QuerySet，
# 可以在QuerySet上继续调用这些方法
Employee.objects.filter(id__gt=3).order_by('-age')
```

切片

```python
# QuerySet可以作取下标操作, 注意：下标不允许为负数:
b[0]  # 取出QuerySet的第一条数据,不存在会抛出IndexError异常
# 若想获得后几条记录，可使用reverse和切片
my_queryset.reverse()[:5]
```

- 特性

惰性查询

```
创建查询集不会访问数据库，直到调用数据时，才会访问数据库，调用数据的情况包括迭代、序列化、与if合用。
```

缓存

```
第一次遍历使用了QuerySet中的所有的对象（比如通过 列表生成式 遍历了所有对象），则django会把数据缓存起来， 第2次再使用同一个QuerySet时，将会使用缓存。注意：使用索引或切片引用查询集数据，将不会缓存，每次都会查询数据库。
```

- 样例

```python
paper_list.filter(name__icontains=name)
Project.objects.get(pk=project_id)
UserPro.objects.filter(pk=system.update_user_id).last()
User.objects.filter(type=1).exclude(status=-1).exclude(employee__training_use=1)
SchoolClass.objects.filter(pk=data.cls_id, status=1).first().project_id
UserBook.objects.filter(userbookclass__status=1).count()
JmsGift.objects.filter(jms_user_id=jms_user_id).exists()
OrderDetail.objects.filter(order_id=order.id).values_list("user_book_id", flat=True))
PDFTask.objects.filter(homework_date=date).order_by("-id")
UserBookClass.objects.filter(user_book__project_id=project_id).aggregate(
        class_name=GROUP_CONCAT("cls__name", distinct=True, separator='，'))
CouponUser.objects.filter(start_time__lte=now).exclude(end_time__lte=now).aggregate(sum=Sum("num"))

project_id_list = Project.objects.filter(status=1).values_list("id", flat=True)
query = SchoolUser.objects.exclude(status=-1).filter(user_type=1)
Project.objects.filter(pk__in=project_id_list).values("id", "name")
query = query.extra(where=[
'exists (select * from yh_user_book where user_id=auth_user.id and status in (0, 1) AND project_id in (%s))'% ','.join(map(str, project_id_list))])

assess_info = assess_info.extra(select={"asses_id": "id"}).\
            values("asses_id", "again_delay_delay").last() or {}

UserBookClass.objects.filter(user_book__project__in=project_id_list,).\
            values("user_id", "user_book__project_id").\
            annotate(class_name=GROUP_CONCAT("cls__name", distinct=True, separator='<br>')).\
            values("user_id", "user_book__project_id", "class_name")

SchoolClass.objects.filter(school_id=school_id, status=1). \
            extra(where=['0 = (select count(*) from yh_user_book_class where cls_id=yh_class.id and user_type=3 and status=1 and user_id !=%s)'
            % tea_user_id]).values("id", "name", 'project_id').order_by("name", "id")
  
  
StudentScore.objects.filter(user_id__in=uids, project_id__in=project_id_list). \
            values("user_id", "project_id").annotate(max_id=Max("id")). \
            values_list("max_id", flat=True).distinct()
users.annotate(live=Case(When(last_login__gte=live_data, then=1), When(date_joined__gte=live_data,then=1), default=0, output_field=IntegerField()))
    
users.filter(Q(date_joined__gt=live_data) | Q(last_login__gt=live_data))
```

### 单表条件

```
模型类.objects.filter(模型类属性名__条件名 = 值)

```

返回QuerySet对象，包含了所有满足条件的数据。

若有多个参数，做AND处理

常见条件

```python
__gt  # 大于
__gte  # 大于等于
__lt  # 小于
__lte  # 小于等于
__exact  # 精确等于
__iexact  # 精确等于忽略大小写 ilike 'aaa'
__contains  # 包含
__startswith  # 以…开头
__istartswith  # 以…开头 忽略大小写
__endswith  # 以…结尾
__iendswith # 以…结尾，忽略大小写
__rang	# 在…范围内
__year  # 日期字段的年份
__month  # 日期字段的月份`
__day  # 日期字段的日
__in  # 在范围内
__isnull  # 判空


注意：
mysql：
date函数： date('2017-1-1')
year函数: year(hire_date)
python：
date类: date(2017,1,1)
```

eg

```python
BookInfo.objects.filter(bpub_date__gt=date(1990,1,1))
Student.objects.filter(age__gte=10)
Student.objects.filter(age__lt=10)
Student.objects.filter(age__lte=10)
BookInfo.objects.filter(id_exact=1)
BookInfo.objects.filter(btitle__contains="天")
BookInfo.objects.filter(btitle__startwith="天")
BookInfo.objects.filter(btitle__endwith="传")
BookInfo.objects.filter(bpub_date__year='1990')
BookInfo.objects.filter(bpub_date__month=11)
Student.objects.filter(age__in=[10, 20, 30])
Student.objects.filter(name__isnull=True)
```

### 外键关联

在类模型中创建关联关系

```
一对多关系，将字段定义在多的一端中
关联属性 = models.ForeignKey("一类类名")

多对多关系，将字段定义在任意一端中
关联属性 = models.ManyToManyField("关联类类名")

一对一关系，将字段定义在任意一端中
关联属性 = models.OneToOneField("关联类类名")
```

关联查询

```python
# 对象进行关联查询
1. 由一类对象查询多类对象
一类对象.多类名小写_set.all()
2. 由多类对象查询一类对象
多类对象.关联属性


# 模型类进行关联查询
1. 查询一类数据(通过多类的条件)：
一类名.objects.filter(多类名小写__多类属性名__条件名=值) 
2. 查询多类数据(通过一类的条件)：
多类名.objects.filter(关联属性__一类属性名__条件名=值)
提示：会生成内连接语句进行查询， 条件名为in,gt, isnull等
```

- 一对多

正向查找

```python
# 对象查找
# 对象.关联字段.字段
book_obj = models.Book.objects.first()  # 第一本书对象
print(book_obj.publisher)  # 得到这本书关联的出版社对象
print(book_obj.publisher.name)  # 得到出版社对象的名称
# 字段查找
# 关联字段__字段
print(models.Book.objects.values_list("publisher__name"))
```

反向查找

```python
# 对象查找
# obj.表名_set
publisher_obj = models.Publisher.objects.first()  # 找到第一个出版社对象
books = publisher_obj.book_set.all()  # 找到第一个出版社出版的所有书
titles = books.values_list("title")  # 找到第一个出版社出版的所有书的书名
# 字段查找
# 表名__字段
titles = models.Publisher.objects.values_list("book__title")
```

- 多对多

```python
# 方式一：手工指定
class NewsType(models.model):
    ntid = models.AutoField(promary_key=True)
    news_id = models.ForeignKey("NewsInfo")
    type_id = models.ForeignKey("TypeInfo")
    
class TypeInfo(models.Model):
    tid = models.AutoField(promary_key=True)
  	tname = models.CharField(max_length=20) 

class NewsInfo(models.Model):
    nid = models.AutoField(promary_key=True)
  	ntitle = models.CharField(max_length=60)
  	ncontent = models.TextField()
  	npub_date = models.DateTimeField(auto_now_add=True)
    # 指定第三张表
  	t2n= models.ManyToManyField('TypeInfo', through="NewsType") 
# 方式二：使用Django
class TypeInfo(models.Model):
  tname = models.CharField(max_length=20) #新闻类别

class NewsInfo(models.Model):
  ntitle = models.CharField(max_length=60) #新闻标题
  ncontent = models.TextField() #新闻内容
  npub_date = models.DateTimeField(auto_now_add=True) #新闻发布时间
  ntype = models.ManyToManyField('TypeInfo') #通过ManyToManyField建立TypeInfo类和NewsInfo类之间多对多的关系

```

> 关联管理器

"关联管理器"是在一对多或者多对多的关联上下文中使用的管理器。

它存在于下面两种情况

```
外键关系的反向查询
多对多关联关系
```

简单来说就是当点后面的对象 可能存在多个的时候就可以使用以下的方法。

- create

创建一个新的对象，保存对象，并将它添加到关联对象集之中，返回新创建的对象

```shell
>>> import datetime
>>> models.Author.objects.first().book_set.create(title="番茄物语", publish_date=datetime.date.today())
```

- add

把指定的model对象添加到关联对象集中

```
# 添加对象
>>> author_objs = models.Author.objects.filter(id__lt=3)
>>> models.Book.objects.first().authors.add(*author_objs)

# 添加id
>>> models.Book.objects.first().authors.add(*[1, 2])
```

- set

更新model对象的关联对象

```
>>> book_obj = models.Book.objects.first()
>>> book_obj.authors.set([2, 3])
```

- remove

从关联对象集中移除执行的model对象

```
>>> book_obj = models.Book.objects.first()
>>> book_obj.authors.remove(3)
```

- clear

从关联对象集中移除一切对象。

```
>>> book_obj = models.Book.objects.first()
>>> book_obj.authors.clear()
```

### 聚合查询

```
模型类.objects.aggregate(聚合类('模型属性'))
```

常用聚合类有：Sum, Count, Max, Min, Avg等
返回值是一个字典, 格式：` {'属性名__聚合函数': 值}`

导入内置函数

```
from django.db.models import Avg, Sum, Max, Min, Count

```

默认名称

```shell
>>> from django.db.models import Avg, Sum, Max, Min, Count
>>> models.Book.objects.all().aggregate(Avg("price"))
{'price__avg': 13.233333}

```

指定名称

```shell
>>> models.Book.objects.aggregate(average_price=Avg('price'))
{'average_price': 13.233333}

```

多个聚合

```shell
>>> models.Book.objects.all().aggregate(Avg("price"), Max("price"), Min("price"))
{'price__avg': 13.233333, 'price__max': Decimal('19.90'), 'price__min': Decimal('9.90')}

```

### 分组查询

```
annotate(args, *kwargs)
```

使用提供的聚合表达式查询对象。

表达式可以是简单的值、对模型（或任何关联模型）上的字段的引用或者聚合表达式（平均值、总和等）。

annotate()的每个参数都是一个annotation，它将添加到返回的QuerySet每个对象中。

- 示例

按照部分分组求平均工资

```shell
select dept,AVG(salary) from employee group by dept;

from django.db.models import Avg
Employee.objects.values("dept").annotate(avg=Avg("salary").values(dept, "avg")

```

连表查询的分组

```shell
select dept.name,AVG(salary) from employee inner join dept on (employee.dept_id=dept.id) group by dept_id;

from django.db.models import Avg
models.Dept.objects.annotate(avg=Avg("employee__salary")).values("name", "avg")

```

统计每一本书的作者个数

```shell
>>> book_list = models.Book.objects.all().annotate(author_num=Count("author"))
>>> for obj in book_list:
...     print(obj.author_num)
...
2
1
1

```

统计出每个出版社买的最便宜的书的价格

```shell
>>> publisher_list = models.Publisher.objects.annotate(min_price=Min("book__price"))
>>> for obj in publisher_list:
...     print(obj.min_price)
...     
9.90
19.90

# 方法二
>>> models.Book.objects.values("publisher__name").annotate(min_price=Min("price"))
<QuerySet [{'publisher__name': '沙河出版社', 'min_price': Decimal('9.90')}, {'publisher__name': '人民出版社', 'min_price': Decimal('19.90')}]>

```

统计不止一个作者的图书

```shell
>>> models.Book.objects.annotate(author_num=Count("author")).filter(author_num__gt=1)
<QuerySet [<Book: 番茄物语>]>

```

根据一本图书作者数量的多少对查询集 QuerySet进行排序

```shell
>>> models.Book.objects.annotate(author_num=Count("author")).order_by("author_num")
<QuerySet [<Book: 香蕉物语>, <Book: 橘子物语>, <Book: 番茄物语>]>

```

查询各个作者出的书的总价格

```shell
>>> models.Author.objects.annotate(sum_price=Sum("book__price")).values("name", "sum_price")
<QuerySet [{'name': '小精灵', 'sum_price': Decimal('9.90')}, {'name': '小仙女', 'sum_price': Decimal('29.80')}, {'name': '小魔女', 'sum_price': Decimal('9.90')}]>

```

### F查询

```
F('字段')

```

F() 的实例可以在查询中引用字段，来比较同一个 model 实例中两个不同字段的值。

查询评论数大于收藏数的书籍

```
from django.db.models import F
models.Book.objects.filter(commnet_num__gt=F('keep_num'))

```

Django 支持 F() 对象之间以及 F() 对象和常数之间的加减乘除和取模的操作

```
models.Book.objects.filter(commnet_num__lt=F('keep_num')*2)

```

修改操作也可以使用F函数,比如将每一本书的价格提高30元

```python
models.Book.objects.all().update(price=F("price")+30)

```

修改char字段

```shell
>>> from django.db.models.functions import Concat
>>> from django.db.models import Value
>>> models.Book.objects.all().update(title=Concat(F("title"), Value("("), Value("第一版"), Value(")")))

```

### Q查询

```
Q(条件1) 逻辑操作符 Q(条件2)
```

组合多个查询条件，可以通过&|~(not and or)对多个Q对象进行逻辑操作。同sql语句中where部分的and关键字

```python
from django.db.models import Q
# 或
list = BookInfo.objects.filter(Q(bread__gt=20) | Q(pk__lt=3))
# 非
list = BookInfo.objects.filter(~Q(pk=3))
# 与
BookInfo.objects.filter(bread_gt=20,id_lt=3)
BookInfo.objects.filter(bread_gt=20).filter(id_lt=3)
BookInfo.objects.filter(Q(bread_gt=20)&(id_lt=3))
```

查询作者名是小仙女或小魔女的

```
models.Book.objects.filter(Q(authors__name="小仙女")|Q(authors__name="小魔女"))
```

查询作者名字是小仙女并且不是2018年出版的书的书名

```
>>> models.Book.objects.filter(Q(author__name="小仙女") & ~Q(publish_date__year=2018)).values_list("title")
<QuerySet [('番茄物语',)]>
```

查询出版年份是2017或2018，书名中带物语的所有书

```
>>> models.Book.objects.filter(Q(publish_date__year=2018) | Q(publish_date__year=2017), title__icontains="物语")
<QuerySet [<Book: 番茄物语>, <Book: 香蕉物语>, <Book: 橘子物语>]>
```

### SQL

- extra

在QuerySet的基础上继续执行子语句

```python
extra(self, select=None, where=None, params=None, tables=None, order_by=None, select_params=None)

# 参数
select和select_params是一组  
where和params是一组
tables用来设置from哪个表
```

示例

```python
Entry.objects.extra(select={'new_id': "select col from sometable where othercol > %s"}, select_params=(1,))

Entry.objects.extra(where=['headline=%s'], params=['Lennon'])

Entry.objects.extra(where=["foo='a' OR bar = 'a'", "baz = 'a'"])

Entry.objects.extra(select={'new_id': "select id from tb where id > %s"}, select_params=(1,), order_by=['-nid'])


models.UserInfo.objects.extra(
                    select={'newid':'select count(1) from app01_usertype where id>%s'},
                    select_params=[1,],
                    where = ['age>%s'],
                    params=[18,],
                    order_by=['-age'],
                    tables=['app01_usertype']
                )
# 等价SQL
"""
select 
    app01_userinfo.id,
    (select count(1) from app01_usertype where id>1) as newid
from app01_userinfo,app01_usertype
where 
    app01_userinfo.age > 18
order by 
    app01_userinfo.age desc
"""

```

- cursor

纯原生sql，更高灵活度的方式执行原生SQL语句

```python
from django.db import connection, connections
cursor = connection.cursor()  # cursor = connections['default'].cursor()
cursor.execute("""SELECT * from auth_user where id = %s""", [1])
row = cursor.fetchone()

```

### API

```python
##################################################################
# PUBLIC METHODS THAT ALTER ATTRIBUTES AND RETURN A NEW QUERYSET #
##################################################################

def all(self)
    # 获取所有的数据对象

def filter(self, *args, **kwargs)
    # 条件查询
    # 条件可以是：参数，字典，Q

def exclude(self, *args, **kwargs)
    # 条件查询
    # 条件可以是：参数，字典，Q

def select_related(self, *fields)
    性能相关：表之间进行join连表操作，一次性获取关联的数据。

    总结：
    1. select_related主要针一对一和多对一关系进行优化。
    2. select_related使用SQL的JOIN语句进行优化，通过减少SQL查询的次数来进行优化、提高性能。

def prefetch_related(self, *lookups)
    性能相关：多表连表操作时速度会慢，使用其执行多次SQL查询在Python代码中实现连表操作。

    总结：
    1. 对于多对多字段（ManyToManyField）和一对多字段，可以使用prefetch_related()来进行优化。
    2. prefetch_related()的优化方式是分别查询每个表，然后用Python处理他们之间的关系。

def annotate(self, *args, **kwargs)
    # 用于实现聚合group by查询

    from django.db.models import Count, Avg, Max, Min, Sum

    v = models.UserInfo.objects.values('u_id').annotate(uid=Count('u_id'))
    # SELECT u_id, COUNT(ui) AS `uid` FROM UserInfo GROUP BY u_id

    v = models.UserInfo.objects.values('u_id').annotate(uid=Count('u_id')).filter(uid__gt=1)
    # SELECT u_id, COUNT(ui_id) AS `uid` FROM UserInfo GROUP BY u_id having count(u_id) > 1

    v = models.UserInfo.objects.values('u_id').annotate(uid=Count('u_id',distinct=True)).filter(uid__gt=1)
    # SELECT u_id, COUNT( DISTINCT ui_id) AS `uid` FROM UserInfo GROUP BY u_id having count(u_id) > 1

def distinct(self, *field_names)
    # 用于distinct去重
    models.UserInfo.objects.values('nid').distinct()
    # select distinct nid from userinfo

    注：只有在PostgreSQL中才能使用distinct进行去重

def order_by(self, *field_names)
    # 用于排序
    models.UserInfo.objects.all().order_by('-id','age')

def extra(self, select=None, where=None, params=None, tables=None, order_by=None, select_params=None)
    # 构造额外的查询条件或者映射，如：子查询

    Entry.objects.extra(select={'new_id': "select col from sometable where othercol > %s"}, select_params=(1,))
    Entry.objects.extra(where=['headline=%s'], params=['Lennon'])
    Entry.objects.extra(where=["foo='a' OR bar = 'a'", "baz = 'a'"])
    Entry.objects.extra(select={'new_id': "select id from tb where id > %s"}, select_params=(1,), order_by=['-nid'])

 def reverse(self):
    # 倒序
    models.UserInfo.objects.all().order_by('-nid').reverse()
    # 注：如果存在order_by，reverse则是倒序，如果多个排序则一一倒序


 def defer(self, *fields):
    models.UserInfo.objects.defer('username','id')
    或
    models.UserInfo.objects.filter(...).defer('username','id')
    #映射中排除某列数据

 def only(self, *fields):
    #仅取某个表中的数据
     models.UserInfo.objects.only('username','id')
     或
     models.UserInfo.objects.filter(...).only('username','id')

 def using(self, alias):
     指定使用的数据库，参数为别名（setting中的设置）


##################################################
# PUBLIC METHODS THAT RETURN A QUERYSET SUBCLASS #
##################################################

def raw(self, raw_query, params=None, translations=None, using=None):
    # 执行原生SQL
    models.UserInfo.objects.raw('select * from userinfo')

    # 如果SQL是其他表时，必须将名字设置为当前UserInfo对象的主键列名
    models.UserInfo.objects.raw('select id as nid from 其他表')

    # 为原生SQL设置参数
    models.UserInfo.objects.raw('select id as nid from userinfo where nid>%s', params=[12,])

    # 将获取的到列名转换为指定列名
    name_map = {'first': 'first_name', 'last': 'last_name', 'bd': 'birth_date', 'pk': 'id'}
    Person.objects.raw('SELECT * FROM some_other_table', translations=name_map)

    # 指定数据库
    models.UserInfo.objects.raw('select * from userinfo', using="default")

    ################### 原生SQL ###################
    from django.db import connection, connections
    cursor = connection.cursor()  # cursor = connections['default'].cursor()
    cursor.execute("""SELECT * from auth_user where id = %s""", [1])
    row = cursor.fetchone() # fetchall()/fetchmany(..)


def values(self, *fields):
    # 获取每行数据为字典格式

def values_list(self, *fields, **kwargs):
    # 获取每行数据为元祖

def dates(self, field_name, kind, order='ASC'):
    # 根据时间进行某一部分进行去重查找并截取指定内容
    # kind只能是："year"（年）, "month"（年-月）, "day"（年-月-日）
    # order只能是："ASC"  "DESC"
    # 并获取转换后的时间
        - year : 年-01-01
        - month: 年-月-01
        - day  : 年-月-日

    models.DatePlus.objects.dates('ctime','day','DESC')

def datetimes(self, field_name, kind, order='ASC', tzinfo=None):
    # 根据时间进行某一部分进行去重查找并截取指定内容，将时间转换为指定时区时间
    # kind只能是 "year", "month", "day", "hour", "minute", "second"
    # order只能是："ASC"  "DESC"
    # tzinfo时区对象
    models.DDD.objects.datetimes('ctime','hour',tzinfo=pytz.UTC)
    models.DDD.objects.datetimes('ctime','hour',tzinfo=pytz.timezone('Asia/Shanghai'))

    """
    pip3 install pytz
    import pytz
    pytz.all_timezones
    pytz.timezone(‘Asia/Shanghai’)
    """

def none(self):
    # 空QuerySet对象


####################################
# METHODS THAT DO DATABASE QUERIES #
####################################

def aggregate(self, *args, **kwargs):
   # 聚合函数，获取字典类型聚合结果
   from django.db.models import Count, Avg, Max, Min, Sum
   result = models.UserInfo.objects.aggregate(k=Count('u_id', distinct=True), n=Count('nid'))
   ===> {'k': 3, 'n': 4}

def count(self):
   # 获取个数

def get(self, *args, **kwargs):
   # 获取单个对象

def create(self, **kwargs):
   # 创建对象

def bulk_create(self, objs, batch_size=None):
    # 批量插入
    # batch_size表示一次插入的个数
    objs = [
        models.DDD(name='r11'),
        models.DDD(name='r22')
    ]
    models.DDD.objects.bulk_create(objs, 10)

def get_or_create(self, defaults=None, **kwargs):
    # 如果存在，则获取，否则，创建
    # defaults 指定创建时，其他字段的值
    obj, created = models.UserInfo.objects.get_or_create(username='root1', defaults={'email': '1111111','u_id': 2, 't_id': 2})

def update_or_create(self, defaults=None, **kwargs):
    # 如果存在，则更新，否则，创建
    # defaults 指定创建时或更新时的其他字段
    obj, created = models.UserInfo.objects.update_or_create(username='root1', defaults={'email': '1111111','u_id': 2, 't_id': 1})

def first(self):
   # 获取第一个

def last(self):
   # 获取最后一个

def in_bulk(self, id_list=None):
   # 根据主键ID进行查找
   id_list = [11,21,31]
   models.DDD.objects.in_bulk(id_list)

def delete(self):
   # 删除

def update(self, **kwargs):
    # 更新

def exists(self):
   # 是否有结果

```

## 查看ORM语句

通过代码

```python
ret = BookInfo.objects.all()
print(ret.query)
```

通过mysql

```
可以通过查看mysql的日志文件，了解Django ORM 生成出来的sql语句。

1、打开mysqld.cnf文件，打开68 69两行的注释：
sudo vi /etc/mysql/mysql.conf.d/mysqld.cnf
2、重启mysql服务
sudo service mysql restart
3、查看mysql日志文件的内容
sudo tail -f /var/log/mysql/mysql.log
tail命令: 默认会显示文件的末尾，会自动刷新显示文件最新内容。退出可按ctrl+c
```

