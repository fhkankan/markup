# 执行原始SQL查询

在[模型查询API](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/queries.html)不够用的情况下，你可以使用原始的SQL语句。 Django 提供两种方法使用原始SQL进行查询：一种是使用[`Manager.raw()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/sql.html#django.db.models.Manager.raw)方法，[进行原始查询并返回模型实例](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/sql.html#performing-raw-queries)；另一种是完全避开模型层，[直接执行自定义的SQL语句](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/sql.html#executing-custom-sql-directly)。

> 警告
编写原始的SQL语句时，应该格外小心。 每次使用的时候，都要确保转义了`params`中任何用户可以控制的字符，以防受到SQL注入攻击。 更多信息请参阅[SQL注入防护](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/security.html#sql-injection-protection)。

## 执行原始查询

`raw()`管理器方法用于原始的SQL查询，并返回模型的实例：

`Manager.raw(raw_query, params=None, translations=None)`

这个方法执行原始的SQL查询，并返回一个`django.db.models.query.RawQuerySet` 实例。 这个`RawQuerySet` 实例可以像一般的[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet)那样，通过迭代来提供对象实例。

这里最好通过例子展示一下， 假设存在以下模型：

```python
class Person(models.Model):
    first_name = models.CharField(...)
    last_name = models.CharField(...)
    birth_date = models.DateField(...)
```

你可以像这样执行自定义的SQL语句：

```shell
>>> for p in Person.objects.raw('SELECT * FROM myapp_person'):
...     print(p)
John Smith
Jane Jones
```

当然，这个例子不是特别有趣 — 和直接使用`Person.objects.all()`的结果一模一样。 但是，`raw()` 拥有其它更强大的使用方法。

> 模型表的名称
在上面的例子中，`Person`表的名称是从哪里得到的？
通常，Django通过将模型的名称和模型的“应用标签”（你在`manage.py startapp`中使用的名称）进行关联，用一条下划线连接他们，来组合表的名称。 在这里我们假定`myapp_person`模型存在于一个叫做`myapp`的应用中，所以表就应该叫做`Person`。
更多细节请查看[`db_table`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/options.html#django.db.models.Options.db_table)选项的文档，它也可以让你自定义表的名称。

> 警告
传递给 `.raw()`方法的sql语句并没有任何检查。 django默认它会返回一个数据集，但这不是强制性的。 如果查询的结果不是数据集，则会产生一个错误。

> 警告
如果你在mysql上执行查询，注意在类型不一致的时候，mysql的静默类型强制可能导致意想不到的结果发生。 如果你在一个字符串类型的列上查询一个整数类型的值，mysql会在比较前强制把每个值的类型转成整数。 例如，如果你的表中包含值`'abc'`和`'def'`，你查询`WHERE mycolumn=0`，那么两行都会匹配。 要防止这种情况，在查询中使用值之前，要做好正确的类型转换。

> 警告
虽然`QuerySet`可以像普通的[`QuerySet`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet)一样迭代，`RawQuerySet`并没有实现可以在 `RawQuerySet`上使用的所有方法。 例如，`__bool__()`和`__len__()`在`RawQuerySet`中没有被定义，所以所有`RawQuerySet`转化为布尔值的结果都是`True`。 `RawQuerySet`中没有实现它们的原因是，在没有内部缓存的情况下会导致性能下降，而且增加内部缓存不向后兼容。

### 将查询字段映射到模型字段

`raw()`方法自动将查询字段映射到模型字段。

字段的顺序并不重要。 换句话说，下面两种查询的作用相同：

```shell
>>> Person.objects.raw('SELECT id, first_name, last_name, birth_date FROM myapp_person')
...
>>> Person.objects.raw('SELECT last_name, birth_date, first_name, id FROM myapp_person')
...
```

Django会根据名字进行匹配。 这意味着你可以使用SQL的`AS`子句来将查询中的字段映射成模型的字段。 所以如果在其他的表中有一些`Person`数据，你可以很容易地把它们映射成`Person`实例:

```shell
>>> Person.objects.raw('''SELECT first AS first_name,
...                              last AS last_name,
...                              bd AS birth_date,
...                              pk AS id,
...                       FROM some_other_table''')
```

只要名字能对应上，模型的实例就会被正确创建。

又或者，你可以在`raw()`方法中使用`translations` 参数。 这个参数是一个字典，将查询中的字段名称映射为模型中的字段名称。 例如，上面的查询可以写成这样：

```shell
>>> name_map = {'first': 'first_name', 'last': 'last_name', 'bd': 'birth_date', 'pk': 'id'}
>>> Person.objects.raw('SELECT * FROM some_other_table', translations=name_map)
```

### 索引查找

`raw()`方法支持索引访问，所以如果只需要第一条记录，可以这样写：

```shell
>>> first_person = Person.objects.raw('SELECT * FROM myapp_person')[0]
```

然而，索引和切片并不在数据库层面上进行操作。 如果数据库中有很多的`Person`对象，更加高效的方法是在SQL层面限制查询中结果的数量：

```shell
>>> first_person = Person.objects.raw('SELECT * FROM myapp_person LIMIT 1')[0]
```

### 延迟模型字段

字段也可以像这样被省略：

```shell
>>> people = Person.objects.raw('SELECT id, first_name FROM myapp_person')
```

查询返回的`Person`对象是一个延迟的模型实例（请见 [`defer()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/ref/models/querysets.html#django.db.models.query.QuerySet.defer)）。 这意味着被省略的字段，在访问时才被加载。 像这样：

```shell
>>> for p in Person.objects.raw('SELECT id, first_name FROM myapp_person'):
...     print(p.first_name, # 这将在开始的查询获取到
...           p.last_name) # 这将根据需要获取
...
John Smith
Jane Jones
```

从表面上来看，看起来这个查询获取了first_name和last_name。 然而，这个例子实际上执行了3次查询。 只有first_name字段在raw()查询中获取，last_name字符在执行打印命令时才被获取。

只有一种字段不可以被省略 — 就是主键。 Django 使用主键来识别模型的实例，所以它在每次原始查询中都必须包含。 如果你忘记包含主键的话，会抛出一个`InvalidQuery`异常。

### 添加注解

你也可以在查询中包含模型中没有定义的字段。 例如，我们可以使用[PostgreSQL 的age() 函数](https://www.postgresql.org/docs/current/static/functions-datetime.html)来获得一群人的列表，带有数据库计算出的年龄：

```shell
>>> people = Person.objects.raw('SELECT *, age(birth_date) AS age FROM myapp_person')
>>> for p in people:
...     print("%s is %s." % (p.first_name, p.age))
John is 37.
Jane is 42.
...
```

### 将参数传递给`raw()` 

如果你需要参数化的查询，可以向`raw()`方法传递`params`参数。

```shell
>>> lname = 'Doe'
>>> Person.objects.raw('SELECT * FROM myapp_person WHERE last_name = %s', [lname])
```

`params`是存放参数的列表或字典。 你可以在查询语句中使用`%s`占位符，或者对于字典使用`%(key)s`占位符（`key`替换成字典中相应的key值），无论你的数据库引擎是什么。 这些占位符将用`params` 参数中的值替换。

> 注
>
> SQLite后端不支持字典参数；使用这个后端，你必须将参数作为列表传递。

> 警告
>
> **不要在原始查询上使用字符串格式化！**
> 它类似于这种样子：
```
>>> query = 'SELECT * FROM myapp_person WHERE last_name = %s' % lname
>>> Person.objects.raw(query)
```

> 使用`params`参数可以完全防止[SQL注入攻击](https://en.wikipedia.org/wiki/SQL_injection)，它是一种普遍的漏洞，使攻击者可以向你的数据库中注入任何SQL语句。 如果你使用字符串格式化，早晚会受到SQL注入攻击。 只要你记住默认使用 `params` 参数，就可以免于攻击。

## 直接执行自定义SQL 

有时[`Manager.raw()`](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/sql.html#django.db.models.Manager.raw)方法并不十分好用，你不需要将查询结果映射成模型，或者你需要执行`DELETE`、 `INSERT`以及`UPDATE`查询。

在这些情况下，你可以直接访问数据库，完全避开模型层。

`django.db.connection`对象提供了常规数据库连接的方式。 为了使用数据库连接，先要调用`connection.cursor()`方法来获取一个游标对象 之后，调用`cursor.execute(sql, [params])`来执行sql语句，调用`cursor.fetchall()`或者`cursor.fetchone()`来返回结果行。

像这样：

```python
from django.db import connection

def my_custom_sql(self):
    with connection.cursor() as cursor:
        cursor.execute("UPDATE bar SET foo = 1 WHERE baz = %s", [self.baz])
        cursor.execute("SELECT foo FROM bar WHERE baz = %s", [self.baz])
        row = cursor.fetchone()

    return row
```

注意如果你的查询中包含百分号字符，你需要写成两个百分号字符，以便能正确传递参数：

```python
cursor.execute("SELECT foo FROM bar WHERE baz = '30%'")
cursor.execute("SELECT foo FROM bar WHERE baz = '30%%' AND id = %s", [self.id])
```

如果你使用了[多个数据库](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/multi-db.html)，你可以使用`django.db.connections`来获取针对特定数据库的连接（以及游标）对象。 `django.db.connections`是一个类似于字典的对象，允许你通过它的别名获取特定的连接。

```python
from django.db import connections
cursor = connections['my_db_alias'].cursor()
# Your code here...
```

默认情况下，Python DB API会返回不带字段的结果，这意味着你得到的是一个`list`，而不是一个`dict`。 使用下面这种方法，花费很小的性能和内存就可以返回结果为`dict`：

```python
def dictfetchall(cursor):
    "Return all rows from a cursor as a dict"
    columns = [col[0] for col in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]
```

另一个选择是使用Python标准库中的[`collections.namedtuple()`](https://docs.python.org/3/library/collections.html#collections.namedtuple)。 `namedtuple`是一个类元组对象，可以通过属性查找访问字段；它同时也是可索引的和可迭代的。 结果是不可变的，可以通过字段名称或索引访问，这可能是有用的：

```python
from collections import namedtuple

def namedtuplefetchall(cursor):
    "Return all rows from a cursor as a namedtuple"
    desc = cursor.description
    nt_result = namedtuple('Result', [col[0] for col in desc])
    return [nt_result(*row) for row in cursor.fetchall()]
```

这里有三个例子之间的区别：

```shell
>>> cursor.execute("SELECT id, parent_id FROM test LIMIT 2");
>>> cursor.fetchall()
((54360982, None), (54360880, None))

>>> cursor.execute("SELECT id, parent_id FROM test LIMIT 2");
>>> dictfetchall(cursor)
[{'parent_id': None, 'id': 54360982}, {'parent_id': None, 'id': 54360880}]

>>> cursor.execute("SELECT id, parent_id FROM test LIMIT 2");
>>> results = namedtuplefetchall(cursor)
>>> results
[Result(id=54360982, parent_id=None), Result(id=54360880, parent_id=None)]
>>> results[0].id
54360982
>>> results[0][0]
54360982
```

### Connection和cursor

`connection`和`cursor`主要实现[**PEP 249**](https://www.python.org/dev/peps/pep-0249)中描述的Python DB API标准 — 除非它涉及到[transaction handling](https://yiyibooks.cn/__trs__/xx/Django_1.11.6/topics/db/transactions.html)。

如果你不熟悉Python DB-API，注意`cursor.execute()`中的SQL语句使用占位符`"%s"`，而不是直接在SQL中添加参数。 如果你使用这种方法，底层数据库的库会在必要时自动转义你的参数。

还要注意，Django希望`"%s"`占位符，*不*是`"?"` 占位符，它用于SQLite的Python绑定。 这是为了一致和清晰。

将cursor作为上下文管理器使用:

```python
with connection.cursor() as c:
    c.execute(...)
```

等价于：

```python
c = connection.cursor()
try:
    c.execute(...)
finally:
    c.close()
```

#### 调用存储过程

`CursorWrapper.callproc(procname, params=None, kparams=None)`

用给定名称调用数据库存储过程。可以提供输入参数的序列（参数）或字典（kparams）。大多数数据库不支持kparams。在Django的内置后端中，只有Oracle支持它。

例如，给定此存储过程在Oracle数据库中：
```python
CREATE PROCEDURE "TEST_PROCEDURE"(v_i INTEGER, v_text NVARCHAR2(10)) AS
    p_i INTEGER;
    p_text NVARCHAR2(10);
BEGIN
    p_i := v_i;
    p_text := v_text;
    ...
END;
```
这个可以调用
```python
with connection.cursor() as cursor:
    cursor.callproc('test_procedure', [1, 'test'])
```
在Django 2.0中进行了更改：
kparams参数已添加。