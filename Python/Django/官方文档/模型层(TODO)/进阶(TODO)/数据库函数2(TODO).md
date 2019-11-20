# 数据库函数2

## 文本函数

### Chr

###  Concat

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

### Left 

### Length

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

### Lower

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

### LPad

### LTrim

### Ord

### Repeat

### Replace

### Right

### RPad

### RTrim

### StrIndex

### Substr

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

### Trim

### Upper

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

## 窗口函数

### CumeDist

### DenseRank

### FirstValue

### Lag

### LastValue

### Lead

### NthValue

### Ntile

### PercentRank

### Rank

### RowNumber
