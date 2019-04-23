[TOC]

# 模型-查询

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

## 增删改

增

```python
from book_app.models import *

# 对象方法
# 调用一个模型类对象的save方法， 就可以实现数据新增或修改，id在表中存在为修改，否则为新增。
book1 = BookInfo(btitle="倚天屠龙记", bpub_date="1990-08-23")
book1.save()

# 模型类方法
BookInfo.objects.create(btitle="倚天屠龙记", bpub_date="1990-08-23"， create_time=datetime.datetime.now())

# 批量写入数据库
Entry.objects.bulk_create([
    Entry(headline='This is a test'),
    Entry(headline='This is only a test'),
])
```

删

```python
# 调用一个模型类对象的delete方法，就可以实现数据删除，会根据id删除
BookInfo.objects.filter(pk=id).delete()

```

改

```python
# 对象方法
# 调用一个模型类对象的save方法， 就可以实现数据新增或修改，id在表中存在为修改，否则为新增。
book = BookInfo.objects.get(id=1)
book.btitle = "射雕英雄传"
book.save()

# 模型类方法
BookInfo.objects.filter(id=1).update(btitle = "射雕英雄传")
catalogs.filter(pk=key).update(is_system=1, system_num=F("system_num") + 20)

```

## 数据库事务

https://yiyibooks.cn/xx/django_182/topics/db/transactions.html

### 管理数据库事务

Django 的默认行为是运行在自动提交模式下。任何一个查询都立即被提交到数据库中，除非激活一个事务。

Django 用事务或者保存点去自动的保证复杂ORM各种查询操作的统一性,尤其是 [*delete()*](https://yiyibooks.cn/__trs__/xx/django_182/topics/db/queries.html#topics-db-queries-delete) 和[*update()*](https://yiyibooks.cn/__trs__/xx/django_182/topics/db/queries.html#topics-db-queries-update) 查询.

Django’s [`测试用例`](https://yiyibooks.cn/__trs__/xx/django_182/topics/testing/tools.html#django.test.TestCase) 也包装了事务性能原因的测试类

- http请求

```
在web上一种简单处理事务的方式是把每个请求用事务包装起来.在每个你想保存这种行为的数据库的配置文件中，设置 ATOMIC_REQUESTS值为 True，

它是这样工作的。在调用一个view里面的方法之前，django开始一个事务如果发出的响应没有问题,Django就会提交这个事务。如果在view这里产生一个异常，Django就会回滚这次事务

你可能会在你的视图代码中执行一部分提交并且回滚，通常使用atomic()context管理器.但是最后你的视图，要么是所有改变都提交执行，要么是都不提交。

缺点：
当流量增长时它会表现出较差的效率。对每个视图开启一个事务是有所耗费的。其对性能的影响依赖于应用程序对数据库的查询语句效率和数据库当前的锁竞争情况。

```

当 [`ATOMIC_REQUESTS`](https://yiyibooks.cn/__trs__/xx/django_182/ref/settings.html#std:setting-DATABASE-ATOMIC_REQUESTS)被启用后，仍然有办法来阻止视图运行一个事务操作

```
non_atomic_requests(using=None)

# 这个装饰器会否定一个由 ATOMIC_REQUESTS设定的视图:

```

示例

```python
# 它将仅工作在设定了此装饰器的视图上。
from django.db import transaction

@transaction.non_atomic_requests
def my_view(request):
    do_stuff()

@transaction.non_atomic_requests(using='other')
def my_other_view(request):
    do_stuff_on_the_other_database()

```

- Atomic

```
atomic(using=None, savepoint=True)

# 参数
using参数是数据库的名字，若没提供，默认使用"default"数据库
savepoint是True，采用如下的事务管理逻辑：
当进入到最外层的 atomic 代码块时会打开一个事务;当进入到内层atomic代码块时会创建一个保存点;当退出内部块时会释放或回滚保存点;当退出外部块时提交或回退事物。
savepoint为False来使对内层的保存点失效。
如果异常发生，若设置了savepoint，Django会在退出第一层代码块时执行回滚，否则会在最外层的代码块上执行回滚。 原子性始终会在外层事物上得到保证。这个选项仅仅用在设置保存点开销很明显时的情况下。它的缺点是打破了上述错误处理的原则。

# 特性
atomic()会将其下的一系列数据库操作视为一个整体，等待同时提交或回滚
可嵌套使用atomic()，会自动创建savepoint以允许部分提交或回滚

```

函数视图

```python
# 加装饰器
from django.db import transaction

@transaction.atomic
def viewfunc(request):
    # This code executes inside a transaction.
    do_stuff()

```

类视图

```python
# 加扩展类
from django.db import transaction

class TransactionMixin(object):
    """为视图添加事务支持的装饰器"""
    @classmethod
    def as_view(cls, *args, **kwargs):
        # super寻找调用类AddressView的下一个父类的as_view()
        view = super(TransactionMixin, cls).as_view(*args, **kwargs)

        view = transaction.atomic(view)

        return view
      
class Demo(TransactionMixin, view):
  pass

```

上下文管理

```python
import os
    
if __name__ == '__main__':
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BMS.settings")
    import django
    django.setup()
    
    import datetime
    from app01 import models
    
    try:
        from django.db import transaction
        with transaction.atomic():
            new_publisher = models.Publisher.objects.create(name="火星出版社")
            models.Book.objects.create(title="橘子物语", publish_date=datetime.date.today(), publisher_id=10)  # 指定一个不存在的出版社id
    except Exception as e:
        print(str(e))

```

嵌套使用

```python
from django.db import IntegrityError, transaction

@transaction.atomic
def viewfunc(request):
    create_parent()

    try:
        with transaction.atomic():
            generate_relationships()
    except IntegrityError:
        handle_exception()

    add_children()
```

- Savepoint

尽可能优先选择`atomic()`控制食物，它遵循数据库的相关特性且防止了非法操作，低级别的API仅仅用于自定义的事务管理场景，可与atomic混合使用

```python
from django.db import transaction


# 创建保存点
save_id = transaction.savepoint()
# 回退（回滚）到保存点
transaction.savepoint_rollback(save_id) 
# 提交保存点	
transaction.savepoint_commit(save_id) 
```

示例

```python
rom django.db import transaction

# open a transaction
@transaction.atomic
def viewfunc(request):

    a.save()
    # transaction now contains a.save()
    sid = transaction.savepoint()
    b.save()
    # transaction now contains a.save() and b.save()
    if want_to_keep_b:
        transaction.savepoint_commit(sid)
        # open transaction still contains a.save() and b.save()
    else:
        transaction.savepoint_rollback(sid)
        # open transaction now contains only a.save()
```

### 关闭事务管理

你可以在配置文件里通过设置[`AUTOCOMMIT`](https://yiyibooks.cn/__trs__/xx/django_182/ref/settings.html#std:setting-DATABASE-AUTOCOMMIT)为 `False` 完全关闭Django的事物管理。如果这样做了，Django将不能启用autocommit,也不能执行任何 commits. 你只能遵照数据库层面的规则行为。

这就需要你对每个事务执行明确的commit操作，即使由Django或第三方库创建的。因此，这最好只用于你自定义的事务控制中间件或者是一些比较奇特的场景。

## 自关联

**自关联关联属性定义：**

```
# 区域表自关联属性：特殊的一对多

关联属性 = models.ForeignKey('self')

```

举例：

```
需求： 查询出广州市的上级区域和下级区域
- 资料中提供了测试数据：area.sql
- 往数据库表插入测试数据
- 广州市的id为232
- 在python环境中，查询出广州市的上级区域和下级区域

实现步骤：
1. 添加区域模型类
class Area(models.Model):
"""区域类： 保存省份 城市 区县"""
	# 区域名称
    title = models.CharField(max_length=30)

    # 关联属性：自关联 (表示上级区域)
    parent = models.ForeignKey('self', null=True, blank=True)

    def __str__(self):
        return self.title
2. 迁移生成表
3. 插入测试数据，并查看（资料：area.sql）
4. 进入python交互环境，编写orm查询代码，查询出广州市的上级区域和下级区域
area = Area.objects.get(id=232)
parent = area.parent;
children = area.area_set.all()

```

## 自定义模型管理器

每个模型类默认都有一个 **objects** 类属性，可以把它叫 **模型管理器**。它由django自动生成，类型为 `django.db.models.manager.Manager`

可以在模型类中自定义模型管理器，自定义后, Django将不再生成默认的 **objects**。（模型类可以自定义多个管理器）

```python
class Department(models.Model):
    # 自定义模型管理器
    manager = models.Manager()
    
# 调用 Department.objects会抛出AttributeError异常，而 Department.manager.all()会返回一个包含所有Department对象的列表。

```

两种情况需要自定义管理器

```
1、修改管理器返回的原始查询集
(1)自定义模型管理器，继承Manager
(2)在模型类中应用自定义的模型管理器

2、封装增删改查的方法到模型管理器中

```

- 修改原始查询集，重写all()方法

```python
# a）打开booktest/models.py文件，定义类BookInfoManager
class BookInfoManager(models.Manager):
	"""图书管理器"""
    def all(self):
        #默认查询未删除的图书信息
        #调用父类的成员语法为：super().方法名
        return super().all().filter(isDelete=False)
        
# b)在模型类BookInfo中定义管理器
class BookInfo(models.Model):
    ...
    books = BookInfoManager() 

```

- 在管理器类中定义创建对象的方法

```python
# a）打开booktest/models.py文件，定义方法create。
class BookInfoManager(models.Manager):
    ...
    #创建模型类，接收参数为属性赋值
    def create_book(self, title, pub_date):
        #创建模型类对象self.model可以获得模型类
        book = self.model()
        book.btitle = title
        book.bpub_date = pub_date
        book.bread=0
        book.bcommet=0
        book.isDelete = False
        # 将数据插入进数据表
        book.save()
        return book
        
# b）为模型类BookInfo定义管理器books语法如下
class BookInfo(models.Model):
    ...
    books = BookInfoManager()
    
# c）调用语法如下：
book=BookInfo.books.create_book("abc",date(1980,1,1))
```

## 迁移生成

```python
# 生成数据库表
python manage.py makemigrations
python manage.py migrate
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

