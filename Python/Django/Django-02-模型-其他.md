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

## 