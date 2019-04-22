# 模型

ORM Object relational mapping 对象关系映射

- 自动生成的数据库表
- 以面向对象的方式操作数据库数据
- 通过方便的配置，切换使用不同的数据库

## 配置使用数据库

### 创建空白数据库(mysql)

操作流程 

```python
1、手动生成mysql数据库
mysql –uroot –p 
show databases;
create database db_django01 charset=utf8;

2、在Django中配置mysql
1)、修改setting.py中的DATABASES
	# Project01/setting.py
DATABASES = {
    'default': {
        # 'ENGINE': 'django.db.backends.sqlite3',
        # 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),

        # 配置mysql数据库
        'ENGINE': 'django.db.backends.mysql',
        'NAME': "db_django01",
        'USER': "root",
        'PASSWORD': "mysql",
        'HOST': "localhost",
        'PORT': 3306,
    }
}

2)、在python虚拟环境下安装mysqlPython包:
pip install mysql-python 	# python2
pip install pymysql			# python3
pip install mysqlclient # python2、3

3)、导入mysql包
在项目或应用的__init__.py中，
import pymysql
pymysql.install_as_MySQLdb()

4)、编写新的modle.py

5)、重新生成数据库表
删除掉应用名/migrations目录下所有的迁移文件
重新执行：
python manage.py makemigrations
python manage.py migrate

3、确认是否已经生成了对应的数据库表
```

### 连接旧有数据库(mysql)

- 自动生成模型类

操作流程

```python
1.在django中配置数据信息
1)、修改setting.py
DATABASES = {
    'default': {
        # 'ENGINE': 'django.db.backends.sqlite3',
        # 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
        # 配置mysql数据库
        'ENGINE': 'django.db.backends.mysql',
        'NAME': "db_django01",
        'USER': "root",
        'PASSWORD': "mysql",
        'HOST': "localhost",
        'PORT': 3306,
    }
}

2)、检查数据库表信息
python manage.py inspectdb

3)、生成modle.py
python manage.py inspectdb > myapp/models.py

4)、修改model.py信息
managed = False  # 表示django不对该表进行创建、修改和删除
managed = True  # 默认状态，django的migrate表记录model中类的改动变化，执行makemigrations和migrate将改动应用到数据库表

5)、安装核心django表
python manage.py migrate

2.检查数据库表与模型之间的对应关系
```

清理生成的Models

```
- 数据库的每一个表都会被转化为一个model类。这意味着你需要为多对多连接表重构其models为ManyToManyField的对象。所生成的每一个model中的每个字段都拥有自己的属性，包括id主键字段。
- 如果某个model没有主键的时候，那么Django会为其自动增加一个id主键字段。你或许想移除这行代码因为这样不仅是冗余的码而且如果当你的应用需要向这些表中增加新纪录时，会导致某些问题。
- 每一个字段都是通过查找数据库列类型来确定的。取过inspectdb无法把某个数据库字段映射导model字段上，它会使用TextField字段进行代替，并且会在所生成的model字段后面加入注释“该字段类型是猜的”。
- 如果你的数据库中的某个字段在Django中找不到合适的对应物，你可以忽略它，因为Django模型层不要求导入数据表中的每个列。
- 如果数据库中某个列的名字是P与桃红的保留字， inspectdb会在每个属性名后加上_field，并将db_column属性设置为真实的字段名。
- 如果数据库中的某张表引用了其他表，就像外键和多键，需要是党的四ugai所生成model的顺序，以使得这种引用能够正确映射。
- 对于PostgreSQL,MySQL和SQLite数据库系统，insoectdb能够自动检测出主键关系。也就是说，它会在合适的位置插入primary_key=True，而对于其他数据库系统，你必须为每个model中至少一个字段插入这样的语句。因为这个主键字段是必须有的。
- 外键检测仅对PostgreSQL,还有MySQL表中的某些特定类型生效。 至于其他数据库,外键字段将在假定其为INT列的情况下被自动生成为IntegerField。
```

- 手动写模型类

操作流程

```python
1.在django中配置数据信息
1)、修改setting.py
DATABASES = {
    'default': {
        # 'ENGINE': 'django.db.backends.sqlite3',
        # 'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
        # 配置mysql数据库
        'ENGINE': 'django.db.backends.mysql',
        'NAME': "db_django01",
        'USER': "root",
        'PASSWORD': "mysql",
        'HOST': "localhost",
        'PORT': 3306,
    }
}

2. 手动写相应的模型类
# 注意：字段名和表名要和数据库中一致
from django.db import models

class ReportDownload(models.Model):
    TYPE_CHOICES = (
        ('pay_jms', '加盟奖励确认'),
        ('pay_bonus', '加盟奖励确认'),
        ('pay_stu', '学生缴费明细'),
        ('pay_order', '加盟商订单查询')
    )
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    type = models.CharField(choices=TYPE_CHOICES, max_length=20, default='')
    url = models.CharField('文件下载地址（绝对地址）', max_length=200)
    add_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "report_download"
```

## 模型类

```python
#定义图书模型类BookInfo
class BookInfo(models.Model):
    btitle = models.CharField(max_length=20)#图书名称
    bpub_date = models.DateField()#发布日期
    bread = models.IntegerField(default=0)#阅读量
    bcomment = models.IntegerField(default=0)#评论量
    isDelete = models.BooleanField(default=False)#逻辑删除
    class Meta:
      db_table = 'book_info'  # 表名
      
      
#定义英雄模型类HeroInfo
class HeroInfo(models.Model):
    hname = models.CharField(max_length=20)#英雄姓名
    hgender = models.BooleanField(default=True)#英雄性别
    isDelete = models.BooleanField(default=False)#逻辑删除
    hcomment = models.CharField(max_length=200)#英雄描述信息
    hbook = models.ForeignKey('BookInfo')#英雄与图书表的关系为一对多，所以属性定义在英雄模型类中
    class Meta:
      db_table = 'hero_info'  # 表名
```

### 字段

```
在模型类中，定义属性，生成对应的数据库表字段
属性名 = models.字段类型(字段选项)

属性名命名限制
不能是python的保留关键字。
不允许使用连续的下划线，这是由django的查询方式决定的。
```

字段类型

| 类型             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| AutoField        | 自动增长的IntegerField，通常不用指定，不指定时Django会自动创建属性名为id的自动增长属性 |
| BooleanField     | 布尔字段，值为True或False                                    |
| NullBooleanField | 支持Null、True、False三种值                                  |
| CharField        | 字符串，参数max_length表示最大字符个数                       |
| TextField        | 大文本字段，一般超过4000个字符时使用                         |
| IntegerField     | 整数                                                         |
| DecimalField     | 十进制浮点数， 参数max_digits表示总位数， 参数decimal_places表示小数位数 |
| FloatField       | 浮点数                                                       |
| DateField        | 日期， 参数auto_now表示每次保存对象时，自动设置该字段为当前时间，用于"最后一次修改"的时间戳，它总是使用当前日期，默认为False； 参数auto_now_add表示当对象第一次被创建时自动设置当前时间，用于创建的时间戳，它总是使用当前日期，默认为False; 参数auto_now_add和auto_now是相互排斥的，组合将会发生错误 |
| TimeField        | 时间，参数同DateField                                        |
| DateTimeField    | 日期时间，参数同DateField                                    |
| FileField        | 上传文件字段                                                 |
| ImageField       | 继承于FileField，对上传的内容进行校验，确保是有效的图片      |

[官方更多字段类型说明](http://python.usyiyi.cn/translate/django_182/ref/models/fields.html)

类型对照(mysql)

| 数据库字段类型 | 模型类字段类型 | python数据类型    |
| -------------- | -------------- | ----------------- |
| datetime       | DatetimeFiled  | datetime.datetime |
| date           | DateFiled      | dateteime.date    |
| decimal(11,2)  | Decimal()      | Int,Decimal       |

例子

```python
# datetime.datetime类型
datetime.datetime.now(),
# 转换为字符串：
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# dateteime.date类型
datetime.datetime.strptime('1999-01-01', "%Y-%m-%d").date()
```

**注意： 只要修改了表字段的类型，就需要重新生成迁移文件并执行迁移操作。**

字段选项

通过选项实现对数据库表字段的约束：

| 选项        | 默认值   | 描述                                                         | 是否要重新迁移修改表结构 |
| ----------- | -------- | ------------------------------------------------------------ | ------------------------ |
| null        | False    | 如果为True，数据库中字段允许为空                             | 是                       |
| unique      | False    | True表示这个字段在表中必须有唯一值                           | 是                       |
| db_column   | 属性名称 | 字段名，如果未指定，则使用属性的名称                         | 是                       |
| db_index    | False    | 若值为True, 则在表中会为此字段创建索引。 查看索引：show index from 表名 | 是                       |
| primary_key | False    | s若为True，则该字段会成为模型的主键字段，一般作为AutoField的选项使用 | 是                       |
| default     |          | 默认值                                                       | 否                       |
| blank       | False    | True，html页面表单验证时字段允许为空                         | 否                       |

**null是数据库范畴的概念，blank是表单验证范畴的**

### 表

模型类如果未指明表名，Django默认以 `应用名小写_模型类名小写` 为数据库表名。

可以通过Meta类来指定数据库表名。若是迁移的，需重新生成迁移文件，并进行生成表

Meta类主要处理的是关于模型的各种元数据的使用和显示。

如：对象的名显示，查询数据库表的默认排序顺序，数据表的名字

```python
class Department(models.Model):    
		"""部门类"""
		name = models.CharField(max_length=20)
		class Meta(object):
	    	"""指定表名"""
	        db_table = "department"
```

### 键

- 主键

django会为表创建自动增长的主键列，每个模型只能有一个主键列，如果使用选项设置某属性为主键列后django不会再创建自动增长的主键列。

默认创建的主键列属性为id，可以使用pk代替，pk全拼为primary key。

- 外键

在设置外键时，需要通过`on_delete`选项指明主表删除数据时，对于外键引用表数据如何处理，在django.db.models中包含了可选常量：

| name        | desc                                                         |
| ----------- | ------------------------------------------------------------ |
| DO_NOTHING  | 不做任何操作，如果数据库前置指明级联性，此选项会抛出IntegrityError异常 |
| CASCADE     | 级联，删除主表数据时连同一起删除外键表中数据                 |
| PROTECT     | 保护，通过抛出ProtectedError异常，来阻止删除主表中被外键应用的数据 |
| SET_NULL    | 设置为NULL，仅在该字段null=True允许为null时可用              |
| SET_DEFAULT | 设置为默认值，仅在该字段设置了默认值时可用                   |
| SET()       | 设置为特定值或者调用特定方法                                 |

`set()`方法

```python
from django.conf import settings
from django.contrib.auth import get_user_model
from django.db import models

def get_sentinel_user():
    return get_user_model().objects.get_or_create(username='deleted')[0]

class MyModel(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET(get_sentinel_user),
    )
```

### Admin选项

注册模型和自定义显示

```python
# app01/admin.py:
from django.contrib import admin
from app01.models import Department, Employee


class DepartmentAdmin(admin.ModelAdmin):
	# 指定后台网页要显示的字段
	list_display = ["id", "name", "create_date"]

class EmployeeAdmin(admin.ModelAdmin):
    # 指定后台网页要显示的字段
    list_display = ["id", "name", "age", "sex", "comment"]
    
# 注册Model类
admin.site.register(Department, DepartmentAdmin)
admin.site.register(Employee, EmployeeAdmin)
```

ModelAdmin选项中的类型

```
# 列表格式化
list_display:显示在列表试图里的变量
list_display_links:激活变量查找和过滤链接
list_filter:

# 表单显示
fields:重写模型里默认表单表现形式
js:添加js
save_on_top:
```

### 方法属性

方法

```python
str()			# 在将对象转换成字符串时会被调用。

save()		# 将模型对象保存到数据表中，ORM框架会转换成对应的insert或update语句。

delete()	# 将模型对象从数据表中删除，ORM框架会转换成对应的delete语句。
```

属性

```python
objects		# 管理器，是models.Manager类型的对象，用于与数据库进行交互。
					# 当没有为模型类定义管理器时，Django会为每一个模型类生成一个名为objects的管理器，自定义管理器后，Django不再生成默认管理器objects。

model  		# 在管理器中，可以通过self.model属性，获取管理器所属的模型类，通过self.model()则可以创建模型类对象
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

````python
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

````

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

事务

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

## 自关联

**自关联关联属性定义：**

    # 区域表自关联属性：特殊的一对多
    
    关联属性 = models.ForeignKey('self')

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

