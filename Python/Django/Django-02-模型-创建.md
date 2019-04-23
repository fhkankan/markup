[TOC]

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

## 创建模型类

模型是你的数据的唯一的、权威的信息源。它包含你所储存数据的必要字段和行为。通常，每个模型对应数据库中唯一的一张表。

基础

```
- 每个模型都是django.db.models.Model的一个Python 子类。
- 模型的每个属性都表示为数据库中的一个字段。
- Django 提供一套自动生成的用于数据库访问的API
```

示例


```python
from django.db import models

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
我们导入from django.db import models，然后使用 models.<Foo>Field的形式使用字段。

在模型类中，定义属性，生成对应的数据库表字段
属性名 = models.字段类型(字段选项)

属性名命名限制
不能是python的保留关键字。
不允许使用连续的下划线，这是由django的查询方式决定的。
```

####  字段类型

注意： 只要修改了表字段的类型，就需要重新生成迁移文件并执行迁移操作。

| 类型                     | 说明                                                         |
| ------------------------ | ------------------------------------------------------------ |
| AutoField                | 自动增长的IntegerField，通常不用指定，不指定时Django会自动创建属性名为id的自动增长属性 |
| BooleanField             | 布尔字段，值为True或False                                    |
| NullBooleanField         | 支持Null、True、False三种值                                  |
| CharField                | 字符串，参数`max_length`(必填)表示最大字符个数               |
| TextField                | 大文本字段，一般超过4000个字符时使用                         |
| IntegerField             | 整数                                                         |
| PositiveIntegerField     | 0和正整数                                                    |
| CommaSeparatedInterField | 逗号分隔的整数字段，参数`max_length`(必填)表示最大长度       |
| DecimalField             | 使用python的`Decimal`实例表示的十进制浮点数， 参数`max_digits`(必填)表示总位数， 参数`decimal_places`(必填)表示小数位数 |
| FloatField               | 使用python的`float`实例表示的浮点数                          |
| DateField                | 使用python的`datetime.date`实例表示的日期。<br/> 参数`auto_now`表示每次保存对象时，自动设置该字段为当前时间，默认为False； 参数`auto_now_add`表示当对象第一次被创建时自动设置当前时间，默认为False; 参数auto_now_add和auto_now是相互排斥的，组合将会发生错误 |
| TimeField                | 使用python的`datetime.time`实例表示时间，参数同DateField     |
| DateTimeField            | 使用python的`datetime.datetime`实例表示的日期时间，参数同DateField |
| UUIDField                | 使用python的`UUID`类，用来存储UUID字段。使用UUID类型相对于使用具有`primary_key`参数的`AutoField`类型是一个更好的解决方案 |
| URLField                 | 一个CharField类型的URL，接收`max_length`参数，若无，则默认值200 |
| EmailField               | 邮件，一个`CharField`用来检查输入地址是否合法，它使用`EmailValidator`来验证输入合法性 |
| GenericIPAddressField    | 一个 IPv4 或 IPv6 地址, 字符串格式                           |
| FileField                | 上传文件字段, 不支持primary_key和unique                      |
| ImageField               | 继承于FileField，对上传的内容进行校验，确保是有效的图片      |

[官方更多字段类型说明](https://yiyibooks.cn/xx/django_182/ref/models/fields.html#model-field-types)

类型对照(mysql)

| 数据库字段类型 | 模型类字段类型 | python数据类型    |
| -------------- | -------------- | ----------------- |
| datetime       | DatetimeFiled  | datetime.datetime |
| date           | DateFiled      | dateteime.date    |
| decimal        | Decimal        | Decimal           |
| char(32)       | UUIDField      | UUID              |

示例

Datetime

```python
# datetime.datetime类型
datetime.datetime.now(),
# 转换为字符串：
datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# dateteime.date类型
datetime.datetime.strptime('1999-01-01', "%Y-%m-%d").date()


class Good(models.Model):
  create_time = models.DatetimeField('添加时间', auto_now_add=True, blank=True)
```

FileField和ImageField

```python
# 在模型中调用需要如下几步
1. 在你的settings文件中, 你必须要定义 MEDIA_ROOT 作为Django存储上传文件的路径(从性能上考虑，这些文件不能存在数据库中。) 定义一个 MEDIA_URL 作为基础的URL或者目录。确保这个目录可以被web server使用的账户写入。
2. 在模型中添加FileField 或 ImageField 字段, 定义 upload_to参数，内容是 MEDIA_ROOT 的子目录，用来存放上传的文件。
3. 数据库中存放的仅是这个文件的路径 （相对于MEDIA_ROOT). 你很可能会想用由Django提供的便利的url 属性。比如说, 如果你的ImageField 命名为 mug_shot, 你可以在template中用 {{ object.mug_shot.url }}获得你照片的绝对路径。
例如，如果你的 MEDIA_ROOT设定为 '/home/media'，并且 upload_to设定为 'photos/%Y/%m/%d'。 upload_to的'%Y/%m/%d'被strftime()所格式化；'%Y' 将会被格式化为一个四位数的年份, '%m' 被格式化为一个两位数的月份'%d'是两位数日份。如果你在Jan.15.2007上传了一个文件，它将被保存在/home/media/photos/2007/01/15目录下.

如果你想获得上传文件的存盘文件名，或者是文件大小，你可以分别使用 name 和 size 属性； 更多可用属性及方法信息，请参见 File 类索引 和 Managing files 主题指导.
```

UUIDField

```python
import uuid
from django.db import models

class MyUUIDModel(models.Model):
    # 这里传递给default是一个可调用的对象（即一个省略了括号的方法），而不是传递一个UUID实例给default
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # other fields
```

#### 字段选项

通过选项实现对数据库表字段的约束：

| 选项        | 默认值   | 描述                                                         | 是否要重新迁移修改表结构 |
| ----------- | -------- | ------------------------------------------------------------ | ------------------------ |
| null        | False    | 如果为True，Django将在数据库中将空值存储为NULL。对于CharFIeld和TextField避免使用，它们存储空字符串而不是null | 是                       |
| unique      | False    | True表示这个字段在表中必须有唯一值                           | 是                       |
| db_column   | 属性名称 | 字段名，如果未指定，则使用属性的名称                         | 是                       |
| db_index    | False    | 若值为True, 则在表中会为此字段创建索引                       | 是                       |
| primary_key | False    | 若为True，则该字段会成为模型的主键字段。若没有指定任何字段，则Django自动添加AutoField字段来充当主键。 | 是                       |
| default     |          | 默认值，可以时一个值或一个可调用对象(不可变),若是可调用对象,则每次创新对象时,将会调用一次 | 否                       |
| blank       | False    | 若为True，则该字段允许为空白，表单验证时将允许输入空值，为False则该字段必填。是表单数据验证范畴，null为数据库范畴 | 否                       |
| choices     |          | 是一个迭代结构(列表或元组)，由可迭代的二元组组成,用来给这个字段提供选择项 |                          |

[更多字段选项](https://yiyibooks.cn/xx/django_182/ref/models/fields.html)

示例

Choice

```python
from django.db import models

class Student(models.Model):
    FRESHMAN = 'FR'
    SOPHOMORE = 'SO'
    JUNIOR = 'JR'
    SENIOR = 'SR'
    YEAR_IN_SCHOOL_CHOICES = (
        (FRESHMAN, 'Freshman'),
        (SOPHOMORE, 'Sophomore'),
        (JUNIOR, 'Junior'),
        (SENIOR, 'Senior'),
    )
    year_in_school = models.CharField(max_length=2,
                                      choices=YEAR_IN_SCHOOL_CHOICES,
                                      default=FRESHMAN)

    def is_upperclass(self):
        return self.year_in_school in (self.JUNIOR, self.SENIOR)
```

default

```python
# 这个默认值不可以是一个可变对象（如字典，列表，等等）,因为对于所有模型的一个新的实例来说，它们指向同一个引用。或者，把他们包装为一个可调用的对象。
# 注意lambdas 函数不可作为如 default 这类可选参数的值.因为它们无法被 migrations命令序列化
def contact_default():
    return {"email": "to1@example.com"}

contact_info = JSONField("ContactInfo", default=contact_default)
```

#### 关系字段

| 类型             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| ForeignKey       | 多对一关系，需要一个位置参数：与该模型关联的类。若要创建一个递归的关系。参数为`self`。会自动创建数据库索引，可设置`db_index`为False取消 |
| ManayToManyField | 多对多关系                                                   |
| OneToOneField    | 一对一关系                                                   |

ForeignKey

```python
# 定义
from django.db import models

class Car(models.Model):
  	# 关联到一个还没有定义的模型
    manufacturer = models.ForeignKey('Manufacturer')
    # 关联一个其他应用中定义的模型
    manufacturer = models.ForeignKey('production.Manufacturer')
    # 关联到一个已经定义的模型
		manufacturer = models.ForeignKey(Manufacturer)
    
# 数据库显示
Django会在字段名上添加"_id" 来创建数据库中的列名。
```

ManayToManyField

```

```

OneToOneField

```

```



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

### 元选项









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

## 使用模型类

改配置文件中的`INSTALLED_APPS` 设置，在其中添加`models.py`所在应用的名称

```python
# settings.py
INSTALLED_APPS = (
    #...
    'myapp',
    #...
)
```

当你在INSTALLED_APPS 中添加新的应用名时，请确保运行命令`manage.py migrate`，可以事先使用`manage.pymakemigrations` 给应用生成迁移脚本。