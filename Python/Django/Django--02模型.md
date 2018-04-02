# 模型

ORM Object relational mapping 对象关系映射

- 自动生成的数据库表
- 以面向对象的方式操作数据库数据
- 通过方便的配置，切换使用不同的数据库

## 配置使用mysql数据库

django项目默认使用的是sqlite3小型数据库， 可以通过配置使用mysql数据库： 

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

3)、导入mysql包
在项目或应用的__init__.py中，
import pymysql
pymysql.install_as_MySQLdb()

4)、重新生成数据库表
删除掉应用名/migrations目录下所有的迁移文件
重新执行：
python manage.py makemigrations
python manage.py migrate

5)确认是否已经生成了对应的数据库表
```

## 字段类型和选项

```
在模型类中，定义属性，生成对应的数据库表字段：

属性名 = models.字段类型(字段选项)

属性名命名限制：
不能是python的保留关键字。
不允许使用连续的下划线，这是由django的查询方式决定的。
```

### 字段类型

```
使用时需要引入django.db.models包，字段类型如下：

AutoField：
自动增长的IntegerField，通常不需要指定，Django会自动创建属性名为id的自动增长属性。

BooleanField：
布尔字段，值为True或False

NullBooleanField:
支持Null、True、False三种值。

CharField (max_length=字符个数)：
字符串 必须指定的参数： max_length 最大字符个数

TextField：
大文本字段，一般超过4000个字符时使用。

DateField：[auto_now=False, auto_now_add=False]：
日期
参数auto_now表示每次保存对象时，自动设置该字段为当前时间，用于"最后一次修改"的时间戳，它总是使用当前日期，默认为false。
参数auto_now_add表示当对象第一次被创建时自动设置当前时间，用于创建的时间戳，它总是使用当前日期，默认为false。
参数auto_now_add和auto_now是互斥的，不能在同时使用到一个类属性中。

TimeField：
时间，参数同DateField。

DateTimeField：
日期时间，参数同DateField。

IntegerField：
整数。在Django所支持的所有数据库中， 从 -2147483648 到 2147483647 范围内的值是合法的

DecimalField (max_digits=None, decimal_places=None)：
十进制浮点数，用python中的Decimal实例来表示，适合用来保存金额。
必须指定参数： max_digits总位数，decimal_places小数位数。
例：最大值：99.99 -- DecimalField (max_digits=4, decimal_places=2)

FloatField：
浮点数，用python中的float来表示，有误差。

FileField：
上传文件字段。

ImageField：
继承于FileField，对上传的内容进行校验，确保是有效的图片。

注意： 只要修改了表字段的类型，就需要重新生成迁移文件并执行迁移操作。
```

[官方更多字段类型说明](http://python.usyiyi.cn/translate/django_182/ref/models/fields.html)

**注意： 只要修改了表字段的类型，就需要重新生成迁移文件并执行迁移操作。**

### 字段选项

通过选项实现对数据库表字段的约束：

| 选项          | 默认值   | 描述                                       | 是否要重新迁移修改表结构 |
| ----------- | ----- | ---------------------------------------- | ------------ |
| null        | False | 如果为True，数据库中字段允许为空                       | 是            |
| unique      | False | True表示这个字段在表中必须有唯一值                      | 是            |
| db_column   | 属性名称  | 字段名，如果未指定，则使用属性的名称                       | 是            |
| db_index    | False | 若值为True, 则在表中会为此字段创建索引。 查看索引：show index from 表名 | 是            |
| primary_key | False | 若为True，则该字段会成为模型的主键字段，一般作为AutoField的选项使用 | 是            |
| default     |       | 默认值                                      | 否            |
| blank       | False | True，html页面表单验证时字段允许为空                   | 否            |

## 定义模型类

```python
#定义图书模型类BookInfo
class BookInfo(models.Model):
    btitle = models.CharField(max_length=20)#图书名称
    bpub_date = models.DateField()#发布日期
    bread = models.IntegerField(default=0)#阅读量
    bcomment = models.IntegerField(default=0)#评论量
    isDelete = models.BooleanField(default=False)#逻辑删除

#定义英雄模型类HeroInfo
class HeroInfo(models.Model):
    hname = models.CharField(max_length=20)#英雄姓名
    hgender = models.BooleanField(default=True)#英雄性别
    isDelete = models.BooleanField(default=False)#逻辑删除
    hcomment = models.CharField(max_length=200)#英雄描述信息
    hbook = models.ForeignKey('BookInfo')#英雄与图书表的关系为一对多，所以属性定义在英雄模型类中
```

## 迁移生成

```python
# 生成数据库表
python manage.py makemigrations
python manage.py migrate
```

## 字段查询

- 每个模型类默认都有一个叫 **objects** 的类属性，它由django自动生成，类型为： `django.db.models.manager.Manager`，可以把它叫 **模型管理器**;

- **objects模型管理器**中提供了一些查询数据的方法： 

  | objects管理器中的方法      | 返回类型                                | 作用                                                         |
  | -------------------------- | --------------------------------------- | ------------------------------------------------------------ |
  | 模型类.objects.get()       | 模型对象                                | **返回一个对象，且只能有一个**: <br>如果查到多条数据，则报：MultipleObjectsReturned <br>如果查询不到数据，则报：DoesNotExist |
  | 模型类.objects.filter()    | QuerySet                                | 返回满足条件的对象                                           |
  | 模型类.objects.all()       | QuerySet                                | 返回所有的对象                                               |
  | 模型类.objects.exclude()   | QuerySet                                | 返回不满条件的对象                                           |
  | 模型类.objects.order_by()  | QuerySet                                | 对查询结果集进行排序                                         |
  | 模型类.objects.aggregate() | 字典，例如：<br>{'salary__avg': 9500.0} | 进行聚合操作</br>Sum, Count, Max, Min, Avg                   |
  | 模型类.objects.count()     | 数字                                    | 返回查询集中对象的数目                                       |


### filter

```python
filter方法用来实现条件查询，返回QuerySet对象，包含了所有满足条件的数据。

通过方法参数，指定查询条件： 

模型类.objects.filter(模型类属性名__条件名 = 值)

判等： exact
模糊查询： contains / endswith / startswith
空查询： isnull
范围查询: in
比较查询: gt、lt、gte、lte
日期查询： year， date类
mysql：
date函数： date('2017-1-1')
year函数: year(hire_date)
python：
date类: date(2017,1,1)
```

### exclude

```
返回不满足条件的数据：   

用法： 模型类.objects.exclude(条件)
```


### F对象

```
作用： 引用某个表字段的值, 生成对应的SQL语句, 用于两个属性的比较

用法： F('字段')

使用之前需要先导入：
from django.db.models import F
list = BookInfo.objects.filter(bread__gt=F('bcomment') * 2)
```

### Q对象

```sql
作用： 组合多个查询条件，可以通过&|~(not and or)对多个Q对象进行逻辑操作。同sql语句中where部分的and关键字

用法： Q(条件1) 逻辑操作符 Q(条件2)

需要先导入：
from django.db.models import Q
list = BookInfo.objects.filter(Q(bread__gt=20) | Q(pk__lt=3))
list = BookInfo.objects.filter(~Q(pk=3))
```

### order_by

```
作用： 对查询结果进行排序, 默认升序

用法：
升序： 模型类.objects.order_by('字段名') 
降序： 模型类.objects.order_by('-字段名')
```

### aggregate

```python
作用： 聚合操作，对多行查询结果中的一列进行操作，返回一个值。

用法： 模型类.objects.aggregate（聚合类（'模型属性'））

常用聚合类有：Sum, Count, Max, Min, Avg等
返回值是一个字典, 格式： {'属性名__聚合函数': 值}

需先导入聚合类：
from django.db.models import Sum, Count, Max, Min, Avg
list = BookInfo.objects.aggregate(Sum('bread'))
```

### count方法

```
作用：统计满足条件的对象的个数，返回值是一个数字

用法： 模型类.objects.count()
```

## 查询集

查询集表示从数据库中获取的对象集合，在管理器上调用某些过滤器方法会返回查询集，查询集可以含有零个、一个或多个过滤器。过滤器基于所给的参数限制查询的结果，从Sql的角度，查询集和select语句等价，过滤器像where和limit子句。

- 返回查询集的过滤器

```
all()：返回所有数据。
filter()：返回满足条件的数据。
exclude()：返回满足条件之外的数据，相当于sql语句中where部分的not关键字。
order_by()：对结果进行排序。
```

- 返回单个值的过滤器

```
get()：返回单个满足条件的对象
	如果未找到会引发"模型类.DoesNotExist"异常。
	如果多条被返回，会引发"模型类.MultipleObjectsReturned"异常。
count()：返回当前查询结果的总条数。
aggregate()：聚合，返回一个字典。
```

- 判断一个查询集中是否有数据

```
exists()：判断查询集中是否有数据，如果有则返回True，没有则返回False。
```

- 方法

```
调用模型管理器的all, filter, exclude, order_by方法会产生一个QuerySet，可以在QuerySet上继续调用这些方法，比如：

Employee.objects.filter(id__gt=3).order_by('-age')
QuerySet可以作取下标操作, 注意：下标不允许为负数:
b[0]
取出QuerySet的第一条数据,
不存在会抛出IndexError异常

QuerySet可以作切片 操作, 切片操作会产生一个新的QuerySet，注意：下标不允许为负数。
如果获取一个对象，直接使用[0]，等同于[0:1].get()，但是如果没有数据，[0]引发IndexError异常，[0:1].get()如果没有数据引发DoesNotExist异常。
list=BookInfo.objects.all()[0:2]



# QuerySet的方法
QuerySet的get()方法
取出QuerySet的唯一一条数据
QuerySet不存在数据，会抛出： DoesNotExist异常
QuerySet存在多条数据，会抛出：MultiObjectsReturned异常
```

- 特性

```
惰性查询：创建查询集不会访问数据库，直到调用数据时，才会访问数据库，调用数据的情况包括迭代、序列化、与if合用。

缓存：第一次遍历使用了QuerySet中的所有的对象（比如通过 列表生成式 遍历了所有对象），则django会把数据缓存起来， 第2次再使用同一个QuerySet时，将会使用缓存。注意：使用索引或切片引用查询集数据，将不会缓存，每次都会查询数据库。
```

## 查看ORM语句

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

## 增删改

```
调用一个模型类对象的save方法， 就可以实现数据新增或修改，id在表中存在为修改，否则为新增。

调用一个模型类对象的delete方法，就可以实现数据删除，会根据id删除
```

## 模型类关系

### 模型类关系

```
在类模型中创建关联关系

一对多关系，将字段定义在多的一端中
关联属性 = models.ForeignKey("一类类名")

多对多关系，将字段定义在任意一端中
关联属性 = models.ManyToManyField("关联类类名")

一对一关系，将字段定义在任意一端中
关联属性 = models.OneToOneField("关联类类名")
```

- eg

```python
# 一对多
#定义图书模型类BookInfo
class BookInfo(models.Model):
    btitle = models.CharField(max_length=20)#图书名称
    bpub_date = models.DateField()#发布日期
    bread = models.IntegerField(default=0)#阅读量
    bcomment = models.IntegerField(default=0)#评论量
    isDelete = models.BooleanField(default=False)#逻辑删除
#定义英雄模型类HeroInfo
class HeroInfo(models.Model):
    hname = models.CharField(max_length=20)#英雄姓名
    hgender = models.BooleanField(default=True)#英雄性别
    isDelete = models.BooleanField(default=False)#逻辑删除
    hcomment = models.CharField(max_length=200)#英雄描述信息
    hbook = models.ForeignKey('BookInfo')#英雄与图书表的关系为一对多，所以属性定义在英雄模型类中

 
# 多对多
class TypeInfo(models.Model):
  tname = models.CharField(max_length=20) #新闻类别

class NewsInfo(models.Model):
  ntitle = models.CharField(max_length=60) #新闻标题
  ncontent = models.TextField() #新闻内容
  npub_date = models.DateTimeField(auto_now_add=True) #新闻发布时间
  ntype = models.ManyToManyField('TypeInfo') #通过ManyToManyField建立TypeInfo类和NewsInfo类之间多对多的关系
```

### 关联查询

```
一、通过对象进行关联查询
用法：
由一类对象查询多类对象：
一类对象.多类名小写_set.all()

由多类对象查询一类对象：
多类对象.关联属性


二、通过模型类进行关联查询
用法：
通过多类的条件查询一类数据：
一类名.objects.filter(多类名小写__多类属性名__条件名=值) 

通过一类的条件查询多类数据：
多类名.objects.filter(关联属性__一类属性名__条件名=值)
提示：会生成内连接语句进行查询， 条件名为in,gt, isnull等
```

- eg

```python
# 对象关联查询
b = BookInfo.objects.get(id=1)
b.heroinfo_set.all()

h = HeroInfo.objects.get(id=1)
h.hbook

# 模型类关联查询
list = BookInfo.objects.filter(heroinfo__hcontent__contains='八')
list = HeroInfo.objects.filter(hbook__btitle='天龙八部')
```

### 自关联

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

## 模型类实例方法

```
str()：在将对象转换成字符串时会被调用。

save()：将模型对象保存到数据表中，ORM框架会转换成对应的insert或update语句。

delete()：将模型对象从数据表中删除，ORM框架会转换成对应的delete语句。
```

## 模型类属性

```
属性objects：管理器，是models.Manager类型的对象，用于与数据库进行交互。

当没有为模型类定义管理器时，Django会为每一个模型类生成一个名为objects的管理器，自定义管理器后，Django不再生成默认管理器objects。

model属性： 在管理器中，可以通过self.model属性，获取管理器所属的模型类，通过self.model()则可以创建模型类对象
```

## 自定义模型管理器

- 每个模型类默认都有一个 **objects** 类属性，可以把它叫 **模型管理器**。它由django自动生成，类型为 `django.db.models.manager.Manager`


- 可以在模型类中自定义模型管理器，自定义后, Django将不再生成默认的 **objects**。（模型类可以自定义多个管理器）

  例如：

  	class Department(models.Model):
  	    # 自定义模型管理器
  	    manager = models.Manager()
  	    
  	调用 Department.objects会抛出AttributeError异常，而 Department.manager.all()会返回一个包含所有Department对象的列表。

- 两种情况需要自定义管理器

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

## 元选项

- Django默认生成的表名：

   应用名小写_模型类名小写

- 可以通过在模型类中定义Meta类来修改表名：

    	class Department(models.Model):    
    		"""部门类"""
    		name = models.CharField(max_length=20)
    		class Meta(object):
    	    	"""指定表名"""
    	        db_table = "department"

   需重新生成迁移文件，并进行生成表