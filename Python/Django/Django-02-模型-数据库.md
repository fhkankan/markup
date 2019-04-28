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

## 