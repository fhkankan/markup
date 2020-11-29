# SQLALchemy

## 安装

**安装数据库 **

```shell
# 安装服务端
sudo apt-get install mysql-server

# 安装客户端
sudo apt-get install mysql-client
sudo apt-get install libmysqlclient-dev

# 数据库基本命令
mysql -u root -p
create database <数据库名> charset=utf8;
show databases;
desc create table 数据表名;
```

**安装flask-sqlalchemy**

```shell
pip install flask-sqlalchemy
pip install flask-mysqlclient
```

## 字段

**字段类型**

| 类型名       | python中类型      | 说明                                                |
| ------------ | ----------------- | --------------------------------------------------- |
| Integer      | int               | 普通整数，一般是32位                                |
| SmallInteger | int               | 取值范围小的整数，一般是16位                        |
| BigInteger   | int或long         | 不限制精度的整数                                    |
| Float        | float             | 浮点数                                              |
| Numeric      | decimal.Decimal   | 普通整数，一般是32位                                |
| String       | str               | 变长字符串                                          |
| Text         | str               | 变长字符串，对较长或不限长度的字符串做了优化        |
| Unicode      | unicode           | 变长Unicode字符串                                   |
| UnicodeText  | unicode           | 变长Unicode字符串，对较长或不限长度的字符串做了优化 |
| Boolean      | bool              | 布尔值                                              |
| Date         | datetime.date     | 时间                                                |
| Time         | datetime.datetime | 日期和时间                                          |
| LargeBinary  | str               | 二进制文件                                          |

**列选项**

| 选项名      | 说明                                              |
| ----------- | ------------------------------------------------- |
| primary_key | 如果为True，代表表的主键                          |
| unique      | 如果为True，代表这列不允许出现重复的值            |
| index       | 如果为True，为这列创建索引，提高查询效率          |
| nullable    | 如果为True，允许有空值，如果为False，不允许有空值 |
| default     | 为这列定义默认值                                  |

**关系选项**

| 选项名         | 说明                                                         |
| -------------- | ------------------------------------------------------------ |
| backref        | 在关系的另一模型中添加反向引用                               |
| primary join   | 明确指定两个模型之间使用的联结条件                           |
| uselist        | 如果为False，不使用列表，而使用标量值                        |
| order_by       | 指定关系中记录的排序方式                                     |
| secondary      | 指定多对多中记录的排序方式                                   |
| secondary join | 在SQLAlchemy中无法自行决定时，指定多对多关系中的二级联结条件 |

## 构建映射

### 配置信息

- 示例

```python
from flask import Flask

app = Flask(__name__)

# 方法一：以类的方式
class Config(object):
    SQLALCHEMY_DATABASE_URI = 'mysql://root:mysql@127.0.0.1:3306/toutiao'
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # 在Flask中是否追踪数据修改
    SQLALCHEMY_ECHO = True  # 显示生成的SQL语句，可用于调试

app.config.from_object(Config)

# 方法二：字典形式
# 设置连接数据库的URL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:mysql@127.0.0.1:3306/Flask_test'
# 设置每次请求结束后会自动提交数据库中的改动（后期会去除）
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# 数据库变更追踪，可以设定为False
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True
```

- 配置项

`SQLALCHEMY_DATABASE_URI` 数据库的连接信息

```python
# Postgres
postgresql://user:password@localhost/mydatabase
# MySQL
mysql://user:password@localhost/mydatabase
# Oracle
oracle://user:password@127.0.0.1:1521/sidname
# SQLite
sqlite:////absolute/path/to/foo.db
```

其他配置参考如下：

| 名字                      | 备注                                                         |
| :------------------------ | :----------------------------------------------------------- |
| SQLALCHEMY_DATABASE_URI   | 用于连接的数据库 URI 。                                      |
| SQLALCHEMY_BINDS          | 一个映射 binds 到连接 URI 的字典。更多 binds 的信息见[*用 Binds 操作多个数据库*](http://docs.jinkan.org/docs/flask-sqlalchemy/binds.html#binds)。 |
| SQLALCHEMY_ECHO           | 如果设置为Ture， SQLAlchemy 会记录所有 发给 stderr 的语句，这对调试有用。(打印sql语句) |
| SQLALCHEMY_RECORD_QUERIES | 可以用于显式地禁用或启用查询记录。查询记录 在调试或测试模式自动启用。更多信息见get_debug_queries()。 |
| SQLALCHEMY_NATIVE_UNICODE | 可以用于显式禁用原生 unicode 支持。当使用 不合适的指定无编码的数据库默认值时，这对于 一些数据库适配器是必须的（比如 Ubuntu 上 某些版本的 PostgreSQL ）。 |
| SQLALCHEMY_POOL_SIZE      | 数据库连接池的大小。默认是引擎默认值（通常 是 5 ）           |
| SQLALCHEMY_POOL_TIMEOUT   | 设定连接池的连接超时时间。默认是 10 。                       |
| SQLALCHEMY_POOL_RECYCLE   | 多少秒后自动回收连接。这对 MySQL 是必要的， 它默认移除闲置多于 8 小时的连接。注意如果 使用了 MySQL ， Flask-SQLALchemy 自动设定 这个值为 2 小时。 |

### 创建对象

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)

# 设置连接数据库的URL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:mysql@127.0.0.1:3306/Flask_test'
# 设置每次请求结束后会自动提交数据库中的改动（后期会去除）
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# 数据库变更追踪，可以设定为False
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True

# 实例化SQLAlchemy对象
# 方法一
db = SQLAlchemy(app)
# 方法二
db = SQLAlchemy()
db.init_app(app)
```

对于创建类的方法二，注意此方式在单独运行调试时，对数据库操作需要在Flask的应用上下文中进行，即

```python
with app.app_context():
    User.query.all()
```

###模型类映射

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy


app = Flask(__name__)

# 设置连接数据库的URL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:mysql@127.0.0.1:3306/Flask_test'
# 设置每次请求结束后会自动提交数据库中的改动（后期会去除）
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# 数据库变更追踪，可以设定为False
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True

# 实例化SQLAlchemy对象
db = SQLAlchemy(app)


class Role(db.Model):
    # 定义表名，若不写，则默认创建为类名的小写格式
    __tablename__ = 'roles'
    # 定义映射对象，若列对象名字和数据库列名字一致，可省略
    id = db.Column('id', db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, doc="角色名称")
    # 创建一对多的外键，在数据库中无实体，第一个参数为对应的类，第二个关键字参数值一般写为对象名小写(可任意)
    us = db.relationship('User', backref='role')

    #repr()方法显示一个可读字符串
    def __repr__(self):
        return 'Role:%s'% self.name

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, index=True)
    email = db.Column(db.String(64),unique=True)
    pswd = db.Column(db.String(64))
    # 多对一的外键，第二个参数为表名.主键
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))

    def __repr__(self):
        return 'User:%s'%self.name
        
if __name__ == '__main__':
	# 删除表
    db.drop_all()
    # 创建表
    db.create_all()
    # 为类对象赋值
    ro1 = Role(name='admin')
    ro2 = Role(name='user')
    # 添加多条数据到会话
    db.session.add_all([ro1,ro2])
    # 提交数据到数据库
    db.session.commit()
    us1 = User(name='wang',email='wang@163.com',pswd='123456',role_id=ro1.id)
    us2 = User(name='zhang',email='zhang@189.com',pswd='201512',role_id=ro2.id)
    us3 = User(name='chen',email='chen@126.com',pswd='987654',role_id=ro2.id)
    us4 = User(name='zhou',email='zhou@163.com',pswd='456789',role_id=ro1.id)
    db.session.add_all([us1,us2,us3,us4])
    db.session.commit()
    app.run(debug=True)
```

## 操作

在Flask-SQLAlchemy中，插入、修改、删除操作，均由数据库会话管理。会话用db.session表示。在准备把数据写入数据库前，要先将数据添加到会话中然后调用commit()方法提交会话。

数据库会话是为了保证数据的一致性，避免因部分更新导致数据不一致。提交操作把会话对象全部写入数据库，如果写入过程发生错误，整个会话都会失效。

数据库会话也可以回滚，通过`db.session.rollback()`方法，实现会话提交数据前的状态。

在Flask-SQLAlchemy中，查询操作是通过query对象操作数据。最基本的查询是返回表中所有数据，可以通过过滤器进行更精确的数据库查询。

### 查询

- 查询过滤器

| 过滤器         | 说明                       |
| ----------- | ------------------------ |
| filter()    | 把过滤器添加到原查询上，返回一个新查询      |
| filter_by() | 把等值过滤器添加到原查询上，返回一个新查询    |
| limit       | 使用指定的值限定原查询返回的结果         |
| offset()    | 偏移原查询返回的结果，返回一个新查询       |
| order_by()  | 根据指定条件对原查询结果进行排序，返回一个新查询 |
| group_by()  | 根据指定条件对原查询结果进行分组，返回一个新查询 |

- 查询执行器

| 方法             | 说明                         |
| -------------- | -------------------------- |
| all()          | 以列表形式返回查询的所有结果             |
| first()        | 返回查询的第一个结果，如果未查到，返回None    |
| first_or_404() | 返回查询的第一个结果，如果未查到，返回404     |
| get()          | 返回指定主键对应的行，如不存在，返回None     |
| get_or_404()   | 返回指定主键对应的行，如不存在，返回404      |
| count()        | 返回查询结果的数量                  |
| paginate()     | 返回一个Paginate对象，它包含指定范围内的结果 |

- 示例

```python
from flask.dbs import *

# first()返回查询到的第一个对象
User.query.first()
# all()返回查询到的所有对象
User.query.all()
# get()，参数为主键，如果主键不存在没有返回内容
User.query.get(主键值)
# 另一种方式
db.session.query(User).all()
db.session.query(User).first()
db.session.query(User).get(2)


# filter_by精确查询，参数只需指定字段名
User.query.filter_by(name='wang').all()
# filter模糊查询，参数需要指定模型类名
User.query.filter(User.name.endswith('g')).all()
# 逻辑非，返回名字不等于wang的所有数据。
User.query.filter(User.name!='wang').all()

# 逻辑与，需要导入and，返回and()条件满足的所有数据。
from sqlalchemy import and_
User.query.filter(and_(User.name!='wang',User.email.endswith('163.com'))).all()
# 逻辑或，需要导入or_
from sqlalchemy import or_
User.query.filter(or_(User.name!='wang',User.email.endswith('163.com'))).all()
# not_ 相当于取反
from sqlalchemy import not_
User.query.filter(not_(User.name=='chen')).all()

# 偏移
User.query.offset(2).all()
# 限制
User.query.limit(3).all()
# order_by
User.query.order_by(User.id).all()  # 正序
User.query.order_by(User.id.desc()).all()  # 倒序

# 复合查询
# 方法一
User.query.filter(User.name.startswith('13')).order_by(User.id.desc()).offset(2).limit(5).all()
# 方法二
query = User.query.filter(User.name.startswith('13'))
query = query.order_by(User.id.desc())
query = query.offset(2).limit(5)
ret = query.all()

# 聚合查询
from sqlalchemy import func

db.session.query(Relation.user_id, func.count(Relation.target_user_id)).filter(Relation.relation == Relation.RELATION.FOLLOW).group_by(Relation.user_id).all()
```

- 查询数据后删除

```python
user = User.query.first()
db.session.delete(user)
db.session.commit()
User.query.all()
```

### 关联查询

- ForeignKey

```python
# 模型类
class Role(db.Model):
   	...
    id = db.Column(db.Integer, primary_key=True)
    # 创建一对多的外键，在数据库中无实体，
    # 参数1为对应的类，参数2一般写为对象名小写(可任意)，参数3为返回值是list，省略时默认为list
    us = db.relationship('User', backref='role', uselist=True)
	...

    
class User(db.Model):
    ...
    id = db.Column(db.Integer, primary_key=True)
    # 多对一的外键，第二个参数为表名.主键
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
	...
    
# 测试
# 多查一
us1 = User.query.get(3)
us1.role
# 一查多
ro1 = Role.query.get(1)
ro1.us
```

- primaryjoin

```python
# 模型类
class Role(db.Model):
   	...
    id = db.Column(db.Integer, primary_key=True)
    us = db.relationship('User', primaryjoin='Role.id==foreign(User.role_id)')
	...

    
class User(db.Model):
    ...
    id = db.Column(db.Integer, primary_key=True)
    role_id = db.Column(db.Integer)
	...   

# 测试
user = User.query.get(1)
user.profile.gender
user.followings
```

- 指定字段关联查询

之前的查询是惰性查询，使用join则可以一次性查询

```python
# 模型类
class Relation(db.Model):
    ...
    target_user = db.relationship('User', primaryjoin='Relation.target_user_id==foreign(User.id)', uselist=False)
    ...

# 测试    
from sqlalchemy.orm import load_only, contains_eager  
# load_only过滤字段，contains_eager加载新的表

Relation.query.join(Relation.target_user).options(load_only(Relation.target_user_id), contains_eager(Relation.target_user).load_only(User.name)).all()
```

### 优化查询

```python
user = User.query.filter_by(id=1).first()  # 查询所有字段
select user_id, mobile......

select * from   # 程序不要使用
select user_id, mobile,.... # 查询指定字段

from sqlalchemy.orm import load_only
User.query.options(load_only(User.name, User.mobile)).filter_by(id=1).first() # 查询特定字段
```

###增删

表

```python
# 创建表
db.create_all()

# 删除表
db.drop_all()
```

数据

```python
# 增加
# 插入一条数据
ro1 = Role(name='admin')
db.session.add(ro1)
db.session.commit()
# 一次插入多条数据
us1 = User(name='wang',email='wang@163.com',pswd='123456',role_id=ro1.id)
us2 = User(name='zhang',email='zhang@189.com',pswd='201512',role_id=ro2.id)

db.session.add_all([us1,us2])
db.session.commit()

# 删除
# 方法一
user = User.query.order_by(User.id.desc()).first()
db.session.delete(user)
db.session.commit()
# 方法二
User.query.filter(User.mobile='18512345678').delete()
db.session.commit()
```

### 更新

```python
# 方法一
user = User.query.first()
user.name = 'dong'
db.session.add(user)
db.session.commit()

# 方法二
User.query.filter_by(name='zhang').update({'name':'li'})
db.session.commit()
```

### 事务

在请求上下文中，falsk默认事务开启。注意：在flask视图函数中，也是在请求上下文中。

```python
environ = {'wsgi.version':(1,0), 'wsgi.input': '', 'REQUEST_METHOD': 'GET', 'PATH_INFO': '/', 'SERVER_NAME': 'itcast server', 'wsgi.url_scheme': 'http', 'SERVER_PORT': '80'}

with app.request_context(environ):
    try:
        user = User(mobile='18911111111', name='itheima')
        db.session.add(user)
        db.session.flush() # 将db.session记录的sql传到数据库中执行
        profile = UserProfile(id=user.id)
        db.session.add(profile)
        db.session.commit()
    except:
        db.session.rollback()
```

