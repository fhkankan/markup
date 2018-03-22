# 数据库

## SQLALchemy

SQLALchemy是python成熟的ORM框架模块，可适用于Django,Flask等。为了便于在Flask中使用，对它进行了再次封装，形成了Flask-SQLALchemy

### 安装数据库 

```
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

### 安装flask-sqlalchemy

```
# ORM(将模型类操作转换为sql语句，将结果转换为模型类对象)
pip install flask-sqlalchemy

# 数据库驱动程序(数据库连接，sql语句传输，执行结果获取)
pip install mysql-python(mysql官方,ORM自动识别为mysqldb)
或
pip install flask-mysqldb(封装过，ORM自动识别为mysqldb)
或
pip install pymysql(需要在程序中执行
import pymysql
pymysql.install_as_mysqldb(),之后ORM才能识别为myslqdb)
```

### 数据库配置

```
# 设置连接数据库的URL(必需)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:mysql@127.0.0.1:3306/Flask_test'
# 设置每次请求结束后会自动提交数据库中的改动（后期会去除）
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
# 数据库变更追踪，可以设定为False
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# 查询时会显示原始SQL语句
app.config['SQLALCHEMY_ECHO'] = True
```

### 字段类型

| 类型名          | python中类型         | 说明                            |
| ------------ | ----------------- | ----------------------------- |
| Integer      | int               | 普通整数，一般是32位                   |
| SmallInteger | int               | 取值范围小的整数，一般是16位               |
| BigInteger   | int或long          | 不限制精度的整数                      |
| Float        | float             | 浮点数                           |
| Numeric      | decimal.Decimal   | 普通整数，一般是32位                   |
| String       | str               | 变长字符串                         |
| Text         | str               | 变长字符串，对较长或不限长度的字符串做了优化        |
| Unicode      | unicode           | 变长Unicode字符串                  |
| UnicodeText  | unicode           | 变长Unicode字符串，对较长或不限长度的字符串做了优化 |
| Boolean      | bool              | 布尔值                           |
| Date         | datetime.date     | 时间                            |
| Time         | datetime.datetime | 日期和时间                         |
| LargeBinary  | str               | 二进制文件                         |

### 列选项

| 选项名         | 说明                            |
| ----------- | ----------------------------- |
| primary_key | 如果为True，代表表的主键                |
| unique      | 如果为True，代表这列不允许出现重复的值         |
| index       | 如果为True，为这列创建索引，提高查询效率        |
| nullable    | 如果为True，允许有空值，如果为False，不允许有空值 |
| default     | 为这列定义默认值                      |

### 关系选项

| 选项名            | 说明                                  |
| -------------- | ----------------------------------- |
| backref        | 在关系的另一模型中添加反向引用                     |
| primary join   | 明确指定两个模型之间使用的联结条件                   |
| uselist        | 如果为False，不使用列表，而使用标量值               |
| order_by       | 指定关系中记录的排序方式                        |
| secondary      | 指定多对多中记录的排序方式                       |
| secondary join | 在SQLAlchemy中无法自行决定时，指定多对多关系中的二级联结条件 |

## 使用数据库

在Flask-SQLAlchemy中，插入、修改、删除操作，均由数据库会话管理。会话用db.session表示。在准备把数据写入数据库前，要先将数据添加到会话中然后调用commit()方法提交会话。

数据库会话是为了保证数据的一致性，避免因部分更新导致数据不一致。提交操作把会话对象全部写入数据库，如果写入过程发生错误，整个会话都会失效。

数据库会话也可以回滚，通过db.session.rollback()方法，实现会话提交数据前的状态。

在Flask-SQLAlchemy中，查询操作是通过query对象操作数据。最基本的查询是返回表中所有数据，可以通过过滤器进行更精确的数据库查询。

### 定义模型类

从数据库表的层面考虑定义模型类

```
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
# 方法一：
# db = SQLALchemy()
# db.init_app(app)
# 方法二：
db = SQLAlchemy(app)

class Role(db.Model):
    # 定义表名，若不写，则默认创建为类名的小写格式
    __tablename__ = 'roles'
    # 定义列对象，db.column数据库中有实体
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    # 创建一对多的外键，在数据库中无实体，第一个参数为对应的类，第二个关键字指明为User类添加了一个额外数据行role
    users = db.relationship('User', backref='role')

    #repr()方法显示一个可读字符串
    def __repr__(self):
        return 'Role:%s'% self.name

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, index=True)
    email = db.Column(db.String(64),unique=True)
    pswd = db.Column(db.String(64))
    # 多对一的外键，关联一的主键，在数据库中有实体，第二个参数中为表名.主键
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
### 一对多表对比

```
 user类和role类
 
 # 在django中的user类创建外键
 role = ForeignKey(Role)
 # 在django中使用
 user1.role 	 ---> Role对象
 role1.user_set  ---> User对象列表
 
 
# 在Flask中的user类创建外键
role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
# 在Flask中的role类创建关联
users = db.relationship('User', backref='role')
# 在Flask中使用
user1.role_id	---> roles表主键的id值 --->role对象值
user1.role		---> Role对象
role1.users		---> User表的对象列表
```
### 增删表数据

```
创建表：
db.create_all()

删除表
db.drop_all()

# 插入一条数据
ro1 = Role(name='admin')
db.session.add(ro1)
db.session.commit()

# 一次插入多条数据
us1 = User(name='wang',email='wang@163.com',pswd='123456',role_id=ro1.id)
us2 = User(name='zhang',email='zhang@189.com',pswd='201512',role_id=ro2.id)

db.session.add_all([us1,us2])
db.session.commit()
```


### 查询过滤器

| 过滤器         | 说明                       |
| ----------- | ------------------------ |
| filter()    | 把过滤器添加到原查询上，返回一个新查询      |
| filter_by() | 把等值过滤器添加到原查询上，返回一个新查询    |
| limit       | 使用指定的值限定原查询返回的结果         |
| offset()    | 偏移原查询返回的结果，返回一个新查询       |
| order_by()  | 根据指定条件对原查询结果进行排序，返回一个新查询 |
| group_by()  | 根据指定条件对原查询结果进行分组，返回一个新查询 |

### 查询执行器

| 方法             | 说明                         |
| -------------- | -------------------------- |
| all()          | 以列表形式返回查询的所有结果             |
| first()        | 返回查询的第一个结果，如果未查到，返回None    |
| first_or_404() | 返回查询的第一个结果，如果未查到，返回404     |
| get()          | 返回指定主键对应的行，如不存在，返回None     |
| get_or_404()   | 返回指定主键对应的行，如不存在，返回404      |
| count()        | 返回查询结果的数量                  |
| paginate()     | 返回一个Paginate对象，它包含指定范围内的结果 |


### 查询数据

```
# 方法一：
db.session.query(模型类).[过滤器]执行器
# 方法二：
模型类.query.[过滤器]执行器

# 注意：在SQL_ALchemy中查询数据不存在，不报错，返回NoneType

在ipython中，from flask.dbs(文件名) import *
```

#### 简单查询

```
# first()返回查询到的第一个对象
db.session.query(User).first()
User.query.first()

# all()返回查询到的所有对象的列表
User.query.all()

# get()，参数为主键，如果主键不存在没有返回内容
User.query.get(主键值)
```

#### 过滤查询

```
# filter_by精确查询，参数只需指定字段名
User.query.filter_by(name='wang').all()
User.query.filter_by(id=3, name='wang').first()

# filter模糊查询，参数需要指定模型类名
User.query.filter(User.name=='wang').all()
User.query.filter(User.id==3, User.name=='wang').first()
User.query.filter(User.name.endswith('g')).first()
```

#### 关联查询

```
# 一对多
# 查询roles表id为1的角色
role1 = Role.query.get(1)
role1.name
# 查询该角色的所有用户
role1.users		---> 对象列表
role1.users[0].name

# 多对一
# 查询users表id为4的用户
user1 = User.query.get(4)
user1.name
user1.role_id
# 查询用户属于什么角色
user1.role		---> 对象
user1.role.name
```
#### 逻辑查询

```
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
```

#### 限制查询

```
# count()，返回符合过滤条件的条数
User.query.filter(过滤条件).count()

# limit(number),返回限制的条数
User.query.filter(过滤条件).limit(3).all()

# offset(number),跳过条目数
User.query.offset(1).all()

# order_by，排序,默认升序
User.query.order_by(User.id).all()
# 降序排序
User.query.order_by(User.id.desc()).all()
```

#### 分组查询

```
# func存储了聚合函数
import sqlalchemy import func
# 返回元组组成的列表,例如[(1L,2L),(2L,2L)]
db.session.query(User.role_id, func.count(User.role_id)).group_by(User.role_id).all()
```

#### 分页查询

```
# 返回一个pagenate对象，参数1：第几页，参数2：每页几条，参数3：是否自动错误返回
page_obj = User.query.pagenate(1， per_page=2, error_out=False)

# 获取总页数
page_obj.pages

# 返回对象的列表
page_obj.items
page_obj.items[0].name
```
### 更新数据

```
# 方式一：对象存在直接赋值
user = User.query.first()
user.name = 'dong'
db.session.commit()
User.query.first()

# 方式二：查询时更新
User.query.filter_by(name='zhang').update({'name':'li'})
# 或
User.query.filter(User.name=='zhang').update({'name':'li'})
db.session.commit()
```
### 删除数据

```
# 方式一：对象存在直接删除
user = User.query.get(3)
db.session.delete(user)
db.session.commit()
User.query.query.get(3)

# 方式二：查询时删除
User.query.filter_by(name='zhang').delete()
db.session.commit()
```

## 数据库迁移

在Flask中可以使用Flask-Migrate扩展，来实现数据迁移。并且集成到Flask-Script中，所有操作通过命令就能完成。

为了导出数据库迁移命令，Flask-Migrate提供了一个MigrateCommand类，可以附加到flask-script的manager对象上。

### 安装Flask-Migrate

```
pip install flask-migrate
```

### 创建模型类文件

database.py

```
#coding=utf-8
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate,MigrateCommand
from flask_script import Shell,Manager

# 创建Flask实例
app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:mysql@127.0.0.1:3306/Flask_test'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# 创建SQLAlchemy实例
db = SQLAlchemy(app)

# 数据库迁移
# 1.创建Manager管理器Flask-Script实例
manager = Manager(app)
# 2.创建迁移框架于程序的关联，参数1:Flask的实例，参数2:Sqlalchemy数据库实例
migrate = Migrate(app,db) 
# 3.添加迁移命令，manager是Flask-Script的实例，这条语句在flask-Script中添加一个db命令
manager.add_command('db',MigrateCommand)

# 定义模型Role
class Role(db.Model):
    # 定义表名
    __tablename__ = 'roles'
    # 定义列对象
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    def __repr__(self):
        return 'Role:'.format(self.name)

# 定义用户
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True)
    def __repr__(self):
        return 'User:'.format(self.username)
        
        
if __name__ == '__main__':
    manager.run()
```

### 创建迁移仓库

```
# 这个命令会创建migrations文件夹，所有迁移文件都放在里面。
python 文件名.py db init
```

### 创建迁移脚本

```
# 创建自动迁移脚本
python 文件名.py db migrate -m 'initial migration'
```

### 更新数据库

```
# 同步模型类的操作至数据库中
python 文件名.py db upgrade
```

### 回退数据库

```
# 查看数据库历史版本
python 文件名.py db history

# 回退数据库至特定版本
python 文件名.py db downgrade 版本号
```

# 邮箱

Flask的扩展包Flask-Mail通过包装了Python内置的smtplib包，可以用在Flask程序中发送邮件。

Flask-Mail连接到简单邮件协议（Simple Mail Transfer Protocol,SMTP）服务器，并把邮件交给服务器发送。

```
from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)

# 配置邮件：服务器／端口／传输层安全协议／邮箱名／密码
# dict.update(多个键值对)（python中字典的内置方法）
app.config.update(
    DEBUG = True,
    MAIL_SERVER='smtp.qq.com',
    MAIL_PROT=465,
    MAIL_USE_TLS = True,
    MAIL_USERNAME = '371673381@qq.com',
    MAIL_PASSWORD = 'goyubxohbtzfbidd',
)

mail = Mail(app)

@app.route('/')
def index():
 	# sender 发送方，recipients 接收方列表
    msg = Message("This is a test ",sender='371673381@qq.com', recipients=['shengjun@itcast.cn','371673381@qq.com'])
    # 邮件内容
    msg.body = "Flask test mail"
    # 发送邮件
    mail.send(msg)
    print "Mail sent"
    return "Sent　Succeed"

if __name__ == "__main__":
    app.run()
```

