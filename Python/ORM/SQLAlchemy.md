# SQLAlchemy
## 概述
[参考](https://blog.csdn.net/qq_36019490/article/details/96883453)

SQLAlchemy是一个基于Python实现的ORM框架。该框架建立在 DB API之上，使用关系对象映射进行数据库操作，简言之便是：将类和对象转换成SQL，然后使用数据API执行SQL并获取执行结果。

安装

```
pip install sqlalchemy
```
组成部分：
```
Engine，框架的引擎
Connection Pooling ，数据库连接池
Dialect，选择连接数据库的DB API种类
Schema/Types，架构和类型
SQL Exprression Language，SQL表达式语言
```

## 同步操作

### 连接数据库

[参考](https://docs.sqlalchemy.org/en/13/dialects/mysql.html)

- mysql

SQLAlchemy本身无法操作数据库，需要数据库驱动才可使用

```
pip install pymysql
pip install mysqlclient
pip install mysql-connector-python
```

连接方式

```python
mysql+frameworkname://username:password@address:port/databasename

# 示例
mysql+pymysql://<username>:<password>@<host>/<dbname>[?<options>]
mysql+mysqldb://<user>:<password>@<host>[:<port>]/<dbname>
mysql+mysqlconnector://<user>:<password>@<host>[:<port>]/<dbname>
```

- postgreSQL

```
pip install psycopg2
```

连接方式

```
postgresql+psycopg2://user:password@host:port/dbname[?key=value&key=value...]
```

- sqlite

```
sqlite:////absolutepath/dbname.db  # Unix/Mac系统
sqlite:///C:\\absolutepath\\dbname.db  # Windows系统
```

示例

```python
from sqlalchemy import create_engine
 
try:
    # 连接MySQL数据库
	MySQLEngine = create_engine(
    	'mysql+pymysql://root:123@localhost:3306/test?charset=utf8', 
   		 encoding='utf-8')
    print('连接MySQL数据库成功', MySQLEngine)
    # 连接SQLite数据库，如果当前目录不存在test.db文件则会自动生成
    SQLiteEngine = create_engine('sqlite:///:test.db', encoding='utf-8')
    print('连接SQLite数据库成功', SQLiteEngine)
except Exception as e:
    print('连接数据库失败', e)
```

### 创建表

```python
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
 
# 创建基类
BASE = declarative_base()
 
# 定义学生对象
class Student(BASE):
    __tablename__ = 'STUDENT'  # 表的名字:STUDENT
    id = Column(Integer, primary_key=True) # id
    sno = Column(String(10))  # 学号
    sname = Column(String(20))  # 姓名
    __table_args__ = {
        "mysql_charset": "utf8"  # 创建表的参数
    }
 
try:
	MySQLEngine = create_engine(
        'mysql+pymysql://root:123@localhost:3306/test?charset=utf8', 
        encoding='utf-8')
    # 创建STUDENT表
    BASE.metadata.create_all(MySQLEngine)
    print('创建STUDENT表成功')
except Exception as e:
    print("连接SQLite数据库失败", e)
```

### 插入数据

```python
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
 
BASE = declarative_base()
 
class Student(BASE):
    __tablename__ = 'STUDENT' 
    id = Column(Integer, primary_key=True)
    sno = Column(String(10))  
    sname = Column(String(20))  
    __table_args__ = {
        "mysql_charset": "utf8"  
    }
 
try:
	MySQLEngine = create_engine(
        'mysql+pymysql://root:123@localhost:3306/test?charset=utf8', 
        encoding='utf-8')
    # 创建与数据库的会话
    MySQLSession = sessionmaker(bind=MySQLEngine)
    # 生成session实例
    session = MySQLSession()
 
    # 使用ORM插入数据
    Stu = Student(sname='张三', sno='2016081111')  
    # 将创建的对象添加进session中
    session.add(Stu)
 
    # 使用原生SQL插入数据
    session.execute(
        "INSERT INTO STUDENT VALUES('2016081115','吴芳'),('2016081116','胡月')")
    # 提交到数据库
    session.commit()
    # 关闭session
    session.close()
    print('插入数据成功')
except Exception as e:
    print("连接SQLite数据库失败", e)
```

### 查询数据

```python
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
 
BASE = declarative_base()
 
class Student(BASE):
    __tablename__ = 'STUDENT'
 	id = Column(Integer, primary_key=True)
    sname = Column(String(20), primary_key=True)
    sno = Column(String(10))
 
    def __str__(self):  # 格式输出查询出的数据
        return '%s,%s' % (self.sname, self.sno)
 
try:
    MySQLEngine = create_engine(
      'mysql+pymysql://root:123@localhost:3306/test?charset=utf8', 
      encoding='utf-8')
    MySQLSession = sessionmaker(bind=MySQLEngine)
    session = MySQLSession()
    # 查询
    Stu = session.query(Student).filter(Student.sno == '2016081111')  # 查询学号为2016081111的学生
    Stuf = session.query(Student).filter_by(sno='2016081111').first()  
    Stus = session.query(Student).all()  # 查询所有数据
    Stu = session.query(Student).filter(
        Student.id>0).filter(Student.id<5).all()  # 多条件查询
    print('查询结果的类型：', type(Stu))
    print('STUDENT表所有  的数据：')
    for row in Stus:
        print(row)
    session.close()
except Exception as e:
	print("连接SQLite数据库失败", e)
```

### 修改数据

```python
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
 
BASE = declarative_base()
 
class Student(BASE):
    __tablename__ = 'STUDENT'
 	id = Column(Integer, primary_key=True)
    sname = Column(String(20), primary_key=True)
    sno = Column(String(10))
 
    def __str__(self):  # 格式输出查询出的数据
        return '%s,%s' % (self.sname, self.sno)
 
 
try:
	MySQLEngine = create_engine(
        'mysql+pymysql://root:123@localhost:3306/test?charset=utf8', 
        encoding='utf-8')
    MySQLSession = sessionmaker(bind=MySQLEngine)
    session = MySQLSession()
    # 修改数据
    Stu = session.query(Student).filter(Student.sno == '2016081111').first()
    print('更改前：', Stu)
    Stu.sname = '李华'  # 更改姓名为李华
    # 回滚
    session.rollback()
    Stu.sname = '李明'
    session.commit()
    print('更改后：', Stu)
    session.close()
except Exception as e:
	print("连接SQLite数据库失败", e)
```

### 删除数据

```python
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
 
BASE = declarative_base()
 
class Student(BASE):
    __tablename__ = 'STUDENT'
 
    sname = Column(String(20), primary_key=True)
    sno = Column(String(10))
 
    def __str__(self): 
        return '%s,%s' % (self.sname, self.sno)
 
try:
	MySQLEngine = create_engine(
        'mysql+pymysql://root:123@localhost:3306/test?charset=utf8', 
        encoding='utf-8')
    MySQLSession = sessionmaker(bind=MySQLEngine)
    session = MySQLSession()
    # 删除数据
    before = session.query(Student).filter(Student.sno == '2016081111').first()
    print('删除前：', before)
    session.query(Student).filter(Student.sno == '2016081111').delete()   # 删除数据
    session.commit()
    after = session.query(Student).filter(Student.sno == '2016081111').first()
    print('删除后：', after)
    session.close()
except Exception as e:
	print("连接SQLite数据库失败", e)
```

### 外键关联

单外键关联

```python
import  sqlalchemy
from sqlalchemy import  create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Integer,String,ForeignKey #区分大小写
from sqlalchemy.orm import sessionmaker,relationship

base=declarative_base()

class user(base):
    __tablename__ = 'users' 
    id = Column(Integer, primary_key=True)
    name = Column(String(32))
    password = Column(String(64))
    def __repr__(self):
        return "<user(id='%d',name='%s',  password='%s')>" % (self.id,
        self.name, self.password)
     
class Address(base):
    __tablename__ = 'addresses'
    id = Column(Integer, primary_key=True)
    email_address = Column(String(32), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    user = relationship("user", backref="addresses") 
    '''允许你在user表里通过backref字段反向查出所有它在addresses表里的关联项，在内存中创建。在addresses表中可以使用user来查询users表中的数据，在users表中可以使用backref后的addresses来查询assresses表中的数据。'''
 
    def __repr__(self):
        return "<Address(email_address='%s',id='%d',user_id='%d')>" % (self.email_address,self.id,self.user_id)

# 创建连接
engine = create_engine(
    "mysql+pymysql://root:123456@localhost/ceshi",
    encoding='utf-8',
    echo=True)    
base.metadata.create_all(engine) #创建表结构
Session_class=sessionmaker(bind=engine) #创建与数据库的会话
Session=Session_class()   #生成session实例
obj = Session.query(user).first()
print(obj.addresses)  #在users表里面通过addresses来查询addresses表中的数据。
 
for i in obj.addresses:
    print i
     
addr_obj = Session.query(Address).first()
print(addr_obj.user)  #在addresses表中通过user来查询users表中的数据。
print(addr_obj.user.name)
 
Session.commit() #提交，使前面修改的数据生效。
```

多外键关联

```python
class Customer(base):
    __tablename__ = 'customer'
    id = Column(Integer, primary_key=True)
    name = Column(String)
  
    billing_address_id = Column(Integer, ForeignKey("address.id")) 
    shipping_address_id = Column(Integer, ForeignKey("address.id"))
 '''#创建的列billing_address_id、shipping_address_id都作为外键关联address表中id列'''
    billing_address = relationship("Address", foreign_keys=[billing_address_id])
    shipping_address = relationship("Address", foreign_keys=[shipping_address_id])#创建两个关联项
  
class Address(base):
    __tablename__ = 'address'
    id = Column(Integer, primary_key=True)
    street = Column(String)
    city = Column(String)
    state = Column(String)
```

### 多对多

创建

```python
from sqlalchemy import Table, Column, Integer,String,DATE, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
 

base = declarative_base()

# 创建book_m2m_author表，关联另外两张表。
book_m2m_author = Table(
    'book_m2m_author', 
    base.metadata,
    Column('book_id',Integer,ForeignKey('books.id')),
    Column('author_id',Integer,ForeignKey('authors.id')),
)  
 
class Book(base):
    __tablename__ = 'books'
    id = Column(Integer,primary_key=True)
    name = Column(String(64))
    pub_date = Column(DATE)
    authors = relationship('Author',secondary=book_m2m_author,backref='books')
 
    def __repr__(self):
        return self.name
 
class Author(base):
    __tablename__ = 'authors'
    id = Column(Integer, primary_key=True)
    name = Column(String(32))
 
    def __repr__(self):
        return self.name
    
engine = create_engine(
    "mysql+pymysql://root:123456@localhost/ceshi",
    encoding='utf-8',echo=True)
base.metadata.create_all(engine) 
Session_class=sessionmaker(bind=engine) 
Session=Session_class()
b1 = Book(name="跟A学Python")
b2 = Book(name="跟A学linux")
b3 = Book(name="跟A学java")
b4 = Book(name="跟C学开发")
  
a1 = Author(name="A")
a2 = Author(name="B")
a3 = Author(name="C")
  
b1.authors = [a1,a2]  #建立关系
b2.authors = [a1,a2,a3]
Session.add_all([b1,b2,b3,b4,a1,a2,a3])
Session.commit()
```

查询

```python
book_obj = Session.query(Book).filter_by(name="跟A学Python").first()
print(book_obj.name, book_obj.authors)  # 这里book_obj.authors只输出name,因为定义类Author时在__repr__(self):定义了返回值
  
author_obj =Session.query(Author).filter_by(name="A").first()
print(author_obj.name , author_obj.books)
```

删除

```python
# 通过指定书里删除作者,删除的是关系,作者不受影响。
author_obj =Session.query(Author).filter_by(name="C").first()
book_obj = Session.query(Book).filter_by(name="跟A学linux").first()
book_obj.authors.remove(author_obj) 
# 删除作者的同时也删除关系
author_obj =Session.query(Author).filter_by(name="A").first()
Session.delete(author_obj) 
```

## 异步操作

### mysql

安装

```
pip install aiomysql
```

- 使用

[参考](https://aiomysql.readthedocs.io/en/latest/sa.html#)

简单示例

```python
import asyncio
import sqlalchemy as sa

from aiomysql.sa import create_engine


metadata = sa.MetaData()

tbl = sa.Table('tbl', metadata,
               sa.Column('id', sa.Integer, primary_key=True),
               sa.Column('val', sa.String(255)))


async def go(loop):
    engine = await create_engine(user='root', db='test_pymysql',
                                 host='127.0.0.1', password='', loop=loop)
    async with engine.acquire() as conn:
        await conn.execute(tbl.insert().values(val='abc'))
        await conn.execute(tbl.insert().values(val='xyz'))

        async for row in conn.execute(tbl.select()):
            print(row.id, row.val)

    engine.close()
    await engine.wait_closed()


loop = asyncio.get_event_loop()
loop.run_until_complete(go(loop))
```

### postgreSQL

安装

```
pip install aiopg
```

- 使用

[参考](https://aiopg.readthedocs.io/en/stable/sa.html)

简单示例

```python
import asyncio
from aiopg.sa import create_engine
import sqlalchemy as sa

metadata = sa.MetaData()

tbl = sa.Table('tbl', metadata,
    sa.Column('id', sa.Integer, primary_key=True),
    sa.Column('val', sa.String(255)))

async def create_table(engine):
    async with engine.acquire() as conn:
        await conn.execute('DROP TABLE IF EXISTS tbl')
        await conn.execute('''CREATE TABLE tbl (
                                  id serial PRIMARY KEY,
                                  val varchar(255))''')

async def go():
    async with create_engine(user='aiopg',
                             database='aiopg',
                             host='127.0.0.1',
                             password='passwd') as engine:

        async with engine.acquire() as conn:
            await conn.execute(tbl.insert().values(val='abc'))

            async for row in conn.execute(tbl.select()):
                print(row.id, row.val)

loop = asyncio.get_event_loop()
loop.run_until_complete(go())
```





