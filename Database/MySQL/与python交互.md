# 与python交互

## 同步交互

### 数据库驱动

数据库驱动是用于连接 MySQL 服务器，可以执行sql语句的发送，执行结果的获取

ORM是将模型类的操作转换为sql语句，将数据库执行的结果转换为模型类对象

在mysql数据库的使用中，若使用ORM，ORM只识别mysqldb作为数据库驱动名

```
- PyMySQL是个人维护的第三方库
支持python2.x和python3.x.
在ORM（Django,sqlalchemy,flask-sqlalchemy）中使用时需执行如下语句
import pymysql
pymsql.install_as_mysqldb()

- mysql-client
支持python2和python3

- mysql-connector-python
mysql官方维护的库，支持python2和python3
```

安装驱动

```
pip install PyMySQL
pip install mysqlclient
pip install mysql-connector-python
```

### 数据库交互

- 引入模块

```
from pymysql import *
```

- connection对象

用于建立与数据库的连接

```python
# 创建对象
conn=connect(参数列表)
# 参数列表
host：连接的mysql主机，如果本机是'localhost'
port：连接的mysql主机的端口，默认是3306
database：数据库的名称
user：连接的用户名
password：连接的密码
charset：通信采用的编码方式，推荐使用utf8
```

对象的方法

```python
cursor()  # 返回Cursor对象，用于执行sql语句并获得结果
commit()  # 对数据库数据进行增删更时需提交    
rollback()  # 提交出错时回滚
close()  # 关闭连接
```

- Cursor对象

用于执行sql语句，使用频度最高的语句为select、insert、update、delete

游标对象

```python
cs1=conn.cursor() # 执行后返回由远组组成的列表
cs2=conn.cursor(pymysql.cursors.DictCursor)  # 执行后返回由字典组成的列表
```

对象方法

```python
close()  # 关闭
execute(operation [, parameters ])  # 执行语句，返回受影响的行数，主要用于执行insert、update、delete语句，也可以执行create、alter、drop等语句
fetchone()  # 执行查询语句时，获取查询结果集的第一个行数据，返回一个元组
fetchall()  # 执行查询时，获取结果集的所有行，一行构成一个元组，再将这些元组装入一个元组返回
```

对象的属性

```python
rowcount  # 只读属性，表示最近一次execute()执行后受影响的行数
connection  # 获得当前连接对象
```

上下文管理器

```python
with conn.cursor() as cursor:
    pass
```

### 参数化

- sql语句的参数化，可以有效防止sql注入
- 注意：此处不同于python的字符串格式化，全部使用%s占位

```python
from pymysql import *

def main():

    find_name = input("请输入物品名称：")

    # 创建Connection连接
    conn = connect(host='localhost',port=3306,user='root',password='mysql',database='jing_dong',charset='utf8')
    # 获得Cursor对象
    cs1 = conn.cursor()


    # # 非安全的方式
    # # 输入 " or 1=1 or "   (双引号也要输入)
    # sql = 'select * from goods where name="%s"' % find_name
    # print("""sql===>%s<====""" % sql)
    # # 执行select语句，并返回受影响的行数：查询所有数据
    # count = cs1.execute(sql)

    # 安全的方式
    # 构造参数列表
    params = [find_name]
    # 执行select语句，并返回受影响的行数：查询所有数据
    count = cs1.execute('select * from goods where name=%s', params)
    # 注意：
    # 如果要是有多个参数，需要进行参数化
    # 那么params = [数值1, 数值2....]，此时sql语句中有多个%s即可 

    # 打印受影响的行数
    print(count)
    # 获取查询的结果
    # result = cs1.fetchone()
    result = cs1.fetchall()
    # 打印查询的结果
    print(result)
    # 关闭Cursor对象
    cs1.close()
    # 关闭Connection对象
    conn.close()

if __name__ == '__main__':
    main()
```

### 综合实例

#### pymsql

```python
import pymysql

# 打开数据库连接
db = pymysql.connect("localhost","testuser","test123","TESTDB" )

# 创建游标对象
cursor = db.cursor()

# 执行sql语句
# 增删改
sql = "DROP TABLE IF EXISTS EMPLOYEE"
sql = """CREATE TABLE EMPLOYEE (
         FIRST_NAME  CHAR(20) NOT NULL,
         LAST_NAME  CHAR(20),
         AGE INT,  
         SEX CHAR(1),
         INCOME FLOAT )"""
sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
         LAST_NAME, AGE, SEX, INCOME)
         VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""
sql = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')
sql = "DELETE FROM EMPLOYEE WHERE AGE > '%d'" % (20)
try:
   cursor.execute(sql)
   db.commit()  # 提交到数据库执行
except:
   db.rollback()  # 如果发生错误则回滚
# 数据查询
sql = "SELECT VERSION()"
sql = "SELECT * FROM EMPLOYEE \
       WHERE INCOME > '%d'" % (1000)
cursor.execute(sql)
data = cursor.fetchone()  # 获取sql执行单行结果
results = cursor.fetchall()  # 获取sql执行后多行结果
for row in results:
     pass

# 关闭游标
cursor.close()
# 关闭数据库连接
db.close()
```

with

```python
import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='user',
                             password='passwd',
                             db='db',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
        cursor.execute(sql, ('webmaster@python.org', 'very-secret'))
    connection.commit()

    with connection.cursor() as cursor:
        sql = "SELECT `id`, `password` FROM `users` WHERE `email`=%s"
        cursor.execute(sql, ('webmaster@python.org',))
        result = cursor.fetchone()
        print(result)
finally:
    connection.close()
```

#### mysqlClient

```python
import MySQLdb

# 创建连接对象
conn= MySQLdb.connect(
        host='localhost',
        port = 3306,
        user='root',
        passwd='123456',
        db ='test',
        )

# 创建游标。
cur = conn.cursor()

# 执行sql
cur.execute("create table student(id int ,name varchar(20),class varchar(30),age varchar(10))")

#插入一条数据
cur.execute("insert into student values('2','Tom','3 year 2 class','9')")


#修改查询条件的数据
cur.execute("update student set class='3 year 1 class' where name = 'Tom'")

#删除查询条件的数据
cur.execute("delete from student where age='9'")

#一次插入多条记录
sqli="insert into student values(%s,%s,%s,%s)"
cur.executemany(sqli,[
    ('3','Tom','1 year 1 class','6'),
    ('3','Jack','2 year 1 class','7'),
    ('3','Yaheng','2 year 2 class','7'),
    ])

# 关闭游标
cur.close()

#conn.commit()方法在提交事物，在向数据库插入一条数据时必须要有这个方法，否则数据不会被真正的插入。
conn.commit()

#conn.close()关闭数据库连接
conn.close()
```

#### m-c-p

mysql经典协议

```python
import mysql.connector

# Connect to server
cnx = mysql.connector.connect(
    host="127.0.0.1",
    port=3306,
    user="mike",
    password="s3cre3t!")

# Get a cursor
cur = cnx.cursor()

# Execute a query
cur.execute("SELECT CURDATE()")

# Fetch one result
row = cur.fetchone()
print("Current date is: {0}".format(row[0]))

# Close connection
cnx.close()
```

使用MySQL X DevAPI

```python
import mysqlx

# Connect to server
session = mysqlx.get_session(
   host="127.0.0.1",
   port=33060,
   user="mike",
   password="s3cr3t!")
schema = session.get_schema("test")

# Use the collection "my_collection"
collection = schema.get_collection("my_collection")

# Specify which document to find with Collection.find()
result = collection.find("name like :param") \
                   .bind("param", "S%") \
                   .limit(1) \
                   .execute()

# Print document
docs = result.fetch_all()
print(r"Name: {0}".format(docs[0]["name"]))

# Close session
session.close()
```

## 异步交互

安装驱动

```
pip install aiomysql
```

- 使用

[参考](https://github.com/aio-libs/aiomysql)

[文档](https://aiomysql.readthedocs.io/en/latest/)

简单使用

```python
import asyncio
import aiomysql


async def test_example(loop):
    pool = await aiomysql.create_pool(host='127.0.0.1', port=3306,
                                      user='root', password='',
                                      db='mysql', loop=loop)
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 42;")
            print(cur.description)
            (r,) = await cur.fetchone()
            assert r == 42
    pool.close()
    await pool.wait_closed()


loop = asyncio.get_event_loop()
loop.run_until_complete(test_example(loop))
```

## 集成Django

在框架中，使用默认的ORM

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
pip install pymysql

3)、导入mysql包
1. 在项目或应用的__init__.py中，
import pymysql
pymysql.install_as_MySQLdb()
2. 修改源码屏蔽版本控制
(1)在python\Lib\site-packages\django\db\backends\mysql/base.py中注释掉如下代码
if version < (1, 3, 13): 　　　　　　　　　　
        raise ImproperlyConfigured(‘mysqlclient 1.3.13 or newer is required; you have %s.’ % Database.version) 　
(2)在Python\lib\site-packages\django\db\backends\mysql\operations.py”, line 146
decode修改为encode

4)、重新生成数据库表
删除掉应用名/migrations目录下所有的迁移文件
重新执行：
python manage.py makemigrations
python manage.py migrate

5)确认是否已经生成了对应的数据库表
```

## 批量读取数据

```python
import time
from math import ceil
import pymysql as MySQLdb
import csv


def mysql_start():
    # 数据库连接属性
    host = '127.0.0.1'
    usr = 'root'
    passwd = ''
    db = 'polls'

    # 连接数据库
    conn = MySQLdb.connect(host=host, user=usr, password=passwd, database=db)
    return conn


def batch_query_write1(conn):
    # 创建游标
    cur = conn.cursor()
    # 总共多少数据
    allData = 1000100
    # 每个批次多少条数据
    dataOfEach = 200000
    # 批次
    batch = ceil(allData / dataOfEach)

    # 文件名
    global IDctrl
    IDctrl = 1
    filename = str(IDctrl) + '.txt'
    sum = 0
    f = open("sync.txt", "w")
    while IDctrl < batch:
        # 读取数据库
        sql = 'SELECT * from polls.question where ID>=' + str(IDctrl) + ' and ID <' + str(IDctrl + dataOfEach)
        cur.execute(sql)
        rows = cur.fetchall()
        sum += len(rows)
        # 同步写文件
        f.writelines(str(rows))
        # 文件名加1
        IDctrl += 1
    print(sum)
    f.close()


def batch_query_write2(conn):
    cur = conn.cursor()
    sql = "SELECT * from polls.question where id < 5"
    cur.execute(sql)
    row = cur.fetchone()
    sum = 0
    f = open("sync.csv", "a")
    f_csv = csv.writer(f)
    while row is not None:
        print(row, type(row))
        sum += 1
        f_csv.writerow(row)
        row = cur.fetchone()
    print(sum)
    f.close()


def batch_query_write3(conn):
    cur = MySQLdb.cursors.SSCursor(conn)
    sql = "SELECT * from polls.question"
    cur.execute(sql)
    sum = 0
    while True:
        row = cur.fetchone()
        if not row:
            break
        else:
            sum += 1

    print(sum)


def mysql_end(conn):
    # 关闭数据库连接
    conn.close()


if __name__ == '__main__':
    start = time.process_time()
    conn = mysql_start()
    # batch_query_write1(conn)
    # batch_query_write2(conn)
    # batch_query_write3(conn)
    mysql_end(conn)
    end = time.process_time()
    print('total_time2', end - start)
```
