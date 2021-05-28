# JSON

从MySQL5.7.8开始，MySQL支持原生的JSON数据类型

[参考](http://www.lnmp.cn/mysql-57-new-features-json.html)

[参考](http://dev.mysql.com/doc/refman/5.7/en/json-search-functions.html)

## 创建

类似 varchar，设置 JSON 主要将字段的 type 是 json, 不能设置长度，可以是 NULL  但不能有默认值。
```mysql
mysql> CREATE TABLE lnmp (
    `id` int(10) unsigned NOT NULL AUTO_INCREMENT,
    `category` JSON,
    `tags` JSON,
    PRIMARY KEY (`id`)
);
```

## 插入

- 字符串

就是插入 json 格式的字符串，可以是对象的形式，也可以是数组的形式

```mysql
mysql> INSERT INTO `lnmp` (category, tags) VALUES ('{"id": 1, "name": "lnmp.cn"}', '[1, 2, 3]');
Query OK, 1 row affected (0.01 sec)
```

- 函数

MySQL 也有专门的函数` JSON_OBJECT`，`JSON_ARRAY `生成 json 格式的数据

```mysql
mysql> INSERT INTO `lnmp` (category, tags) VALUES (JSON_OBJECT("id", 2, "name", "php.net"), JSON_ARRAY(1, 3, 5));
Query OK, 1 row affected (0.00 sec)
```

## 查询

- 查询结果

查询 json 中的数据用` column->path `的形式，其中对象类型 path 这样表示 `$.path`, 而数组类型则是` $[index]`

```mysql
mysql> SELECT id, category->'$.id', category->'$.name', tags->'$[0]', tags->'$[2]' FROM lnmp;
+----+------------------+--------------------+--------------+--------------+
| id | category->'$.id' | category->'$.name' | tags->'$[0]' | tags->'$[2]' |
+----+------------------+--------------------+--------------+--------------+
|  1 | 1                | "lnmp.cn"          | 1            | 3            |
|  2 | 2                | "php.net"          | 1            | 5            |
+----+------------------+--------------------+--------------+--------------+
2 rows in set (0.00 sec)
```

可以看到对应字符串类型的 `category->'$.name' `中还包含着双引号，这其实并不是想要的结果，

可以用 `JSON_UNQUOTE` 函数将双引号去掉，从 MySQL 5.7.13 起也可以通过这个操作符` ->> `这个和 `JSON_UNQUOTE` 是等价的

```mysql
mysql> SELECT id, category->'$.name', JSON_UNQUOTE(category->'$.name'), category->>'$.name' FROM lnmp;
+----+--------------------+----------------------------------+---------------------+
| id | category->'$.name' | JSON_UNQUOTE(category->'$.name') | category->>'$.name' |
+----+--------------------+----------------------------------+---------------------+
|  1 | "lnmp.cn"          | lnmp.cn                          | lnmp.cn             |
|  2 | "php.net"          | php.net                          | php.net             |
+----+--------------------+----------------------------------+---------------------+
2 rows in set (0.00 sec)
```

- 下面说下 JSON 作为条件进行搜索。

> 字段比较

因为 JSON 不同于字符串，所以如果用字符串和 JSON 字段比较，是不会相等的

```mysql
mysql> SELECT * FROM lnmp WHERE category = '{"id": 1, "name": "lnmp.cn"}';
Empty set (0.00 sec)
```

这时可以通过 CAST 将字符串转成 JSON 的形式

```python
mysql> SELECT * FROM lnmp WHERE category = CAST('{"id": 1, "name": "lnmp.cn"}' as JSON);
+----+------------------------------+-----------+
| id | category                     | tags      |
+----+------------------------------+-----------+
|  1 | {"id": 1, "name": "lnmp.cn"} | [1, 2, 3] |
+----+------------------------------+-----------+
1 row in set (0.00 sec)
```

> 查询

对象的`column->path`

通过 JSON 中的元素进行查询, 对象型的查询同样可以通过 `column->path`

```python
mysql> SELECT * FROM lnmp WHERE category->'$.name' = 'lnmp.cn';
+----+------------------------------+-----------+
| id | category                     | tags      |
+----+------------------------------+-----------+
|  1 | {"id": 1, "name": "lnmp.cn"} | [1, 2, 3] |
+----+------------------------------+-----------+
1 row in set (0.00 sec)
```

上面有提到` column->path` 形式从 select 中查询出来的字符串是包含双引号的，但作为条件这里其实没什么影响，`-> `和 `->> `结果是一样的

```python
mysql> SELECT * FROM lnmp WHERE category->>'$.name' = 'lnmp.cn';
+----+------------------------------+-----------+
| id | category                     | tags      |
+----+------------------------------+-----------+
|  1 | {"id": 1, "name": "lnmp.cn"} | [1, 2, 3] |
+----+------------------------------+-----------+
1 row in set (0.00 sec)
```

要特别注意的是，JSON 中的元素搜索是严格区分变量类型的，比如说整型和字符串是严格区分的

```python
mysql> SELECT * FROM lnmp WHERE category->'$.id' = '1';
Empty set (0.00 sec)
# 搜索字符串 1 和整型 1 的结果是不一样的。
mysql> SELECT * FROM lnmp WHERE category->'$.id' = 1;
+----+------------------------------+-----------+
| id | category                     | tags      |
+----+------------------------------+-----------+
|  1 | {"id": 1, "name": "lnmp.cn"} | [1, 2, 3] |
+----+------------------------------+-----------+
1 row in set (0.00 sec)
```

`JSON_CONTAINS`

除了用 `column->path`的形式搜索，还可以用`JSON_CONTAINS` 函数，但和 `column->path` 的形式有点相反的是，`JSON_CONTAINS `第二个参数是不接受整数的，无论 json 元素是整型还是字符串，否则会出现这个错误

```python
mysql> SELECT * FROM lnmp WHERE JSON_CONTAINS(category, 1, '$.id');
ERROR 3146 (22032): Invalid data type for JSON data in argument 2 to function json_contains; a JSON string or JSON type is required.
  
# 这里必须是要字符串 1
mysql> SELECT * FROM lnmp WHERE JSON_CONTAINS(category, '1', '$.id');
+----+------------------------------+-----------+
| id | category                     | tags      |
+----+------------------------------+-----------+
|  1 | {"id": 1, "name": "lnmp.cn"} | [1, 2, 3] |
+----+------------------------------+-----------+
1 row in set (0.01 sec)
```

对于数组类型的 JSON 的查询，比如说 tags 中包含有 2 的数据，同样要用 `JSON_CONTAINS `函数，同样第二个参数也需要是字符串

```python
mysql> SELECT * FROM lnmp WHERE JSON_CONTAINS(tags, '2');
+----+------------------------------+-----------+
| id | category                     | tags      |
+----+------------------------------+-----------+
|  1 | {"id": 1, "name": "lnmp.cn"} | [1, 2, 3] |
+----+------------------------------+-----------+
1 row in set (0.00 sec)
```

## 更新

- 如果是整个 json 更新的话，和插入时类似的。

```mysql
mysql> UPDATE lnmp SET tags = '[1, 3, 4]' WHERE id = 1;
Query OK, 1 row affected (0.00 sec)
Rows matched: 1  Changed: 1  Warnings: 0
```

- 但如果要更新 JSON 下的元素

MySQL 并不支持` column->path`的形式

``` mysql                                                  
mysql> UPDATE lnmp SET category->'$.name' = 'lnmp', tags->'$[0]' = 2 WHERE id = 1;
ERROR 1064 (42000): You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near '->'$.name' = 'lnmp', tags->'$[0]' = 2 WHERE id = 1' at line 1
```

则可能要用到以下几个函数

| name             | Desc                             |
| ---------------- | -------------------------------- |
| `JSON_INSERT()`  | 插入新值，但不会覆盖已经存在的值 |
| `JSON_SET()`     | 插入新值，并覆盖已经存在的值     |
| `JSON_REPLACE()` | 只替换存在的值                   |
| `JSON_REMOVE()`  | 删除 JSON 元素                   |
| ``               |                                  |
| ``               |                                  |

示例

```mysql
# JSON_INSERT()
mysql> UPDATE lnmp SET category = JSON_INSERT(category, '$.name', 'lnmp', '$.url', 'www.lnmp.cn') WHERE id = 1;
Query OK, 1 row affected (0.00 sec)
Rows matched: 1  Changed: 1  Warnings: 0
# JSON_SET()
mysql> UPDATE lnmp SET category = JSON_SET(category, '$.host', 'www.lnmp.cn', '$.url', 'http://www.lnmp.cn') WHERE id = 1;
Query OK, 1 row affected (0.00 sec)
Rows matched: 1  Changed: 1  Warnings: 0
# JSON_REPLACE()
mysql> UPDATE lnmp SET category = JSON_REPLACE(category, '$.name', 'php', '$.url', 'http://www.php.net') WHERE id = 2;
Query OK, 1 row affected (0.01 sec)
Rows matched: 1  Changed: 1  Warnings: 0
# JSON_REMOVE()
mysql> UPDATE lnmp SET category = JSON_REMOVE(category, '$.url', '$.host') WHERE id = 1;
Query OK, 1 row affected (0.01 sec)
Rows matched: 1  Changed: 1  Warnings: 0
```

## 空数组

```
JSON类型的数，可以支持null，{}, []，{'k':'v'}，这些类型。

null：默认值就是null，可以单独设置null；

{}：空的键值对，可以用cast('{}' as JSON)来设置；

[]：空的数组，可以用cast('[] as JSON')来设置，注意这里并不是集合的概念，里面的值是允许重复的；

{'k':'v'}：有键值对的数组，可以用cast来设置；
```

示例

```mysql
mysql> select * from test_json;
+----+------------------------------------+---------+
| id | j                                  | name    |
+----+------------------------------------+---------+
|  1 | {"url": "lnmp.cn", "name": "lnmp"} |         |
|  2 | NULL                               |         |
|  3 | {}                                 |         |
|  4 | NULL                               | brother |
|  5 | [100, 100, 200]                    | sister  |
+----+------------------------------------+---------+
rows in set (0.00 sec)
```

## key为int

- key需将int转换为string

添加子key为int类型的数据，直接添加int型的子key是有问题的

```mysql
insert into test_json (j) values(cast('{0:"100",1:"200"}' as JSON));
ERROR 3141 (22032): Invalid JSON text in argument 1 to function cast_as_json: "Missing a name for object member." at position 1.
```

添加子key为string类型，但是值为数字型的数据

```mysql
mysql> insert into test_json (j) values(cast('{"0":"100","1":"200"}' as JSON));
Query OK, 1 row affected (0.01 sec)

mysql> select * from test_json;
+----+------------------------------------+---------+
| id | j                                  | name    |
+----+------------------------------------+---------+
|  1 | {"url": "lnmp.cn", "name": "lnmp"} |         |
|  2 | NULL                               |         |
|  3 | {}                                 |         |
|  4 | NULL                               | brother |
|  5 | [100, 100, 200]                    | sister  |
|  6 | {"100": "100", "200": "200"}       |         |
|  7 | {"0": "100", "1": "200"}           |         |
+----+------------------------------------+---------+
rows in set (0.00 sec)
```

结论：如果要添加数字型的子key，必须包含引号，int型转成string型才可以

- 按照key查找条目

```mysql
mysql> select * from test_json where JSON_CONTAINS(j, '"100"', '$."0"');
+----+--------------------------+------+
| id | j                        | name |
+----+--------------------------+------+
|  7 | {"0": "100", "1": "200"} |      |
+----+--------------------------+------+
row in set (0.00 sec)
```

注意：

```
1. 是第二个参数必须是带印号，
2. 是第三个参数的键值名称必须带双引号，而不是之前的'$.name'这样的方式。
```

- select中带有子key

```mysql
mysql> select j->'$."0"' from test_json where id=7;
+------------+
| j->'$."0"' |
+------------+
| "100"      |
+------------+
row in set (0.00 sec)
```





# 视图

视图只能用于查询，当原始表中数据变化时，视图会自动更新。

```python
# 创建视图
create view 视图名 as 查询的SQL语句

# 查看创建视图的语句
show create view 视图名

# 删除视图
drop view 视图名

# 更新视图
# 方法一：drop + create
# 方法二：create or replace view
# 方法三：alter view 视图名 as 查询的SQL语句
```

# 事务

```python
# 开始
begin; 或者 start transaction;
# 提交
commit;
# 回滚
rollback;

# 使用保留点
# 创建占位符
savepoint 保留点名
# 回退至保留点
rollback to 保留点名
```

# 触发器

使用触发器可以定制用户对表进行【增、删、改】操作时前后的行为，注意：没有查询

触发器无法由用户直接调用，而知由于对表的【增/删/改】操作被动引发的。

- 创建

```python
# 创建触发器
# 行触发器,MySQL不支持语句触发器
create trigger 触发器名{before|after}
insert|update|delete}
on 表名
for each row
# 开启
begin
#触发器执行的操作
#如果要获取表中的数据当作条件,insert时用NEW，delete时用OLD,update时都可以
# 结束
end;
```
示例
```shell
# 插入前
CREATE TRIGGER tri_before_insert_tb1 BEFORE INSERT ON tb1 FOR EACH ROW
# 插入后
CREATE TRIGGER tri_after_insert_tb1 AFTER INSERT ON tb1 FOR EACH ROW
# 删除前
CREATE TRIGGER tri_before_delete_tb1 BEFORE DELETE ON tb1 FOR EACH ROW
# 删除后
CREATE TRIGGER tri_after_delete_tb1 AFTER DELETE ON tb1 FOR EACH ROW
# 更新前
CREATE TRIGGER tri_before_update_tb1 BEFORE UPDATE ON tb1 FOR EACH ROW
# 更新后
CREATE TRIGGER tri_after_update_tb1 AFTER UPDATE ON tb1 FOR EACH ROW

# 样例
# 创建表
create table user(
    id int primary key auto_increment,
    name varchar(20) not null,
    reg_time datetime, # 注册用户的时间
    affirm enum('yes','no') # no表示该用户执行失败
);

create table userLog(
    id int primary key auto_increment,
    u_name varchar(20) not null,
    u_reg_time datetime # 注册用户的时间
);

# 创建触发器 delimiter 默认情况下，delimiter是分号 触发器名称应遵循命名约定[trigger time]_[table name]_[trigger event]
delimiter //
create trigger after_user_insert after insert on user for each row
begin
    if new.affirm = 'yes' then
        insert into userLog(u_name,u_reg_time) values(new.name,new.reg_time);
    end if;

end //
delimiter ;


#往用户表中插入记录，触发触发器，根据if的条件决定是否插入数据
insert into user(name,reg_time,affirm) values ('张三',now(),'yes'),('李四',now(),'yes'),('王五',now(),'no');


# 查看日志表，发现多了两条记录 ，大家应该看到for each row就明白了

mysql> select * from userlog;
+----+--------+---------------------+
| id | u_name | u_reg_time          |
+----+--------+---------------------+
|  1 | 张三   | 2018-06-14 17:52:49 |
|  2 | 李四   | 2018-06-14 17:52:49 |
+----+--------+---------------------+
2 rows in set (0.00 sec)

# 注意：请注意，在为INSERT定义的触发器中，可以仅使用NEW关键字。不能使用OLD关键字。但是，在为DELETE定义的触发器中，没有新行，因此您只能使用OLD关键字。在UPDATE触发器中，OLD是指更新前的行，而NEW是更新后的行
```
- 删除

```
# 删除触发器
drop trigger 触发器名

```





# 存储过程

```python
# 创建存储过程
create procedure 存储过程名(参数)
# 开启
begin
# 要执行的操作
# 终止
end;

# 检查存储过程
show create procedure 存储过程名
# 调用存储过程
call 存储过程名([参数]);

# 删除存储过程
drop procedure 存储过程名;


eg:
#  创建
create procedure ordertotal(
	in onumber int,
    out ototal decimal(8,2)
)
begin
	select sum(item_price*quantity)
	from orderitems
    where order_num = onumber
    into ototal;
end;
# 调用
call ordertotal(20005, @total);
select @total
```

# 游标

```sql
# mysql中的游标只能用于存储过程(和函数)
# 创建游标
create procedure processorders()
begin
	declare ordernumbers cursor
	for
	select order_num from orders;
end;

# 打开游标
open ordernumbers

# 关闭游标
close ordernumbers

eg:
# 创建
create procedure processorders()
begin
	-- declare local variable
	declare done boolean default 0;
	decalre o int;
	declare t decimal(8,2);
	-- declare the cursor
	declare ordernumbers cursor
	for 
	select order_num from orders;
	-- declare continue handler
	declare continue handler for sqlstate '02000' set done=1;
	-- create a table to store the results
	create table if not exists ordertotals
	(order_num int, total decimal(8,2));
	-- open the cursor
	open ordernumbers
	-- loop through all rows
	repeat
		-- get order number
		fetch ordernumbers into o;
		--get the total for this order
		call ordertotal(o, 1, t)
		--insert order and total into ordertotals
		insert into ordertotals(order_num, total)
		values(o, t);
	-- nd of loop
	until done end repeat;
	-- close the cursor
	close ordernumbers;
end;

# 使用
select * from  ordertotals;	
```

# 事件

## 创建事件

每天凌晨两点自动删除de_records表中七天以前的数据

```sql
CREATE EVENT event_delete_de_records_7days ON SCHEDULE EVERY 1 DAY STARTS '2018-01-01 02:00:00' DO DELETE FROM de_records WHERE timestamp <DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 7 DAY);
```

每天凌晨三点自动删除as_records表中七天以前的数据

```sql
CREATE EVENT event_delete_as_records_7days ON SCHEDULE EVERY 1 DAY STARTS '2018-01-01 03:00:00' DO DELETE FROM as_records WHERE timestamp <DATE_SUB(CURRENT_TIMESTAMP, INTERVAL 7 DAY);
```

## 开启事件

```python
# 检测事件是否开启
show variables like 'event_scheduler';
# 开启事件
set global event_scheduler = on;
# 登录mysql中
show databases;
use mysql;
```

## 查看事件

```python
# 方法一
select name from event;
# 方法二
use chatroom;
show events;
```

## 删除事件

```python
# 语法：
drop event 表名

# 示例
drop event 4332432143243
```

# 导入导出sql文件

## Windows

```
1.导出整个数据库
mysqldump -u 用户名 -p 数据库名 > 导出的文件名
mysqldump -u dbuser -p dbname > dbname.sql

2.导出一个表
mysqldump -u 用户名 -p 数据库名 表名> 导出的文件名
mysqldump -u dbuser -p dbname users> dbname_users.sql

3.导出一个数据库结构
mysqldump -u dbuser -p -d --add-drop-table dbname >d:/dbname_db.sql
-d 没有数据 --add-drop-table 在每个create语句之前增加一个drop table

4.导入数据库
常用source 命令
进入mysql数据库控制台，如
mysql -u root -p
mysql>use 数据库
然后使用source命令，后面参数为脚本文件(如这里用到的.sql)
mysql>source d:/dbname.sql
注意：要在盘符的根目录放置sql文件，或在登录mysql时的目录下
```

## Linux

```
一、导出数据库用mysqldump命令（注意mysql的安装路径，即此命令的路径）：
1、导出数据和表结构：
mysqldump -u用户名 -p密码 数据库名 > 数据库名.sql
#/usr/local/mysql/bin/   mysqldump -uroot -p abc > abc.sql
敲回车后会提示输入密码

2、只导出表结构
mysqldump -u用户名 -p密码 -d 数据库名 > 数据库名.sql
#/usr/local/mysql/bin/   mysqldump -uroot -p -d abc > abc.sql

注：/usr/local/mysql/bin/  --->  mysql的data目录


二、导入数据库
1、首先建空数据库
mysql>create database abc;

2、导入数据库
方法一：
（1）选择数据库
mysql>use abc;
（2）设置数据库编码
mysql>set names utf8;
（3）导入数据（注意sql文件的路径）
mysql>source /home/abc/abc.sql;
方法二：
mysql -u用户名 -p密码 数据库名 < 数据库名.sql
#mysql -uabc_f -p abc < abc.sql
```



