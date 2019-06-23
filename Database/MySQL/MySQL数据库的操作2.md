# 视图

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

12351

ps -elf |grep python



