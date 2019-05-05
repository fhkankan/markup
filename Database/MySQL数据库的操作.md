# 启动环境

## ubuntu

```python
# 启动服务
sudo service mysql start 
# 查看进程中是否存在mysql服务
ps ajx|grep mysql
# 停止服务
sudo service mysql stop
# 重启服务
sudo service mysql restart

# 帮助文档
mysql --help
# 基本连接
mysql -uroot -pmysql
# 远程连接
mysql -h 192.168.5.400 -uroot -pmysql -P 5001
# 退出
quit/exit/ctrl+d
```

## windows

```python
# 启动服务
net start mysql
# 关闭服务
net stop mysql
# 帮助文档
mysql --help
# 基本连接
mysql -uroot -pmysql
# 退出
quit/exit
```

#数据库引擎

```python
# 查看默认引擎
show engines

# 更改默认引擎
# 方法一：在配置文件中更改，后重启
default-storage-engine=InnoDB
# 方法二：建表的时候指定
create table mytbl(   
    id int primary key,   
    name varchar(50)   
)type=MyISAM;
# 方法三：建表后修改
alter table mytbl2 type = InnoDB;
```

mysql的两大数据库引擎区别

```
	Innodb引擎提供了对数据库事务的支持，支持行级锁和外键，不支持FULLTEXT类型的索引，而且它没有保存表的行数，当SELECT COUNT(*) FROM TABLE时需要扫描全表。由于锁的粒度更小，写操作不会锁定全表，所以在并发较高时，使用Innodb引擎会提升效率。但是使用行级锁也不是绝对的，如果在执行一个SQL语句时MySQL不能确定要扫描的范围，InnoDB表同样会锁全表。索引结构是B+Tree,索引文件本身就是数据文件，索引的key就是主键，辅助索引数据域存储的也是相应记录主键的值而不是地址，所以当以辅助索引查找时，会先根据辅助索引找到主键，再根据主键索引找到实际的数据。

	MyISAM不支持数据库事务的支持，也不支持行级锁和外键，支持FULLTEXT类型的索引，存储了表的行数，于是SELECT COUNT(*) FROM TABLE时只需要直接读取已经保存好的值而不需要进行全表扫描。如果表的读操作远远多于写操作且不需要数据库事务的支持，那么MyIASM也是很好的选择。索引结构是B+Tree,数据域存储的内容为实际数据的地址，它的索引和实际的数据是分开的，只不过是用索引指向了实际的数据

	InnoDB引擎：大尺寸的数据集，支持事务，故障恢复，写操作比较多时无表锁，主键查询比较快，并发建议
	MyISAM引擎：支持FULLTEXT类型的索引，不支持事务，无行锁，读操作多时选择
```
# 全球化与本地化

```sql
# 字符集：为字母和符号的集合
# 编码：为摸个字符集成员的内部表示
# 校对：为规定字符如何比较的指令

# 查看支持的字符集
show character set;
# 查看支持的校对
show collation;

eg:
create table mytable
(
	column1 int,
    column2 varchar(10),
    column3 varchar(10) character set latin1 collate latin1_general_ci
)
default character set hebrew
collate hebrew_general_ci
```

# 安全管理

```
# 管理用户
use mysql;
select user from user;

# 创建用户账户
create user ben identified by 'p@$$w0rd'

# 重命名用户账号
rename user ben to bforta

# 删除用户账户
drop user bforta

# 查看权限
show grants for bforta

# 授予权限
grant select on mydatabase.* to bforta

# 撤销权限
revoke select on mydatabase.* from bforta

# 更改密码
set password for bforta = password('new password')
# 当前用户更改
set password = password('new password')
```



# 对用户的操作

```python
# 查看用户表结构
desc user;
# 查看用户
select host,user,authentication_string from user;
# 创建用户
# ip用% 时表示所有ip均可
create user '用户名'@'允许连接的主机IP' identified by '密码';
create user 'test'@'%' identified by 'test';
create user 'test'@'127.0.0.1' identified by 'test';
create user 'test'@'localhost' identified by 'test';
# 删除用户
drop user '用户名'@'允许连接的主机IP';
delete from mysql.user where user = '用户名' and host = '允许连接的主机IP';
# 修改root用户密码
select user();
set password=password('123456'); 
flush privileges;
exit;
# 修改用户密码
update mysql.user set authentication_string = password('新密码') where user = '用户名' and host = '允许连接的主机IP'; 
# 设置用户权限
grant 权限 on 数据库.* to '用户名'@'允许连接的主机IP'; //需要用户存在
grant 权限 on 数据库.* to '用户名'@'允许连接的主机IP' identified by '密码'; //自动创建一个用户
eg.
grant insert,delete,update,select on test.* to 'test'@'localhost';
grant all privileges on test.* to 'test'@'localhost'; //为test用户设置test数据库的所有权限
# 回收权限
revoke insert on python.* from 'py1'@'%';
# 修改完成，需刷新权限
flush privileges;

# 忘记密码
在配置文件中添加
skip-grant-tables
重启服务，再设置密码
退出服务，注释掉代码
重启服务
```

# 数据库的操作

```python
# 查看所有数据库
show databases;
# 查看创建数据库的语句
show create database xxx;
# 切换数据库
use xxx;
# 显示数据库版本
select version();
# 显示时间
select now();
# 查看当前使用得数据库
select database();

# 创建数据库
# 默认字符集是latin1(拉丁文，不支持中文)
create database xxx charset=utf8;
# 修改数据库字符集
alter database xxx charset=utf8;

# 删除数据库
drop database xxx;
# 修改输入提示符
prompt python>

# 数据库的备份
# ubuntu下运行
mysqldump –uroot –p 数据库名 > python.sql;
# 数据库的恢复
# 连接mysql，创建新的数据库，退出连接，执行如下命令
mysql -uroot –p 新数据库名 < python.sql
```

# 数据表的操作

```python
# 查看当前数据库中所有表
show tables;
# 查看表结构
desc zzz;
# 查看表的创建语句
show create table zzz;

# 创建表
 create table zzz(
		id int unsigned not null primary key auto_increment,
		name varchar(20) not null,
   		age tinyint unsigned default 0,
    	height decimal(5,2),
    	gender enum('男','女','保密'),
    	cls_id int unsigned default 0,
     	foreign key(cls_id) references classes(id)
	);
# 修改表名字
rename table zzz to new_name;
# 修改表字符集
alter table zzz charset utf8
# 修改表---添加字段
alter table students add birthday datetime;
# 修改表-修改字段：重命名版
alter table students change birthday birth datetime not null;
# 修改表-修改字段：不重命名版
alter table students modify birth date not null;
# 修改表-删除字段
alter table students drop birthday

# 删除表
drop table zzz;
```

# 数据行的操作

## 增

```python
# 全列插入一行
insert into students values(0,'郭靖',1,'蒙古','2016-1-2');
# 部分列插入一行
insert into students(name,hometown) values('黄蓉','桃花岛');
# 多行插入
insert into classes values(0,'python1'),(0,'python2');
# 插入select语句
insert into  classes select clas_id from students where id=3;
# 多行多列插入(字段需一致)
INSERT INTO TPersonnelChange(
    UserId,
    DepId,
    SubDepId,
    PostionType,
    AuthorityId,
    ChangeDateS,
    InsertDate,
    UpdateDate,
    SakuseiSyaId
)SELECT
    UserId,
    DepId,
    SubDepId,
    PostionType,
    AuthorityId,
    DATE_FORMAT(EmployDate, '%Y%m%d'),
    NOW(),
    NOW(),
    1
FROM
    TUserMst
WHERE
    `Status` = 0
AND QuitFlg = 0
AND UserId > 2
```

## 改

```python
# 改特定的行和列
update students set gender=0,hometown='北京' where id=5;
```

## 删

```python
# 物理删除
delete from students where id=5;
# 逻辑删除
update students set isdelete=1 where id=1;
# 删除数据保留表结构

```

## 查

由于Console的宽度有限，因此在查询数据库记录时，就会出现不能在一行完全显示全部字段内容的情况，于是为查询带了很大不便。现在只需在查询语句后面加一个“**|G**”，就可以实现记录的竖行显示

### 查询所有列

```
select * from 表名;
```

### 查询指定列

```
select 字段名,字段名 from 表名;
```

###查询指定行

```
select 字段名 from 表名 where 字段名 = ?;
```

### 查询多行

```
select 字段名 from 表名 limit 0,10;	# 前10行，第一个参数是开始的索引，第二个是个数
```

### 字段别名

```
select 字段名 as 别名 from 表名;
```

### 去掉重复数据

```python
# 当指定多个字段时，多个字段必须全部匹配才会成功
select distinct 字段名 from 表名; 
```

### 滤空修正

```
select ifnull(字段名,替换值) from 表名;
```

### 使用算术表达式

```python
select 字段名 + 字段名 as '别名' from 表名;
```

### 字符串拼接

```
select concat(str1,str2,...) as '别名' from 表名;

CONCAT(str1,str2,…)  
返回结果为连接参数产生的字符串。如有任何一个参数为NULL ，则返回值为 NULL。
注意：
如果所有参数均为非二进制字符串，则结果为非二进制字符串。 
如果自变量中含有任一二进制字符串，则结果为一个二进制字符串。
一个数字参数被转化为与之相等的二进制字符串格式；若要避免这种情况，可使用显式类型 cast,
SELECT CONCAT(CAST(int_col AS CHAR), char_col)

CONCAT_WS(separator,str1,str2,...)
CONCAT_WS() 代表 CONCAT With Separator ，是CONCAT()的特殊形式。第一个参数是其它参数的分隔符。分隔符的位置放在要连接的两个字符串之间。分隔符可以是一个字符串，也可以是其它参数。
注意：
如果分隔符为 NULL，则结果为 NULL。函数会忽略任何分隔符参数后的 NULL 值。

group_concat([DISTINCT] 要连接的字段 [Order BY ASC/DESC 排序字段] [Separator '分隔符'])

```

### 常用函数

```
select min(字段名) from 表名;
select max(字段名) from 表名;
select avg(字段名) from 表名;
select round(avg(字段名), 2) from 表名;
select count(字段名) from 表名;
select sum(字段名) from 表名;
```

###条件查询运算符

```python
# =,>,>=,<,<=
select * from 表名 where 字段名 > 3; 
# !=,<>
select * from 表名 where 字段名 != 3; 
# 多值
select * from 表名 where 字段名 in (3,5); 
# 区间
select * from 表名 where 字段名 between 3 and 5; 
# 空值
select * from 表名 where 字段名 is null; 
# and,or,not
select * from 表名 where 字段名 is not null;
select * from 表名 where 字段名 = ? and 字段名 = ?; 
select * from 表名 where 字段名 = ? or 字段名 = ?;
```

### 模糊查询

```python
# % 表示任意个数的任意字符
select * from 表名 where 字段名 like '%xxx'; 
# _ 表示单个的任意字符
select * from 表名 where 字段名 like '_xxx';
# [] 表示单个字符的取值范围
select * from 表名 where 字段名 rlike '[0-9]abc'; 
# [^] 表示单个字符的非取值范围
select * from 表名 where 字段名 rlike '[^0-9]abc';
# \ 表示转义,查询字段中包含%的数据
select * from 表名 where 字段名 like '%\%%'; 
```

### 分组查询

```python
# 特定条件
select 字段名 from 表名where 查询条件 group by 字段名 having 查询条件;
# group_concat
select 字段名1,group_concat(字段名2) from 表名where 查询条件 group by 字段名1 having 查询条件;
# 聚合函数max,min,avg,sum,count
select name, avg(score), sum(score>60) as cnt from student group by name
# 在最后新增一行，记录当前列的所有记录总和
select gender,count(*) from students group by gender with rollup;
```

### 排序查询

```python
# asc 表示升序
select * from 表名 where 查询条件 order by 字段名 asc; 
# desc 表示降序排列
select * from 表名 where 查询条件 order by 字段名 desc;
```

### 嵌套查询

```python
# 当子查询返回的值只有一个时，才可以直接使用 =,>,< 等运算符;
# 当子查询返回的值是一个结果集时，需要使用 in,not in,any,some,all 等操作符;
# all : 匹配结果集中的所有结果, 例如 > all 表示比所有结果都大;
# any : 匹配结果信中任意一个结果, 例如 < any 表示只要比其中任意一个小就符合条件;
# in : 查询的数据在子查询的结果集中,相当于 = any;
# not in : 查询的数据不在子查询结果集中,相当于 <> all;
# some : 匹配结果集中的部分结果，基本等同于 any,不常用;
# 子查询通常只用于返回单个字段的结果集，如果需要匹配多个字段，可以使用多表联查(后面讲);
select * from 表名 where 字段名 = (select 字段名 from 表名 where 字段名=?);
eg.
select * from 表名 where 字段名 > all (select 字段名 from 表名);
```

### 多表联查

```python
# 数据表的别名
select * from 表名 as 别名; 
#  交叉连接(笛卡尔积)
# 显示两张表的乘积
select * from 表1 cross join 表2 on 连接条件 where 查询条件; //cross join on 可省略
# 默认就是交叉连接,如果 where 条件中有等值判断则称为等值连接
select * from 表1,表2 where 查询条件;
# 内连接
# 显示两张表的交集
select * from 表1 inner join 表2 on 连接条件 where 查询条件; //inner 可省略
# 左外连接
# 左显全集,右显交集
select * from 表1 left outer join 表2 on 连接条件 where 查询条件; //outer 可省略
# 右外连接
# 左显交集,右显全集
select * from 表1 right outer join 表2 on 连接条件 where 查询条件; //outer 可省略
# 联合查询
# 显示两张表的并集,要求两张表的查询字段名必须保持一致
select id,name from 表1 union select id,name from 表2;
# 全外连接(MySQL不支持)
# 显示两张表的并集
select * from 表1 full outer join 表2 on 连接条件 where 查询条件; //outer 可省略
# 可以通过左外连接和右外连接的联合查询来实现
select * from 表1 left join 表2 on 连接条件 where 查询条件 union select * from 表1 right join 表2 on 连接条件 where 查询条件;
```

### 常用字符串函数

```python
# upper 和 ucase
# 把所有字符转换为大写字母
select upper(name) from 表名;
# lower 和 lcase
# 把所有字符转换为小写字母
select lcase(name) from 表名;
# replace(str, from_str, to_str)
# 把str中的from_str替换为to_str
select replace(字段名,替换前的值,替换后的值) from 表名;
#  repeat(str,count)
# 返回str重复count次的字符串
select repeat('abc',2) from 表名; // abcabc
#  reverse(str)
# 逆序字符串
select reverse('abc') from 表名; // cba
#  insert(str,pos,len,newstr)
# 把str中pos位置开始长度为len的字符串替换为newstr
select insert('abcdef',2,3,'hhh'); // ahhhef
#  substring(str from pos)
# 从str中的pos位置开始返回一个新字符串 
select substring('abcdef',3); // cdef
# substring_index(str,delim,count)
# 返回str中第count次出现的delim之前的所有字符,如果count为负数,则从右向左
select substring_index('abacadae','a',3); // abac
# ltrim(str)
# 去除字符串左边的空格
select ltrim(' abc');
# rtrim(str)
# 去除字符串右边的空格
select rtrim('abc ');
#  trim(str)
# 去除字符串左右两边的空格
select trim(' abc ');
# mid(str,pos,len)
# 从str中的pos位置开始返回len个长度的字符串
select mid('abcdef',2,3); // bcd
# lpad(str,len,padstr)
# 在str左边填充padstr直到str的长度为len
select lpad('abc',8,'de'); // dededabc
# rpad(str,len,padstr)
# 在str右边填充padstr直到str的长度为len
select rpad('abc',8,'de'); // abcdeded
#  left(str,len)
# 返回str左边的len个字符
select left('abcd',2); // ab
# right(str,len)
# 返回str右边的len个字符
select right('abcd',2); // cd
# position(substr in str)
# 返回substr在str中第一次出现的位置
select position('c' in 'abcdc'); // 3
# length(str)
# 返回字符串的长度
select length('abcd'); // 4
# concat(str1,str2,...)
# 合并字符串
select concat('abc','def','gh'); // abcdefgh
```

### 日期时间函数

```python
# dayofweek(date)
# 返回date是星期几,1代表星期日,2代表星期一...
select dayofweek('2017-04-09');
# weekday(date)
# 返回date是星期几,0代表星期一,1代表星期二...
select weekday('2017-04-09');
# dayname(date)
# 返回date是星期几(按英文名返回)
select dayname('2017-04-09');
# dayofmonth(date)
# 返回date是一个月中的第几天(范围1-31)
select dayofmonth('2017-04-09');
# dayofyear(date)
# 返回date是一年中的第几天(范围1-366)
select dayofyear('2017-04-09');
# month(date)
#返回date中的月份数值
select month('2017-04-09');
# monthname(date)
# 返回date是几月(按英文名返回)
select monthname('2017-04-09');
# quarter(date)
# 返回date是一年中的第几个季度
select quarter('2017-04-09');
# week(date,first)
# 返回date是一年中的第几周(first默认值是0,表示周日是一周的开始,取值为1表示周一是一周的开始)
select week('2017-04-09');
select week('2017-04-09',1);
# year(date)
# 返回date的年份
select year('2017-04-09');
# hour(time)
# 返回time的小时数
select hour('18:06:53');
# minute(time)
# 返回time的分钟数 select minute('18:06:53');
# second(time)
# 返回time的秒数
select second('18:06:53');
# period_add(p,n)
# 增加n个月到时期p并返回(p的格式为yymm或yyyymm)
select period_add(201702,2);
# period_diff(p1,p2)
# 返回在时期p1和p2之间的月数(p1,p2的格式为yymm或yyyymm)
select period_diff(201605,201704);
# date_format(date,format)
# 根据format字符串格式化date
select date_format('2017-04-09','%d-%m-%y');
# time_format(time,format)
# 根据format字符串格式化time
select time_format('12:22:33','%s-%i-%h');
# curdate() 和 current_date()
# 以'yyyy-mm-dd'或yyyymmdd的格式返回当前日期值
select curdate();
select current_date();
# curtime() 和 current_time()
# 以'hh:mm:ss'或hhmmss格式返回当前时间值
select curtime();
select current_date();
# now(),sysdate(),current_timestamp()
# 以'yyyy-mm-dd hh:mm:ss'或yyyymmddhhmmss格式返回当前日期时间 select now();
select sysdate();
select current_timestamp();
# unix_timestamp()
# 返回一个unix时间戳(从'1970-01-01 00:00:00'开始到当前时间的秒数)
select unix_timestamp();
# sec_to_time(seconds)
# 把秒数seconds转化为时间time
select sec_to_time(3666);
# time_to_sec(time)
# 把时间time转化为秒数seconds
select time_to_sec('01:01:06');
# top N 问题
select * from 表名 limit 0,N; //第0行取N行的数据
```

### 对查询出的数据进行筛选替换 

```python
# 此方式仅限等于条件 select case when 字段条件 then '要显示的值' end from 表名 //此方式条件更多
select case sex when 1 then '男' when 2 then '女' else '人妖' end from 表名 
```

# 数据表列的查询

```
SHOW DATABASES                                //列出 MySQL Server 数据库。
SHOW TABLES [FROM db_name]                    //列出数据库数据表。
SHOW CREATE TABLES tbl_name                    //导出数据表结构。
SHOW TABLE STATUS [FROM db_name]              //列出数据表及表状态信息。
SHOW COLUMNS FROM tbl_name [FROM db_name]     //列出资料表字段
SHOW FIELDS FROM tbl_name [FROM db_name]，DESCRIBE tbl_name [col_name]。
SHOW FULL COLUMNS FROM tbl_name [FROM db_name]//列出字段及详情
SHOW FULL FIELDS FROM tbl_name [FROM db_name] //列出字段完整属性
SHOW INDEX FROM tbl_name [FROM db_name]       //列出表索引。
SHOW STATUS                                  //列出 DB Server 状态。
SHOW VARIABLES                               //列出 MySQL 系统环境变量。
SHOW PROCESSLIST                             //列出执行命令。
SHOW GRANTS FOR user                         //列出某用户权限
```

# 全文本搜索

```sql
# mysql数据库的MyISAM支持全文本搜索，不支持事务
# 启用全文本搜索
create table 表名
（
	字段名  	类型  	约束，
	note_text  text 	null,
	fulltext(note_text)
）engine=myisam

# 使用全文本搜索
select note_text from 表名
where match(note_text) against('匹配字符串')

# 使用查询扩展
select note_text from 表名
where match(note_text) against('匹配字符串' with query expansion)

# 使用布尔文本搜索
select note_text from 表名
where match(note_text) against('匹配字符串的布尔操作' in boolean mode)
```

| 布尔操作符 | 说明                                                         |
| ---------- | ------------------------------------------------------------ |
| +          | 包含，词必须存在                                             |
| -          | 排除，词必须不出现                                           |
| >          | 包含，增加等级值                                             |
| <          | 包含，减少等级值                                             |
| ()         | 把词组成子表达式<br>允许子表达式作为一个组被包含、排除、排列等 |
| ~          | 取消一个词的排序值                                           |
| *          | 词尾的通配符                                                 |
| ""         | 定义一个短句<br>匹配整个短语以便包含或排除这个短语           |

#索引

##普通索引
```python
create index 索引名 on 表名(字段名(索引长度));
alter table 表名 add index 索引名 (字段名(索引长度));
create table 表名(字段名 字段类型,字段名 字段类型,index 索引名(字段名(索引长度));
```
##唯一索引
```
create unique index 索引名 on 表名(字段名(索引长度));
alter table 表名 add unique 索引名 (字段名(索引长度));
create table 表名(字段名 字段类型,字段名 字段类型,unique 索引名 (字段名(索引长度));
```
##全文索引
```
# 只支持 MyISAM 引擎
create fulltext index 索引名 on 表名(字段名);
alter table 表名 add fulltext 索引名(字段名);
create table 表名(字段名 字段类型,字段名 字段类型,fulltext (字段名)）;
```
##组合索引
```
create index 索引名 on 表名(字段名(索引长度),字段名(索引长度),...);
alter table 表名 add index 索引名 (字段名(索引长度),字段名(索引长度),...;
create table 表名(字段名 字段类型,字段名 字段类型,index 索引名 (字段名(索引长度),字段名(索引长度));
```
## 查看索引
```
show index from 表名;
```
##删除索引
```
alter table 表名 drop index 索引名;
```
# 约束

```python
# 主键约束
create table 表名(字段名 字段类型 primary key,字段 字段类型,...);
//一个表只能有一个主键,这个主键可以由一列或多列组成
create table 表名(字段1 字段类型,字段2 字段类型,primary key(字段1,字段2);
# 唯一键约束
create table 表名(字段名 字段类型 unique,字段名 字段类型,...);
# 外键约束
create table 表1(字段1 字段类型,字段2 字段类型,foreign key(字段1) references 表2(字段名),...);
# 非空约束
create table 表名(字段名 字段类型 not null,字段名 字段类型,...);
# 默认值约束
create table 表名(字段名 字段类型 default 默认值,字段名 字段类型,...);
# check约束
# (MySQL 不支持)
create table 表名(字段1 字段类型,字段2 字段类型,check(字段1 > 30),...);
```
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