# Hive

Hive 由 Facebook 实现并开源，是基于 Hadoop 的一个数据仓库工具，可以将结构化的数据映射为一张数据库表，并提供 HQL(Hive SQL)查询功能，底层数据是存储在 HDFS 上。

**Hive 本质**: 将 SQL 语句转换为 MapReduce 任务运行，使不熟悉 MapReduce 的用户很方便地利用 HQL 处理和计算 HDFS 上的结构化的数据,是一款基于 HDFS 的 MapReduce **计算框架**

**主要用途**：用来做离线数据分析，比直接用 MapReduce 开发效率更高。

## 安装部署

- Hive 安装前需要安装好 JDK 和 Hadoop。配置好环境变量。
- 下载Hive的安装包 http://archive.cloudera.com/cdh5/cdh/5/ 并解压

```shell
 tar -zxvf hive-1.1.0-cdh5.7.0.tar.gz  -C ~/app/
```

- 配置

配置文件

```shell
# 进入到解压后的hive目录找到 conf目录, 修改配置文件
cp hive-env.sh.template hive-env.sh
vi hive-env.sh
# 在hive-env.sh中指定hadoop的路径
HADOOP_HOME=/home/hadoop/app/hadoop-2.6.0-cdh5.7.0
```

环境变量

 ```shell
 vi ~/.bash_profile
 
 export HIVE_HOME=/home/hadoop/app/hive-1.1.0-cdh5.7.0
 export PATH=$HIVE_HOME/bin:$PATH
 
 source ~/.bash_profile
 ```

- 元数据存储

根据元数据存储的介质不同，分为下面两个版本，其中 derby 属于内嵌模式。实际生产环境中则使用 mysql 来进行元数据的存储。

内置derby版

```shell
bin/hive  # 启动即可使用

# 缺点：不同路径启动hive，每一个hive拥有一套自己的元数据，无法共享
```
mysql 版
```shell
# 1.上传 mysql驱动到 hive安装目录的lib目录下
mysql-connector-java-5.*.jar
# 2.配置 Mysql 元数据库信息(MySql安装见文档)
vi conf/hive-site.xml 
# 添加入下信息
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
<!-- 插入以下代码 -->
    <property>
        <name>javax.jdo.option.ConnectionUserName</name>
        <value>hive</value><!-- 指定mysql用户名 -->
    </property>
    <property>
        <name>javax.jdo.option.ConnectionPassword</name>
        <value>hive</value><!-- 指定mysql密码 -->
    </property>
   <property>
    <name>javax.jdo.option.ConnectionURL</name>mysql
        <value>jdbc:mysql://127.0.0.1:3306/hive</value>
    </property><!-- 指定mysql数据库地址 -->
    <property>
        <name>javax.jdo.option.ConnectionDriverName</name>
        <value>com.mysql.jdbc.Driver</value><!-- 指定mysql驱动 -->
    </property>
        <!-- 到此结束代码 -->
  <property>
    <name>hive.exec.script.wrapper</name>
    <value/>
    <description/>
  </property>
</configuration>   
```
- hive启动

```shell
# 启动docker 
service docker start

# 通过docker启动mysql
docker start mysql  

# 启动hive的metastore元数据服务
hive --service metastore

# 启动hive
hive

# 如上所配，mysql的用户hive密码hive
```

## 架构组件

<img src="images/hive2.jpg" alt="hive2"/>

- 用户接口

包括 CLI、JDBC/ODBC、WebGUI。

```
- CLI(command line interface)为 shell 命令行
- JDBC/ODBC 是 Hive 的 JAVA 实现，与传统数据库JDBC 类似
- WebGUI 是通过浏览器访问 Hive。
- HiveServer2基于Thrift, 允许远程客户端使用多种编程语言如Java、Python向Hive提交请求
```
-  元数据存储

通常是存储在关系数据库如 mysql/derby 中。

Hive 将元数据存储在数据库中。

Hive 中的元数据包括
```
- 表的名字
- 表的列
- 分区及其属性
- 表的属性（是否为外部表等）
- 表的数据所在目录等。
```
- 解释器、编译器、优化器、执行器

完成 HQL 查询语句从词法分析、语法分析、编译、优化以及查询计划的生成。

生成的查询计划存储在 HDFS 中，并在随后由 MapReduce 调用执行

- Hive 与 Hadoop 的关系

Hive 利用 HDFS 存储数据，利用 MapReduce 查询分析数据。

Hive是数据仓库工具，没有集群的概念，如果想提交Hive作业只需要在hadoop集群 Master节点上装Hive就可以了

## 数据库特征

与传统关系数据库对比

<table>
  <tr>
    <th></th>
    <th>Hive</th>
    <th>关系型数据库</th>
  </tr>
  <tr>
    <td> ANSI SQL </td>
    <td> 不完全支持 </td>
    <td> 支持 </td>
  </tr>
  <tr>
    <td> 更新 </td>
    <td> INSERT OVERWRITE\INTO TABLE(默认) </td>
    <td> UPDATE\INSERT\DELETE </td>
  </tr>
  <tr>
    <td> 事务 </td>
    <td> 不支持(默认) </td>
    <td> 支持 </td>
  </tr>
  <tr>
    <td> 模式 </td>
    <td> 读模式 </td>
    <td> 写模式 </td>
  </tr>
  <tr>
    <td> 查询语言 </td>
    <td> HQL  </td>
    <td> SQL</td>
  </tr>
  <tr>
    <td> 数据存储 </td>
    <td> HDFS </td>
    <td> Raw Device or Local FS </td>
  </tr>
  <tr>
    <td> 执行 </td>
    <td> MapReduce </td>
    <td> Executor</td>
  </tr>
  <tr>
    <td> 执行延迟 </td>
    <td> 高 </td>
    <td> 低 </td>
  </tr>
  <tr>
    <td> 子查询 </td>
    <td> 只能用在From子句中 </td>
    <td> 完全支持 </td>
  </tr>
  <tr>
    <td> 处理数据规模 </td>
    <td> 大 </td>
    <td> 小 </td>
  </tr>
  <tr>
    <td> 可扩展性 </td>
    <td> 高 </td>
    <td> 低 </td>
  </tr>
  <tr>
    <td> 索引 </td>
    <td> 0.8版本后加入位图索引 </td>
    <td> 有复杂的索引 </td>
  </tr>
</table>

hive支持的数据类型
```shell
# 原子数据类型  
TINYINT 
SMALLINT 
INT 
BIGINT 
BOOLEAN 
FLOAT 
DOUBLE 
STRING 
BINARY 
TIMESTAMP 
DECIMAL 
CHAR 
VARCHAR 
DATE

# 复杂数据类型
ARRAY
MAP
STRUCT
```

hive中表的类型
```
托管表 (managed table) (内部表)
外部表
```

## 数据模型

- Hive 中所有的数据都存储在 HDFS 中，没有专门的数据存储格式
- 在创建表时指定数据中的分隔符，Hive 就可以映射成功，解析数据。
- Hive 中包含以下数据模型：
    - db：在 hdfs 中表现为 hive.metastore.warehouse.dir 目录下一个文件夹
    - table：在 hdfs 中表现所属 db 目录下一个文件夹
    - external table：数据存放位置可以在 HDFS 任意指定路径
    - partition：在 hdfs 中表现为 table 目录下的子目录
    - bucket：在 hdfs 中表现为同一个表目录下根据 hash 散列之后的多个文件

## HQL操作

### 简单操作

```sql
# 创建数据库
CREATE DATABASE test;
# 显示所有数据库
SHOW DATABASES;

# 查询表数据
select * from student;

# 分组group by和count
select classNo, count(score) from student where score>=60 group by classNo;
```

### 内部外部表

<table>
  <tr>
    <th></th>
    <th>内部表(managed table)</th>
    <th>外部表(external table)</th>
  </tr>
  <tr>
    <td> 概念 </td>
    <td> 创建表时无external修饰 </td>
    <td> 创建表时被external修饰 </td>
  </tr>
  <tr>
    <td> 数据管理 </td>
    <td> 由Hive自身管理 </td>
    <td> 由HDFS管理 </td>
  </tr>
  <tr>
    <td> 数据保存位置 </td>
    <td> hive.metastore.warehouse.dir  （默认：/user/hive/warehouse） </td>
    <td> hdfs中任意位置 </td>
  </tr>
  <tr>
    <td> 删除时影响 </td>
    <td> 直接删除元数据（metadata）及存储数据 </td>
    <td> 仅会删除元数据，HDFS上的文件并不会被删除 </td>
  </tr>
  <tr>
    <td> 表结构修改时影响 </td>
    <td> 修改会将修改直接同步给元数据  </td>
    <td> 表结构和分区进行修改，则需要修复（MSCK REPAIR TABLE table_name;）</td>
  </tr>
</table>

创建表

```sql
# 内部表
CREATE TABLE student(classNo string, stuNo string, score int) row format delimited fields terminated by ',';
# row format delimited fields terminated by ','  指定了字段的分隔符为逗号，所以load数据的时候，load的文本也要为逗号，否则加载后为NULL。hive只支持单个字符的分隔符，hive默认的分隔符是\001

# 外部表
CREATE EXTERNAL TABLE student2 (classNo string, stuNo string, score int) row format delimited fields terminated by ',' location '/tmp/student';
# 位置任意
```

导入数据

```sql
# 导入数据
# 本地文件/home/hadoop/tmp/student.txt
# C01,N0101,82
# C01,N0102,59
load data local inpath '/home/hadoop/tmp/student.txt'overwrite into table student;  
# 这个命令将student.txt文件复制到hive的warehouse目录中，这个目录由hive.metastore.warehouse.dir配置项设置，默认值为/user/hive/warehouse。
# Overwrite选项将导致Hive事先删除student目录下所有的文件, 并将文件内容映射到表中。Hive不会对student.txt做任何格式处理，因为Hive本身并不强调数据的存储格式。

load data local inpath '/home/hadoop/tmp/student.txt' overwrite into table student2;
```

显示表信息

```sql
desc formatted student;
```

删除表查看结果

```sql
delete table student;
```

### 分区表

概念

```
- 随着表的不断增大，对于新纪录的增加，查找，删除等(DML)的维护也更加困难。对于数据库中的超大型表，可以通过把它的数据分成若干个小表，从而简化数据库的管理活动，对于每一个简化后的小表，我们称为一个单个的分区。
- hive中分区表实际就是对应hdfs文件系统上独立的文件夹，该文件夹内的文件是该分区所有数据文件。
- 分区可以理解为分类，通过分类把不同类型的数据放到不同的目录下。
- 分类的标准就是分区字段，可以一个，也可以多个。
- 分区表的意义在于优化查询。查询时尽量利用分区字段。如果不使用分区字段，就会全部扫描。
```

创建分区表

```sql
# 数据
tom,4300
jerry,12000
mike,13000
jake,11000
rob,10000

# sql
create table employee (name string, salary bigint) partitioned by (date1 string) row format delimited fields terminated by ',' lines terminated by '\n' stored as textfile;
```

查看表的分区

```sql
show partitions employee;
```

加载数据到分区

```sql
load data local inpath '/home/hadoop/tmp/employee.txt' into table employee partition(date1='2018-12-01');

# 如果重复加载同名文件，不会报错，会自动创建一个*_copy_1.txt
```

添加分区

```sql
alter table employee add if not exists partition(date1='2018-12-01');

# 外部分区表即使有分区的目录结构, 也必须要通过hql添加分区, 才能看到相应的数据
hadoop fs -mkdir /user/hive/warehouse/emp/dt=2018-12-04
hadoop fs -copyFromLocal /tmp/employee.txt /user/hive/warehouse/test.db/emp/dt=2018-12-04/employee.txt
# 此时查看表中数据发现数据并没有变化, 需要通过hql添加分区
alter table emp add if not exists partition(dt='2018-12-04');
```

总结
```
- 利用分区表方式减少查询时需要扫描的数据量
    - 分区字段不是表中的列, 数据文件中没有对应的列
    - 分区仅仅是一个目录名
    - 查看数据时, hive会自动添加分区列
    - 支持多级分区, 多级子目录
```

### 动态分区

在写入数据时自动创建分区(包括目录结构)

创建表

```sql
create table employee2 (name string, salary bigint) partitioned by (date1 string) row format delimited fields terminated by ',' lines terminated by '\n' stored as textfile;
```

导入数据

```sql
insert into table employee2 partition(date1) select name,salary,date1 from employee;
```

使用动态分区需要设置参数

```shell
set hive.exec.dynamic.partition.mode=nonstrict;
```

### 函数

[参考文档](https://cwiki.apache.org/confluence/display/Hive/LanguageManual+UDF)

- 内置运算符

```
- 关系运算符
- 算术运算符
- 逻辑运算符
- 复杂运算
```

- 内置函数

```
- 简单函数: 日期函数 字符串函数 类型转换 
- 统计函数: sum avg distinct
- 集合函数
- 分析函数
```

常用命令
```shell
# 显示所有函数
show functions; 

# 显示某个函数详情
desc function 函数名;
esc function extended 函数名;
```

- UDF

当 Hive 提供的内置函数无法满足你的业务处理需要时，此时就可以考虑使用用户自定义函数（UDF：user-defined function）

**UDF**：就是做一个mapper，对每一条输入数据，映射为一条输出数据。

**UDAF**:就是一个reducer，把一组输入数据映射为一条(或多条)输出数据。

一个脚本至于是做mapper还是做reducer，又或者是做udf还是做udaf，取决于我们把它放在什么样的hive操作符中。放在select中的基本就是udf，放在distribute by和cluster by中的就是reducer。

**运行java已经编写好的UDF**

```shell
# 在hdfs中创建 /user/hive/lib目录
hadoop fs -mkdir /user/hive/lib

# 把 hive目录下 lib/hive-contrib-hive-contrib-1.1.0-cdh5.7.0.jar 放到hdfs中
hadoop fs -put hive-contrib-1.1.0-cdh5.7.0.jar /user/hive/lib/

# 把集群中jar包的位置添加到hive中
hive>add jar hdfs:///user/hive/lib/hive-contrib-1.1.0-cdh5.7.0.jar;

# 在hive中创建**临时**UDF
hive>CREATE TEMPORARY FUNCTION row_sequence as 'org.apache.hadoop.hive.contrib.udf.UDFRowSequence'

# 在之前的案例中使用**临时**自定义函数(函数功能: 添加自增长的行号)
hive>Select row_sequence(),* from employee;

# 创建**非临时**自定义函数
hive>CREATE FUNCTION row_sequence as 'org.apache.hadoop.hive.contrib.udf.UDFRowSequence' using jar 'hdfs:///user/hive/lib/hive-contrib-1.1.0-cdh5.7.0.jar';
```

**python创建一个UDF**

```shell
# 准备案例环境
# 创建表
CREATE table u(fname STRING,lname STRING);  
# 向表中插入数据
insert into table u values('George','washington');
insert into table u values('George','bush');
insert into table u values('Bill','clinton');
insert into table u values('Bill','gates');


# 编写map风格脚本udf.py
import sys
for line in sys.stdin:
    line = line.strip()
    fname , lname = line.split('\t')
    l_name = lname.upper()
    print '\t'.join([fname, str(l_name)])
    
# 本地模拟测试
cat test.txt | python udf.py

# 向hive中ADD file
# 通过hdfs加载
hadoop fs -put udf.py /user/hive/lib/   # 加载文件到hdfs
ADD FILE hdfs:///user/hive/lib/udf.py;  # hive从hdfs中加载python脚本
# 通过本地加载
ADD FILE /root/tmp/udf1.py;  

# transform
SELECT TRANSFORM(fname, lname) USING 'python udf.py' AS (fname, l_name) FROM u;
```

### 示例

内容推荐数据处理：根据用户行为以及文章标签筛选出用户最感兴趣(阅读最多)的标签

![hive3](images/hive3.png)

- 相关数据

 user_id article_id event_time

```
11,101,2018-12-01 06:01:10
22,102,2018-12-01 07:28:12
33,103,2018-12-01 07:50:14
11,104,2018-12-01 09:08:12
22,103,2018-12-01 13:37:12
33,102,2018-12-02 07:09:12
11,101,2018-12-02 18:42:12
35,105,2018-12-03 09:21:12
22,104,2018-12-03 16:42:12
77,103,2018-12-03 18:31:12
99,102,2018-12-04 00:04:12
33,101,2018-12-04 19:10:12
11,101,2018-12-05 09:07:12
35,102,2018-12-05 11:00:12
22,103,2018-12-05 12:11:12
77,104,2018-12-05 18:02:02
99,105,2018-12-05 20:09:11
```

文章数据

```
artical_id,artical_url,artical_keywords
101,http://www.itcast.cn/1.html,kw8|kw1
102,http://www.itcast.cn/2.html,kw6|kw3
103,http://www.itcast.cn/3.html,kw7
104,http://www.itcast.cn/4.html,kw5|kw1|kw4|kw9
105,http://www.itcast.cn/5.html,
```

- 环境准备

数据上传hdfs

```shell
hadoop fs -mkdir /tmp/demo
hadoop fs -mkdir /tmp/demo/user_action
```

创建外部表

```sql
# 用户行为表
drop table if exists user_actions;
CREATE EXTERNAL TABLE user_actions(
    user_id STRING,
    article_id STRING,
    time_stamp STRING
)
ROW FORMAT delimited fields terminated by ','
LOCATION '/tmp/demo/user_action';

# 文章表
drop table if exists articles;
CREATE EXTERNAL TABLE articles(
    article_id STRING,
    url STRING,
    key_words array<STRING>
)
ROW FORMAT delimited fields terminated by ',' 
COLLECTION ITEMS terminated BY '|' 
LOCATION '/tmp/demo/article_keywords';
/*
key_words array<STRING>  数组的数据类型
COLLECTION ITEMS terminated BY '|'  数组的元素之间用'|'分割
*/
```

查看数据

```sql
select * from user_actions;
select * from articles;
```

- 查询数据

分组查询每个用户的浏览记录
```sql
# collect_set：将group by中的某列转为一个数组返回并去重
select user_id,collect_set(article_id) 
from user_actions group by user_id;

"""
11      ["101","104"]
22      ["102","103","104"]
33      ["103","102","101"]
35      ["105","102"]
77      ["103","104"]
99      ["102","105"]
"""

# collect_list：将group by中的某列转为一个数组返回不去重
select user_id,collect_list(article_id) 
from user_actions group by user_id;

"""
11      ["101","104","101","101"]
22      ["102","103","104","103"]
33      ["103","102","101"]
35      ["105","102"]
77      ["103","104"]
99      ["102","105"]
"""

# sort_array: 对数组排序
select user_id,sort_array(collect_list(article_id)) as contents 
from user_actions group by user_id;
"""
11      ["101","101","101","104"]
22      ["102","103","103","104"]
33      ["101","102","103"]
35      ["102","105"]
77      ["103","104"]
99      ["102","105"]
"""
```

查看每一篇文章的关键字 lateral view explode

```sql
# explode函数 将array 拆分
select explode(key_words) from articles;

# lateral view 和 explode 配合使用,将一行数据拆分成多行数据，在此基础上可以对拆分的数据进行聚合
select article_id,kw from articles lateral view explode(key_words) t as kw;
"""
101     kw8
...
104     kw9
"""
select article_id,kw from articles lateral view outer explode(key_words) t as kw;
"""
101     kw8
...
105     NULL
#含有outer
"""
```

根据文章id找到用户查看文章的关键字

```sql
# 原始数据
101     http://www.itcast.cn/1.html     ["kw8","kw1"]
102     http://www.itcast.cn/2.html     ["kw6","kw3"]
103     http://www.itcast.cn/3.html     ["kw7"]
104     http://www.itcast.cn/4.html     ["kw5","kw1","kw4","kw9"]
105     http://www.itcast.cn/5.html     []

# sql
select a.user_id, b.kw from user_actions 
as a left outer JOIN (select article_id,kw from articles
lateral view outer explode(key_words) t as kw) b
on (a.article_id = b.article_id)
order by a.user_id;
"""
11      kw1
...
99      NULL
"""
```

根据文章id找到用户查看文章的关键字并统计频率

```sql
select a.user_id, b.kw,count(1) as weight 
from user_actions as a 
left outer JOIN (select article_id,kw from articles
lateral view outer explode(key_words) t as kw) b
on (a.article_id = b.article_id)
group by a.user_id,b.kw 
order by a.user_id,weight desc;

"""
11      kw1     4
...
99      kw6     1
"""
```

将用户查看的关键字和频率合并成 key:value形式

```sql
select a.user_id, concat_ws(':',b.kw,cast (count(1) as string)) as kw_w 
from user_actions as a 
left outer JOIN (select article_id,kw from articles
lateral view outer explode(key_words) t as kw) b
on (a.article_id = b.article_id)
group by a.user_id,b.kw;
"""
11      kw1:4
...
99      kw6:1
"""
```

将用户查看的关键字和频率合并成 key:value形式并按用户聚合

```sql
select cc.user_id,concat_ws(',',collect_set(cc.kw_w))
from(
select a.user_id, concat_ws(':',b.kw,cast (count(1) as string)) as kw_w 
from user_actions as a 
left outer JOIN (select article_id,kw from articles
lateral view outer explode(key_words) t as kw) b
on (a.article_id = b.article_id)
group by a.user_id,b.kw
) as cc 
group by cc.user_id;
"""
11      kw1:4,kw4:1,kw5:1,kw8:3,kw9:1
22      kw1:1,kw3:1,kw4:1,kw5:1,kw6:1,kw7:2,kw9:1
33      kw1:1,kw3:1,kw6:1,kw7:1,kw8:1
35      1,kw3:1,kw6:1
77      kw1:1,kw4:1,kw5:1,kw7:1,kw9:1
99      1,kw3:1,kw6:1
"""
```

将上面聚合结果转换成map

```sql
select cc.user_id,str_to_map(concat_ws(',',collect_set(cc.kw_w))) as wm
from(
select a.user_id, concat_ws(':',b.kw,cast (count(1) as string)) as kw_w 
from user_actions as a 
left outer JOIN (select article_id,kw from articles
lateral view outer explode(key_words) t as kw) b
on (a.article_id = b.article_id)
group by a.user_id,b.kw
) as cc 
group by cc.user_id;
"""
11      {"kw1":"4","kw4":"1","kw5":"1","kw8":"3","kw9":"1"}
22      {"kw1":"1","kw3":"1","kw4":"1","kw5":"1","kw6":"1","kw7":"2","kw9":"1"}
33      {"kw1":"1","kw3":"1","kw6":"1","kw7":"1","kw8":"1"}
35      {"1":null,"kw3":"1","kw6":"1"}
77      {"kw1":"1","kw4":"1","kw5":"1","kw7":"1","kw9":"1"}
99      {"1":null,"kw3":"1","kw6":"1"}
"""
```

将用户的阅读偏好结果保存到表中

```sql
create table user_kws as 
select cc.user_id,str_to_map(concat_ws(',',collect_set(cc.kw_w))) as wm
from(
select a.user_id, concat_ws(':',b.kw,cast (count(1) as string)) as kw_w 
from user_actions as a 
left outer JOIN (select article_id,kw from articles
lateral view outer explode(key_words) t as kw) b
on (a.article_id = b.article_id)
group by a.user_id,b.kw
) as cc 
group by cc.user_id;
```

从表中通过key查询map中的值

```sql
select user_id, wm['kw1'] from user_kws;
"""
11      4
22      1
33      1
35      NULL
77      1
99      NULL
"""
```

从表中获取map中所有的key 和 所有的value

```sql
select user_id,map_keys(wm),map_values(wm) from user_kws;
"""
11      ["kw1","kw4","kw5","kw8","kw9"] ["4","1","1","3","1"]
22      ["kw1","kw3","kw4","kw5","kw6","kw7","kw9"]     ["1","1","1","1","1","2","1"]
33      ["kw1","kw3","kw6","kw7","kw8"] ["1","1","1","1","1"]
35      ["1","kw3","kw6"]       [null,"1","1"]
77      ["kw1","kw4","kw5","kw7","kw9"] ["1","1","1","1","1"]
99      ["1","kw3","kw6"]       [null,"1","1"]
"""
```

用lateral view explode把map中的数据转换成多列

```sql
select user_id,keyword,weight from user_kws lateral view explode(wm) t as keyword,weight;
"""
11      kw1     4
11      kw4     1
11      kw5     1
11      kw8     3
11      kw9     1
22      kw1     1
22      kw3     1
22      kw4     1
22      kw5     1
22      kw6     1
22      kw7     2
22      kw9     1
33      kw1     1
33      kw3     1
33      kw6     1
33      kw7     1
33      kw8     1
35      1       NULL
35      kw3     1
35      kw6     1
77      kw1     1
77      kw4     1
77      kw5     1
77      kw7     1
77      kw9     1
99      1       NULL
99      kw3     1
99      kw6     1
"""
```

## python交互

