# Sqoop

## 概述

Sqoop是一款进行数据传输的工具，可在hadoop的hdfs和关系型数据库之间传输数据。

可以使用sqoop把数据从MySQL或Oracle导入到hdfs中，也可以把hdfs数据导入到MySQL或Oracle

sqoop可以自动执行数据传输的大部分过程，使用MapReduce导入和导出数据，提供并行操作和容错

**原理**：通过编写sqoop命令把sqoop命令翻译成mapreduce，通过mapreduce连接各种数据源实现数据的传递。

## 安装部署

- 下载安装包url
- 解压文件夹

```
tar -zxvf /home/hadoop/software/sqoopxxx.tar.gz -C ~/app/
```

- 配置

环境变量

```shell
vi ~/.bash_profile
export SQOOP_HOME=/home/hadoop/app/sqoopxxx
export PATH=$SQOOP_HOME/bin:$PATH
source ~/.bash_profile
```

拷贝mysql驱动到`$SQOOP_HOME/lib`下

```shell
cp /home/hadoop/app/hivexxx/lib/mysql-connector-java-5.1.47.jqr  /home/hadoop/app/sqoopxxx/lib
```

配置相关服务

```shell
cd /home/hadoop/app/sqoopxxx/conf
cp sqoop-env-template.sh sqoop-env.sh
vi sqoop-env.sh
export HADOOP_COMMON_HOME==/home/hadoop/app/hadoopxxx/
export HADOOP_MAPRED_HOME==/home/hadoop/app/hadoopxxx/
export HIVE_HOME==/home/hadoop/app/hivexxx/
```

- 测试

```
sqoop-version
```

## 导入数据

- 准备mysql数据

```sql
# 创建表
CREATE table u(id int primary key auto_increment, fname varchar(20),lname varchar(20));  
# 向表中插入数据
insert into u (fname, lanme) values('George','washington');
insert into u (fname, lanme) values('George','bush');
insert into u (fname, lanme) values('Bill','clinton');
insert into u (fname, lanme) values('Bill','gates');
```

- sqoop导入

语法

```shell
sqoop import [控制参数] [导入参数]
```

示例

```shell
sqoop import --connect jdbc:mysql://192.168.19.137:3306/test --usename root -password password --table u -m 1  # 导入操作、数据源、访问方式、导入控制、目标地址

sqoop import --connect jdbc:mysql://192.168.19.137:3306/test --usename root -password password --table u --target-dir /tmp/u -m 1 # 可指定hdfs上数据存放的目录，默认地址为hdfs上/user/linux用户名/mysql表名/
```

- 通过hive建立外表导入数据到hive

```shell
create external table u4(id INT, fname STRING, lname STRING)
row format delimited fields terminated by ','
location '/user/root/u/'
```
