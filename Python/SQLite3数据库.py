# SQLite3
# 嵌入式关系型数据库，它的数据库就是一个文件。

"""
数据类型
"""
# smallint       ---> 16位整数
# integer        ---> 32位整数
# decimal(p,s)   ---> p是精确值，s是小数位数
# float          ---> 32位实数
# double         ---> 64位实数
# char(n)        ---> n长度字符串，不能超过254
# varchar(n)     ---> 长度不固定最大字符串长度为n,n不超过4000
# graphic(n)     ---> 和char(n)一样，但是单位是两个字符double-bytes,n不超过127(中文字)
# vargraphic(n)  ---> 可变长度且最大长度为n
# date           ---> 包含了年、月、日
# time           ---> 包含了时、分、秒
# timestamp      ---> 包含了年、月、日、时、分、秒、千分之一秒


"""
函数
"""
# 时间/日期
# datetime(日期/时间，修整符，修正符...) ---> 产生日期和时间
# date(日期/时间，修整符，修正符...)     ---> 产生日期
# time()                                 ---> 产生时间
# strtime(格式，日期/时间，修整符，修正符...) ---> 对上面三个函数产生的日期和时间进行格式化

# 算数
# abs(x)            ---> 返回绝对值  
# max(x,y[,...])    ---> 返回最大值
# min(x,y[,...])    ---> 返回最小值
# random(*)         ---> 返回随机数
# round(x[,y])      ---> 四舍五入

# 字符串处理
# length(x)         ---> 返回字符串字符个数
# lower(x)          ---> 大写转小写
# upper(x)          ---> 小写转大写
# substr(x,y,z)     ---> 截取子串
# like(A,B)         ---> 确定给定的字符串与在制定的模式是否匹配

# 其他
# typeof(x)         ---> 返回数据的类型
# last_insert_rowid() ---> 返回最后插入数据的ID


"""
模块
"""
# Sqlite3.Version：常量，版本号
# Sqlite3.Connect(database)：函数，链接到数据库，返回connect对象
# Sqlite3.Cursor：游标对象
# Sqlite3.Row：行对象


"""
数据库编程
"""

# 导入模块
import sqlite3
# 建立连接，返回connection对象
con = sqlite3.connect(connectstring)
# 创建游标对象
cur = con.curso()
# 使用游标对象的execute执行SQL语句
# cur.execute(sql)   ---> 执行SQL语句
# cur.execute(sql,parameters) --->执行带参数的SQL语句
# cur.executemany(sql,seq_of_parameters) ---> 根据参数执行多次SQL语句
# cur.executescript(sql_script) --->执行SQL脚本

# 获取游标查询的结果
# cur.fetchone()     ---> 返回结果集的下一行，无数据时返回None
# cur.fetchall()     ---> 返回结果集的剩余行，无数据时返回None
# cur.fetchmany()    ---> 返回结果集的多行，无数据时返回None

# 数据库的提交和回滚
# con.commit()      ---> 事务提交
# con.rokkback()    ---> 事务回滚

# 关闭
# cur.close()       ---> 关闭游标对象
# con.close()       ---> 关闭连接对象




















