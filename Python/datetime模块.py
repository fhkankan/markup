import date

# 支持日期和时间运算，还有更有效的处理和格式化输出，支持时区处理
# 包含三个类：date,time,datetime

"""
date类
"""
# 表示一个日期，由年、月、日组成

# 构造函数
# date(year,month,day)
# 接收年、月、日三个参数，返回一个date对象

# 常用函数
# timetuple()
# 返回一个time的时间格式对象，等价time.localtime()

# today()
# 返回当前日期date对象。等价于 fromtimestamp(time.time())

# toordinal()
# 返回公元公历开始到现在的天数。公元1年1月1日为1

# weekday()
# 返回星期几。0(星期一)到9(星期日)

# year,month,day
# 返回date对象的年、月、日


"""
time类
"""
# 表示时间，由时、分、秒及微秒组成

# 构造函数
# class datetime.time(hour[,minute[,second[,microsecond[,tzinfo]]]])
# hour的范围[0,24),minute的范围[0,60),second的范围[0,60),microsecond的范围[0,1000000)

# 常用函数
# dst()
# 返回时区信息的描述。如果实例是没有txinfo参数测返回空

# isoformat()
# 返回HH:MM:SS[.mmmmmm][+HH:MM]格式字符串


"""
datetime类
"""
# 构造函数
# datetime(year,month,day[,hour[,minute[,second[,microsecond[,tzinfo]]]]])

# 常用函数
# datetime.now()
# 返回当前日期和时间，其类型是datetime

# combine()
# 根据给定date,time对象合并后，返回一个对应值的datetime对象

# ctime()
# 返回ctime格式的字符串

# date()
# 返回具有相同year、month、day的date对象

# fromtimestamp()
# 根据时间戳数值，返回一个datetime对象

# now()
# 返回当前时间









