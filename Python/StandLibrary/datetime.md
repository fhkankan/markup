# datetime

datetime是date和time的结合体，包括date和time的所有信息

最大最小年份

```
datetime.MINYEAR是1
datetime.MAXYEAR是9999
```

datetime定义了5个类

```
datetime.date	# 日期的类
datetime.time  # 时间的类
datetime.datetime  # 日期时间
datetime.timedelta  # 时间间隔
datetime.tzinfo  # 与时区有关的相关信息
```

引用

```
import datetime
```

## datetime.date

表示一个日期，由年、月、日组成 

- 构造函数

```python
date(year,month,day)
# 接收年、月、日三个参数，返回一个date对象
```

- 属性

```python
year,month,day
# 返回date对象的年、月、日
```

- 方法

```python
timetuple()
# 返回一个time的时间格式对象，等价time.localtime()

today()
# 返回当前日期date对象。等价于 fromtimestamp(time.time())

toordinal()
# 返回公元公历开始到现在的天数。公元1年1月1日为1

weekday()
# 返回星期几。0(星期一)到9(星期日)
```

## datetime.time

表示时间，由时、分、秒及微秒组成 

- 构造函数

```python
class datetime.time(hour[,minute[,second[,microsecond[,tzinfo]]]])
# hour的范围[0,24),minute的范围[0,60),second的范围[0,60),microsecond的范围[0,1000000)
```

- 属性

```python
hour,minute,second,microsecond
# time对象的小时，分钟，秒，毫秒数
```

- 方法

```python
dst()
# 返回时区信息的描述。如果实例是没有txinfo参数测返回空

isoformat()
# 返回HH:MM:SS[.mmmmmm][+HH:MM]格式字符串
```

## datetime.datetime

- 构造函数

```
datetime(year,month,day[,hour[,minute[,second[,microsecond[,tzinfo]]]]])
```

- 方法

```python
today()
# 返回一个表示当前本地时间的datetime对象
now([tz])
# 返回当前日期和时间的datetime对象,tz指定时区
utcnow()
# 返回当前utc时间的datetime对象
fromtimestamp(timestamp[,tz])
# 根据时间戳数值，返回一个datetime对象,tz指定时区
utcfromtimestamp(timestamp)
# 根据时间戳数值，返回一个datetime对象
strptime(date_string, format)
# 将格式字符串转换为datetime对象
strftime(format)
# 将格式字符串转换为datetime对象
combine()
# 根据给定date,time对象合并后，返回一个对应值的datetime对象
ctime()
# 返回ctime格式的字符串
date()
# 返回具有相同year、month、day的date对象
```

# 示例

## 获取当前的日期和时间

```
>>> from datetime import datetime
>>> now = datetime.now() # 获取当前datetime
>>> print(now)
2015-05-18 16:28:07.198690
>>> print(type(now))
<class 'datetime.datetime'>
```

## 获取指定日期和时间

```
>>> from datetime import datetime
>>> dt = datetime(2015, 4, 19, 12, 20) # 用指定日期时间创建datetime
>>> print(dt)
2015-04-19 12:20:00
```

## 时间戳表示

```
在计算机中，时间实际上是用数字表示的。我们把1970年1月1日 00:00:00 UTC+00:00时区的时刻称为epoch time，记为0（1970年以前的时间timestamp为负数），当前时间就是相对于epoch time的秒数，称为timestamp。

timestamp = 0 = 1970-1-1 00:00:00 UTC+0:00
# 对应的北京时间是：
timestamp = 0 = 1970-1-1 08:00:00 UTC+8:00

timestamp的值与时区毫无关系，因为timestamp一旦确定，其UTC时间就确定了，转换到任意时区的时间也是完全确定的，这就是为什么计算机存储的当前时间是以timestamp表示的，因为全球各地的计算机在任意时刻的timestamp都是完全相同的（假定时间已校准）。
```

## datetime--->timestamp

把一个`datetime`类型转换为timestamp只需要简单调用`timestamp()`方法

注意Python的timestamp是一个浮点数。如果有小数位，小数位表示毫秒数。

某些编程语言（如Java和JavaScript）的timestamp使用整数表示毫秒数，这种情况下只需要把timestamp除以1000就得到Python的浮点表示方法

```
>>> from datetime import datetime
>>> dt = datetime(2015, 4, 19, 12, 20) # 用指定日期时间创建datetime
>>> dt.timestamp() # 把datetime转换为timestamp
1429417200.0


time.mktime(dateTime.timetuple())
```

## timestamp--->datetime

要把timestamp转换为`datetime`，使用`datetime`提供的`fromtimestamp()`方法

注意到timestamp是一个浮点数，它没有时区的概念，而datetime是有时区的。上述转换是在timestamp和本地时间做转换。

```
>>> from datetime import datetime
>>> t = 1429417200.0
>>> print(datetime.fromtimestamp(t)) # 本地时间
2015-04-19 12:20:00
>>> print(datetime.utcfromtimestamp(t)) # UTC时间
2015-04-19 04:20:00
```

## str ---> datetime

通过`datetime.strptime()`实现，需要一个日期和时间的格式化字符串，详见https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

注意转换后的datetime是没有时区信息的

```
>>> from datetime import datetime
>>> cday = datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
>>> print(cday)
2015-06-01 18:19:59
```

## datetime ---> str

通过`strftime()`实现的，同样需要一个日期和时间的格式化字符串：

```
>>> from datetime import datetime
>>> now = datetime.now()
>>> print(now.strftime('%a, %b, %Y %m %d %H:%M:%S'))
```

## str ---> timestamp

```
time.mktime(string_toDatetime(strTime).timetuple()
```

 ## timestamp --->str

```
time.strftime("%Y-%m-%d-%H", tiem.localtime(stamp))
```



## datetime加减

对日期和时间进行加减实际上就是把datetime往后或往前计算，得到新的datetime。加减可以直接用`+`和`-`运算符，不过需要导入`timedelta`这个类：

```
>>> from datetime import datetime, timedelta
>>> now = datetime.now()
>>> now
datetime.datetime(2015, 5, 18, 16, 57, 3, 540997)
>>> now + timedelta(hours=10)
datetime.datetime(2015, 5, 19, 2, 57, 3, 540997)
>>> now - timedelta(days=1)
datetime.datetime(2015, 5, 17, 16, 57, 3, 540997)
>>> now + timedelta(days=2, hours=12)
datetime.datetime(2015, 5, 21, 4, 57, 3, 540997)
```

## 本地时间转换为UTC时间

本地时间是指系统设定时区的时间，例如北京时间是UTC+8:00时区的时间，而UTC时间指UTC+0:00时区的时间。

一个`datetime`类型有一个时区属性`tzinfo`，但是默认为`None`，所以无法区分这个`datetime`到底是哪个时区，除非强行给`datetime`设置一个时区：

```
>>> from datetime import datetime, timedelta, timezone
>>> tz_utc_8 = timezone(timedelta(hours=8)) # 创建时区UTC+8:00
>>> now = datetime.now()
>>> now
datetime.datetime(2015, 5, 18, 17, 2, 10, 871012)
>>> dt = now.replace(tzinfo=tz_utc_8) # 强制设置为UTC+8:00
>>> dt
datetime.datetime(2015, 5, 18, 17, 2, 10, 871012, tzinfo=datetime.timezone(datetime.timedelta(0, 28800)))
```

## 时区转换

先通过`utcnow()`拿到当前的UTC时间，再转换为任意时区的时间

时区转换的关键在于，拿到一个`datetime`时，要获知其正确的时区，然后强制设置时区，作为基准时间。

利用带时区的`datetime`，通过`astimezone()`方法，可以转换到任意时区。

注：不是必须从UTC+0:00时区转换到其他时区，任何带时区的`datetime`都可以正确转换

```
# 拿到UTC时间，并强制设置时区为UTC+0:00:
>>> utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
>>> print(utc_dt)
2015-05-18 09:05:12.377316+00:00

# astimezone()将格林威治标准时间转换时区为北京时间:
>>> bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
>>> print(bj_dt)
2015-05-18 17:05:12.377316+08:00

# astimezone()将格林威治标准时间转换时区为东京时间:
>>> tokyo_dt = utc_dt.astimezone(timezone(timedelta(hours=9)))
>>> print(tokyo_dt)
2015-05-18 18:05:12.377316+09:00

# astimezone()将bj_dt转换时区为东京时间:
>>> tokyo_dt2 = bj_dt.astimezone(timezone(timedelta(hours=9)))
>>> print(tokyo_dt2)
2015-05-18 18:05:12.377316+09:00
```

