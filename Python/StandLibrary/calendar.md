# calendar

此模块与日历相关。

星期一是默认的每周第一天，星期日是默认的最后一天

更该设置需调用calendar.setfirstweekday()

引用

```
import calendar
```

## 函数

```python
calendar(year, w=2, l=1, c=6)
# 返回一个多行字符串格式的year年年历，3个月一行，间隔距离为c。每日宽度间隔为w个字符。每行长度为21*w+18+2*c。l是每星期行数

firstweekday()
# 返回当前每周起始日期的设置。默认情况下，首次才入calendar模块时返回0，星期一

isleap(year)
# 是闰年返回True，否则返回False

leapdays(y1,y2)
# 返回在y1，y2两年之间的闰年总数

month(year,month,w=2,l=1)
# 返回一个多行字符串格式的year年month月日历，两行标题，一周一行，每日宽度间隔为w个字符。每行长度为7*w+6。l是每星期行数

monthcalendar(year,month)
# 返回一个整数的单层嵌套列表。每个子列表装载代表一个星期的整数。year年month月外的日期都设为0；范围内的日子都由该月第几日表示，从1开始

monthrange(year,month)
# 返回两个整数。第一个是该月的星期几的日期码，第二个是该月的日期码。日从0(星期一)到6(星期日)；月从1到12

setfirstweekday(weekday)
# 设置每周的起始日期码，0(星期一)到6(星期日)

timegm(tupletime)
# 和time.gmtime相反，接收一个时间元组形式，返回该时刻的时间戳(1970纪元后经过的浮点秒数)

weekday(year,month,day)
# 返回给定日期的日期码。0(星期一)到6(星期日)。月从1到12
```

