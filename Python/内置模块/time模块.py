import time

"""
时间表示方式
"""

# 在python中，通常由两种方式表示时间
# 1、时间戳：从1970年1月1日00:00:00开始到现在的秒数
# 2、时间元组struct_time，其中有九个元素
# tm_year 年
# tm_mon 月
# tm_mday 日
# tm_hour 小时0~23
# tm_min 分0~59
# tm_sec 秒0~59
# tm_wday 星期0~6
# tm_yday 一年中的第几天，1~366
# tm_isdst 是否是夏令时，默认为1夏令时


"""
函数
"""
# time.time()
# 返回当前时间的时间戳(1970纪元后经过的浮点秒数)

# time.localtime([secs])
# 接收时间戳(1970纪元后经过的浮点秒数)并返回当地时间的时间元组t

# time.ctime([secs])
# 作用相当于asctime(localtime(secs)),获取当前时间字符串

# time.asctime([tupletime])
# 接受时间元组并返回一个可读的形式为'Tue Dec 11 18:07:14 2008'的24个字符的字符串

# time.sleep(secs)
# 推迟调用线程的运行，secs指秒数

# time.clock()
# 用以浮点数计算的秒数返回当前的CPU时间。用来衡量不同成都的耗时，比time.time()更有用

# time.gmtime([secs])
# 接收时间戳(1970纪元后经过的浮点秒数)并返回时间元组t

# time.mktime(tupletime)
# 接收时间元组并返回时间戳(1970纪元后经过的浮点秒数)

# time.strftime(fmt[,tupletime])
# 接收以时间元组，并返回以可读字符串表示的当地时间，格式由ftm决定

# time.strptime(str,fmt = '%a%b%d%H:%M:%S%Y')
# 根据fmt的格式把一个时间字符串解析为时间元组









