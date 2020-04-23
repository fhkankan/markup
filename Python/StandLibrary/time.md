# time

```
import time
```

## 函数

| 函数                                        | 输入   | 输出   | 功能                                                         |
| ------------------------------------------- | ------ | ------ | ------------------------------------------------------------ |
| time.time()                                 | 无     | 时间戳 | 返回当前时间的时间戳(1970纪元后经过的浮点秒数)               |
| time.clock()                                | 无     | 时间戳 | 用以浮点数计算的秒数返回当前的CPU时间。用来衡量不同程序的耗时，比time.time()更有用；python3.3后不建议使用 |
| time.process_time()                         | 无     | 时间戳 | 分析的处理时间：内核和用户空间CPU时间之和                    |
| Time.perf_counter()                         | 无     | 时间戳 | 基准测试的性能计数器                                         |
| time.localtime([secs])                      | 时间戳 | 元组   | 接收时间戳(1970纪元后经过的浮点秒数)并返回当地时间的时间元组t |
| time.gmtime([secs])                         | 时间戳 | 元组   | 接收时间戳(1970纪元后经过的浮点秒数)并返回时间元组t          |
| time.ctime([secs])                          | 时间戳 | 字符串 | 作用相当于asctime(localtime(secs)),获取当前时间字符串        |
| time.mktime(tupletime)                      | 元组   | 时间戳 | 接收时间元组并返回时间戳(1970纪元后经过的浮点秒数)           |
| time.strftime(fmt[,tupletime])              | 元组   | 字符串 | 接收时间元组，并返回以可读字符串表示的当地时间，格式由ftm决定 |
| time.asctime([tupletime])                   | 元组   | 字符串 | 接受时间元组并返回一个可读的形式为'Tue Dec 11 18:07:14 2008'的24个字符的字符串 |
| time.strptime(str,fmt = '%a%b%d%H:%M:%S%Y') | 字符串 | 元组   | 根据fmt的格式把一个时间字符串解析为时间元组                  |
| time.sleep(secs)                            | 秒数   |        | 推迟调用线程的运行，secs指秒数                               |

eg

```python
import time
# 时间文本转时间戳，精确到秒
a = '2016-10-01 10:00:00'
a = int(time.mktime(time.strptime(a, '%Y-%m-%d %H:%M:%S')))
print a
# 时间戳转时间文本
b = int(time.time())
b = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(b))
```

代码执行速度时要使用到time中区别

```
cpu 的运行机制：cpu是多任务的，例如在多进程的执行过程中，一段时间内会有对各进程被处理。一个进程从从开始到结束其实是在这期间的一些列时间片断上断断续续执行的。所以这就引出了程序执行的cpu时间（该程序单纯在cpu上运行所需时间）和墙上时钟wall time。
time.time()是统计的wall time(即墙上时钟)，也就是系统时钟的时间戳（1970纪元后经过的浮点秒数）。所以两次调用的时间差即为系统经过的总时间。
time.clock()是统计cpu时间 的工具，这在统计某一程序或函数的执行速度最为合适。两次调用time.clock()函数的插值即为程序运行的cpu时间。
```

eg

```python
import time

def procedure():
    a = 0
    for i in range(100000):
        a += 1

t1 = time.clock()
t2 = time.time()
procedure()
time.sleep(2)
t1_1 = time.clock()
t2_1 = time.time()

print 'CPU:', t1_1 - t1
print 'time:', t2_1 - t2 
```

