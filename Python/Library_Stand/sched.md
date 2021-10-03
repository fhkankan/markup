# sched

## 概述

sched模块实现了一个通用事件调度器，在调度器类使用一个延迟函数等待特定的时间，执行任务。同时支持多线程应用程序，在每个任务执行后会立刻调用延时函数，以确保其他线程也能执行

```python
# 构造类
class sched.scheduler(timefunc=time.monotonic, delayfunc=time.sleep)
# scheduler 类定义了一个调度事件的通用接口。 它需要两个函数来实际处理“外部世界” —— timefunc 应当不带参数地调用，并返回一个数字（“时间”，可以为任意单位）。 delayfunc 函数应当带一个参数调用，与 timefunc 的输出相兼容，并且应当延迟其所指定的时间单位。 每个事件运行后还将调用 delayfunc 并传入参数 0 以允许其他线程有机会在多线程应用中运行。

# 方法属性
enterabs(time, priority, action, argument=(), kwargs={})
# 安排一个新事件。 time 参数应该有一个数字类型兼容的返回值，与传递给构造函数的 timefunc 函数的返回值兼容。 计划在相同 time 的事件将按其 priority 的顺序执行。 数字越小表示优先级越高。
# 执行事件意为执行 action(*argument, **kwargs)。 argument 是包含有 action 的位置参数的序列。 kwargs 是包含 action 的关键字参数的字典。
# 返回值是一个事件，可用于以后取消事件
enter(delay, priority, action, argument=(), kwargs={})
# 安排延后 delay 时间单位的事件。 除了相对时间，其他参数、效果和返回值与 enterabs() 的相同。
cancel(event)
# 从队列中删除事件。 如果 event 不是当前队列中的事件，则此方法将引发 ValueError。
empty()
# 如果事件队列为空则返回 True。
run(blocking=True)
# 运行所有预定事件。 此方法将等待（使用传递给构造函数的 delayfunc() 函数）进行下一个事件，然后执行它，依此类推，直到没有更多的计划事件。
# 如果 blocking 为false，则执行由于最快到期（如果有）的预定事件，然后在调度程序中返回下一个预定调用的截止时间（如果有）。
# action 或 delayfunc 都可以引发异常。 在任何一种情况下，调度程序都将保持一致状态并传播异常。 如果 action 引发异常，则在将来调用 run() 时不会尝试该事件。
# 如果一系列事件的运行时间比下一个事件之前的可用时间长，那么调度程序将完全落后。 不会发生任何事件；调用代码负责取消不再相关的事件。
queue
# 只读属性按照将要运行的顺序返回即将发生的事件列表。 每个事件都显示为 named tuple ，包含以下字段：time、priority、action、argument、kwargs。
```

## 使用

[参考](https://www.cnblogs.com/luminousjj/p/9340082.html)

### 延迟运行

在一个延迟或规定时间之后执行事件，需要采用enter()方法

如果多个事件是同一时间执行，通过设置他们的优先级值，用于确定顺序运行

```python
import sched
import time

#生成调度器
scheduler = sched.scheduler(time.time, time.sleep)

def print_event(name="default"):
    print('EVENT:', time.time(), name)

def print_some_times():
	print('start:', time.time())
	#分别设置在执行后10秒、2秒、3秒之后执行调用函数
    scheduler.enter(10, 1, print_event)
	scheduler.enter(2, 1, print_event, argument=('first',))
	scheduler.enter(3, 1, print_event, kwargs={"a": 'second'}) 
    scheduler.enter(3, 2, print_event, kwargs={"a": 'third'})
    scheduler.run()
    print("end:", time.time)

if __name__ == "__main__":
    print_some_times()

```

### 重叠事件

调用run()块执行所有的事件。每个事件都在同一线程中运行，所以如果一个事件需要更长的时间，延迟事件将会有重叠。为了不丢失事件，延迟事件将会在之前事件运行完再被执行，但一些延迟事件可能会晚于原本计划的事件。

```python
import sched
import time

scheduler = sched.scheduler(time.time, time.sleep)

def long_event(name):
    print('BEGIN EVENT :', time.time(), name)
    time.sleep(2)
    print('FINISH EVENT:', time.time(), name)

print('START:', time.time())
scheduler.enter(2, 1, long_event, ('first',))

#事件无法在设想的3秒后执行，将会顺延执行
scheduler.enter(3, 1, long_event, ('second',))

scheduler.run()
```

### 取消事件

利用`enter()`和`enterabs()`返回一个引用事件用来取消它。

```python
import sched
import threading
import time

scheduler = sched.scheduler(time.time, time.sleep)

#建立一个全局 线程计数器
counter = 0

def increment_counter(name):
    global counter
    print('EVENT:', time.time(), name)
    counter += 1
    print('NOW:', counter)

print('START:', time.time())
e1 = scheduler.enter(2, 1, increment_counter, ('E1',))
e2 = scheduler.enter(3, 1, increment_counter, ('E2',))

# 开始一个线程执行事件
t = threading.Thread(target=scheduler.run)
t.start()

# 在主线程,取消第一个预定事件
scheduler.cancel(e1)

# 等待线程调度程序完成运行
t.join()
```

