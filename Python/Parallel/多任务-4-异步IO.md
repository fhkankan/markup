[TOC]

# 异步IO

## 协程

```
协程，又称微线程，纤程。英文名Coroutine。

协程是python个中另外一种实现多任务的方式，只不过比线程更小占用更小执行单元（理解为需要的资源）。类似CPU中断，可由用户调度，可在一个子程序内中断去执行其他的子程序。 为啥说它是一个执行单元，因为它自带CPU上下文。这样只要在合适的时机，我们可以把一个协程切换到另一个协程。只要这个过程中保存或恢复 CPU上下文那么程序还是可以运行的。

通俗的理解：在一个线程中的某个函数，可以在任何地方保存当前函数的一些临时变量等信息，然后切换到另外一个函数中执行，注意不是通过调用函数的方式做到的，并且切换的次数以及什么时候再切换到原来的函数都由开发者自己确定
```

进程、线程、协程差异

```
进程是资源分配的单位
进程切换需要的资源很最大，效率很低

线程是操作系统调度的单位
线程切换需要的资源一般，效率一般（在不考虑GIL的情况下）

协程切换任务资源很小，效率高
多进程、多线程根据cpu核数不一样可能是并行的，但是协程是在一个线程中 所以是并发

由于协程在一个线程内执行，因此对于多核CPU平台，当并发量很大时，可以使用多进程+协程的方式。
```

协程概念

```
协程有两种含义：
1.用来定义写成的函数，亦可称为协程函数
2.调用协程函数的得到协程对象，表示一个最终会完成的计算或者IO操作

协程的引入使得编写单线程并发代码称为可能，事件循环在单个线程中运行并在同一个线程中执行所有的回调函数和任务，当事件循环中正在运行一个任务时，该线程中不会再同时运行其他任务，一个事件循环在某个时刻只运行一个任务。但是如果该任务执行yield from语句等待某个Future对象的完成，则当前任务被挂起，事件循环执行下一个任务。当然，不同线程中的事件循环可以并发执行多个任务

在语法形式上，协程可以通过async def语句或生成器来实现，若不需要考虑和旧版本python兼容，则优先考虑前者；基于生成器的协程函数需要使用@asyncio.coroutine进行修饰，并且使用yield from而不是yield语句

Future类代表可代哦用对象的异步执行，Task类是Future的子类，用来调度协程，负责在事件循环中执行协程对象，若果在协程中使用yield from语句从一个Future对象中返回值的话，Task对象会挂起协程的执行并且等待Future对象的完成，当Future对象完成后，协程会重新启动并得到Future对象的结果或异常

于普通函数不同，调用一个协程函数并不会立刻启动代码的执行，返回的协程对象在被调度之前不会做什么事情。启动协程对象的执行有两种方法：1.在一个正在运行的协程中使用await或yield from语句等待线程对象的返回结果；2.使用ensure_future()函数或者AbstractEventLoop.create_task()方法创建任务(Task对象)并调度协程的执行
```

### 生成器创建

Python对协程的支持是通过generator实现的。

在generator中，我们不但可以通过`for`循环来迭代，还可以不断调用`next()`函数获取由`yield`语句返回的下一个值。

但是Python的`yield`不但可以返回一个值，它还可以接收调用者发出的参数。

传统的生产者-消费者模型是一个线程写消息，一个线程取消息，通过锁机制控制队列和等待，但一不小心就可能死锁。

如果改用协程，生产者生产消息后，直接通过`yield`跳转到消费者开始执行，待消费者执行完毕后，切换回生产者继续生产，效率极高：

```python
def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    # 启动生成器
    c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        # 切换到consumer()函数执行
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

c = consumer()
produce(c)
```

注意到`consumer`函数是一个`generator`，把一个`consumer`传入`produce`后：

1. 首先调用`c.send(None)`启动生成器；
2. 然后，一旦生产了东西，通过`c.send(n)`切换到`consumer`执行；
3. `consumer`通过`yield`拿到消息，处理，又通过`yield`把结果传回；
4. `produce`拿到`consumer`处理的结果，继续生产下一条消息；
5. `produce`决定不生产了，通过`c.close()`关闭`consumer`，整个过程结束。

整个流程无锁，由一个线程执行，`produce`和`consumer`协作完成任务，所以称为“协程”，而非线程的抢占式多任务。

```python
#生成器是可以暂停的函数
import inspect
# def gen_func():
#     value=yield from
#     #第一返回值给调用方， 第二调用方通过send方式返回值给gen
#     return "bobby"
#1. 用同步的方式编写异步的代码， 在适当的时候暂停函数并在适当的时候启动函数
import socket
def get_socket_data():
    yield "bobby"

def downloader(url):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setblocking(False)
    try:
        client.connect((host, 80))  # 阻塞不会消耗cpu
    except BlockingIOError as e:
        pass

    selector.register(self.client.fileno(), EVENT_WRITE, self.connected)
    source = yield from get_socket_data()
    data = source.decode("utf8")
    html_data = data.split("\r\n\r\n")[1]
    print(html_data)

def download_html(html):
    html = yield from downloader()

if __name__ == "__main__":
    #协程的调度依然是 事件循环+协程模式 ，协程是单线程模式
    pass
```

### async/await

python位了将语义变得更加明确，python3.5以后引入async/await，避免了生成器和协程混用

```python
#python为了将语义变得更加明确，就引入了async和await关键词用于定义原生的协程
async def downloader(url):
    return "bobby"

# import types
# @types.coroutine  # 装饰器封装为await
# def downloader(url):
#     yield "bobby"

async def download_url(url):
    #dosomethings
    html = await downloader(url)
    return html

if __name__ == "__main__":
    coro = download_url("http://www.imooc.com")
    # next(None)  # 原协程不能如此调用
    coro.send(None)
```

### greenlet

为了更好使用协程来完成多任务，python中的greenlet模块对其封装，从而使得切换任务变的更加简单

- 安装

```
sudo pip3 install greenlet
```

- 使用

```
#coding=utf-8

from greenlet import greenlet
import time

def test1():
    while True:
        print ("---A--")
        gr2.switch()
        time.sleep(0.5)

def test2():
    while True:
        print ("---B--")
        gr1.switch()
        time.sleep(0.5)

gr1 = greenlet(test1)
gr2 = greenlet(test2)

#切换到gr1中运行
gr1.switch()
```

### gevent

使用C语言实现协程

python还有一个比greenlet更强大的并且能够自动切换任务的模块`gevent`

其原理是当一个greenlet遇到IO(指的是input output 输入输出，比如网络、文件操作等)操作时，比如访问网络，就自动切换到其他的greenlet，等到IO操作完成，再在适当的时候切换回来继续执行。

由于IO操作非常耗时，经常使程序处于等待状态，有了gevent为我们自动切换协程，就保证总有greenlet在运行，而不是等待IO

**安装**

```
pip3 install gevent
```

**使用**

- 依次运行

```
import gevent

def f(n):
    for i in range(n):
        print(gevent.getcurrent(), i)

g1 = gevent.spawn(f, 5)
g2 = gevent.spawn(f, 5)
g3 = gevent.spawn(f, 5)
g1.join()
g2.join()
g3.join()
```

- 切换执行

```
import gevent

def f(n):
    for i in range(n):
        print(gevent.getcurrent(), i)
        #用来模拟一个耗时操作，注意不是time模块中的sleep
        gevent.sleep(1)

g1 = gevent.spawn(f, 5)
g2 = gevent.spawn(f, 5)
g3 = gevent.spawn(f, 5)
g1.join()
g2.join()
g3.join()
```

- 打补丁

```python
from gevent import monkey
import gevent
import random
import time

# 有耗时操作时需要
# 将程序中用到的耗时操作的代码，换为gevent中自己实现的模块
monkey.patch_all()  


def coroutine_work(coroutine_name):
    for i in range(10):
        print(coroutine_name, i)
        time.sleep(random.random())

gevent.joinall([
        gevent.spawn(coroutine_work, "work1"),
        gevent.spawn(coroutine_work, "work2")
])
```

**并发下载**

```python
from gevent import monkey
import gevent
import urllib.request

# 有耗时操作时需要
monkey.patch_all()

def my_downLoad(url):
    print('GET: %s' % url)
    resp = urllib.request.urlopen(url)
    data = resp.read()
    print('%d bytes received from %s.' % (len(data), url))

gevent.joinall([
        gevent.spawn(my_downLoad, 'http://www.baidu.com/'),
        gevent.spawn(my_downLoad, 'http://www.itcast.cn/'),
        gevent.spawn(my_downLoad, 'http://www.itheima.com/'),
])
```

**多视频下载**

```python
from gevent import monkey
import gevent
import urllib.request

#有IO才做时需要这一句
monkey.patch_all()

def my_downLoad(file_name, url):
    print('GET: %s' % url)
    resp = urllib.request.urlopen(url)
    data = resp.read()

    with open(file_name, "wb") as f:
        f.write(data)

    print('%d bytes received from %s.' % (len(data), url))

gevent.joinall([
        gevent.spawn(my_downLoad, "1.mp4", 'http://oo52bgdsl.bkt.clouddn.com/05day-08-%E3%80%90%E7%90%86%E8%A7%A3%E3%80%91%E5%87%BD%E6%95%B0%E4%BD%BF%E7%94%A8%E6%80%BB%E7%BB%93%EF%BC%88%E4%B8%80%EF%BC%89.mp4'),
        gevent.spawn(my_downLoad, "2.mp4", 'http://oo52bgdsl.bkt.clouddn.com/05day-03-%E3%80%90%E6%8E%8C%E6%8F%A1%E3%80%91%E6%97%A0%E5%8F%82%E6%95%B0%E6%97%A0%E8%BF%94%E5%9B%9E%E5%80%BC%E5%87%BD%E6%95%B0%E7%9A%84%E5%AE%9A%E4%B9%89%E3%80%81%E8%B0%83%E7%94%A8%28%E4%B8%8B%29.mp4'),
])
```

## asyncio

Python 3.4版本引入的标准库asyncio，以生成器对象为基础，直接内置了对异步IO的支持。Python 3.5又提供了语法`async`和`await`层面的支持，可以让coroutine的代码更简洁易读。

### 概述

asyncio提供了管理事件、协程、任务和线程的功能，以及编写并发代码的同步原语(synchronization primitives)。

```
包含各种特定系统实现的模块化事件循环
传输和协议抽象
对TCP、UDP、SSL、子进程、延时调用以及其他的具体支持 
模仿futures模块但适用于时间循环使用的Future类
基于yield from的协议和任务，可用顺序的方式编写并发代码
必须使用一个将产生阻塞IO的调用时，有接口可以把这个事件转移到线程池中
```

该模块主要由以下组件构成

| 概念        | 说明                                                         |
| ----------- | ------------------------------------------------------------ |
| event_loop  | 事件循环，程序开启一个无限循环，开发者把一些函数注册到事件循环中，当满足事件发生的条件时，调用相应的协程函数。asyncio模块支持每个进程拥有一个事件循环 |
| coroutine   | 协程对象，指一个使用async关键字定义的函数，它的调用不会立即执行函数，而是返回一个协程对象。协程对象需要注册到事件循环中，由事件循环调用。这是子例程(subroutine)概念的泛化。写成在执行时可以暂停，以等待外部处理程序完成(I/O中的某个例行程序)，外部处理程序结束后则从暂停之处返回 |
| task        | 任务，一个协程对象就是一个原生可以挂起的函数，任务则是对协程进一步的封装，其中包含任务的各种状态。用于封装并管理并行模式下的协程。 |
| future      | 代表将来执行或没有执行任务的结果，与task没有本质的区别       |
| async/await | python3.5用于定义协程的关键字，async定义一个协程，await用于挂起阻塞的异步调用接口 |

常用方法

| 方法                                                         | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `asyncio.get_event_loop()`                                   | 可以获得当前上下文的事件循环                                 |
| `asyncio.set_event_loop(loop)`                               | 将当前上下文的事件循环设置为给定循环                         |
| `asyncio.new_event_loop()`                                   | 根据此函数的规则创建并返回一个新的事件循环对象               |
| `asyncio.Future()`                                           | 创建Future对象                                               |
| `asyncio.ensure_future(coro_or_future, *, loop=None)`        | 将任务注册到事件循环，返回Task对象；在python3.4.3中使用`asyncio.async()` |
| `asyncio.gather(*coros_or_futures, loop=None, return_exceptions=False)` | 接受一系列协程或任务，并返回将那些任务聚合后的单个任务(包装其接收的任何适用协程)；也可作为整体的一组任务添加回调的机制 |
| `asyncio.wait(fs, *, loop=None, timeout=None, return_when=ALL_COMPLETED)` | 接受一系列协程或任务构成的列表，根据参数确定何时返回结果，默认全部完成返回，可设定超时时长 |
| `asyncio.run_coroutine_threadsafe(coro,loop)`                | 在主线程中将协程加入到子线程中开始的事件循环中               |
| `asyncio.Queue()`                                            | 最基本的一步队列                                             |
| `asyncio.sleep(delay)`                                       | 异步中延迟执行时长秒数                                       |

python3.4与python3.5区别

```python
用asyncio提供的@asyncio.coroutine可以把一个generator标记为coroutine类型，然后在coroutine内部用yield from调用另一个coroutine实现异步操作。

请注意，async和await是针对coroutine的新语法，要使用新的语法，只需要做两步简单的替换：
@asyncio.coroutine <==> async
yield from <==> await

# python3.4
@asyncio.coroutine
def hello():
   print("Hello world!")
   r = yield from asyncio.sleep(1)
   print("Hello again!")

# python3.5
async def hello():
    print("Hello world!")
    r = await asyncio.sleep(1)
    print("Hello again!")
```

`asyncio`的编程模型就是一个消息循环。我们从`asyncio`模块中直接获取一个`EventLoop`的引用，然后把需要执行的协程扔到`EventLoop`中执行，就实现了异步IO。

异步操作需要在`coroutine`中通过`yield from`完成；多个`coroutine`可以封装成一组Task然后并发执行。

### 事件循环

在计算机系统中，能够产生事件的尸体被称为事件源(evnet source)，而负责协商管理事件的实体被称为事件处理器(evnet handler)。有时可能还存在被称为事件循环的第三个实体。它实现了管理计算代码中所有事件的功能。更准确地说，在沉痼执行期间事件循环不断周期反复，追踪某个数据结构内部发生的事件，将其纳入队列，如果主线程空闲则调用事件处理器一个一个地处理这些事件。如下是一段事件循环管理器的伪代码。while循环中的所有事件被事件处理器捕捉，然后逐一处理。事件的处理器是系统中唯一进行的活动，在处理器结束后，控制被传递给下一个执行的事件。

```c
while (1) {
    events = getEvents();
    for (e in events)
        processEvent(e);
}
```

大多数异步应用程序实现异步的机制时通过在后台执行的事件循环。当代码需要执行时，这些代码才会被注册到事件循环中。几乎所有的服务器都是一个事件循环。

将一个函数注册到事件循环会导致它变成一个任务。事件循环负责在获得任务后，马上执行它。另一种方式是事件循环有时在等待一定时间后，再执行任务

管理事件循环的方法

| 方法                                        | 说明                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| `loop = asyncio.get_event_loop()`           | 可以获得当前上下文的事件循环                                 |
| `loop = asyncio.set_event_loop()`           | 将当前上下文的事件循环设置为给定循环                         |
| `loop = asyncio.new_event_loop()`           | 根据此函数的规则创建并返回一个新的事件循环对象               |
| `loop.call_later(delay,callback,*args)`     | 安排在给定事件delay秒后，调用某个回调对象                    |
| `loop.call_soon(callback,*args)`            | 安排一个将马上被调用的回调对象，在`call_soon()`返回、控制回到事件循环之后，回调对象就被调用 |
| `loop.call_soon_threadsafe(callback,*args)` | 线程安全的`call_soon()`                                      |
| `loop.call_at(when,callback,*args)`         | 在指定的与`loop.time()`相比较的时间时执行回调                |
| `loop.time()`                               | 以浮点值的形式返回根据事件循环的内部时钟确定当前时间         |
| `loop.create_task(coro)`                    | 创建一个task任务                                             |
| `loop.run_until_complete(future)`           | 任务结束前执行循环，完任务后自动停止事件循环                 |
| `loop.run_forever()`                        | 一直执行，直到调用`stop()`为止                               |
| `loop.stop()`                               | 停止事件循环                                                 |
| `loop.close()`                              | 关闭事件循环                                                 |


- 创建循环

在大多数情况下，并不需要自己创建一个事件循环对象。可以通过`asyncio.get_event_loop()`函数返回一个BaseEventLoop对象。实际上，获得的是一个子类，具体是哪个子类会根据平台的不同而不同，不必过多考虑细节。所有平台的API相同，但在某些平台上会有功能限制

```python
import asyncio

loop = asyncio.get_event_loop()
loop.is_runing()  # False,第一次获得循环对象，其并未执行
```

- 执行循环

下面的事件循环还没有注册任何内容，但可以执行它

```python
loop.run_forever()
```

若执行上述代码，将失去对python解释器的控制权，程序陷入了死循环。使用Ctrl+C来结束循环重新获得解释器的控制权。对于大多数应用程序来说，编写一个服务或者守护程序的目的获取是在前台执行，并等待其他进程发起命令，所以死循环并不是一个大障碍。但是在测试或实验中，应避免陷入死循环。

- 注册任务并执行循环

任务主要适用call_soon注册到循环，注册顺序是FIFO(先进先出)队列

```python
import functools

def hello_word():
	print('Hello word!')
    
def stop_loop(loop):
    print('Stopping loop.')
    loop.stop()

# 注册任务
loop.call_soon(hello_word)  
loop.call_soon(functools.partial(stop_loop, loop))
# 执行循环
loop.run_forever()
```

- 延迟调用

call_later方法接受延迟时间(秒)和被调用的函数名称作为参数，注册一个延迟执行的任务

```python
# 注册延时调用任务
loop.call_later(10, hello_word)
loop.call_later(20, functools.partial(stop_loop, loop))
# 执行循环
loop.run_forever()
```

若是在同一时间出现多个延时调用，先后顺序无法确定

- 偏函数

大多数接受函数的asyncio方法仅仅接受函数对象(或被其他调用元素)，但这些函数在被调用时没有带参数。若需参数，则用functools.partial，此方法本身接受参数与关键字参数，在底层函数被调用时传给底层函数

通常可以将这类调用封装到不需要参数的函数中，但是之所以要用partial，主要由于调试时更有用。Partial对象知道调用函数用的哪个参数，partial函数以数据的形式表示参数，在被调用时使用这些数据执行合适的函数调用

创建一个partial函数，查看其底层函数与参数找出函数与便函数之间区别

```python
>>>partial = functools.partial(stop_loop, loop)
>>>partial.dunc  # <function stop_loop at xxx>
>>>partial.args  # (<asyncio.unix_events._UnixSelectorEventLoop object at xxx>)
```

- 任务结束前执行循环

```python
async def trivial():
    return 'Hello world!'

# 调用run_until_complete时，将任务注册并在任务结束前执行循环
# 由于该任务时队列中唯一任务，完成后退出循环，返回任务结果
loop.run_until_complete(trivial())
```

- 执行一个后台循环

```python
import asyncio
import threading

def run_loop_forever_in_background(loop):
    def thread_func(l):
        asyncio.set_event_loop(l)
        l.run_forever()
    thread = threading.Thread(target=thread_func, args=(loop,))
    thread.start()
    return thread

loop = asyncio.get_event_loop()
run_loop_forever_in_background(loop)  # <Thread(Thread-1, started xxx)>
loop.is_running()  # True
```

该例可做测试但不会应用在项目中，原因：停止循环很难，`loop.stop()`将不再生效

```python
# 把任务注册到循环并令其立刻执行
# call_soon_threadsafe方法用于通知循环立刻异步执行给定函数，由于很少会利用线程执行事件循环，故大多情况下仅仅call_soon函数就够了，返回一个Handle对象。该对象只有一个方法:cancel,在合适时，完全可以取消任务
loop.call_soon_threadsafe(functools.partial(print, 'Hello word'))
```

- 综合实例

```python
import asyncio
import datetime
import time

# 程序行为
# end_time:定义函数内部的时间上限
# loop:事件循环
def function_1(end_time, loop):
    print("function_1 called")
    if (loop.time() + 1.0) < end_time:
        loop.call_later(1, function_2, end_time, loop)
    else:
        loop.stop()

def function_2(end_time, loop):
    print("function_2 called")
    if (loop.time() + 1.0) < end_time:
        loop.call_later(1, function_3, end_time, loop)
    else:
        loop.stop()

def function_3(end_time, loop):
    print("function_3 called")
    if (loop.time() + 1.0) < end_time:
        loop.call_later(1, function_1, end_time, loop)
    else:
        loop.stop()

def function_4(end_time, loop):
    print("function_4 called")
    if (loop.time() + 1.0) < end_time:
        loop.call_later(1, function_4, end_time, loop)
    else:
        loop.stop()

# 捕获整个事件循环
loop = asyncio.get_event_loop()
end_loop = loop.time() + 9.0
# 安排调用执行
loop.call_soon(function_1, end_loop, loop)
# loop.call_soon(function_4, end_loop, loop)
# 如果用完全部执行使劲啊，循环事件停止
loop.run_forever()
loop.close()
```

### 协程

当程序变得冗长复杂事，将其划分成子例程的方式会使处理变得更加便利，每个子例程完成一个特定的任务，并针对任务实现合适的算法。子例程无法独立运行，只能在主程序的要求下才能运行，主程序负责协调子例程的使用。协程就是子例程的泛化。与子例程类似，协程执行一个计算步骤，但不同的是，不存在可用于协调结果的主程序。这是因为协程之间可以相互连接在一起，形成一个管道，不需要任何监督式函数来按照顺序调用协程。在协程中，可以暂停执行点，同时保存干预时的本地状态，便于后续继续执行。有了协程池之后，协程计算就能够相互交错：运行第一个协程，直到其返回控制权，然后运行第二个协程，以此类推。

协程相互交错的控制组件就是事件循环，事件循环追踪全部的协程，并安排其执行的时间。

协程的重要特性

```
协程支持多个进入点，可以多次声称(yield)
协程能够将执行转移至任何其他协程
```

“生成”(yield)用于描述那些暂停并将控制流传递给另一个协程的携程。由于协程可以同时传递控制流和值。“生成一个值”(yielding a value)这个短语用于描述生成并将值传递给获得控制流的协程。

在asyncio中使用的大多数函数都是协程(coroutines)。协程是一种设计用在事件循环中执行的特殊函数。此外，若创建了协程但未执行它，那么将会在日志中记录一个错误.

```python
import asyncio

async def coro_sum(*args):
    anser = 0
    for i in args:
        answer += i
    return answer

loop = asyncio.get_event_loop()
loop.run_until_complete(coro_sum(1, 2, 3, 4, 5))  # 15
```

创建的coro_sum函数不再是一个普通的函数，而是一个协程，由事件循环调用。注意：不能再以常规方式调用该函数并返回预期结果

```python
>>>coro_sum(1, 2, 3, 4, 5)  # <generator object coro at xxx>
```

协程实际上时一个由事件循环消费的特殊生成器，这也是为什么run_until_complete方法可以接受的参数看起来像一个标准函数调用的原因所在。函数此时实际并未执行。由事件循环消费生成器，并最终返回结果

```python
# 底层实际看起来类似如下代码
try:
    next(coro_sum(1, 2, 3, 4, 5))
except StopIteration as ex:
	ex.value
```

 生成器并未返回任何值，而是立刻引发了StopIteration异常。StopIteration被赋予函数的返回值，然后事件循环可以提取该值并正确处理该值

- 嵌套的协程

协程提供了一种以模仿顺序编程的方式来调用其他携程的特殊机制(Future实例)。通过使用yield from，一个携程可以执行另外一个协程，并由语句返回结果。这是一种以顺序方式编写异步代码的可用机制

```python
import asyncio


async def nested(*args):
    print('The "nested" function ran with args: %r' % (args,))
    return [i+1 for i in args]

async def outer(*args):
    print('The "outer" function ran with args: %r' % (args,))
    # outer协程遇到await时挂起，将nested的协程放入事件循环并执行。outer协程在nested完成并返回结果之前不会继续执行
    # 返回它执行协程的结果
    answer = await nested(*[i*2 for i in args])
    return answer

loop = asyncio.get_event_loop()
loop.run_until_complete(outer(2, 3, 5, 8))
```

- 使用anyncio的协程机制模拟一个具备5个状态的有限状态机。

![有限状态机](images/有限状态机.png)



示例

```python
import asyncio
import time
from random import randint


# 状态S0
async def startState():
    print("start State called \n")
    input_value = randint(0, 1)
    time.sleep(1)
    if input_value == 0:
        result = await state2(input_value)
    else:
        result = await state1(input_value)
    print("Resume of the Transition:\n start State calling " + result)

# 状态S1
async def state1(transition_value):
    outputValue = str(("state 1 with transition value = %s \n" % (transition_value)))
    input_value = randint(0, 1)
    time.sleep(1)
    print("...Evaluating...")
    if input_value == 0:
        result = await state3(input_value)
    else:
        result = await state2(input_value)
    result = "state 1 calling " + result
    return (outputValue + str(result))

# 状态S2
async def state2(transition_value):
    outputValue = str(("state 2 with transition value = %s \n" % (transition_value)))
    input_value = randint(0, 1)
    time.sleep(1)
    print("...Evaluating...")
    if input_value == 0:
        result = await state1(input_value)
    else:
        result = await state3(input_value)
    result = "state 2 calling " + result
    return (outputValue + str(result)) 

# 状态S3
async def state3(transition_value):
    outputValue = str(("state 3 with transition value = %s \n" % (transition_value)))
    input_value = randint(0, 1)
    time.sleep(1)
    print("...Evaluating...")
    if input_value == 0:
        result = await state1(input_value)
    else:
        result = await endState(input_value)
    result = "state 3 calling " + result
    return (outputValue + str(result)) 

# 状态S4
async def endState(transition_value):
    outputValue = str(("end state with transition value = %s \n" % (transition_value)))
    print("...stop computation...")
    return (outputValue)

if __name__ == "__main__":
    print("Finite State Machine simulation with Asyncio Coroutine")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(startState())
```

### 管理任务

Asyncio的宗旨是处理事件循环中的异步进程和并发执行任务。它还提供了一个叫做`asyncio.Task()`类，用于将协程封装在任务中。该类的用途在于，支持独立运行的任务与同一个事件循环中的其他任务并发执行。协程被封装进任务中后，它将该任务与实际那循环相连接，并在循环开始时自动运行，因此算是提供了一种自动驱动协程的机制。

Ayncio模块提供了一个处理任务计算的方法:`asyncio.Task(coroutine)`。该方法用于调度协程的执行。任务负责执行事件循环中的协程对象。如果被封装的协程丛future生成，任务将暂停执行被封装的协程，并等待future执行完毕。

在future执行完毕后，被封装的协程以future返回的结果或异常重新开始执行。另外，必须注意：一个事件循环一次只执行一个任务。如果其他事件循环通过不同的线程运行，则可以并行执行其他任务。在任务等待future执行完毕时，事件循环将执行一个新任务

示例

```python
import asyncio


async def factorial(number):
    f = 1
    for i in range(2, number+1):
        print("Asyncio.Task: Compute factorial(%s)" % i)
        await asyncio.sleep(1)
        f *= i
    print("Asyncio.Task - factorial(%s) = %s" % (number, f))

async def fibonacci(number):
    a, b = 0, 1
    for i in range(2, number):
        print("Asyncio.Task: Compute fibonacci (%s)" % i)
        await asyncio.sleep(1)
        a, b = b, a + b
    print("Ayncio.Task - fibonacci(%s) = %s" % (number, a))

async def binomialCoeff(n, k):
    result = 1
    for i in range(1, k+1):
        result = result * (n-i+1) / i
        print("Asyncio.Task: Compute binomialCoeff (%s)" % i)
        await asyncio.sleep(1)
    print("Asyncio.Task - binomialCoeff(%s, %s) = %s" % (n, k, result))

if __name__ == "__main__":
    # 任务列表，并发执行3个数学函数
    tasks = [
        asyncio.Task(factorial(10)),
        asyncio.Task(fibonacci(10)),
        asyncio.Task(binomialCoeff(20, 10))
    ]
    # 获取事件循环
    loop = asyncio.get_event_loop()
    # 运行任务
    loop.run_until_complete(asyncio.wait(tasks))
    # 关闭事件循环
    loop.close()
```

### Future/Task

由于使用asyncio完成的大多工作都是异步的，因此在处理异步方式执行时的返回值要小心。为此，yield from语句提供了一种方式，但是另外一些时候需要其他处理方式，比如，需要并行执行异步函数

- Future对象

在遇到特殊问题时一种对应机制是使用Future对象。本质上讲，Future是一个用于通知异步函数状态的对象。这包括函数的状态(执行中、已完成、已取消)，还包括函数的结果，或者是当函数引发异常时，返回对应的异常和回溯

Future是一个独立的对象，并不依赖正在执行的函数。该对象仅仅用于存储状态和结果信息，此外无它用

Future类与`concurrent.futures.Futures`非常类似，但是已经按照Asyncio的事件循环机制做了调整。`asyncio.Future`类代表一个还不可用的结果(也可能是个异常)。因此，它是对尚需完成的任务的抽象表示。

定义Future对象

```python
import asyncio
future = asyncio.Future()
```

该类具备如下方法

| 方法                       | 说明                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `cancel()`                 | 取消future，并安排回调对象                                   |
| `result()`                 | 返回future所代表的结果                                       |
| `execption()`              | 返回future上设置的异常                                       |
| `add_done_callback(fn)`    | 添加一个在future执行时运行的回调对象                         |
| `remove_done_callback(fn)` | 从“结束后调用(call when done)”列表中移除一个回调对象的所有实例 |
| `set_result(result)`       | 将future标记为已完成，并设置其结果                           |
| `set_exception(exception)` | 将future标记为已完成，并设置一个异常                         |

示例

```python
import asyncio
import sys

# 求n个整数的和

async def first_coroutine(future, N):
    count = 0
    for i in range(1, N+1):
        count = count + i
    await asyncio.sleep(3)
    # 标记已完成，设置其结果
    future.set_result("first corountine (sum of N integers) result = " + str(count))

# 求n的阶乘

async def second_coroutine(future, N):
    count = 1
    for i in range(2, N+1):
        count *= i
    await asyncio.sleep(4)
    future.set_result("second corountine (factorial) result = " + str(count))

# 打印future最后的结果
def got_result(future):
    print(future.result())


if __name__ == "__main__":
    # 从命令行接收参数
    N1 = int(sys.argv[1])
    N2 = int(sys.argv[2])

    loop = asyncio.get_event_loop()
    # 定义两个future对象，与协程关联
    future1 = asyncio.Future()
    future2 = asyncio.Future()
    # 定义任务，将future对象作为协程的实参传入
    tasks = [
        first_coroutine(future1, N1),
        second_coroutine(future2, N2)
    ]
    # 添加一个future执行时将运行的回调对象
    future1.add_done_callback(got_result)
    future2.add_done_callback(got_result)

    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()
```

- Task对象

Task对象是Future对象的子类，在使用asyncio时常用的对象。每当一个协程在事件循环中被安排执行后，协程就会被一个Task对象包装。

当调用`run_until_complete(coro)`，该协程参数会被包装到一个Task对象中并执行。Task对象的任务时存储结果并为`await`语句提供值

除了`run_until_complete(coro)`方法外，还有如下方式创建task：

1. `asyncio.ensure_future(coro)`，返回对应的Task对象

注意：若是python3.4.4以上版本，使用ensure_future，若是3.4.3使用async

2. `loop.create_task(coro)`，返回对应的Task对象

```python
import asyncio

async def make_tea(variety):
    print('Now making %s tea.' % variety)
    # 获取事件循环
    asyncio.get_event_loop().stop()
    return '%s tea' % variety

# 方式一：将任务注册到事件循环，但循环未执行
task = asyncio.ensure_future(make_tea('chamomile'))
# 查看Task对象
print(task.done())  # False
# task.result()  # 抛出InvalidStateError异常
# 开始循环，任务完成后，由于调用loop.stop()，task将立即停止执行
loop = asyncio.get_event_loop()
# 方式二：
# task = loop.create_task(make_tea('chamomile'))
loop.run_forever()
# 查看Task对象
print(task.done())  # True
print(task.result()) # 'chamomile tea'
```

- 状态

`future`对象有几个状态：
```
- `Pending`
- `Running`
- `Done`
- `Cacelled`
```
创建`future`的时候，`task`为`pending`，事件循环调用执行的时候当然就是`running`，调用完毕自然就是`done`，如果需要停止事件循环，就需要先把`task`取消。可以使用`asyncio.Task`获取事件循环的`task`

```python
import asyncio
import time

now = lambda :time.time()

async def do_some_work(x):
    print("Waiting:",x)
    await asyncio.sleep(x)
    return "Done after {}s".format(x)

coroutine1 =do_some_work(1)
coroutine2 =do_some_work(2)
coroutine3 =do_some_work(2)

tasks = [
    asyncio.ensure_future(coroutine1),
    asyncio.ensure_future(coroutine2),
    asyncio.ensure_future(coroutine3),
]

start = now()

loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(asyncio.wait(tasks))
except KeyboardInterrupt as e:
    print(asyncio.Task.all_tasks())  # 获取所有task的状态
    for task in asyncio.Task.all_tasks():
        print(task.cancel())
    loop.stop()
    loop.run_forever()
finally:
    loop.close()

print("Time:",now()-start)
```

启动事件循环之后，马上 ctrl+c，会触发`run_until_complete`的执行异常 `KeyBorardInterrupt`。然后通过循环`asyncio.Task`取消`future`。

### 回调

Future对象(以及Task对象，因为Task是Future的子类)的另一个功能是能够将回调注册到Future。回调就是一个在Future完成后执行的一个函数(协程)，该函数接受Future作为参数

在某种程度上，回调代表了一个与yield from模型相反的模型。当一个协程使用yield from时，该协程会确保嵌套协程在其之前或同时被执行。当注册一个回调时，顺序则相反。回调被附加到原始的任务，它在任务执行之后再执行回调

可以使用对象的`add_done_callback`方法将一个回调添加到任何Future对象。回调接受一个参数，即为Future对象本身(该对象包含状态和结果信息，若存在底层任务，则为底层任务的状态或结果)

```python
import asyncio

loop = asyncio.get_event_loop()
# 生成协程，本协程不会停止
async def make_tea(variety):
    print('Now making %s tea.' % variety)
    return '%s tea' % variety

# 该函数接受被注册到其中的Future对象(本例中是task变量)，Future对象包含协程的结果
def confirm_tea(future):
    print('The %s is made.' % future.result())
    
task = asyncio.ensure_future(make_tea('green'))
# 将confirm_tea方法作为回调赋值给task，该函数被赋值给task(对一个协程的特殊调用),而不是赋值给协程本身
# 若用调用同一个协程的asyncio.ensure_future方法将另一个任务注册到循环，该任务不会得到该回调
task.add_done_callback(confirm_tea)

loop.run_until_complete(task)  
# Now making green tea. 
# The green tea is made.
# 'green tea'
```

- 不保证成功

Future仅仅是被执行，但并不能保证它能够执行成功。本例仅仅是假设`future.result()`结果值被正确返回，但事实可能并非如此。Task的执行可能会引发异常，在这种情况下，尝试访问`future.result()`将会引发该异常

同样地，也有可能取消任务(使用`Future.cancel()`方法或其他方式)。若这么做，则任务会被标记为Cancelled，会安排回调。在这种情况下，尝试访问`future.result()`将会引发CancelledError异常

- 幕后

在内部，由aysncio通知Future对象已经完成。Future对象接受接下来对所有已注册到Future的回调，并对其调用`call_soon_threadsafe`函数

需要注意的是，对于回调并不能保证执行顺序，完全有可能(且不会引起问题)将多个回调注册到一个任务中。然而，无法控制是否只执行某些回调以及回调之间的执行顺序

- 带参数的回调

回调系统的一个限制是回调接收作为位置参数的Future对象，但不接收其他参数

可以通过使用`functools.partial`函数将其他参数发送给回调。但若这样做，回调必须仍然接受作为位置参数的Future。实际上，在回调被调用之前，Future会附加到位置参数列表的结尾处

```python
# 接受其他参数的回调
import asyncio
import functools

loop = asyncio.get_event_loop()

async def make_tea(variety):
    print('Now making %s tea.' % variety)
    return '%s tea' % variety

# 接受两个位置参数
def add_ingredient(ingredient, future):
    print('Now adding %s to the %s.' % (ingredient, future.result()))
    
task = asyncio.ensure_future(make_tea('herbal'))
# 回调的注册方式是通过实例化一个带有位置参数的functools.partial对象实现
# partial仅接受一个参数，Future对象作为最后一个位置参数被发送
task.add_done_callback(functools.partial(add_ingredient, 'honey'))

loop.run_until_complete(task)  
# Now making herbal tea.
# Now adding honey to the herbal tea.
# 'herbal tea'
```

### 任务聚合

asyncio模块提供了一种聚合任务的便利方法，聚合任务主要归因于两个原因。1是在一组任务中的任何任务完成后采取某些行动。2是在所有任务都完成后采取某些行动

- 聚集任务

asyncio为聚集任务目的提供的第一种机制是通过gather函数。gather接受一系列协程或任务，并返回将那些任务聚合后的单个任务(包装其接收的任何适用协程)

```python
import asyncio

loop = asyncio.get_event_loop()

async def make_tea(variety):
    print('Now making %s tea.' % variety)
    return '%s tea' % variety

# 接收3个协程对象，在该函数中将所有协程包装到一个任务中，并返回一个充当3个携程聚合的单独任务
# meta_task对象高效地对3个被聚集的任务进行调度，一旦开始执行循环，3个子任务全部开始执行
meta_task = asyncio.gather(
	make_tea('chamomile'),
    make_tea('green'),
    mkae_tea('herbal')
)

meta_task.done()  # False

# asyncio.gather创建的任务，返回的结果是一个列表，该列表包含被聚集的单个任务的结果。返回的列表汇总任务的顺序保证于任务聚集的顺序一致(但任务的执行并不保证按照该顺序执行)。因此，返回的字符串列表与在asyncio.gather调用中协程的注册顺序保持一致
loop.run_until_complete(meta_task)
# Now mkaing chamomile tea
# Now mkaing herbal tea
# Now mkaing green tea
# ['chamomile tea', 'herbal tea', 'green tea']

meta_task.done()  # True
```

asyncio.gather还提供了针对作为整体的一组任务添加回调的机制，而不是针对每一个单独对象添加回调。若只想在所有任务完成后执行一次回调，但不关心任务完成的顺序，可以如下所做

```python
import asyncio

loop = asyncio.get_event_loop()

async def make_tea(variety):
    print('Now making %s tea.' % variety)
    return '%s tea' % variety

# 函数接受的Future对象是meta_task,而不是单个任务。
# result方法返回的是这两个任务返回值连接后的列表
def mix(future):
    print('Mixing the %s together.' % ' and '.join(future.result()))
    
meta_task = asyncio.gather(make_tea('herbal'), make_tea('green'))
meta_task.add_done_callback(mix)

loop.run_until_complete(meta_task)
# Now making green tea.
# Now making herbal tea.
# Mixing the green tea and herbal tea together.
# ['green tea', 'herbal tea']
```

- 等待任务

asyncio模块提供的另一个工具是内置的wait协程。asyncio.wait协程接受一系列协程或任务(在任务中包装任意协程)，一旦完成后就返回结果。注意该协程的签名与asyncio.gather不同。每一个协程或任务都是gather的一个单独位置参数，而wait接受一个列表作为参数

wait接受一个用于在任何任务完成后返回的参数，而无须等待所有任务完成。无论该标记位是否设置，wait方法总是返回两部分：第一个元素为一完成的Future对象；第二个元素为还未完成的部分

```python
# 类似之前asycio.gather
import asyncio

loop = asyncio.get_event_loop()


async def make_tea(variety):
    print('Now making %s tea.' % variety)
    return '%s tea' % variety

# wait方法返回一个协程，该协程带有值，可以在yield from中使用该协程
# 无法将回调直接附加到wait返回的协程上，若希望如此，则必须使用asyncio.ensure_future将该协程包装到一个任务中
coro = asyncio.wait([make_tea('chamomile'), make_tea('herbal')])

# asyncio.wait的返回值是一个包含Future对象(其自身包含返回值)的两部分容器。
# Future对象被重新组织，asyncio.wait协程江其分为两部分，一部分是已经完成的，另一部分是还未完成的，由于集合自身是一个未排序的结构，这意味着必须依赖Future对象来找出哪一个结果与哪一个任务对应
loop.run_until_complete(coro)
# Now making chamomile tea.
# Now making herbal tea.
# ({Task(<coro>)<result='herbal tea'>, Task(<coro>)<result='chamomile tea'>}, set())
```

> 超时

可以使用asyncio.wait协程在指定时间后返回结果，无论所有任务是否都已完成。为此，将timeout关键字参数传递给asyncio.wait

```python
import asyncio

loop = asyncio.get_event_loop()

# asyncio.sleep提供一个协程。该协程仅仅等待指定的秒数，然后返回None
# 在本例中超时时间设置使得其中一个任务在超时之前完成(第二个任务)，而另一个任务无法完成
# 使用timeout并不需要等到指定的超时时间过后才完成，若在超时时间到达之前所有的任务都执行完成，则协程江辉立刻完成
coro = asyncio.wait([asyncio.sleep(5), asyncio.sleep(1)], timeout=3)

# 两部分中的第二个元组现在包含一个未完成的任务；未完成的sleep协程仍然处于挂起状态，另一个已完成的协程又一个返回值(None)
loop.run_until_complete(coro)
# ({Task(<sleep><result=None>)}, {Task(<sleep>)<PENDING>})
```

> 等待任意任务

asyncio.wait的一个重要功能是可以在其包含的任意Future对象完成后，即可返回协程。asyncio.wait函数还接受一个return_when关键字参数。通过给该关键字传递一个特殊常量(asyncio.FIRST_COMPLETED)，一旦任意任务完成后，即可完成该协程，不再需要等所有任务都完成

```python
import asyncio

loop = asyncio.get_event_loop()

# wait的第一个参数是asyncio.sleep协程列表，当代协程被调用时，会执行所有其包含的任务。只等待1秒的asyncio.sleep协程首先执行完成，从而使得wait协程完成，因此，返回两部分结果集，其中第一部分结果集只包含一个项(已完成的任务)，第二个结果集包含两个项(仍然挂起的任务)
coro = asyncio.wait([
    asyncio.sleep(3),
    asyncio.sleep(2),
    asyncio.sleep(1)
], return_when=asyncio.FIRST_COMPLETED)

loop.run_until_complete(coro)
# ({Task(<sleep>)<result=None>},{Task(<sleep>)<PENDING>},{Task(<sleep>)<PENDING>})
```

> 等待异常

也可以使得在一个任务引发异常，而不是正常完成时，对于asyncio.wait的调用已经完成，在希望尽早补货并处理异常的情况下，这时很有价值的工具

可以使用return_from关键字参数出发该行为，但是这次使用asyncio.FIRST_EXCEPTION常量

```python
import asyncio

loop = asyncio.get_event_loop()

async def raise_ex_after(seconds):
    yield from asyncio.sleep(seconds)
    raise RuntimeError('Raising an exception.')
    
coro = asyncio.wait([
    asyncio.sleep(1),
    asyncio.sleep(2),
    asyncio.sleep(3),
    ], return_when=asyncio.FIRST_EXCEPTION)

# wait协程在其中一个任务引发异常后立刻停止。1秒的asyncio.sleep成功执行，因此其在返回值的第一个结果集中。raise_ex_after协程也已经完成，因此，也在第一个结果集中。然而，事实是该协程出发wait在等待3秒的协程完成之前完成，因此等待3秒的协程在第二个结果集中。
loop.run_until_complete(coro)
# ({Task(<raise_ex_after>)<exception=RuntimeError('Raising an exception.',)>, Task(<sleep>)<result=None>},{Task(<sleep>)<PENDING>})
```

有时，所有任务都不引发异常，在此情况下，就和正常情况一样，需要等待所有的任务完成后wait才完成

```python
import asyncio

loop = asyncio.get_event_loop()

coro = asyncio.wait([
    asyncio.sleep(1),
    asyncio.sleep(2),
], return_when=asyncio.FIRST_EXCEPTION)

loop.reun_until_complete(coro)
# ({Task(<sleep>)<result=NOne>, Task(<sleep>)<result=None>}, set())
```



### 多线程

很多时候，我们的事件循环用于注册协程，而有的协程需要动态的添加到事件循环中。一个简单的方式就是使用多线程。当前线程创建一个事件循环，然后再新建一个线程，在新线程中启动事件循环。当前线程不会被`block(阻塞)`。

```python
import asyncio
from threading import Thread
import time

now = lambda :time.time()

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

def more_work(x, start):
    print('More work {}'.format(x))
    time.sleep(x)  # 同步阻塞
    print('Finished more work {}'.format(x))
    print('Done time:{}'.format(time.time()-start))
    print('thread id:{}'.format(Thread.ident))

start = now()
new_loop = asyncio.new_event_loop()
t = Thread(target=start_loop, args=(new_loop,))
t.start()
print('TIME: {}'.format(time.time() - start)) 

# 事件循环中加入函数
new_loop.call_soon_threadsafe(more_work, 6, start)
new_loop.call_soon_threadsafe(more_work, 3, start)
```

启动上述代码之后，当前线程不会被`block`，新线程中会按照顺序执行`call_soon_threadsafe`方法注册的`more_work`方法， 后者因为`time.sleep`操作是同步阻塞的，因此运行完毕`more_work`需要大致6 + 3

```python
import asyncio
import time
from threading import Thread

now = lambda :time.time()

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def do_some_work(x, start):
    print('Waiting {}'.format(x))
    await asyncio.sleep(x)
    print('Done after {}s'.format(x))
    print('Done time:{}'.format(time.time() - start))

def more_work(x):
    print('More work {}'.format(x))
    time.sleep(x)
    print('Finished more work {}'.format(x))

start = now()
new_loop = asyncio.new_event_loop()
t = Thread(target=start_loop, args=(new_loop,))
t.start()
print('TIME: {}'.format(time.time() - start))

# 线程中加入协程
asyncio.run_coroutine_threadsafe(do_some_work(6, start), new_loop)
asyncio.run_coroutine_threadsafe(do_some_work(4, start), new_loop)
```

上述的例子，主线程中创建一个`new_loop`，然后在另外的子线程中开启一个无限事件循环。 主线程通过`run_coroutine_threadsafe`新注册协程对象。这样就能在子线程中进行事件循环的并发操作，同时主线程又不会被`block`。一共执行的时间大概在6s左右。

### 线程池

在协程中集成阻塞IO

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import socket
from urllib.parse import urlparse


def get_url(url):
    #通过socket请求html
    url = urlparse(url)
    host = url.netloc
    path = url.path
    if path == "":
        path = "/"

    #建立socket连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client.setblocking(False)
    client.connect((host, 80)) #阻塞不会消耗cpu

    #不停的询问连接是否建立好， 需要while循环不停的去检查状态
    #做计算任务或者再次发起其他的连接请求

    client.send("GET {} HTTP/1.1\r\nHost:{}\r\nConnection:close\r\n\r\n".format(path, host).encode("utf8"))

    data = b""
    while True:
        d = client.recv(1024)
        if d:
            data += d
        else:
            break

    data = data.decode("utf8")
    html_data = data.split("\r\n\r\n")[1]
    print(html_data)
    client.close()


if __name__ == "__main__":
    import time
    start_time = time.time()
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(3)
    tasks = []
    for url in range(20):
        url = "http://shop.projectsedu.com/goods/{}/".format(url)
        # 把阻塞代码放置到线程池中运行，将线程中的future封装为协程中的future
        task = loop.run_in_executor(executor, get_url, url)
        tasks.append(task)
    loop.run_until_complete(asyncio.wait(tasks))
    print("last time:{}".format(time.time()-start_time))
```

### 同步通信

asyncio中同步通信的机制有Lock、Event、Condition、Semaphere、BoundedSemaphore，使用方法和线程类似

```python
import asyncio
from asyncio import Lock
cache = {}
lock = Lock()

async def get_stuff(utl):
    # await lock.aquire()  # 协程
    # if url in cache:
    #     return cache[url]
    # stuff = await aiohttp.request('GET', url)
    # cache[url] = stuff
    # return stuff
	# lock.release()
    async with lock:  # lock中实现了__await__方法
        if url in cache:
            return cache[url]
        stuff = await aiohttp.request('GET', url)
        cache[url] = stuff
        return stuff 
    
    
async def parse_stuff():
    stuff = await get_stuff()
    # do some parsing
    
async def user_stuff():
    stuff = await get_stuff()
    # use stuff to do something interesting
    
    
tasks = [parse_stuff(), use_stuff()]
loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(tasks))
```

### 队列

asyncio模块提供了一些建立在事件循环与Future对象的基本代码段之上的通用模式。其中之一是基本的队列系统队列是一个由任务执行器处理的任务集合。python生态系统包含很多第三方的任务队列工具，其中常用的一种是celery。而由asyncio模块提供的队列仅仅是最基本的队列，并不是一个全功能的队列应用程序，若需要，可以基于该基本队列按照需求进行自定义开发

Queue是asyncio的组成部分，由于：Queue类提供的方法被用于顺序或异步上下文中

```python
# Queue简单示例
import asyncio

queue = asyncio.Queue()
# 立刻从队列中添加项
queue.put_nowait('foo')
queue.qsize()  # 1
# 立刻从队列中移除项
queue.get_nowait()  # 'foo'
queue.asize()  # 0

# 若是尝试从空队列中调用get_nowait(),抛错
queue.get_nowait()  # 抛QueueEmpty异常
```

Queue类还提供了一个名称为get的方法，get方法在队列为空时并不会引发异常，而是耐心等待项被添加到队列中，然后再从队列中获得该项并立刻返回，与get_nowait不同，该方法是一个协程，在异步上下文中执行

```python
import asyncio

loop = asyncio.get_evnet_loop()

queue = asyncio.Queue()

queue.pu_nowait('foo')
# 由于队列中已经有一项，故get方法会立刻返回。若队列中没有项，则直接调用loop.run_until_complete会永远处于执行状态，并阻塞解释器
loop.run_until_complete(queue.get())  # 'foo'
```

可以使用asyncio.wait中的timeout参数来查看实际的执行情况

```python
import asyncio

loop = asyncio.get_event_loop()
queue = asyncio.Queue()

task = asyncio.ensure_future(queue.get())
coro = asyncio.wait([task], timeout=1)

loop.run_until_complete(coro)
# (set(), {Task(<get>)<PENDING>})

# 此时，队列中依然为空，因此从队列汇总获得项的任务继续执行
task.done()  # False

# 入队列一项
queue.put_nowait('bar')

# 该任务并未完成，这是由于事件循环不再处于执行状态。
# 由于任务任然注册到该事件循环上，因此将一个用于在执行完之后停止循环的回调注册到事件循环上，就可以再次启动任务
import functools

def stop(l, future):
    l.stop()
    
task.add_done_callback(functoola.partial(stop, loop))

loop.run_forever()

# 由于队列汇总已经包含一项，任务已完成，且任务的结果为队列中的项('bar')
task.done()  # True
task.result()  # 'bar'
```

- 最大队列长度

允许设置Queue对象的最大长度，在创建队列时，可以通过设置maxsize关键字参数实现这一点

```python
import asyncio

queue = asyncio.Queue(maxsize=5)
```

若设置了最大长度，Queue将不再允许入队超过最大值的项，调用put方法将会等待之前的项被移除之后(且只能是之后)将项入队。若在队列满时调用put_nowait，将会引发QueueFull异常

### 示例

> Demo1

用`asyncio`实现`Hello world`代码如下：

```python
import asyncio

async def hello():
    print("Hello world!")
    # 异步调用asyncio.sleep(1):
    r = await asyncio.sleep(1)
    print("Hello again!")

# 获取EventLoop:
loop = asyncio.get_event_loop()
# 执行coroutine
loop.run_until_complete(hello())
# 关闭
loop.close()
```

实现过程

```
@asyncio.coroutine把一个generator标记为coroutine类型，然后，我们就把这个coroutine扔到EventLoop中执行。

hello()会首先打印出Hello world!，然后，yield from语法可以让我们方便地调用另一个generator。由于asyncio.sleep()也是一个coroutine，所以线程不会等待asyncio.sleep()，而是直接中断并执行下一个消息循环。当asyncio.sleep()返回时，线程就可以从yield from拿到返回值（此处是None），然后接着执行下一行语句。

把asyncio.sleep(1)看成是一个耗时1秒的IO操作，在此期间，主线程并未等待，而是去执行EventLoop中其他可以执行的coroutine了，因此可以实现并发执行。
```

> Demo2

我们用Task封装两个`coroutine`试试：

```python
import threading
import asyncio

async def hello():
    print('Hello world! (%s)' % threading.currentThread())
    await asyncio.sleep(1)
    print('Hello again! (%s)' % threading.currentThread())

loop = asyncio.get_event_loop()
tasks = [hello(), hello()]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
```

观察执行过程：

```
Hello world! (<_MainThread(MainThread, started 140735195337472)>)
Hello world! (<_MainThread(MainThread, started 140735195337472)>)
(暂停约1秒)
Hello again! (<_MainThread(MainThread, started 140735195337472)>)
Hello again! (<_MainThread(MainThread, started 140735195337472)>)
```

由打印的当前线程名称可以看出，两个`coroutine`是由同一个线程并发执行的。

> Demo3

如果把`asyncio.sleep()`换成真正的IO操作，则多个`coroutine`就可以由一个线程并发执行。

我们用`asyncio`的异步网络连接来获取sina、sohu和163的网站首页：

```python
import asyncio

async def wget(host):
    print('wget %s...' % host)
    connect = asyncio.open_connection(host, 80)
    reader, writer = yield from connect
    header = 'GET / HTTP/1.0\r\nHost: %s\r\n\r\n' % host
    writer.write(header.encode('utf-8'))
    await writer.drain()
    while True:
        line = await reader.readline()
        if line == b'\r\n':
            break
        print('%s header > %s' % (host, line.decode('utf-8').rstrip()))
    # Ignore the body, close the socket
    writer.close()

loop = asyncio.get_event_loop()
tasks = [wget(host) for host in ['www.sina.com.cn', 'www.sohu.com', 'www.163.com']]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
```

执行结果如下：

```
wget www.sohu.com...
wget www.sina.com.cn...
wget www.163.com...
(等待一段时间)
(打印出sohu的header)
www.sohu.com header > HTTP/1.1 200 OK
www.sohu.com header > Content-Type: text/html
...
(打印出sina的header)
www.sina.com.cn header > HTTP/1.1 200 OK
www.sina.com.cn header > Date: Wed, 20 May 2015 04:56:33 GMT
...
(打印出163的header)
www.163.com header > HTTP/1.0 302 Moved Temporarily
www.163.com header > Server: Cdn Cache Server V2.0
...
```

可见3个连接由一个线程通过`coroutine`并发完成。

> Demo4

```python
import asyncio, time


now = lambda: time.time()
async def func(x):
    print('Waiting for %d s' % x)
    await asyncio.sleep(x)
    return  'Done after {}s'.format(x)

start = now()

coro1 = func(1)
coro2 = func(2)
coro3 = func(3)

tasks = [
    asyncio.ensure_future(coro1),
    asyncio.ensure_future(coro2),
    asyncio.ensure_future(coro3)
]

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.wait(tasks))

for task in tasks:
    print('Task return:', task.result())

print('Program consumes: %fs' % (now()-start))
```

执行过程

```
导入asyncio和time模块，定义一个计时的lambda表达式
通过async关键字定义一个协程函数func()，分别定义3个协程
在func()内部使用sleep模拟IO的耗时操作，遇到耗时操作，await将协程的控制权让出
定义一个tasks列表，列表中分别通过ensure_future()创建3个tasks
协程不能直接运行，需将其加入事件循环中，get_event_loop()用于创建一个事件循环
通过run_until_complete()将tasks列表加入事件循环中
通过tasks的result方法获取协程运行状态
最后计算整个程序的运行耗时
```

在单线程中使用事件循环同时计算多个整数的阶乘

```python
import asyncio

async def factorial(name, number);
    f = 1
    for i in range(2, number+1):
        print("Task %s: Compute factorial (%s)..." % (name, i))
        await asyncio.sleep(0.5)
        f *= 1
    print("Task %s: factorial (%s)=%s"%(name, number, f))
    # 返回当前上下文中实现AbstractEventLoop接口的事件循环对象
    loop = asyncio.get_event_loop()
    tasks = [
        asyncio.ensure_future(factorial("A", 14)),
        asyncio.ensure_future(factorial("B", 13)),
        asyncio.ensure_future(factorial("C", 16))
    ]
    # gather()用来返回一个从给定的协程对象或Future对象得到的聚集结果，
    # 要求所有的Future对象共享同一个事件循环，若所有的任务顺利完成，该函数返回结果列表
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
```

显示当前日期时间

```python
import asyncio.subprocess
import sys 

async def get_date():
    code = 'import datetime; print(datetime.datetime.now())'
    # 创建子进程，并把标准输出重定向道管道
    create = asyncio.create_subprocess_exec(sys.executable, '-c', code, stdout=asyncio.subprocess.PIPE)
    proc = await create
    # 读取一行输出
    data = await proc.stdout.readline()
    line = data.decode('ascii').rstrip()
    # 等待子进程退出
    yield from proc.wait()
    return line

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)
else:
    loop = asyncio.get_event_loop()

date = loop.run_until_complete(get_date())
print("Current date: %s" % date)
loop.close()
```

使用协程计算阶乘

```python
import asyncio
import operator
import functools

async def slow_operation(future, n):
    await asyncio.sleep(1)
    result = functools.reduce(operator.mul, range(1, n+1))
    # 设置计算结果
    future.set_result(result)

loop = asyncio.get_event_loop()
future = asyncio.Future()
# 创建并启动任务，计算50的阶乘
asyncio.ensure_future(slow_operation(future, 50))
loop.run_until_complete(future)
# 输出计算结果
print(future.result())
loop.close()
```

在事件循环中执行函数

```python
import asyncio

def hello_word(loop):
    print('hello word')
    # 结束事件循环
    loop.stop()

loop = asyncio.get_event_loop()
# 在制定的事件循环中执行函数
loop.call_soon(hello_word, loop)
# 一直运行事件循环，阻塞当前线程，直到调用loop.stop()
loop.run_forever()
loop.close()
```

