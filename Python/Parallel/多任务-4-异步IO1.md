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

