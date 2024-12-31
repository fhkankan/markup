# Celery

[Celery 官网](http://www.celeryproject.org/)

[flower官方文档](https://flower.readthedocs.io/en/latest/)

[celery中文](http://celery.xgqyq.com/Celery%E8%B5%B7%E6%AD%A5.html)

参考：http://www.cnblogs.com/landpack/p/5564768.html    http://blog.csdn.net/happyAnger6/article/details/51408266

http://www.cnblogs.com/forward-wang/p/5970806.html

分布式队列神器 Celery：https://segmentfault.com/a/1190000008022050

celery最佳实践：https://my.oschina.net/siddontang/blog/284107

Celery 分布式任务队列快速入门：http://www.cnblogs.com/alex3714/p/6351797.html

异步任务神器 Celery 快速入门教程：https://blog.csdn.net/chenqiuge1984/article/details/80127446

定时任务管理之python篇celery使用：http://student-lp.iteye.com/blog/2093397

异步任务神器 Celery：http://python.jobbole.com/87086/

celery任务调度框架实践：https://blog.csdn.net/qq_28921653/article/details/79555212

Celery安装及使用：https://blog.csdn.net/u012325060/article/details/79292243

Celery学习笔记（一）：https://blog.csdn.net/sdulsj/article/details/73741350

https://blog.csdn.net/cuomer/article/details/81214438

celery合集：https://www.cnblogs.com/hunterxiong/p/17450464.html

## 内容简介

celery是一个用于管理分布式任务的python框架，采用的是面向对象中间件的方法实现。其主要特性包括处理大量小型任务，并将其分发给大量计算节点。最后，每个任务的结果重新组合，构成最终的答案。

Celery是由Python开发、简单、灵活、可靠的分布式任务队列，其本质是生产者消费者模型，生产者发送任务到消息队列，消费者负责处理任务。Celery侧重于实时操作，但对调度支持也很好，其每天可以处理数以百万计的任务。

Celery由以下三部分构成：消息中间件Broker、任务执行单元Worker、结果存储Backend

![消息中介架构](images/消息中介架构.png)

celery通过消息进行通信，通常使用一个叫Broker(中间人)来协client(任务的发出者)和worker(任务的处理者). clients发出消息到队列中，broker将队列中的信息派发给worker来处理。

工作原理

```
- 任务模块Task包含异步任务和定时任务。其中，异步任务通常在业务逻辑中被触发并发往消息队列，而定时任务由Celery Beat进程周期性地将任务发往消息队列；

- 任务执行单元Worker实时监视消息队列获取队列中的任务执行；

- Woker执行完任务后将结果保存在Backend中;
```

应用场景

```
- web应用
当用户在网站进行某个操作需要很长时间完成时，我们可以将这种操作交给Celery执行，直接返回给用户，等到Celery执行完成以后通知用户，大大提好网站的并发以及用户的体验感。

- 任务场景
比如在运维场景下需要批量在几百台机器执行某些命令或者任务，此时Celery可以轻松搞定。

- 定时任务
向定时导数据报表、定时发送通知类似场景，虽然Linux的计划任务可以帮我实现，但是非常不利于管理，而Celery可以提供管理接口和丰富的API。
```

- task

这个任务就是异步任务或者是定时任务，即为 task，我们可以定义这些任务，然后发送到 broker

- broker

消息中间件，用于获取异步或者定时任务，形成一个或多个消息队列，然后发送给 worker 处理这些消息。

消息中介是一个不依赖于celery的软件组件，是一个中间件，用于向分布式任务工作进程发送和接收消息。它负责通信网络中的消息交换。这类中间件的编址方案(addressing scheme)不再是点对点式的，而是面向消息式的，其中最知名的就是发布/订阅范式。Celery支持多种类型的消息中介，如RabbitMQ、Redis、Amazon SQS、MongoDB、Memcached 等，其中最为完整的是RabbitMQ和Redis。

- worker

处理消息的程序，获取 broker 中的消息，然后在 worker 中执行，然后根据配置决定将处理结果发送到 backend

Worker是任务执行单元，负责从消息队列中取出任务执行，它可以启动一个或者多个，也可以启动在不同的机器节点，这就是其实现分布式的核心

- backend

Backend结果存储官方也提供了诸多的存储方式支持：RabbitMQ、 Redis、Memcached,SQLAlchemy, Django ORM、Apache Cassandra、Elasticsearch。

- beat

主要用于调用定时任务，根据设定好的定时任务，比如每天晚上十点执行某个函数，beat 则会在相应的时间将这个 task 发送给 broker，然后 worker 获取任务进行处理

定时任务除了说的每天晚上十点这种周期任务，也可以是间隔任务，比如说每隔多少秒，多少分钟执行一次

**注意**：异步任务的发送是不经过 beat 处理，直接发送给 broker 的

## 安装使用

```
pip install -U Celery
```

或从官方直接下载安装包:<https://pypi.python.org/pypi/celery/>

```
tar xvfz celery-0.0.0.tar.gz
cd celery-0.0.0
python setup.py build
python setup.py install
```

### 简单使用

使用celery包含三方面内容

```
1.定义任务函数
2.运行celery服务
3.客户应用程序调用
```

文件目录结构

```
proj/tasks.py  # 定义celery和task
main.py   # 调用task
```

- 定义任务

创建文件

```python
# tasks.py
from celery import Celery

# 配置broker和backend
broker = 'redis://127.0.0.1:6379/1'
backend='redis://127.0.0.1:6379/2'
# 创建celery实例，指定任务名，传入broker和backend
app = Celery('tasks', broker=broker, backend=backend)

# 创建任务函数add
@app.task
def add(x, y):
    return x + y
```

- 启动服务

```shell
celery -A tasks worker  -l info
```

- 应用调用

注意：如果把返回值赋值给一个变量，那么原来的应用程序也会被阻塞，需要等待异步任务返回的结果。因此，实际使用中，不需要把结果赋值。

```python
# main.py
from proj.tasks import add

# 执行,返回执行的对象
res = add.delay(2, 2)
print(res.get())
```

- result资源释放

因为 backend 会使用资源来保存和传输结果，为了确保资源被释放，所以在执行完异步任务后，你必须对每一个结果调用 `get()` 或者 `forget()` 函数

查看是否资源被释放也很简单，登录到对应的 backend，我这里是 redis，使用 redis-cli 或者通过 docker 进入 redis：

```shell
select 1

keys*
```

查看相应的 task id 是否还在列表就可以知道该资源是否被释放

如果不想手动释放资源，可以在配置里设置一个过期时间，那么结果就会在指定时间段后被释放：

```shell
app.conf.update(result_expires=60)
```

### 单任务

文件目录

```
proj/celery.py
proj/config.py
proj/tasks.py
main.py
```

代码

```python
# celery.py
from celery import Celery
 
app = Celery('proj', include=['proj.tasks'])
app.config_from_object('proj.config')
 
if __name__ == '__main__':
	app.start()
    
# config.py    
result_backend = 'redis://127.0.0.1:6379/5'
broker_url = 'redis://127.0.0.1:6379/6'

# tasks.py
from proj.celery import app

@app.task
def add(x, y):
	return x + y

# main.py
from proj.task import add

res = add.delay(2, 2)
print(res.get())
```

运行服务

```
celery -A project worker -l info
```

### 多任务

文件目录

```
proj/celery.py
proj/config.py
proj/tasks.py
main.py
```

`celery.py`

```python
from celery import Celery
 
 
app = Celery()
app.config_from_object('proj.config')
```

`config.py`

```python
from kombu import Queue

 
BROKER_URL = "redis://192.168.3.248:31379/1" 
CELERY_RESULT_BACKEND = "redis://192.168.3.248:31379/2"

CELERY_QUEUES = (
    Queue('default'),
    Queue('videos'),
    Queue('images')
)

CELERY_DEFAULT_QUEUE = 'default'

CELERY_ROUTES = {
    'proj.tasks.image_compress': {'queue': 'images'},
    'proj.tasks.video_upload': {'queue': 'videos'},
    'proj.tasks.video_compress': {'queue': 'videos'}
}
 
CELERY_INCLUDE = ['proj.tasks']
```

`tasks.py`

```python
from proj.celery import app


@app.task
def video_compress(video_name):
    time.sleep(10)
    print('Compressing the:', video_name)
    return 'success'
 
@app.task
def video_upload(video_name):
    time.sleep(5)
    print('正在上传视频', video_name)
    return 'success'
 
@app.task
def image_compress(image_name):
    time.sleep(10)
    print('Compressing the:', image_name)
    return 'success'
 
@app.task
def other(str):
    time.sleep(10)
    print('Do other things')
```

启动，把不同类的任务路由到不同的worker上处理

```python
# 启动默认的worker
celery -A proj worker -Q default -l info -P eventlet
# 启动处理视频的worker
celery -A proj worker -Q videos -l info
# 启动处理图片的worker
celery -A proj worker -Q images -linfo
```

`main.py`

```python
from proj.tasks import video_compress, image_compress, other
from proj.celery import app

res = video_compress.delay("1")
print(res.get())

res = image_compress.delay("2")
print(res.get())

res= other.delay("3")
print(res.get())
```

## 配置处理

### 常用配置

```python
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/1'
include = []
enable_utc = False
timezone = 'Asia/Shanghai'
beat_schedule = {}
# 资源释放
result_expires = 3600
routes = {}
queues = ()


# 老的格式
BROKER_URL
CELERY_RESULT_BACKEND
CELERY_INCLUDE
CELERY_TIMEZONE
CELERYBEAT_SCHEDULE
CELERY_ROUTES
CELERY_QUEUES
```
### 配置加载

函数传参形式

```python
from celery import Celery

# 配置broker和backend
broker = 'redis://192.168.3.248:31379/1'
backend = 'redis://192.168.3.248:31379/2'
# 创建celery实例，指定任务名，传入broker和backend
app = Celery('tasks', broker=broker, backend=backend)
```

配置更新

```python
# 直接更新
app.conf.task_serializer = 'json'

# 集中更新
app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Ignore other content
    result_serializer='json',
    timezone='Europe/Oslo',
    enable_utc=True,
)
```

文件形式加载

```python
"""
proj/celery.py
    /config.py
    /tasks.py
"""
# celeryconfig.py
broker_url = 'redis://localhost/0'
result_backend = 'redis://localhost/1'
include = ['proj.tasks']


# celery.py
from celery import Celery

app = Celery()
app.config_from_object("proj.config")  # 方式一， 推荐
from . import config
app.config_from_object(config)  # 方式二


if __name__ == '__main__':
    app.start()
```

类的方式加载

```python
"""
proj/celery.py
    /tasks.py
"""
# celery.py

from celery import Celery

app = Celery()

class Config:
    include = ['proj.tasks']
    broker_url = 'redis://localhost:6379/0'
    result_backend = 'redis://localhost:6379/1'
    
app.config_from_object(Config)

if __name__ == '__main__':
    app.start()
```

## 任务task

### 基础定义

```python
@app.task
def add(x, y):
    return x + y


# 多个装饰器
@app.task
@decorator1
@decorator2
def add(x, y):
    return x + y

# 任务名称
# 一个task有唯一的名称，如果在定义的时候不指定，系统会为我们默认一个名称
# tasks1.py
from .celery import app

@app.task(name="tasks1.add")
def add(x, y):
    return x + y
```

### 日志处理

我们可以在启动 worker 的时候指定日志的输出，定义格式如下：

```python
celery -A proj worker -l INFO --logfile=/Users/hunter/python/celery_log/celery.log
```

在 task 中的定义

```shell
# 可以使用celery中方法：
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

# 也可以直接使用 logging 模块
import logging

logger1 = logging.getLogger(__name__)

# 直接在 task 中输出
@app.task(name="tasks1.add")
def add(x, y):
    logger.info("this is from logger")
    return x + y
```

### 任务重试

对于一个 task，我们可以对其设置 retry 参数来指定其在任务执行失败后会重试几次，以及隔多长时间重试

```python
# tasks1.py

# autoretry_for 表示的是某种报错情况下重试；default_retry_delay 表示重试间隔时长,以秒为单位，默认值是 3 * 60s；retry_kwargs 是一个 dict，其中有一个 max_retries 参数，表示的是最大重试次数
@app.task(autoretry_for=(Exception, ),  default_retry_delay=10, retry_kwargs={'max_retries': 5})
def div(x, y):
    return x / y


# 如果你的 task 依赖另一个 service 服务，比如会调用其他系统的 API，retry_backoff和retry_backoff_max可以用于避免请求过多的占用服务。
# retry_backoff为 True 的话，自动重试的时间间隔会成倍的增长；retry_backoff_max是重试的最大的间隔时间，比如重试次数设置的很大，retry_backoff 的间隔时间重复达到了这个值之后就不再增大了。这个值默认是 600s，也就是 10分钟。
# retry_jitter 值设为 False， 时间间隔将不是一个随机值
@app.task(autoretry_for=(Exception, ), retry_backoff=2, retry_backoff_max=40, retry_jitter=False, retry_kwargs={'max_retries': 8})
def div(x, y):
    return x / y
```

### 任务调用

基础调用

```python
# 该任务发送一个任务消息
apply_async(args[, kwargs[, ...]])
# 发送任务消息的便捷方法，不支持添加执行选项
delay(*args, **kwargs)
# 直接调用对象,任务不会被 worker 执行,而是在当前线程中执行(消息不会被发送)
calling(call)

# 使用样例
task.delay(arg1, arg2, kwarg1='x', kwarg2='y')
task.apply_async(args=[arg1, arg2], kwargs={'kwarg1':'x', 'kwarg2':'y'})
```
计时和超时
```python
add.apply_async((1, 2), countdown=10) # 10s以后执行
add.apply_async((1, 2), eta=datetime.now() + timedelta(seconds=10))  # 10s以后执行
add.apply_async((1, 2), countdown=60, expires=120)  # 距现在60秒后开始执行，两分钟后过期
```

消息重发

```python
add.apply_async((2, 2), retry=False)  # 默认重试，关闭重试

add.apply_async((2, 2), retry=True, retry_policy={
    'max_retries': 3,  # 最大重试次数，设置为None表示无限重试，默认值是3
    'interval_start': 0,  # 重试间隔，浮点数或整数。默认是0.
    'interval_step': 0.2, # 连续重试时，该数字会被添加到重试间隔时间上，浮点数或者整数。默认是0.2.
    'interval_max': 0.2,  # 重试之间等待的最大秒数（浮点数或整数）。默认值为0.2
})


# 传输连接丢失或无法启动连接时
from celery.utils.log import get_logger
logger = get_logger(__name__)
try:
    add.delay(2,2)
except add.OperationalError as exc:
    logger.exception('Sending task raised: %r', exc)
```

### 结果追踪

```python
res = add.delay(1, 2)

# 结果信息
res.id  # 获取任务编号
res.ready()   # 判断函数运行是否完成
res.result  # 获取结果
res.get()  # 获取异步任务结果，默认阻塞
res.get(timeout=2)


# 任务状态
res.failed()  # 任务执行是否失败，返回 布尔型数据
res.successful()  # 任务执行是否成功，返回布尔型数据
res.state  # 执行的任务所处的状态，state 的值会在 PENDING，STARTED，SUCCESS，RETRY，FAILURE 这几种状态中，分别是 待处理中，任务已经开始，成功，重试中，失败

# 报错处理
res.state # FAILURE
res.get() # 返回报错
res.get(propagate=False)  # 忽略程序的报错，把程序报错的信息作为结果返回
# 当延时任务在程序中报错，它的返回值就不会是正确的，通过 res3.traceback 是否有值来判断函数运行过程中是有报错
if res.traceback:
    print("延时任务报错")
else:
    print("程序正常运行，可以获取返回值")
```

### 忽略结果

有时候延时任务的结果我们并不想保存，但是我们配置了 result_backend 参数，这个时候我们有三种方式不保存运行结果。

```python
# 定义特定任务时
@app.task(ignore_result=True)
def add(x, y):
    return x + y

# 所有任务都不保存
app.conf.update(
    task_ignore_result=True
)

# 执行单个任务时
from proj.tasks1 import add
add.apply_async((1, 2), ignore_result=True)
```

### 任务撤回

撤销任务是通过发送一个广播消息给所有的workers，workers会在内存中维护一个已撤销任务的list。当一个worker启动时，它会同集群中的其他workers同步这个列表。当worker执行任务时会跳过执行该任务。对于已经开始执行的任务并不会终止。除非设置了terminate

```python
# 命令行
celery -A proj control revoke <task_id>

# 代码
result.revoke()

from celery.result import AsyncResult
AsyncResult(id).revoke()

app.control.revoke('d9078da5-9915-40a0-bfa1-392c7bde42ed') 
# 终止多个任务
app.control.revoke(["d9078da5-9915-40a0-bfa1-392c7bde42ed", "d9078da5-9915-40a0-bfa1-392c7bde42ed"])

# 如果设置了terminate 那么执行任务的worker的子进程会被终止， 该进程可能在执行其他任务，所以不能以编程形式调用
app.control.revoke('d9078da5-9915-40a0-bfa1-392c7bde42ed', terminate=True)  # 默认发送的信号是TERM
app.control.revoke('d9078da5-9915-40a0-bfa1-392c7bde42ed', terminate=True, signal='SIGKILL')
```

已撤销任务列表是在内存中的，因此当重启所有的worker时，这个列表会消失。如果你想在重启后依旧保留，可以通过给**celery** **worker** 指定`--statedb`参数来保存到一个文件

```shell
celery -A proj worker -l INFO --statedb=/var/run/celery/worker.state

# 希望每个worker实例保存到单独的文件中，使用类似于%n这样的格式代表节点名称：
celery multi start 2 -l INFO --statedb=/var/run/celery/%n.state
```

## worker

### 启动

常规启动

```shell
# celery启动命令
celery -A proj worker  -l info  # tasks为Celery实例所在的文件名
celery -A proj worker  --loglevel=info
celery -A proj worker --loglevel=INFO --logfile=/Users/hunter/python/celery_log/celery.log  # 指定日志文件

# 开启Eventlet pool支持
celery -A proj worker -P eventlet -c 1000
```

并发

```shell
# 一般来说，当直接启动worker的时候，会默认同时起好几个worker进程。如果不指定worker的数量，worker的进程会默认是所在机器的CPU的数量。我们也可以通过 concurrency 参数来指定启动 worker 的进程数。
celery -A proj worker --concurrency=3 -l INFO
celery -A proj worker -c 3 -l INFO

# 一个机器可以启动多个worker，但是必须指定worker名称并且附加--hostname
celery -A proj worker --loglevel=INFO --concurrency=10 -n worker1@%h
celery -A proj worker --loglevel=INFO --concurrency=10 -n worker2@%h
# 如当前hostname为george.example.com，%h表示george.example.com，%n表示george，%d表示example.com
```

查看/杀死

```shell
# 查看进程
ps aux |grep 'celery -A proj'

# 杀死 worker 进程
ps aux | grep 'celery -A proj' | awk '{print $2}' |xargs sudo kill -9
```

重启

```shell
# 后台启动 celery worker进程 
celery multi start work_1 -A proj -l INFO -c 4 --pidfile=/var/run/celery/%n.pid

# 多进程相关
celery multi stop WOERNAME      # 停止worker进程,有的时候这样无法停止进程，就需要加上-A 项目名，才可以删掉
celery multi restart WORKNAME   # 重启worker进程

# 查看进程数
celery status -A celery_task       # 查看该项目运行的进程数   celery_task同级目录下
```

### 限制

时间限制

```python
# 时间限制
# 时间限制是一个任务在执行它的进程被终止并被新进程取代之前可以运行的最大秒数。
# 时间限制设置为软和硬两个值。 软时间限制允许任务在被杀死之前捕获异常以进行清理：硬超时是不可捕获的并且强制终止任务。

# 命令行设置
--time-limit
--soft-time-limit

# 配置项设置
task_time_limit
task_soft_time_limit

# 通过远程控制在运行时更改时间限制
app.control.time_limit('tasks.crawl_the_web', soft=60, hard=120, reply=True)  # 软限制为1分钟，硬限制为2分钟：


# 开启软限制后会触发一个可捕获的异常以便在硬时间到来之前清理该任务。
from myapp import app
from celery.exceptions import SoftTimeLimitExceeded

@app.task
def mytask():
    try:
        do_work()
    except SoftTimeLimitExceeded:
        clean_up_in_a_hurry()
```

速率限制

```shell
# 影响所有实例
app.control.rate_limit('myapp.mytask', '200/m')  # 使其每分钟最多执行200个该类型的任务

# 影响特定实例
app.control.rate_limit('myapp.mytask', '200/m',  destination=['celery@worker1.example.com'])
```

### 队列

一个worker实例可以从任意多队列中消费。默认情况下会从task_queues设置中定义的队列中消费（如果没指定，默认队列`celery`）。

启动worker时指定 消费特定队列，这个特定队列，若是多个用逗号分隔开。

```shell
celery -A proj worker -l INFO -Q queue_1,queue_2
```

增加队列

```shell
# 命令行
celery -A proj control add_consumer foo  # 所有worker
celery -A proj control add_consumer foo -d celery@worker1.local  # 特定worker

# 代码
app.control.add_consumer('foo', reply=True)
app.control.add_consumer('foo', reply=True, destination=['worker1@example.com'])
pp.control.add_consumer(
...     queue='baz',
...     exchange='ex',
...     exchange_type='topic',
...     routing_key='media.*',
...     options={
...         'queue_durable': False,
...         'exchange_durable': False,
...     },
...     reply=True,
...     destination=['w1@example.com', 'w2@example.com'])
```

取消队列

```shell
# 命令行
celery -A proj control cancel_consumer foo  # 所有worker
celery -A proj control cancel_consumer foo -d celery@worker1.local  # 特定worker

# 代码
app.control.cancel_consumer('foo', reply=True)
```

查看队列

```shell
# 命令行
celery -A proj inspect active_queues
celery -A proj inspect active_queues -d celery@worker1.local

# 代码
app.control.inspect().active_queues()  # 所有worker
app.control.inspect(['worker1@localhost']).active_queues()  # 特定worker
```

### 检测

获取worker信息

```python
from hunter.celery import app

# 获取信息
i = app.control.inspect()  # 所有worker
i = app.control.inspect(['worker1@localhost', 'worker2@localhost'])  # 指定的多个worker
i = app.control.inspect('worker1@localhost')  # 指定worker

# 获取worker中注册的任务列表
i.registered()
# [{'worker1.example.com': ['tasks.add','tasks.sleeptask']}]

# 获取活动任务列表
i.active()
# [{'worker1.example.com':[{'name': 'tasks.sleeptask','id': '32666e9b-809c-41fa-8e93-5ae0c80afbbf','args': '(8,)','kwargs': '{}'}]}]

# 获取待调度的任务列表
i.scheduled()
# {'worker1@localhost': [{'eta': '2022-08-02T22:56:49.503517+08:00', 'priority': 6, 'request': {'id': '23080c03-a906-4cc1-9ab1-f27890c58adb', 'name': 'blog.tasks.add', 'args': [1, 1], 'kwargs': {}, 'type': 'blog.tasks.add', 'hostname': 'worker1@localhost', 'time_start': None, 'acknowledged': False, 'delivery_info': {...}}]}

# queue队列中等待执行的task
i.reserved()
# [{'worker1.example.com':[{'name': 'tasks.sleeptask','id': '32666e9b-809c-41fa-8e93-5ae0c80afbbf','args': '(8,)','kwargs': '{}'}]}]
```

检测还活着的worker

```python
from hunter.celery import app

app.control.ping(timeout=0.5)  # [{'worker1@localhost': {'ok': 'pong'}}]

# 可以指定 worker 来操作
app.control.ping(['worker1@localhost'])
```

## 定时任务

Celery beat 是一个调度器，它定期启动任务，然后由集群中的可用工作节点执行。

默认情况下，这些条目取自beat_schedule设置，但也可以使用自定义存储，例如将条目存储在SQL数据库中。

必须确保一个调度只有一个调度程序进行调度，否则任务会重复运行。

### 配置

**timedelta**

```python
# config.py
from datetime import timedelta
 

CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/5'
BROKER_URL = 'redis://127.0.0.1:6379/6'
# 配置时区
CELERY_TIMEZONE = 'Asia/Shanghai'

# 每隔30秒执行add函数
CELERYBEAT_SCHEDULE = {
	'add-every-30-seconds': {
		'task': 'proj.tasks.add',
		'schedule': timedelta(seconds=30),
		'args': (16, 16)
	},
}

# 执行多个定时任务
CELERYBEAT_SCHEDULE = {
	'taskA_schedule' : {
		'task':'tasks.taskA',
		'schedule':20,  # 间隔20s
		'args':(5,6)  # 参数
	},
	'add_schedule': {
		"task":"tasks.add",
		"schedule":10,
		"args":(1,2)
	}
}
```

**crontab**

```python
# config.py
from celery.schedules import crontab

 
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/5'
BROKER_URL = 'redis://127.0.0.1:6379/6'

CELERY_TIMEZONE = 'Asia/Shanghai'
 
CELERYBEAT_SCHEDULE = {
	# Executes every Monday morning at 7:30 A.M
	'add-every-monday-morning': {
		'task': 'tasks.add',
		'schedule': crontab(hour=7, minute=30, day_of_week=1),
		'args': (16, 16),
	},
}
```

### 启动

```shell
celery -A celery_task beat  # 启动celery beat服务
celery -A celery_task worker --loglevel=info  # 启动celery的worker
celery -A proj worker -B -l info  # 如果不会运行多个worker节点，可以将beat嵌入到worker中

# Beat 需要将任务的最后运行时间存储在本地数据库文件中（默认命名为 celerybeat-schedule），因此它需要访问当前目录的写入权限，
# 文件指定自定义位置
celery -A proj beat -s /home/celery/var/run/celerybeat-schedule
```

### 实例

目录结构

```
celery_task
├── celeryconfig.py    # celeryconfig配置文件
├── celery.py   # celery对象
├── epp_scripts   # 任务函数
│   ├── test1.py
│   ├── test2.py
```

cleeryconfig.py

```python
from __future__ import absolute_import # 拒绝隐式引入，因为celery.py的名字和celery的包名冲突，需要使用这条语句让程序正确地运行
from celery.schedules import crontab


broker_url = "redis://127.0.0.1:6379/2"   # 使用redis存储任务队列
result_backend = "redis://127.0.0.1:6379/6"  # 使用redis存储结果

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = "Asia/Shanghai"  # 时区设置
worker_hijack_root_logger = False  # celery默认开启自己的日志，可关闭自定义日志，不关闭自定义日志输出为空
result_expires = 60 * 60 * 24  # 存储结果过期时间（默认1天）

# 导入任务所在文件
imports = [
    "celery_task.epp_scripts.test1",  # 导入py文件
    "celery_task.epp_scripts.test2",
]


# 需要执行任务的配置
beat_schedule = {
    "test1": {
        "task": "celery_task.epp_scripts.test1.celery_run",  #执行的函数
        "schedule": crontab(minute="*/1"),   # every minute 每分钟执行 
        "args": ()  # # 任务函数参数
    },

    "test2": {
        "task": "celery_task.epp_scripts.test2.celery_run",
        "schedule": crontab(minute=0, hour="*/1"),   # every minute 每小时执行
        "args": ()
    },
}
```

celery初始化文件

```python
# coding:utf-8
from __future__ import absolute_import # 拒绝隐式引入，因为celery.py的名字和celery的包名冲突，需要使用这条语句让程序正确地运行
from celery import Celery

# 创建celery应用对象
app = Celery("celery_demo")

# 导入celery的配置信息
app.config_from_object("celery_task.celeryconfig")
```

任务函数（epp_scripts目录下）

```python
# test1.py
from celery_task.celery import app

def test11():
    print("test11----------------")

def test22():
    print("test22--------------")
    test11()

@app.task
def celery_run():
    test11()
    test22()

if __name__ == '__main__':
    celery_run()
    
------------------------------------------------------------
# test2.py
from celery_task.celery import app

def test33():
    print("test33----------------")
    # print("------"*50)

def test44():
    print("test44--------------")
    # print("------" * 50)
    test33()

@app.task
def celery_run():
    test33()
    test44()


if __name__ == '__main__':
    celery_run()
```

启动

```shell
# 在celery_task同级目录下执行
# 执行worker
celery -A celery_task worker --loglevel=info

# 发布beat任务
celery -A celery_task beat
```

## 消息队列

task 的处理方式，将 task 发送到队列 queue，然后 worker 从 queue 中一个个的获取 task 进行处理。task 的队列 queue 可以是多个，处理 task 的 worker 也可以是多个，worker 可以处理任意 queue 的 task，也可以处理指定 queue 的 task。

### 默认队列

当我们运行一个最简单的延时任务比如 `add.delay(1, 2)` 时，并没有设置一个消息队列，因为如果我们没有指定，系统会为我们创建一个默认队列。

这个默认的队列被命名为 celery，值在 app.conf.task_default_queue，我们可以查看一下：

```python
# main.py
from hunter.celery import app
app.conf.task_default_queue
# 输出为 'celery'
```

### 定义队列

只有一个 worker 处理 task，每个 task 需要处理的时间很长，因为 worker 被占用，这样在我们的任务队列里就会积压很多的 task。

有一些需要即时处理的任务则会被推迟处理，这样的情况下，我们理想的设计是设置多个 worker，多个 worker 分别处理指定队列的 task。

文件目录

```
proj/celery.py
proj/config.py
proj/tasks.py
main.py
```

`celery.py`

```python
from celery import Celery
 
 
app = Celery()
app.config_from_object('proj.config')
```

`config.py`

```python
from kombu import Exchange, Queue
 
# 配置时区
CELERY_TIMEZONE = 'Asia/Shanghai'
# 配置broker
CELERY_BROKER = 'amqp://localhost'
CELERY_RESULT_BACKEND = "redis://127.0.0.1:31379/2"

# 定义一个默认交换机
default_exchange = Exchange('default', type='direct')
 # 定义一个媒体交换机
media_exchange = Exchange('media', type='direct')
 
# 创建三个队列，一个是默认队列，一个是video、一个image
CELERY_QUEUES = (
    Queue('default', default_exchange, routing_key='default'),
    Queue('videos', media_exchange, routing_key='media.video'),
    Queue('images', media_exchange, routing_key='media.image')
)
 
CELERY_DEFAULT_QUEUE = 'default'
CELERY_DEFAULT_EXCHANGE = 'default'
CELERY_DEFAULT_ROUTING_KEY = 'default'

# 任务路由
CELERY_ROUTES = {
    'proj.tasks.image_compress': {'queue': 'images','routing_key': 'media.image'},
    'proj.tasks.video_upload': {'queue': 'videos', 'routing_key': 'media.video'},
    'proj.tasks.video_compress': {'queue': 'videos', 'routing_key': 'media.video'}
}
```

`tasks.py`

```python
from proj.celery import app


@app.task
def video_compress(video_name):
    time.sleep(10)
    print('Compressing the:', video_name)
    return 'success'
 
@app.task
def video_upload(video_name):
    time.sleep(5)
    print('正在上传视频', video_name)
    return 'success'
 
@app.task
def image_compress(image_name):
    time.sleep(10)
    print('Compressing the:', image_name)
    return 'success'
 
@app.task
def other(str):
    time.sleep(10)
    print('Do other things')
```

启动，把不同类的任务路由到不同的worker上处理

```python
# 启动默认的worker
celery -A proj worker -Q default -l info
# 启动处理视频的worker
celery -A proj worker -Q videos -l info
# 启动处理图片的worker
celery -A proj worker -Q images -linfo
```

`main.py`

```python
from proj.tasks import video_compress, image_compress, other
from proj.celery import app

res = video_compress.delay("1")
print(res.get())

res = image_compress.delay("2")
print(res.get())

res= other.delay("3")
print(res.get())

res = other.apply_async(("4",), queue="videos")  # 优先级最高
print(res.get())

```

## 监控管理

```
1.自带命令
2.使用flower
3.使用日志
4.使用backend
5.自定义监控
6.集成APM工具
7.使用Prometheus和Grafana
```

- 自带命令

```shell
# 查看
celery -A proj status  # 列出集群中的活动节点
celery -A proj inspect active  # 列出活动任务
celery -A proj inspect scheduled  # 列出计划的ETA任务
celery -A proj inspect reserved  # 列出保留任务(预读的所有任务，和当前正在等待执行的任务,不包括ETA)
celery -A proj inspect revoked  # 列出已撤消任务的历史记录
celery -A proj inspect registered  # 列出注册任务
celery -A proj inspect stats  # 展示worker统计数据


# 具体任务
celery -A proj inspect query_task e9f6c8f0-fec9-4ae8-a8c6-cf8c8451d4f8  # 查看任务信息
celery -A proj inspect query_task id1 id2 ... idN  # 查看多个任务信息
celery -A proj result -t tasks.add 4e196aa4-0141-4601-8138-7aa33db0f577 # 显示一个任务的结果

# 清除（慎用，无法撤回）
celery -A proj purge  # 清除所有已配置的CELERY_QUEUES任务队列中的消息
celery -A proj purge -Q celery,foo,bar  # 指定要清除的队列
celery -A proj purge -X celery  # 排除某个队列

# 管理
celery -A proj control enable_events  # 启用事件
celery -A proj control disable_events  # 关闭事件
celery -A proj migrate redis://localhost amqp://localhost  # 将任务从一个broker迁移到另一个broker(实验性)

# 特定worker
celery -A proj inspect -d w1@e.com,w2@e.com reserved
celery -A proj control -d w1@e.com,w2@e.com enable_events
```

- 使用flower

[flower](https://flower.readthedocs.io/en/latest/)是一个用于监控任务(运行进度、任务详情、图标、数据)的web工具。

[页面介绍](https://www.jianshu.com/p/4a408657ef76)

安装配置

```python
# 安装
pip install -U flower


# 运行flower命令启动web-server
celery -A <your_project_name> flower

# your_project_name是Celery 应用实例的名称

# 常用启动项
--port=<port>：指定 Flower 的监听端口，默认为 5555。
--broker=<broker_url>：指定 Celery 使用的消息代理（broker）的 URL 地址，例如 Redis 或 RabbitMQ。
--basic_auth=<username>:<password>：设置基本身份认证，允许你通过用户名和密码保护 Flower 控制台。
--address=<address>：设置 Flower 绑定的 IP 地址，默认是 127.0.0.1。
--persistent=True：启用持久化，将任务结果和状态保存到磁盘上。

# 示例
celery -A <your_project_name> flower --port=5555 --broker=amqp://guest:guest@localhost:5672//  
celery -A <your_project_name> flower --address='0.0.0.0' --port=5555 --broker='redis://guest:guest@localhost:6379/0'
```

使用

```shell
# 浏览器访问
open http://localhost:5555
        
# api使用
# 获取woker信息
curl http://127.0.0.1:5555/api/workers
# 获取任务信息
curl http://127.0.0.1:5555/api/task/info/<task_id>
```

Dashboard

```
# 状态
Dashboard 页面是展示异步任务队列的主要情况，该页面包括如下几种状态的任务：

Active: 表示worker从队列中获取到任务，且正在执行的任务数
Processed: 表示worker从队列中获取到任务的总数据量
Failed: 表示worker从队列中获取到任务，且执行失败了的（异常退出）
Successed: 表示worker从队列中获取到任务，且执行成功了的
Retried: 表示worker从队列中获取到任务，因为一些其他原因重新执行的数量

所以，上述这些数量的关系如下：
Processed = Active + Received + Failed + Successed + Retried
其中 Received 表示该任务分配到了worker中，但是还没有被处理的任务数量

# worker
Worker Name 表示的是执行celery任务的worker名称；
Status 表示的是该worker的状态，包括 Online (在线) 、 Offline（离线），重启flower进程，即可将Offline状态的worker剔除掉；
Active / Processed / Failed / Successed / Retried 分别表示该worker正在执行的任务数、该worker处理的总任务数、处理失败的任务数、处理成功的任务数、重试的任务数；
Load Average 表示系统在 1min / 5min / 15min 内的CPU平均负载（百分比）
```

tasks

```
Tasks 页面是展示所有worker接收到的任务的处理情况。下面对该表格中的做一些介绍

Name: 表示该任务的名称，默认规则为该函数的路径规则，例如 {模块名}.{文件名}.{函数名}
UUID: 表示一个唯一字符串ID用于表示该任务
State: 表示该任务的状态，包括： SUCCESS / FAILURE / STARTED / RECEIVED
SUCCESS 表示该任务执行完毕且成功
FAILURE 表示该任务执行失败
STARTED 表示该任务正在执行
RECEIVED 表示该任务在worker中，只是被接收而已
args: 表示该任务的列表参数
kwargs: 表示该任务的字典参数
Result: 表示该任务函数的返回结果
Received: 表示该任务在worker接收到的时间
Started: 表示该任务在worker开始执行的时间
Runtime: 表示该任务在worker真正执行的耗时（单位：秒）
Worker: 表示该任务所在的worker名称
```

Broker

```
Broker 页面展示的是celery连接消息队列的信息，包括消息队列的访问URL，下面的截图展示的是链接的RabbitMQ，当然也可以链接Redis等。

Name: 表示队列的名称
Messages: 表示该队列的消息数量
Unacked: 表示该队列中还没收到ack确认的消息数量（该消息可能处于 RECEIVED 或是 STARTED）
Ready: 表示该队列中还未分配到worker的消息数量
Consumers: 表示消费者数量（即worker数量）
Idle since: 表示该队列空闲的最初时间，否则为 N/A
```

Monitor

```
Monitor 页面展示的是celery后台任务的曲线展示状况。
```

- 使用日志

```python
# 配置 Celery 的日志记录，可以帮助你追踪任务的执行情况。可以在 Celery 配置中设置日志级别和日志文件路径
import logging

logging.basicConfig(
    filename='celery.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

- backend

如果你使用了结果后端（如 Redis、Database），可以进入后端中查看task信息

```shell
# rabbitmq
rabbitmqctl list_queues name messages messages_ready  messages_unacknowledged  #  查找队列中的任务数量
rabbitmqctl list_queues name consumers  # 正在从当前队列消费的worker的数量
rabbitmqctl list_queues name memory  # 给某个队列分配的内存量

# redis
redis-cli -h HOST -p PORT -n DATABASE_NUMBER llen QUEUE_NAME  # 查找队列中的任务数量
redis-cli -h HOST -p PORT -n DATABASE_NUMBER keys \*  # 默认队列名是celery，获取所有可用的队列
```

也可以写脚本定期查询任务状态和结果。这样可以手动监控任务的成功率和失败原因。

```python
# get_result.py
from celery.result import AsyncResult
from tasks import app

result = AsyncResult(<task_id>, app)
if result.successful():
    print(result.get())
elif result.failed():
    print("Task failed:", result.result)
elif result.status == "pending":
    print("pending")
elif result.status == "retry":
    print("retry")
elif result.status == "started":
    print("started")
```

- 自定义监控

可以编写自定义的监控逻辑，将任务执行情况存储在数据库或外部监控系统中。例如，在任务开始和结束时记录相关信息：

```python
@app.task(bind=True, acks_late=True)
def my_task(self):
    # 记录任务开始
    log_task_start(self.request.id)
    try:
        # 任务逻辑
        pass
    except Exception as exc:
        log_task_failure(self.request.id, exc)
        raise self.retry(exc=exc)
    else:
        log_task_success(self.request.id)

```

-  集成 APM 工具

使用应用性能监控（APM）工具（如 New Relic、DataDog 或 Sentry）可以深入了解 Celery 任务的性能和错误。这些工具通常提供更全面的监控和分析功能。

- 使用Prometheus和Grafana

通过 Prometheus 收集指标，并使用 Grafana 可视化这些指标。可以为 Celery 设置自定义指标，如任务处理时间、成功率等。
