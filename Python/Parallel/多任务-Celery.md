# Celery

## 参考

**Celery 官网：http://www.celeryproject.org/**

**Celery 官方文档英文版**：[**http://docs.celeryproject.org/en/latest/index.html**](http://docs.celeryproject.org/en/latest/index.html)

**Celery 官方文档中文版：http://docs.jinkan.org/docs/celery/**

celery配置：http://docs.jinkan.org/docs/celery/configuration.html#configuration

参考：http://www.cnblogs.com/landpack/p/5564768.html    http://blog.csdn.net/happyAnger6/article/details/51408266

http://www.cnblogs.com/forward-wang/p/5970806.html

分布式队列神器 Celery：https://segmentfault.com/a/1190000008022050

celery最佳实践：https://my.oschina.net/siddontang/blog/284107

Celery 分布式任务队列快速入门：http://www.cnblogs.com/alex3714/p/6351797.html

异步任务神器 Celery 快速入门教程：https://blog.csdn.net/chenqiuge1984/article/details/80127446

定时任务管理之python篇celery使用：http://student-lp.iteye.com/blog/2093397

异步任务神器 Celery：http://python.jobbole.com/87086/

celery任务调度框架实践：https://blog.csdn.net/qq_28921653/article/details/79555212

Celery-4.1 用户指南: Monitoring and Management Guide：https://blog.csdn.net/libing_thinking/article/details/78592801

Celery安装及使用：https://blog.csdn.net/u012325060/article/details/79292243

Celery学习笔记（一）：https://blog.csdn.net/sdulsj/article/details/73741350

https://blog.csdn.net/cuomer/article/details/81214438

## 简介

celery是一个用于管理分布式任务的python框架，采用的是面向对象中间件的方法实现。其主要特性包括处理大量小型任务，并将其分发给大量计算节点。最后，每个任务的结果重新组合，构成最终的答案。

Celery是由Python开发、简单、灵活、可靠的分布式任务队列，其本质是生产者消费者模型，生产者发送任务到消息队列，消费者负责处理任务。Celery侧重于实时操作，但对调度支持也很好，其每天可以处理数以百万计的任务。

Celery由以下三部分构成：消息中间件(Broker)、任务执行单元Worker、结果存储(Backend)

![消息中介架构](/home/henry/project/markup/Python/Parallel/images/消息中介架构.png)

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

### 安装

使用python的包管理器pip来安装:

```
pip install -U Celery
```

从官方直接下载安装包:<https://pypi.python.org/pypi/celery/>

```
tar xvfz celery-0.0.0.tar.gz
cd celery-0.0.0
python setup.py build
python setup.py install
```

### broker

消息中介是一个不依赖于celery的软件组件，是一个中间件，用于向分布式任务工作进程发送和接收消息。消息中介也被称为消息中间件。它负责通信网络中的消息交换。这类中间件的编址方案(addressing scheme)不再是点对点式的，而是面向消息式的，其中最知名的就是发布/订阅范式。Celery支持多种类型的消息中介，如RabbitMQ、Redis、Amazon SQS、MongoDB、Memcached 等，其中最为完整的是RabbitMQ和Redis。

- RabbitMQ

它是一个面向消息的中间件(中介消息传递，broker messaging)，它实现了高级消息队列协议(AMQP)。RabbitMQ采用Erlang语言编写，其基础是用于管理集群和故障切换的开放电信平台(OTP)框架。

使用RabbitMQ的细节参照以下链接：<http://docs.celeryproject.org/en/latest/getting-started/brokers/rabbitmq.html#broker-rabbitmq>

```
RabbitMQ是一个功能完备，稳定的并且易于安装的broker. 它是生产环境中最优的选择。

如果我们使用的是Ubuntu或者Debian发行版的Linux，可以直接通过下面的命令安装RabbitMQ: sudo apt-get install rabbitmq-server 安装完毕之后，RabbitMQ-server服务器就已经在后台运行。如果您用的并不是Ubuntu或Debian, 可以在以下网址： http://www.rabbitmq.com/download.html 去查找自己所需要的版本软件。
```

- Redis

关于是有那个Redis作为Broker，可访下面网址：<http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html#broker-redis>

```
Redis也是一款功能完备的broker可选项，但是其更可能因意外中断或者电源故障导致数据丢失的情况。
```

### worker

Worker是任务执行单元，负责从消息队列中取出任务执行，它可以启动一个或者多个，也可以启动在不同的机器节点，这就是其实现分布式的核心

```python
# 我们首先创建tasks.py模块
from celery import Celery

# 我们这里案例使用redis作为broker
# Celery第一个参数是给其设定一个名字， 第二参数我们设定一个中间人broker
app = Celery('demo', broker='redis://127.0.0.1/1')
# app = Celery('demo', broker='amqp://guest@localhost//')
# app = Celery('demo', broker='amqp://name:pwd@url:port//')

# 创建任务函数, 通过加上装饰器app.task, 将其注册到broker的队列中
@app.task
def my_task():
    print("任务函数正在执行....")
```

现在我们在创建一个worker， 等待处理队列中的任务.打开终端，cd到tasks.py同级目录中，执行命令:

```
celery -A tasks worker --loglevel=info

celery -A tasks worker -l info
```

### backend

Backend结果存储官方也提供了诸多的存储方式支持：RabbitMQ、 Redis、Memcached,SQLAlchemy, Django ORM、Apache Cassandra、Elasticsearch。

使用Redis作为存储结果的方案，任务结果存储配置我们通过Celery的backend参数来设定。我们将tasks模块修改如下:

```python
# 修改tasks.py
from celery import Celery

# 我们这里案例使用redis作为broker
# 我们给Celery增加了backend参数，指定redis作为结果存储,并将任务函数修改为两个参数，并且有返回值。
app = Celery('demo',
backend='redis://127.0.0.1:6379/2',
broker='redis://127.0.0.1:6379/1')

# 创建任务函数
@app.task
def my_task(a, b):
    print("任务函数正在执行....")
    return a + b
```

### 发送任务

[参考](http://docs.celeryproject.org/en/master/userguide/calling.html)

[参考](http://docs.celeryproject.org/en/master/reference/celery.html#celery.Celery.send_task)

如何将任务函数加入到队列中，可使用如下方法

```python
# 该任务发送一个任务消息
apply_async(args[, kwargs[, ...]])
# 发送任务消息的便捷方法，不支持添加执行选项
delay(*args, **kwargs)
# 通过任务名来发送任务，与apply_async支持同样的变量
send_task(name, args=None, kwargs=None, countdown=None, eta=None, task_id=None, producer=None, connection=None, router=None, result_cls=None, expires=None, publisher=None, link=None, link_error=None, add_to_parent=True, group_id=None, retries=0, chord=None, reply_to=None, time_limit=None, soft_time_limit=None, root_id=None, parent_id=None, route_name=None, shadow=None, chain=None, task_type=None, **options)

# 使用样例
task.delay(arg1, arg2, kwarg1='x', kwarg2='y')
task.apply_async(args=[arg1, arg2], kwargs={'kwarg1':'x', 'kwarg2':'y'})
task.send_task('tasks.test1', args=[hotplay_id, start_dt, end_dt], exchange='for_task_A', routing_key='task_a')
```

默认多进程执行，也可以多协程处理(greenlet/gevent)

## 使用

### 简单使用

使用celery包含三方面内容

```
1.定义任务函数
2.运行celery服务
3.客户应用程序调用
```

- 定义任务

创建文件tasks.py

```python
from celery import Celery

# 配置broker和backend
broker = 'redis://127.0.0.1:6379/1'
backend='redis://127.0.0.1:6379/2'
# 创建celery实例，指定任务名，传入broker和backend
app = Celery('tasks', broker=broker, backend=backend)

# 创建任务函数add
@app.task
def add(x, y):
    retunr x + y
```

- 启动服务

```
celery -A tasks worker  --loglevel=info
celery -A tasks worker  -l info
```

- 应用调用

注意：如果把返回值赋值给一个变量，那么原来的应用程序也会被阻塞，需要等待异步任务返回的结果。因此，实际使用中，不需要把结果赋值。

```python
# main.py
from tasks import add

# 执行,返回执行的对象
r = add.delay(2, 2)
print(r.id)  # 获取任务编号
print(r.ready()) 
print(r.result)  # 获取结果
print(r.get())  # 获取异步任务结果，默认阻塞
```

- 常用命令

```shell
# 后台启动 celery worker进程 
celery multi start work_1 -A appcelery  
# work_1 为woker的名称，可以用来进行对该进程进行管理

# 多进程相关
celery multi stop WOERNAME      # 停止worker进程,有的时候这样无法停止进程，就需要加上-A 项目名，才可以删掉
celery multi restart WORKNAME        # 重启worker进程

# 查看进程数
celery status -A celery_task       # 查看该项目运行的进程数   celery_task同级目录下

执行完毕后会在当前目录下产生一个二进制文件，celerybeat-schedule 。
该文件用于存放上次执行结果：
　　1、如果存在celerybeat-schedule文件，那么读取后根据上一次执行的时间，继续执行。
　　2、如果不存在celerybeat-schedule文件，那么会立即执行一次。
　　3、如果存在celerybeat-schedule文件，读取后，发现间隔时间已过，那么会立即执行。
```

### 配置文件

Celery 的配置比较多，可以在 官方配置文档：http://docs.celeryproject.org/en/latest/userguide/configuration.html  查询每个配置项的含义。

创建目录结构如下

```shell
# proj tree
__init__.py
celery.py  # celery实例
config.py  # 配置文件
tasks.py  # 任务函数
```

- 单任务

celery.py

```python
# 拒绝隐式引入，因为celery.py的名字和celery的包名冲突，需要使用这条语句让程序正确地运行
from __future__ import absolute_import  
from celery import Celery
 
app = Celery('proj', include=['proj.tasks'])
app.config_from_object('proj.config')
 
if __name__ == '__main__':
	app.start()
```

config.py

```python
from __future__ import absolute_import

CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/5'
BROKER_URL = 'redis://127.0.0.1:6379/6'
```

tasks.py

```python
from __future__ import absolute_import
from proj.celery import app

@app.task
def add(x, y):
	return x + y
```

运行服务

```
celery -A project worker -l info
```

- 多任务1

celery.py

```python
from celery import Celery

app = Celery()
app.config_from_obj('config.py')
```

config.py

```python
from kombu import Exchange,Queue

 
BROKER_URL = "redis://10.32.105.227:6379/0" 
CELERY_RESULT_BACKEND = "redis://10.32.105.227:6379/0"

    
CELERY_QUEUES = (
     # 注意：使用Redis作为broker时，Exchange的名字必须和Queue名字一样
　　　Queue("default",Exchange("default"),routing_key="default"),
　　　Queue("for_task_A",Exchange("for_task_A"),routing_key="task_a"),
　　　Queue("for_task_B",Exchange("for_task_B"),routing_key="task_b")
　)

 
CELERY_ROUTES = {
	'tasks.taskA':{"queue":"for_task_A","routing_key":"task_a"},
	'tasks.taskB':{"queue":"for_task_B","routing_key:"task_b"}
}
```

tasks.py

```python
from celery import app

@app.task
def taskA(x,y):
	return x + y

 
@app.task
def taskB(x,y,z):
	return x + y + z

 
@app.task
def add(x,y):
	return x + y
```

运行任务

```shell
# 启动worker只执行for_task_A队列中的消息，通过指定队列名来指定
celery -A tasks worker -l info -n worker.%h -Q for_task_A
# 启动worker默认名字为celery的Queue
celery -A tasks worker -l info -n worker.%h -Q celey
# gevent多协程启动
celery -A tasks -P gevent -c 5 -l ingo -n worker.%h -Q for_task_A
```

- 多任务2

tasks.py

```python

from celery import Celery
import time
 
 
app = Celery()
app.config_from_object('celeryconfig')
 
# 视频压缩
@app.task
def video_compress(video_name):
    time.sleep(10)
    print 'Compressing the:', video_name
    return 'success'
 
@app.task
def video_upload(video_name):
    time.sleep(5)
    print u'正在上传视频'
    return 'success'
 
# 压缩照片
@app.task
def image_compress(image_name):
    time.sleep(10)
    print 'Compressing the:', image_name
    return 'success'
 
# 其他任务
@app.task
def other(str):
    time.sleep(10)
    print 'Do other things'

```

celeryconfig.py

```python
from kombu import Exchange, Queue
from routers import MyRouter
 
# 配置时区
CELERY_TIMEZONE = 'Asia/Shanghai'
# 配置broker
CELERY_BROKER = 'amqp://localhost'

# 定义一个默认交换机
default_exchange = Exchange('dedfault', type='direct')
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
#
CELERY_ROUTES = (
    {
        'tasks.image_compress': {
            'queue': 'images',
            'routing_key': 'media.image'
        }
    },
    {
        'tasks.video_upload': {
             'queue': 'videos',
             'routing_key': 'media.video'
        }
    },
    {
        'tasks.video_compress': {
             'queue': 'videos',
              'routing_key': 'media.video'
         }
    }, 
)
 
# 在出现worker接受到的message出现没有注册的错误时，使用下面一句能解决
CELERY_IMPORTS = ("tasks",)
```

启动，把不同类的任务路由到不同的worker上处理

```python
# 启动默认的worker
celery worker -Q default --loglevel=info
# 启动处理视频的worker
celery worker -Q videos --loglevel=info
# 启动处理图片的worker
celery worker -Q images --loglevel=info
```

### 定时任务

在celery中执行定时任务非常简单，只需要设置celery对象的CELERYBEAT_SCHEDULE属性即可

- timedelta

config.py

```python
from __future__ import absolute_import
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
	'taskB_scheduler' : {
		'task':"tasks.taskB",
		"schedule":200,
		"args":(10,20,30)
	},
	'add_schedule': {
		"task":"tasks.add",
		"schedule":10,
		"args":(1,2)
	}
}
```

启动时需加`-B`参数

```python
# 在celery_task同级目录下执行   celery worker/beat xxx
celery -A celery_task beat  # 发布任务
celery -A celery_task worker --loglevel=info  # 执行任务
celery -B -A celery_task worker --loglevel=info  # 合并成一条

celery -A proj worker -B -l info
```

- crontab

celey也有crontab模式

config.py

```python
from __future__ import absolute_import
from celery.schedules import crontab

 
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379/5'
BROKER_URL = 'redis://127.0.0.1:6379/6'

CELERY_TIMEZONE = 'Asia/Shanghai'

from celery.schedules import crontab
 
CELERYBEAT_SCHEDULE = {
	# Executes every Monday morning at 7:30 A.M
	'add-every-monday-morning': {
		'task': 'tasks.add',
		'schedule': crontab(hour=7, minute=30, day_of_week=1),
		'args': (16, 16),
	},
}
```

- django中使用

```python
# 创建和开启任务
from celery import Celery
import os

os.environ["DJANGO_SETTINGS_MODULE"] = "dj_py2_demo.settings"

# django不不需开启，celery端需要开启
# import django
# django.setup()

app = Celery(
    'sum_two',
    broker='redis://127.0.0.1:6379/1',
    backend='redis://127.0.0.1:6379/2',
)

@app.task
def sum_two(a, b):
    c = a + b
    return c

# 发布和获取结果
from celery_tasks.sum_two import sum_two

result = sum_two.delay(5,7)
result.get()
```

- 一个案例

目录结构

```
shylin@shylin:~/Desktop$ tree celery_task
celery_task
├── celeryconfig.py    # celeryconfig配置文件
├── celeryconfig.pyc
├── celery.py   # celery对象
├── celery.pyc
├── epp_scripts   # 任务函数
│   ├── __init__.py
│   ├── __init__.pyc
│   ├── test1.py
│   ├── test1.pyc
│   ├── test2.py
│   └── test2.pyc
├── __init__.py
└── __init__.pyc
```

cleeryconfig.py

```python
from __future__ import absolute_import # 拒绝隐式引入，因为celery.py的名字和celery的包名冲突，需要使用这条语句让程序正确地运行
from celery.schedules import crontab

broker_url = "redis://127.0.0.1:6379/5"  
result_backend = "redis://127.0.0.1:6379/6"

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

"schedule": crontab（）与crontab的语法基本一致
"schedule": crontab(minute="*/10",  # 每十分钟执行
"schedule": crontab(minute="*/1"),   # 每分钟执行
"schedule": crontab(minute=0, hour="*/1"),    # 每小时执行
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

发布任务

```shell
# 在celery_task同级目录下执行
celery -A celery_task beat
```

执行任务

```shell
# 在celery_task同级目录下执行
celery -A celery_task worker --loglevel=info
```

### 任务监控

flower是一个用于监控任务(运行进度、任务详情、图标、数据)的web工具。

```python
# 安装
pip install -U flower

# 运行flower命令启动web-server
celery -A proj flower

# 设置端口
# 缺省的端口是http://localhost:5555, 可以使用–port参数改变
celery -A proj flower --port=5555

# 设置broker的URL
celery flower --broker=amqp://guest:guest@localhost:5672//
celery flower --broker=redis://guest:guest@localhost:6379/0    

# 浏览器访问
open http://localhost:5555
        
# api使用
# 获取woker信息
curl http://127.0.0.1:5555/api/workers
```

