# Celery

celery通过消息进行通信，通常使用一个叫Broker(中间人)来协client(任务的发出者)和worker(任务的处理者). clients发出消息到队列中，broker将队列中的信息派发给worker来处理。

一个celery系统可以包含很多的worker和broker，可增强横向扩展性和高可用性能。

## 安装

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

## 中间人(broker)

RabbitMQ

使用RabbitMQ的细节参照以下链接：<http://docs.celeryproject.org/en/latest/getting-started/brokers/rabbitmq.html#broker-rabbitmq>

```
RabbitMQ是一个功能完备，稳定的并且易于安装的broker. 它是生产环境中最优的选择。

如果我们使用的是Ubuntu或者Debian发行版的Linux，可以直接通过下面的命令安装RabbitMQ: sudo apt-get install rabbitmq-server 安装完毕之后，RabbitMQ-server服务器就已经在后台运行。如果您用的并不是Ubuntu或Debian, 可以在以下网址： http://www.rabbitmq.com/download.html 去查找自己所需要的版本软件。
```

Redis

关于是有那个Redis作为Broker，可访下面网址：<http://docs.celeryproject.org/en/latest/getting-started/brokers/redis.html#broker-redis>

```
Redis也是一款功能完备的broker可选项，但是其更可能因意外中断或者电源故障导致数据丢失的情况。
```

## 创建应用

我们首先创建tasks.py模块, 其内容为:

```python
from celery import Celery

# 我们这里案例使用redis作为broker
# Celery第一个参数是给其设定一个名字， 第二参数我们设定一个中间人broker
app = Celery('demo', broker='redis://127.0.0.1/1')

# 创建任务函数, 通过加上装饰器app.task, 将其注册到broker的队列中
@app.task
def my_task():
    print("任务函数正在执行....")
```

现在我们在创建一个worker， 等待处理队列中的任务.打开终端，cd到tasks.py同级目录中，执行命令:

```
celery -A tasks worker --loglevel=info
```

## 存储结果

如果我们想跟踪任务的状态，Celery需要将结果保存到某个地方。有几种保存的方案可选:SQLAlchemy、Django ORM、Memcached、 Redis、RPC (RabbitMQ/AMQP)。

使用Redis作为存储结果的方案，任务结果存储配置我们通过Celery的backend参数来设定。我们将tasks模块修改如下:

```python
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
## 执行任务

任务加入到broker队列中，以便刚才我们创建的celery workder服务器能够从队列中取出任务并执行。如何将任务函数加入到队列中，可使用delay()。

默认多进程执行，也可以多协程处理(greenlet/gevent)

进入python终端, 执行如下代码:

```python
from tasks import my_task
# 执行,返回执行的对象
task_obj = my_task.delay()
# 获取任务编号
print(task_obj.id)
# 获取异步任务结果，默认阻塞
task_obj.get()
```

## django中使用

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

