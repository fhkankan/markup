

# RabbitMQ

## 概述

MQ是一种应用程序对应用程序的通信方法。应用程序通过读出（写入）队列的消息（针对应用程序的数据）来通信，而无需使用专用连接来链接它们。消息传递指的是程序之间通过在消息中发送数据进行通信，而不是通过直接调用彼此来通信，排队指的是应用程序通过 队列来通信。队列的使用排除了接收和发送应用程序同时执行的要求。

RabbitMQ是流行的开源消息队列系统，用erlang语言开发。RabbitMQ是AMQP（高级消息队列协议）的标准实现。

RabbitMQ也是前面所提到的生产者消费者模型，一端发送消息（生产任务），一端接收消息（处理任务）。

rabbitmq的详细使用（包括各种系统的安装配置）可参见其[官方文档](http://www.rabbitmq.com/documentation.html)

作用

```
程序解耦、提升性能、降低多业务逻辑复杂度
```

## 安装配置

### 安装

原始安装

```

```

docker安装

```shell
# 拉取
docker pull rabbitmq:3.7.2-management

# 运行
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 -e RABBITMQ_DEFAULT_USER=root -e RABBITMQ_DEFAULT_PASS=Wb2wvLc2knHyEy42WPs4 docker.io/rabbitmq:3.7.2-management

# 检查
http://ip:15672

账号：root
密码：Wb2wvLc2knHyEy42WPs4
```

### 配置

### 使用

```shell
# 一步启动Erlang node和Rabbit应用
sudo rabbitmq-server
# 在后台启动Rabbit node
sudo rabbitmq-server -detached
# 关闭整个节点（包括应用）
sudo rabbitmqctl stop
```

## python交互

pika是python与Rabbitmq交互的客户端工具

```shell
# 安装
pip install pika
```

### producer/consumer

消息传递消费过程中，可以在rabbit web管理页面实时查看队列消息信息。

Producer

```python
# send.py
import pika
import time
auth=pika.PlainCredentials('ywq','qwe') # 保存作者信息
connection = pika.BlockingConnection(pika.ConnectionParameters(
  '192.168.0.158',5672,'/',auth)) # 连接至rabbitmq
channel = connection.channel() # 创建chanel
channel.queue_declare(queue='hello') # 定义hello队列
#n RabbitMQ a message can never be sent directly to the queue, it always needs to go through an exchange.
channel.basic_publish(exchange='',
   routing_key='hello',  # 告诉rabbitmq将消息发送到hello队列中
   body='Hello World!')  # 发送消息的内容
print(" [x] Sent 'Hello World!'")
connection.close()  # 关闭与rabbitmq的连接
```

consumer

```python
# recive.py
import pika
auth=pika.PlainCredentials('ywq','qwe') #auth info
connection = pika.BlockingConnection(pika.ConnectionParameters(
  '192.168.0.158',5672,'/',auth)) #connect to rabbit
channel = connection.channel()  #create channel

channel.queue_declare(queue='hello')  # 在接收端定义队列，参数与发送端的相同
def callback(ch, method, properties, body):
 		print(" [x] Received %r" % body)
    # time.sleep(30)

channel.basic_consume(callback,
   queue='hello',  # 告诉rabbitmq此程序从hello队列中接收消息
   no_ack=True)
print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()  # 开始接收，未收到消息阻塞
```

说明

```python
注1：我们可以打开time.sleep()的注释（模仿任务处理所需的时间），将no_ack设为默认值（不传参数），同时运行多个receive.py， 运行send.py发一次消息，第一个开始运行的receive.py接收到消息，开始处理任务，如果中途宕机（任务未处理完）；那么第二个开始运行的receive.py就会接收到消息，开始处理任务；如果第二个也宕机了，则第三个继续；如果依次所有运行的receive都宕机（任务未处理完）了，则下次开始运行的第一个receive.py将继续接收消息处理任务，这个机制防止了一些必须完成的任务由于处理任务的程序异常终止导致任务不能完成。如果将no_ack设为True，中途宕机，则后面的接收端不会再接收消息处理任务。

注2：如果发送端不停的发消息，则接收端分别是第一个开始运行的接收，第二个开始运行的接收，第三个开始运行接收，依次接收，这是rabbitmq的消息轮循机制（相当于负载均衡，防止一个接收端接收过多任务卡死，当然这种机制存在弊端，就是如果接收端机器有的配置高有的配置低，就会使配置高的机器得不到充分利用而配置低的机器一直在工作）。这一点可以启动多个receive.py，多次运行send.py验证。

上面的例子我们介绍了消息的接收端（即任务的处理端）宕机，我们该如何处理。接下来，我们将重点放在消息的发送端（即服务端），与接收端不同，如果发送端宕机，则会丢失存储消息的队列，存储的消息（要发送给接收端处理的任务），这些信息一旦丢失会造成巨大的损失，所以下面的重点就是消息的持久化，即发送端异常终止，重启服务后，队列消息都将自动加载进服务里。其实只要将上面的代码稍微修改就可实现。
```

### 消息持久化

consumer端无需改变，在producer端代码内加上两个属性，分别使消息持久化、队列持久化，只选其一还是会出现消息丢失，必须同时开启：

```python
delivery_mode=2  # 使消息持久化
durable=True  # 使队列持久化
```

sending

```python
# new_task.py
import pika
connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue='task_queue', durable=True)  # 使队列持久化
message = "Hello World"
channel.basic_publish(exchange='',
      routing_key='task_queue',
      body=message,
      properties=pika.BasicProperties(
       delivery_mode=2,  # 使消息持久化
      ))
print(" [x] Sent %r" % message)
connection.close()
```

Reciving

```python
# worker.py
import pika
import time

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()

channel.queue_declare(queue='task_queue', durable=True) #再次申明队列，和发送端参数应一样
print(' [*] Waiting for messages. To exit press CTRL+C')

def callback(ch, method, properties, body):
 print(" [x] received %r" % body)
 time.sleep(2)
 print(" [x] Done")
 # 因为没有设置no_ask=True, 所以需要告诉rabbitmq消息已经处理完毕，rabbitmq将消息移出队列。
 ch.basic_ack(delivery_tag=method.delivery_tag)

#同一时间worker只接收一条消息，等这条消息处理完在接收下一条
channel.basic_qos(prefetch_count=1)
channel.basic_consume(callback,
      queue='task_queue')


channel.start_consuming()
```

说明

```python
注1：worker.py中的代码如果不设置，则new_task.py意外终止在重启后，worker会同时接收终止前没有处理的所有消息。两个程序中的queue设置的参数要相同，否则程序出错。no_ask=True如果没设置，则worker.py中的ch.basic_ack(delivery_tag=method.delivery_tag)这行代码至关重要，如果不写，则不管接收的消息有没有处理完，此消息将一直存在与队列中。

注2:这句代码---channel.basic_qos(prefetch_count=1)，解决了上例中消息轮循机制的代码，即接收端（任务的处理端）每次只接收一个任务（参数为几接收几个任务），处理完成后通过向发送端的汇报（即注1中的代码）来接收下一个任务，如果有任务正在处理中它不再接收新的任务。

前面所介绍的例一，例二都是一条消息，只能被一个接收端收到。那么该如何实现一条消息多个接收端同时收到（即消息群发或着叫广播模式）呢？

其实，在rabbitmq中只有consumer（消费者，即接收端）与queue绑定，对于producer（生产者，即发送端）只是将消息发送到特定的队列。consumer从与自己相关的queue中读取消息而已。所以要实现消息群发，只需要将同一条放到多个消费者队列即可。在rabbitmq中这个工作由exchange来做，它可以设定三种类型，它们分别实现了不同的需求，我们分别来介绍。
```

### 公平分发

在多consumer的情况下，默认rabbit是轮询发送消息的，但有的consumer消费速度快，有的消费速度慢，为了资源使用更平衡，引入ack确认机制。consumer消费完消息后会给rabbit发送ack，一旦未ack的消息数量超过指定允许的数量，则不再往该consumer发送，改为发送给其他consumer。

producer端代码不用改变，需要给consumer端代码插入两个属性

```python
channel.basic_qos(prefetch_count= *) #define the max non_ack_count
channel.basic_ack(delivery_tag=deliver.delivery_tag) #send ack to rabbitmq
```

consumer

```python
import pika,time
auth_info=pika.PlainCredentials('ywq','qwe')
connection=pika.BlockingConnection(
	pika.ConnectionParameters(
		'192.168.0.158',5672,'/',auth_info
	)
)
channel=connection.channel()
channel.queue_declare(queue='test2',durable=True)
def callback(chann,deliver,properties,body):
  	'''
注意，no_ack=False 注意，这里的no_ack类型仅仅是告诉rabbit该消费者队列是否返回ack，若要返回ack，需要在callback内定义
prefetch_count=1,未ack的msg数量超过1个，则此consumer不再接受msg，此配置需写在channel.basic_consume上方，否则会造成non_ack情况出现。
		'''
 		print('Recv:',body)
 		time.sleep(5)
 		chann.basic_ack(delivery_tag=deliver.delivery_tag) #send ack to rabbit
channel.basic_qos(prefetch_count=1)

channel.basic_consume(
 callback,
 queue='test2'
)

channel.start_consuming()
```

### 发布/订阅

上方的几种模式都是producer端发送一次，则consumer端接收一次，能不能实现一个producer发送，多个关联的consumer同时接收呢？of course，rabbit支持消息发布订阅，共支持三种模式，通过组件exchange转发器，实现3种模式

```
fanout: 所有bind到此exchange的queue都可以接收消息，类似广播。

direct: 通过routingKey和exchange决定的哪个唯一的queue可以接收消息，推送给绑定了该queue的consumer，类似组播。

topic:所有符合routingKey(此时可以是一个表达式)的routingKey所bind的queue可以接收消息，类似前缀列表匹配路由。
```

使用exchange模式时

```
1.producer端不再申明queue，直接申明exchange

2.consumer端仍需绑定队列并指定exchange来接收message

3.consumer最好创建随机queue，使用完后立即释放。
```

#### fanout

当exchange的类型为fanout时，所有绑定这个exchange的队列都会收到发来的消息。

producer

```python
# emit.py
import pika,sys,time
auth_info=pika.PlainCredentials('ywq','qwe')
connection=pika.BlockingConnection(
  pika.ConnectionParameters(
		'192.168.0.158',5672,'/',auth_info
	)
)
channel=connection.channel()
# 申明一个exchange,两个参数分别为exchange的名字和类型；当exchang='fanout'时，所有绑定到此exchange的消费者队列都将收到消息
channel.exchange_declare(exchange='hello',
    exchange_type='fanout'
    )
# 消息可以在命令行启动脚本时以参数的形式传入
# msg=''.join(sys.argv[1:]) or 'Hello world %s' %time.time()
msg = 'Hello World!'
channel.basic_publish(
 exchange='hello',
 routing_key='',
 body=msg,
 properties=pika.BasicProperties(
 delivery_mode=2
 )
)
print('send done')
connection.close()
```

consumer

```python
# receive.py
import pika
auth_info=pika.PlainCredentials('ywq','qwe')
connection=pika.BlockingConnection(
  pika.ConnectionParameters(
 		'192.168.0.158',5672,'/',auth_info
	)
)
channel=connection.channel()
channel.exchange_declare(
 exchange='hello',
 exchange_type='fanout'
)
# 随机与rabbit建立一个queue，comsumer断开后，该queue立即删除释放
random_num=channel.queue_declare(exclusive=True)
# 得到随机生成消费者队列的名字
queue_name=random_num.method.queue
channel.basic_qos(prefetch_count=1)
# 将消费者队列与exchange绑定
channel.queue_bind(
 queue=queue_name,
 exchange='hello'
)
def callback(chann,deliver,properties,body):
 print('Recv:',body)
 chann.basic_ack(delivery_tag=deliver.delivery_tag) #send ack to rabbit

channel.basic_consume(
 callback,
 queue=queue_name,
)
channel.start_consuming()
```

说明

```
注1：emit.py为消息的发送端，receive.py为消息的接收端。可以同时运行多个receive.py，当emit.py发送消息时，可以发现所有正在运行的receive.py都会收到来自发送端的消息。

注2：类似与广播，如果消息发送时，接收端没有运行，那么它将不会收到此条消息，即消息的广播是即时的。
```

#### direct

当exchange的类型为direct时，发送端和接收端都要指明消息的级别，接收端只能接收到被指明级别的消息。

producer

```python
import pika,sys
auth_info=pika.PlainCredentials('ywq','qwe')
connection=pika.BlockingConnection(
  pika.ConnectionParameters(
 			'192.168.0.158',5672,'/',auth_info
 )
)
channel=connection.channel()
channel.exchange_declare(
  	exchange='direct_log',
		exchange_type='direct',
 )

# 命令行启动时，以参数的的形式传入发送消息的级别，未传默认设置info
# route_key = sys.argv[1] if len(sys.argv) > 2 else 'info'
# 命令行启动时，以参数的的形式传入发送消息的内容，未传默认设置Hello World!
# message = ' '.join(sys.argv[2:]) or 'Hello World!'
route_key = 'info' #作为例子直接将消息的级别设置为info,也可为'warning'
message = 'Hello World'
channel.basic_publish(
		exchange='direct_log',
		routing_key=route_key,
		body=message,
		properties=pika.BasicProperties(
 				delivery_mode=2
		)
)
connection.close()
```

consumer

```python
import pika,sys
auth_info=pika.PlainCredentials('ywq','qwe')
connection=pika.BlockingConnection(
  pika.ConnectionParameters(
 		'192.168.0.158',5672,'/',auth_info
	)
)
channel=connection.channel()
channel.exchange_declare(
		exchange='direct_log',
		exchange_type='direct'
)
queue_num=channel.queue_declare(exclusive=True)
queue_name=queue_num.method.queue

# 命令行启动时以参数的形式传入要接收哪个级别的消息，可以传入多个级别
# severities = sys.argv[1:]
# 演示使用，实际运用应该用上面的方式指明消息级别
# 作为演示，直接设置两个接收级别，info 和 warning
severities = ['info', 'warning']
if not severities:
 		"""如果要接收消息的级别不存在则提示用户输入级别并退出程序"""
		sys.stderr.write("Usage: %s [info] [warning] [error]\n" % sys.argv[0])
		sys.exit(1)
for route_key in severities:
  	"""依次为每个消息级别绑定queue"""
		channel.queue_bind(
 				queue=queue_name,
 				exchange='direct_log',
 				routing_key=route_key
		)
    
def callback(chann,deliver,property,body):
		print('Recv:[level:%s],[msg:%s]' %(route_key,body))
 		chann.basic_ack(delivery_tag=deliver.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(
 		callback,
 		queue=queue_name
)
channel.start_consuming()
```

说明

```python
注1：exchange_type=direct时，rabbitmq按消息级别发送和接收消息，接收端只能接收被指明级别的消息，其他消息，即时是由同一个发送端发送的也无法接收。当在接收端传入多个消息级别时，应逐个绑定消息队列。

注2：exchange_type=direct时，同样是广播模式，也就是如果给多个接收端指定相同的消息级别，它们都可以同时收到这一级别的消息。
```

#### topic

当exchange的类型为topic时，在发送消息时，应指明消息消息的类型（比如mysql.log、qq.info等），我们可以在接收端指定接收消息类型的关键字（即按关键字接收，在类型为topic时，这个关键字可以是一个表达式）。

rabbitmq通配符规则：

符号“#”匹配一个或多个词，符号“”匹配一个词。因此“abc.#”能够匹配到“abc.m.n”，但是“abc.*‘' 只会匹配到“abc.m”。‘.'号为分割符。使用通配符匹配时必须使用‘.'号分割。

producer

```python
import pika,sys
auth_info=pika.PlainCredentials('ywq','qwe')
connection=pika.BlockingConnection(
		pika.ConnectionParameters(
 				'192.168.0.158',5672,'/',auth_info
 		)
)
channel=connection.channel()
channel.exchange_declare(
  	exchange='topic_log',
   	exchange_type='topic'
)
# 以命令行的方式启动发送端，以参数的形式传入发送消息类型的关键字
route_key = sys.argv[1] if len(sys.argv[1]) > 2 else 'anonymous.info' 
# route_key = 'anonymous.info'
# route_key = 'abc.orange.abc'
# route_key = 'abc.abc.rabbit'
# route_key = 'lazy.info'
msg=''.join(sys.argv[2:]) or 'Hello'
channel.basic_publish(
		exchange='topic_log',
		routing_key=route_key,
		body=msg,
		properties=pika.BasicProperties(
		 delivery_mode=2
		)
)
print(" [x] Sent %r:%r" % (route_key, message))
connection.close()
```

consumer

```python
import pika,sys
auth_info=pika.PlainCredentials('ywq','qwe')
connection=pika.BlockingConnection(
  	pika.ConnectionParameters(
 			'192.168.0.158',5672,'/',auth_info
		)
)
channel=connection.channel()
channel.exchange_declare(
		exchange='topic_log',
 		exchange_type='topic'
)
queue_num=channel.queue_declare(exclusive=True)
queue_name=queue_num.method.queue

route_key = sys.argv[1:]
# route_key = '#'  #接收所有的消息
# route_key = ['*.info']  #接收所有以".info"结尾的消息
# route_key = ['*.orange.*'] #接收所有含有".orange."的消息
# route_key = ['*.*.rabbit', 'lazy.*'] #接收所有含有两个扩展名且结尾是".rabbit"和所有以"lazy."开头的消息
if not binding_keys:
 		sys.stderr.write("Usage: %s [binding_key]...\n" % sys.argv[0])
 		sys.exit(1)
for binding_key in binding_keys:
		channel.queue_bind(
 				queue=queue_name,
 				exchange='topic_log',
 				routing_key=route_key
		)
print(' [*] Waiting for logs. To exit press CTRL+C')
def callback(chann,deliver,property,body):
 		print('Recv:[type:%s],[msg:%s]' %(route_key,body))
 		chann.basic_ack(delivery_tag=deliver.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(
 		callback,
 		queue=queue_name
)
channel.start_consuming()
```

说明

```
注：当exchange的类型为topic时，发送端与接收端的代码都跟类型为direct时很像（基本只是变一个类型，如果接收消息类型的指定不用表达式，它们几乎一样），但是topic的应用场景更广。
```

### RPC

实现

```
如图我们可以看出生产端client向消费端server请求处理数据，他会经历如下几次来完成交互。
1.生产端 生成rpc_queue队列，这个队列负责帮消费者 接收数据并把消息发给消费端。
2.生产端 生成另外一个随机队列，这个队列是发给消费端，消费这个用这个队列把处理好的数据发送给生产端。
3.生产端 生成一组唯一字符串UUID，发送给消费者，消费者会把这串字符作为验证在发给生产者。
4.当消费端处理完数据，发给生产端，时会把处理数据与UUID一起通过随机生产的队列发回给生产端。
5.生产端，会使用while循环 不断检测是否有数据，并以这种形式来实现阻塞等待数据，来监听消费端。
6.生产端获取数据调用回调函数，回调函数判断本机的UUID与消费端发回UID是否匹配，由于消费端，可能有多个，且处理时间不等所以需要判断，判断成功赋值数据，while循环就会捕获到，完成交互。
```

Client生产端

```python
# rpc_client.py
import pika
import uuid
import time

# 斐波那契数列 前两个数相加依次排列
class FibonacciRpcClient(object):
    def __init__(self):
        # 链接远程
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(
                host='localhost'))
        self.channel = self.connection.channel()

        # 生成随机queue
        result = self.channel.queue_declare(exclusive=True)
        # 随机取queue名字，发给消费端
        self.callback_queue = result.method.queue

        # self.on_response 回调函数:只要收到消息就调用这个函数。
        # 声明收到消息后就 收queue=self.callback_queue内的消息
        self.channel.basic_consume(self.on_response, no_ack=True,
                                   queue=self.callback_queue)

    # 收到消息就调用
    # ch 管道内存对象地址
    # method 消息发给哪个queue
    # body数据对象
    def on_response(self, ch, method, props, body):
        # 判断本机生成的ID 与 生产端发过来的ID是否相等
        if self.corr_id == props.correlation_id:
            # 将body值 赋值给self.response
            self.response = body

    def call(self, n):
        # 赋值变量，一个循环值
        self.response = None
        #　随机一次唯一的字符串
        self.corr_id = str(uuid.uuid4())
        # routing_key='rpc_queue' 发一个消息到rpc_queue内
        self.channel.basic_publish(
          	exchange='',
          	routing_key='rpc_queue',
            properties=pika.BasicProperties(
            		# 执行命令之后结果返回给self.callaback_queue这个队列中
            		reply_to = self.callback_queue,
            		# 生成UUID 发送给消费端
            		correlation_id = self.corr_id,
            ),
            # 发的消息，必须传入字符串，不能传数字
            body=str(n)
        )
        # 没有数据就循环收
        while self.response is None:
            # 非阻塞版的start_consuming()
            # 没有消息不阻塞
            self.connection.process_data_events()
            print("no msg...")
            time.sleep(0.5)
        return int(self.response)

#　实例化
fibonacci_rpc = FibonacciRpcClient()

print(" [x] Requesting fib(30)")
response = fibonacci_rpc.call(6)
print(" [.] Got %r" % response)
```

server消费端

```python
# rpc_server.py
import pika
import time
# 链接socket
connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
channel = connection.channel()

# 生成rpc queue
channel.queue_declare(queue='rpc_queue')

#　斐波那契数列
def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)


# 收到消息就调用
# ch 管道内存对象地址
# method 消息发给哪个queue
# props 返回给消费的返回参数
# body数据对象
def on_request(ch, method, props, body):
    n = int(body)

    print(" [.] fib(%s)" % n)
    # 调用斐波那契函数 传入结果
    response = fib(n)

    ch.basic_publish(exchange='',
                     # 生产端随机生成的queue
                     routing_key=props.reply_to,
                     # 获取UUID唯一 字符串数值
                     properties=pika.BasicProperties(correlation_id = \
                                                   props.correlation_id),
                     # 消息返回给生产端
                     body=str(response))
    # 确保任务完成,未申明no_ack = True, 消息处理完毕需向rabbitmq确认
    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_qos(prefetch_count=1) # 每次只处理一条消息
# rpc_queue收到消息:调用on_request回调函数
# queue='rpc_queue'从rpc内收
channel.basic_consume(on_request, queue='rpc_queue')

print(" [x] Awaiting RPC requests")
channel.start_consuming()  # 开始接收消息，未收到消息处于阻塞状态
```

说明

```
注1：测试时，先运行rpc_server.py，再运行rpc_client.py。

注2：客户端之所以每隔一秒检测一次服务端有没有返回结果，是因为客户端接收时时无阻塞的，在这一端时间内（不一定是1秒，但执行的任务消耗的时间不要太长）客户端可以执行其他任务提高效率。

注3：为什么客户端和服务端不使用一个队列来传递消息？ 答：如果使用一个队列，以客户端为例，它一边在检测这个队列中有没有它要接收的消息，一边又往这个队列里发送消息，会形成死循环。
```

