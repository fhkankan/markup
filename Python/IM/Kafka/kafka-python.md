# kafka-python

[首页](https://github.com/dpkp/kafka-python)

[文档](https://kafka-python.readthedocs.io/en/master/usage.html#)

## 安装

```
pip install kafka-python
```

发送消息

```python
from kafka import KafkaProducer
producer = KafkaProducer(bootstrap_servers=["ip:9092"])
producer.send("test",b"Hello world")
```

## 生产者

- 官网

```python
from kafka import KafkaProducer
# 手动指定服务
producer = KafkaProducer(bootstrap_servers='localhost:1234')
for _ in range(100):
...     producer.send('foobar', b'some_message_bytes')
# Block until a single message is sent (or timeout)
future = producer.send('foobar', b'another_message')
result = future.get(timeout=60)
# Block until all pending messages are at least put on the network。NOTE: This does not guarantee delivery or success! It is really only useful if you configure internal batching using linger_ms
producer.flush()
# Use a key for hashed-partitioning
producer.send('foobar', key=b'foo', value=b'bar')
# Serialize json messages
import json
producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'))
producer.send('fizzbuzz', {'foo': 'bar'})
# Serialize string keys
producer = KafkaProducer(key_serializer=str.encode)
producer.send('flipflap', key='ping', value=b'1234')
# Compress messages
producer = KafkaProducer(compression_type='gzip')
for i in range(1000):
    producer.send('foobar', b'msg %d' % i)
# Include record headers. The format is list of es with string key and bytes value.
producer.send('foobar', value=b'c29tZSB2YWx1ZQ==', headers=[('content-encoding', b'base64')])
# Get producer performance metrics
metrics = producer.metrics()
```

- 示例

```python
'''
callback也是保证分区有序的, 比如2条消息, a先发送, b后发送, 对于同一个分区, 那么会先回调a的callback, 再回调b的callback
'''
 
import json
from kafka import KafkaProducer
 
topic = 'demo'
 
 
def on_send_success(record_metadata):
  	print(record_metadata.topic)
  	print(record_metadata.partition)
  	print(record_metadata.offset)
 
 
def on_send_error(excp):
  	print('I am an errback: {}'.format(excp))
 
 
def main():
  	producer = KafkaProducer(bootstrap_servers='localhost:9092')
  	producer.send(topic, value=b'{"test_msg":"hello world"}').add_callback(on_send_success).add_callback(on_send_error)
  	# close() 方法会阻塞等待之前所有的发送请求完成后再关闭 KafkaProducer
 	producer.close()
 
 
def main2():
  	'''
  	发送json格式消息
  	:return:
  	'''
  	producer = KafkaProducer(
  	  bootstrap_servers='localhost:9092',
  	  value_serializer=lambda m: json.dumps(m).encode('utf-8')
  	)
  	producer.send(topic, value={"test_msg": "hello world"}).add_callback(on_send_success).add_callback(
  	  on_send_error)
  	# close() 方法会阻塞等待之前所有的发送请求完成后再关闭 KafkaProducer
  	producer.close()
if __name__ == '__main__':
  	# main()
  	main2()
```

## 消费者

- 官方

```python
# 最简单
from kafka import KafkaConsumer
consumer = KafkaConsumer('my_favorite_topic')
for msg in consumer:
    print (msg)   
# 添加group_id
consumer = KafkaConsumer('my_favorite_topic', group_id='my_favorite_group')
for msg in consumer:
    print (msg)
# 手动指定服务
consumer = KafkaConsumer(bootstrap_servers='localhost:1234')
consumer.assign([TopicPartition('foobar', 2)])
msg = next(consumer)
# 序列化
consumer = KafkaConsumer(value_deserializer=msgpack.loads)
consumer.subscribe(['msgpackfoo'])
for msg in consumer:
    assert isinstance(msg.value, dict)
    
# 属性
for msg in consumer:
    print (msg.headers)
    
metrics = consumer.metrics()
```

- 不使用消费组(group_id=None)

不使用消费组的情况下可以启动很多个消费者, 不再受限于分区数, 即使消费者数量 > 分区数, 每个消费者也都可以收到消息

```python
from kafka import KafkaConsumer

topic = 'demo'

def main():
  	consumer = KafkaConsumer(
  	  topic,
  	  bootstrap_servers='localhost:9092',
  	  auto_offset_reset='latest',
  	  # auto_offset_reset='earliest',
  	)
  	for msg in consumer:
  	  	print(msg)
  	  	print(msg.value)
  	consumer.close()
if __name__ == '__main__':
 	main()
```

- 指定消费组

以下使用pool方法来拉取消息

pool 每次拉取只能拉取一个分区的消息, 比如有2个分区1个consumer, 那么会拉取2次

pool 是如果有消息马上进行拉取, 如果timeout_ms内没有新消息则返回空dict, 所以可能出现某次拉取了1条消息, 某次拉取了max_records条

```python
from kafka import KafkaConsumer

topic = 'demo'
group_id = 'test_id'


def main():
  	consumer = KafkaConsumer(
  	  topic,
  	  bootstrap_servers='localhost:9092',
  	  auto_offset_reset='latest',
  	  group_id=group_id,
  	)
  	while True:
  	  	try:
  	    	# return a dict
  	    	batch_msgs = consumer.poll(timeout_ms=1000, max_records=2)
  	    	if not batch_msgs:
  	      		continue
  	    	'''
  	    	{TopicPartition(topic='demo', partition=0): [ConsumerRecord(topic='demo', partition=0, offset=42, timestamp=1576425111411, timestamp_type=0, key=None, value=b'74', headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=2, serialized_header_size=-1)]}
   	   		'''
   	   		for tp, msgs in batch_msgs.items():
   	     		print('topic: {}, partition: {} receive length: '.format(tp.topic, tp.partition, len(msgs)))
   	     		for msg in msgs:
   	       		print(msg.value)
   	 	except KeyboardInterrupt:
   	   		break

  	consumer.close()


if __name__ == '__main__':
  	main()
```

- 多线程

```python
import threading

import os
import sys
from kafka import KafkaConsumer, TopicPartition, OffsetAndMetadata
from collections import OrderedDict

threads = []


class MyThread(threading.Thread):
    def __init__(self, thread_name, topic, partition):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.partition = partition
        self.topic = topic

    def run(self):
        print("Starting " + self.name)
        Consumer(self.thread_name, self.topic, self.partition)

    def stop(self):
        sys.exit()


def Consumer(thread_name, topic, partition):
    broker_list = 'ip1:9092,ip2:9092'

    '''
    fetch_min_bytes（int） 
    # 服务器为获取请求而返回的最小数据量，否则请等待
    fetch_max_wait_ms（int） 
    # 如果没有足够的数据立即满足fetch_min_bytes给出的要求，服务器在回应提取请求之前将阻塞的最大时间量（以毫秒为单位）
    fetch_max_bytes（int）
    # 服务器应为获取请求返回的最大数据量。这不是绝对最大值，如果获取的第一个非空分区中的第一条消息大于此值，则仍将返回消息以确保消费者可以取得进展。注意：使用者并行执行对多个代理的提取，因此内存使用将取决于包含该主题分区的代理的数量。支持的Kafka版本> = 0.10.1.0。默认值：52428800（50 MB）。
    enable_auto_commit（bool） 
    # 如果为True，则消费者的偏移量将在后台定期提交。默认值：True。
    max_poll_records（int） 
    # 单次调用中返回的最大记录数poll()。默认值：500
    max_poll_interval_ms（int） 
    # poll()使用使用者组管理时的调用之间的最大延迟 。这为消费者在获取更多记录之前可以闲置的时间量设置了上限。如果 poll()在此超时到期之前未调用，则认为使用者失败，并且该组将重新平衡以便将分区重新分配给另一个成员。默认300000
    '''
    consumer = KafkaConsumer(
        bootstrap_servers=broker_list,
        group_id="test000001",
        client_id=thread_name,
        enable_auto_commit=False,
        fetch_min_bytes=1024 * 1024,  # 1M
        # fetch_max_bytes=1024 * 1024 * 1024 * 10,
        fetch_max_wait_ms=60000,  # 30s
        request_timeout_ms=305000,
        # consumer_timeout_ms=1,
        # max_poll_records=5000,)
        
    # 设置topic partition
    tp = TopicPartition(topic, partition)
    # 分配该消费者的TopicPartition，也就是topic和partition，根据参数，每个线程消费者消费一个分区
    consumer.assign([tp])
    # 获取上次消费的最大偏移量
    offset = consumer.end_offsets([tp])[tp]
    print(thread_name, tp, offset)

    # 设置消费的偏移量
    consumer.seek(tp, offset)

    print
    u"程序首次运行\t线程:", thread_name, u"分区:", partition, u"偏移量:", offset, u"\t开始消费..."
    num = 0  # 记录该消费者消费次数
    while True:
        msg = consumer.poll(timeout_ms=60000)
        end_offset = consumer.end_offsets([tp])[tp]
        '''可以自己记录控制消费'''
        print
        u'已保存的偏移量', consumer.committed(tp), u'最新偏移量，', end_offset
        if len(msg) > 0:
            print
            u"线程:", thread_name, u"分区:", partition, u"最大偏移量:", end_offset, u"有无数据,", len(msg)
            lines = 0
            for data in msg.values():
                for line in data:
                    print
                    line
                    lines += 1
                '''
                do something
                '''
            # 线程此批次消息条数
            print(thread_name, "lines", lines)
            if True:
                # 可以自己保存在各topic, partition的偏移量
                # 手动提交偏移量 offsets格式：{TopicPartition:OffsetAndMetadata(offset_num,None)}
                consumer.commit(offsets={tp: (OffsetAndMetadata(end_offset, None))})
                if True == 0:
                    # 系统退出？这个还没试
                    os.exit()
                    '''
                    sys.exit()  只能退出该线程，也就是说其它两个线程正常运行，主程序不退出
                    '''
            else:
                os.exit()
        else:
            print
            thread_name, '没有数据'
        num += 1
        print
        thread_name, "第", num, "次"


if __name__ == '__main__':
    try:
        t1 = MyThread("Thread-0", "test", 0)
        threads.append(t1)
        t2 = MyThread("Thread-1", "test", 1)
        threads.append(t2)
        t3 = MyThread("Thread-2", "test", 2)
        threads.append(t3)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        print("exit program with 0")
    except:
        print("Error: failed to run consumer program")

```



- 关于消费组

我们根据配置参数分为以下几种情况

```
- group_id=None
    - auto_offset_reset='latest': 每次启动都会从最新出开始消费, 重启后会丢失重启过程中的数据
    - auto_offset_reset='latest': 每次从最新的开始消费, 不会管哪些任务还没有消费

- 指定group_id
    - 全新group_id
        - auto_offset_reset='latest': 只消费启动后的收到的数据, 重启后会从上次提交offset的地方开始消费
        - auto_offset_reset='earliest': 从最开始消费全量数据
    - 旧group_id(即kafka集群中还保留着该group_id的提交记录)
        - auto_offset_reset='latest': 从上次提交offset的地方开始消费
        - auto_offset_reset='earliest': 从上次提交offset的地方开始消费
```

## 性能测试

- 生产者

```python
'''
producer performance

environment:
  mac
  python3.7
  broker 1
  partition 2
'''

import json
import time
from kafka import KafkaProducer

topic = 'demo'
nums = 1000000


def main():
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda m: json.dumps(m).encode('utf-8')
    )
    st = time.time()
    cnt = 0
    for _ in range(nums):
        producer.send(topic, value=_)
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)

    producer.flush()

    et = time.time()
    cost_time = et - st
    print('send nums: {}, cost time: {}, rate: {}/s'.format(nums, cost_time, nums // cost_time))


if __name__ == '__main__':
    main()

'''
send nums: 1000000, cost time: 61.89236712455749, rate: 16157.0/s
send nums: 1000000, cost time: 61.29534196853638, rate: 16314.0/s
'''

```

- 消费者

```python
'''
consumer performance
'''

import time
from kafka import KafkaConsumer

topic = 'demo'
group_id = 'test_id'


def main1():
    nums = 0
    st = time.time()

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        group_id=group_id
    )
    for msg in consumer:
        nums += 1
        if nums >= 500000:
            break
    consumer.close()

    et = time.time()
    cost_time = et - st
    print('one_by_one: consume nums: {}, cost time: {}, rate: {}/s'.format(nums, cost_time, nums // cost_time))


def main2():
    nums = 0
    st = time.time()

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='latest',
        group_id=group_id
    )
    running = True
    batch_pool_nums = 1
    while running:
        batch_msgs = consumer.poll(timeout_ms=1000, max_records=batch_pool_nums)
        if not batch_msgs:
            continue
        for tp, msgs in batch_msgs.items():
            nums += len(msgs)
            if nums >= 500000:
                running = False
                break

    consumer.close()

    et = time.time()
    cost_time = et - st
    print('batch_pool: max_records: {} consume nums: {}, cost time: {}, rate: {}/s'.format(batch_pool_nums, nums,
                                                                                           cost_time,
                                                                                           nums // cost_time))


if __name__ == '__main__':
    # main1()
    main2()

'''
one_by_one: consume nums: 500000, cost time: 8.018627166748047, rate: 62354.0/s
one_by_one: consume nums: 500000, cost time: 7.698841094970703, rate: 64944.0/s


batch_pool: max_records: 1 consume nums: 500000, cost time: 17.975456953048706, rate: 27815.0/s
batch_pool: max_records: 1 consume nums: 500000, cost time: 16.711708784103394, rate: 29919.0/s

batch_pool: max_records: 500 consume nums: 500369, cost time: 6.654940843582153, rate: 75187.0/s
batch_pool: max_records: 500 consume nums: 500183, cost time: 6.854053258895874, rate: 72976.0/s

batch_pool: max_records: 1000 consume nums: 500485, cost time: 6.504687070846558, rate: 76942.0/s
batch_pool: max_records: 1000 consume nums: 500775, cost time: 7.047331809997559, rate: 71058.0/s
'''

```

## 实时传输

```python
import sys
import json
import pandas as pd
import os
from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError

KAFAKA_HOST = "xxx.xxx.x.xxx"  # 服务器端口地址
KAFAKA_PORT = 9092  # 端口号
KAFAKA_TOPIC = "topic0"  # topic

data = pd.read_csv(os.getcwd() + '\\data\\1.csv')
key_value = data.to_json()


class Kafka_producer():
    '''
    生产模块：根据不同的key，区分消息
    '''

    def __init__(self, kafkahost, kafkaport, kafkatopic, key):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.key = key
        self.producer = KafkaProducer(bootstrap_servers='{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort)
        )

    def sendjsondata(self, params):
        try:
            parmas_message = params  # 注意dumps
            producer = self.producer
            producer.send(self.kafkatopic, key=self.key, value=parmas_message.encode('utf-8'))
            producer.flush()
        except KafkaError as e:
            print(e)


class Kafka_consumer():

    def __init__(self, kafkahost, kafkaport, kafkatopic, groupid, key):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.groupid = groupid
        self.key = key
        self.consumer = KafkaConsumer(self.kafkatopic, group_id=self.groupid,
                                      bootstrap_servers='{kafka_host}:{kafka_port}'.format(
                                          kafka_host=self.kafkaHost,
                                          kafka_port=self.kafkaPort)
                                      )

    def consume_data(self):
        try:
            for message in self.consumer:
                yield message
        except KeyboardInterrupt as e:
            print(e)


def sortedDictValues(adict):
    items = adict.items()
    items = sorted(items, reverse=False)
    return [value for key, value in items]


def main(xtype, group, key):
    '''
    测试consumer和producer
    '''
    if xtype == "p":
        # 生产模块
        producer = Kafka_producer(KAFAKA_HOST, KAFAKA_PORT, KAFAKA_TOPIC, key)
        print("===========> producer:", producer)
        params = key_value
        producer.sendjsondata(params)

    if xtype == 'c':
        # 消费模块
        consumer = Kafka_consumer(KAFAKA_HOST, KAFAKA_PORT, KAFAKA_TOPIC, group, key)
        print("===========> consumer:", consumer)

        message = consumer.consume_data()
        for msg in message:
            msg = msg.value.decode('utf-8')
            python_data = json.loads(msg)  ##这是一个字典
            key_list = list(python_data)
            test_data = pd.DataFrame()
            for index in key_list:
                print(index)
                if index == 'Month':
                    a1 = python_data[index]
                    data1 = sortedDictValues(a1)
                    test_data[index] = data1
                else:
                    a2 = python_data[index]
                    data2 = sortedDictValues(a2)
                    test_data[index] = data2
                    print(test_data)

            # print('value---------------->', python_data)
            # print('msg---------------->', msg)
            # print('key---------------->', msg.kry)
            # print('offset---------------->', msg.offset)


if __name__ == '__main__':
    main(xtype='p', group='py_test', key=None)
    main(xtype='c', group='py_test', key=None)

```

