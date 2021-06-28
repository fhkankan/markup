# aiokafka

[官网](https://aiokafka.readthedocs.io/en/stable/)

## 概述

aiokafka 是使用 asyncio 的 Apache Kafka 分布式流处理系统的客户端。它基于 kafka-python 库，并重用其内部结构进行协议解析、错误等。客户端的功能与官方 Java 客户端非常相似，并带有一些 Pythonic 接口。aiokafka 可与 0.9 Kafka brokers 和支持完全协调的消费者组——即动态分区分配给同一组中的多个消费者。

安装

```shell
pip install aiokafka
```

生产者

```python
from aiokafka import AIOKafkaProducer
import asyncio

async def send_one():
    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092')
    # Get cluster layout and initial topic/partition leadership information
    await producer.start()
    try:
        # Produce message
        await producer.send_and_wait("my_topic", b"Super message")
    finally:
        # Wait for all pending messages to be delivered or expire.
        await producer.stop()

asyncio.run(send_one())
```

消费者

```python
from aiokafka import AIOKafkaConsumer
import asyncio

async def consume():
    consumer = AIOKafkaConsumer(
        'my_topic', 'my_other_topic',
        bootstrap_servers='localhost:9092',
        group_id="my-group")
    # Get cluster layout and join group `my-group`
    await consumer.start()
    try:
        # Consume messages
        async for msg in consumer:
            print("consumed: ", msg.topic, msg.partition, msg.offset,
                  msg.key, msg.value, msg.timestamp)
    finally:
        # Will leave consumer group; perform autocommit if enabled.
        await consumer.stop()

asyncio.run(consume())
```

## 生产者

`AIOKafkaProducer` 是将记录发布到 Kafka 集群的客户端。

```python
producer = aiokafka.AIOKafkaProducer(bootstrap_servers='localhost:9092')
await producer.start()
try:
    await producer.send_and_wait("my_topic", b"Super message")
finally:
    await producer.stop()
```

在底层，Producer 在消息传递方面做了很多工作，包括批处理、重试等。所有这些都可以配置，如下所示。

### 信息缓冲

虽然用户希望上面的示例将消息”直接发送给代理，但实际上并没有立即发送，而是添加到缓冲区空间。然后后台任务将获取成批的消息并将它们发送到集群中的适当节点。这种批处理方案允许更高的吞吐量和更有效的压缩。为了更清楚地看到它，让我们避免使用 send_and_wait 快捷方式：

```python
# Will add the message to 1st partition's batch. If this method times out,
# we can say for sure that message will never be sent.

fut = await producer.send("my_topic", b"Super message", partition=1)

# Message will either be delivered or an unrecoverable error will occur.
# Cancelling this future will not cancel the send.
msg = await fut
```

批处理本身是为每个分区创建的，最大大小为`max_batch_size`。批处理中的消息严格按附加顺序发送，每个分区一次仅发送 1 个批处理（aiokafka 不支持 Java 客户端中存在的 `max.inflight.requests.per.connection` 选项）。这对分区中的消息顺序进行了严格的保证。

 默认情况下，新的批次会在前一个批次之后立即发送（即使它未满）。如果你想减少请求的数量，你可以将 `linger_ms `设置为 0 以外的值。如果它尚未满，这将在发送下一批之前增加额外的延迟。

### 重试和消息确认

aiokafka 会自动重试大多数错误，但只会重试到`request_timeout_ms`。如果请求已过期，则会向应用程序引发最后一个错误。错误后在应用程序级别重试消息可能会导致重复，因此由用户决定。

例如，如果引发 `RequestTimedOutError`，Producer 无法确定 Broker 是否编写了请求。 

`acks` 选项控制当生产请求被认为被确认时。

最持久的设置是 `acks="all"`。 Broker 将等待所有可用副本写入请求，然后再回复 Producer。 Broker 将参考它的 `min.insync.replicas` 设置以了解要写入的最小副本数量。如果同步副本不足，则将引发 `NotEnoughReplicasError` 或 `NotEnoughReplicasAfterAppendError`。在这些情况下，用户应该怎么做，因为错误是不可重试的。

默认是 `ack=1` 设置。它不会等待副本写入，只等待 Leader 写入 request。

最不安全的是 `ack=0` 当 Broker 没有确认时，这意味着客户端永远不会重试，因为它永远不会看到任何错误。 

### 幂等生产

从 Kafka 0.11 开始，代理支持幂等生产，这将防止生产者在重试时创建重复项。 aiokafka 通过将参数 `enable_idempotence=True` 传递给 AIOKafkaProducer 来支持这种模式：

```python
producer = aiokafka.AIOKafkaProducer(
    bootstrap_servers='localhost:9092',
    enable_idempotence=True)
await producer.start()
try:
    await producer.send_and_wait("my_topic", b"Super message")
finally:
    await producer.stop()
```

选项会稍微改变消息传递的逻辑：

- 上面提到的 `ack="all"` 将被强制执行。如果使用 `enable_idempotence=True` 显式传递任何其他值，则会引发 ValueError。
- 与一般模式相比，不会引发 `RequestTimedOutError` 错误，并且不会在通过 `request_timeout_ms` 后使批量交付过期。

### 事务生产

从 Kafka 0.11 开始，Brokers 支持事务性消息生产者，这意味着发送到一个或多个主题的消息仅在事务提交后对消费者可见。要使用事务性生产者和伴随的 API，您必须设置 `transactional_id` 配置属性：

```python
producer = aiokafka.AIOKafkaProducer(
    bootstrap_servers='localhost:9092',
    transactional_id="transactional_test")
await producer.start()
try:
    async with producer.transaction():
        res = await producer.send_and_wait(
            "test-topic", b"Super transactional message")
finally:
    await producer.stop()
```

如果设置了 `transactional_id`，则幂等性与幂等性所依赖的生产者配置一起自动启用。此外，事务中包含的主题应配置为持久性。尤其是`replication.factor`至少应该是3，这些topic的`min.insync.replicas`应该被设置为2。 最后，为了实现端到端的事务保证，消费者必须是配置为仅读取已提交的消息。请参阅[Reading Transactional Messages](https://aiokafka.readthedocs.io/en/stable/consumer.html#transactional-consume)。 

`transactional_id` 的目的是启用跨单个生产者实例的多个会话的事务恢复。它通常来自分区的、有状态的应用程序中的分片标识符。因此，它对于在分区应用程序中运行的每个生产者实例应该是唯一的。使用相同的 `transactional_id` 将导致前一个实例引发不可重试的异常 `ProducerFenced `并强制其退出。 

`transaction() `快捷方式生产者还支持一组类似于 Java Client 中的 API。请参阅[AIOKafkaProducer](https://aiokafka.readthedocs.io/en/stable/api.html#aiokafka-producer)  API 文档。 

除了能够以原子方式提交多个主题之外，由于偏移量也存储在单独的系统主题中，因此可以将消费者偏移量作为同一事务的一部分提交：

```python
async with producer.transaction():
    commit_offsets = {
        TopicPartition("some-topic", 0): 100
    }
    await producer.send_offsets_to_transaction(
        commit_offsets, "some-consumer-group")
```

更多例子参见[Transactional Consume-Process-Produce](https://aiokafka.readthedocs.io/en/stable/examples/transaction_example.html#transaction-example).

## 消费者

`AIOKafkaConsumer` 是一个从 Kafka 集群消费记录的客户端。

```python
consumer = aiokafka.AIOKafkaConsumer(
    "my_topic",
    bootstrap_servers='localhost:9092'
)
await consumer.start()
try:
    async for msg in consumer:
        print(
            "{}:{:d}:{:d}: key={} value={} timestamp_ms={}".format(
                msg.topic, msg.partition, msg.offset, msg.key, msg.value,
                msg.timestamp)
        )
finally:
    await consumer.stop()
    
```

`msg.value` 和 `msg.key` 是原始字节，如果需要解码它们，请使用 `key_deserializer` 和 `value_deserializer `配置。

消费者维护 TCP 连接以及一些后台任务来获取数据和协调分配。在消费者使用后未能调用 `Consumer.stop()` 将使后台任务继续运行。

消费者透明地处理 Kafka 代理的故障，并透明地适应它获取的主题分区在集群内迁移。它还与代理交互以允许消费者组使用消费者组对消费进行负载平衡。

### 偏移量和位置

Kafka 为分区中的每条记录维护一个数字偏移量。此偏移量充当该分区内记录的唯一标识符，还表示消费者在分区中的位置。

```python
msg = await consumer.getone()
print(msg.offset)  # Unique msg autoincrement ID in this topic-partition.

tp = aiokafka.TopicPartition(msg.topic, msg.partition)

position = await consumer.position(tp)
# Position is the next fetched offset
assert position == msg.offset + 1

committed = await consumer.committed(tp)
print(committed)
```

在这里，如果消费者在位置 5，它已经消费了偏移量 0 到 4 的记录，接下来将接收偏移量 5 的记录。

实际上有两个位置概念： 

- 位置给出了应该给出的下一条记录的偏移量。它将比消费者在该分区中看到的最高偏移大一。每次消费者在 `getmany() `或 `getone()` 调用中产生消息时，它都会自动增加。
- 提交位置是已安全存储的最后一个偏移量。如果进程重新启动，这是消费者将从中开始的偏移量。消费者可以定期自动提交偏移量，也可以选择通过调用` await consumer.commit() `手动控制此提交位置。

这种区别使消费者可以控制何时认为记录已被消费。

#### 手动/自动提交

对于大多数应用，自动提交是较好选择

```python
consumer = AIOKafkaConsumer(
    "my_topic",
    bootstrap_servers='localhost:9092',
    group_id="my_group",           # Consumer must be in a group to commit
    enable_auto_commit=True,       # Is True by default anyway
    auto_commit_interval_ms=1000,  # Autocommit every second
    auto_offset_reset="earliest",  # If committed offset not found, start from beginning                 
)
await consumer.start()

async for msg in consumer:  # Will periodically commit returned messages.
    # process message
    pass
```

这个例子可以有“至少一次”传递语义，但前提是我们一次处理一条消息。如果您想要批处理操作的“至少一次”语义，您应该使用手动提交

```python
consumer = AIOKafkaConsumer(
    "my_topic",
    bootstrap_servers='localhost:9092',
    group_id="my_group",           # Consumer must be in a group to commit
    enable_auto_commit=False,      # Will disable autocommit
    auto_offset_reset="earliest",  # If committed offset not found, start from beginning                                
)
await consumer.start()

batch = []
async for msg in consumer:
    batch.append(msg)
    if len(batch) == 100:
        await process_msg_batch(batch)
        await consumer.commit()
        batch = []
```

> 警告
>
> 使用手动提交时，建议提供一个 ConsumerRebalanceListener，它将处理批处理中的待处理消息，并在允许重新加入之前提交。如果您的组在处理期间重新平衡，提交将失败并出现 CommitFailedError，因为分区可能已经被其他消费者处理了。

此示例将保留消息，直到我们有足够的消息进行批量处理。该算法可以通过利用以下优势来增强：

- `await consumer.getmany() `以避免多次调用以获取一批消息。
- `await consumer.highwater(partition)` 以了解我们是否有更多未使用的消息或这是最后一个一个在分区中。

如果你想对提交的分区和消息有更多的控制，你可以手动指定偏移量：

```python
while True:
    result = await consumer.getmany(timeout_ms=10 * 1000)
    for tp, messages in result.items():
        if messages:
            await process_msg_batch(messages)
            # Commit progress only for this partition
            await consumer.commit({tp: messages[-1].offset + 1})
```

> 注意
>
> 提交的偏移量应始终是应用程序将读取的下一条消息的偏移量。因此，当调用 `commit(offsets)` 时，您应该在处理的最后一条消息的偏移量上加一个。

在这里，我们为每个分区处理一批消息，并没有提交所有消耗的偏移量，而是仅针对我们处理的分区。

#### 控制消费者的位置

在大多数用例中，消费者将简单地从头到尾消费记录，定期提交其位置（自动或手动）。

如果您只希望您的使用者处理最新消息，您可以要求它从最新偏移量开始

```python
consumer = AIOKafkaConsumer(
    "my_topic",
    bootstrap_servers='localhost:9092',
    auto_offset_reset="latest",
)
await consumer.start()

async for msg in consumer:
    # process message
    pass
```

> 注意
>
> 如果您有一个有效的提交位置，消费者将使用它。 auto_offset_reset 仅在位置无效时使用。

Kafka 还允许消费者手动控制其位置，使用 `consumer.seek() `在分区中随意向前或向后移动。例如，您可以重新使用记录：

```python
msg = await consumer.getone()
tp = TopicPartition(msg.topic, msg.partition)

consumer.seek(tp, msg.offset)
msg2 = await consumer.getone()

assert msg2 == msg
```

您也可以将其与 `offset_for_times` API 结合使用，以根据时间戳查询特定的偏移量。

#### 在kafka外存储偏移量

在 Kafka 中存储偏移是可选的，您可以将偏移存储在另一个地方并使用`consumer.seek()` API 从保存的位置开始。其主要用例是允许应用程序以原子方式存储结果和偏移量的方式在同一系统中存储偏移量和消耗结果。例如，如果我们在 Redis 中按键计数聚合保存：

```python
import json
from collections import Counter

redis = await aioredis.create_redis(("localhost", 6379))
REDIS_HASH_KEY = "aggregated_count:my_topic:0"

tp = TopicPartition("my_topic", 0)
consumer = AIOKafkaConsumer(
    bootstrap_servers='localhost:9092',
    enable_auto_commit=False,
)
await consumer.start()
consumer.assign([tp])

# Load initial state of aggregation and last processed offset
offset = -1
counts = Counter()
initial_counts = await redis.hgetall(REDIS_HASH_KEY, encoding="utf-8")
for key, state in initial_counts.items():
    state = json.loads(state)
    offset = max([offset, state['offset']])
    counts[key] = state['count']

# Same as with manual commit, you need to fetch next message, so +1
consumer.seek(tp, offset + 1)

async for msg in consumer:
    key = msg.key.decode("utf-8")
    counts[key] += 1
    value = json.dumps({
        "count": counts[key],
        "offset": msg.offset
    })
    await redis.hset(REDIS_HASH_KEY, key, value)
```

因此，要在 Kafka 之外保存结果，您需要：

- 配置 `enable.auto.commit=false`
- 使用每个 ConsumerRecord 提供的偏移量来保存您的位置
- 在重新启动或重新平衡时使用` consumer.seek()` 恢复消费者的位置

这个并不总是可能的，但当它是时，它将使消费完全原子化，并提供比默认的“至少一次”语义更强的“恰好一次”语义，您使用 Kafka 的偏移提交功能获得的语义。

这种类型的用法是最简单的当分区分配也是手动完成的（就像我们上面做的那样）。如果分区分配是自动完成的，则需要特别注意处理分区分配更改的情况。有关更多详细信息，请参阅 [Local state and storing offsets outside of Kafka](https://aiokafka.readthedocs.io/en/stable/examples/local_state_consumer.html#local-state-consumer-example) 。

### 消费组和主题订阅

Kafka 使用消费者组的概念来允许进程池来划分消费和处理记录的工作。这些进程可以在同一台机器上运行，也可以分布在多台机器上，为处理提供可扩展性和容错能力。

共享相同 `group_id` 的所有 `Consumer` 实例将属于同一个 `Consumer Group`

```python
# Process 1
consumer = AIOKafkaConsumer(
    "my_topic", bootstrap_servers='localhost:9092',
    group_id="MyGreatConsumerGroup"  # This will enable Consumer Groups
)
await consumer.start()
async for msg in consumer:
    print("Process %s consumed msg from partition %s" % (
          os.getpid(), msg.partition))

# Process 2
consumer2 = AIOKafkaConsumer(
    "my_topic", bootstrap_servers='localhost:9092',
    group_id="MyGreatConsumerGroup"  # This will enable Consumer Groups
)
await consumer2.start()
async for msg in consumer2:
    print("Process %s consumed msg from partition %s" % (
          os.getpid(), msg.partition))
```

组中的每个消费者都可以通过 `consumer.subscribe(...)` 调用动态设置它想要订阅的主题列表。 Kafka 会将订阅的主题中的每条消息仅传递给每个消费者组中的一个进程。这是通过平衡消费者组中所有成员之间的分区来实现的，以便将每个分区分配给该组中的一个消费者。所以如果有一个topic有四个partition，一个consumer group有两个进程，那么每个进程都会从两个partition中消费。

一个consumer group中的成员身份是动态维护的：如果一个进程失败了，分配给它的partition会重新分配给同一组的其他consumer。同样，如果新的消费者加入该组，分区将从现有消费者移动到新消费者。这称为重新平衡组。

此外，当组重新分配自动发生时，可以通过 `ConsumerRebalanceListener` 通知消费者，这允许他们完成必要的应用程序级逻辑，例如状态清理、手动偏移提交等。有关更多详细信息，请参阅[`aiokafka.AIOKafkaConsumer.subscribe()`](https://aiokafka.readthedocs.io/en/stable/api.html#aiokafka.AIOKafkaConsumer.subscribe)

> 警告
>
> 小心 ConsumerRebalanceListener 以避免死锁。消费者将等待定义的处理程序并阻止对 `getmany()` 和` getone() `的后续调用。例如这段代码会死锁：
>
> ```python
> lock = asyncio.Lock()
> consumer = AIOKafkaConsumer(...)
> 
> class MyRebalancer(aiokafka.ConsumerRebalanceListener):
> 
>     async def on_partitions_revoked(self, revoked):
>         async with self.lock:
>             pass
> 
>     async def on_partitions_assigned(self, assigned):
>         pass
> 
> async def main():
>     consumer.subscribe("topic", listener=MyRebalancer())
>     while True:
>         async with self.lock:
>             msgs = await consumer.getmany(timeout_ms=1000)
>             # process messages
> ```
>
> 您需要将 `consumer.getmany(timeout_ms=1000) `调用放在锁之外。

更多关于消费组的组织见 [Official Kafka Docs](https://kafka.apache.org/documentation/#intro_consumers).

#### 按模式的主题订阅

消费者在后台执行定期元数据刷新，并会注意到何时将新分区添加到订阅主题之一或何时创建与订阅正则表达式匹配的新主题。

```python
consumer = AIOKafkaConsumer(
    bootstrap_servers='localhost:9092',
    metadata_max_age_ms=30000,  # This controls the polling interval
)
await consumer.start()
consumer.subscribe(pattern="^MyGreatTopic-.*$")

async for msg in consumer:  # Will detect metadata changes
    print("Consumed msg %s %s %s" % (msg.topic, msg.partition, msg.value))
```

如果您使用消费者组，则组的领导者将在注意到元数据更改时触发组重新平衡。这是因为只有领导者才能完全了解分配给该组的主题。

#### 手动分区分配

消费者也可以使用`assign([tp1, tp2])`手动分配特定分区。在这种情况下，动态分区分配和消费者组协调将被禁用。

```python
consumer = AIOKafkaConsumer(
    bootstrap_servers='localhost:9092'
)
tp1 = TopicPartition("my_topic", 1)
tp2 = TopicPartition("my_topic", 2)
consumer.assign([tp1, tp2])

async for msg in consumer:
    print("Consumed msg %s %s %s", msg.topic, msg.partition, msg.value)
```

group_id 仍然可以用于提交位置，但要小心避免与共享同一组的多个实例发生冲突。

不能混合手动分区分配 `consumer.assign()` 和主题订阅 `consumer.subscribe()`。尝试这样做将导致 `IllegalStateError`。

#### 消费流控制

默认情况下，消费者将从所有分区中获取，从而有效地为这些分区赋予相同的优先级。但是，在某些情况下，您可能希望某些分区具有更高的优先级（比如它们有更多延迟并且您想赶上）。

```python
consumer = AIOKafkaConsumer("my_topic", ...)

partitions = []  # Fetch all partitions on first request
while True:
    msgs = await consumer.getmany(*partitions)
    # process messages
    await process_messages(msgs)

    # Prioritize partitions, that lag behind.
    partitions = []
    for partition in consumer.assignment():
        highwater = consumer.highwater(partition)
        position = await consumer.position(partition)
        position_lag = highwater - position
        timestamp = consumer.last_poll_timestamp(partition)
        time_lag = time.time() * 1000 - timestamp
        if position_lag > POSITION_THRESHOLD or time_lag > TIME_THRESHOLD:
            partitions.append(partition)
```

在这里，如果它们不落后，我们将消耗所有分区，但如果某些分区超过某个阈值，我们将消耗它们以赶上。这可以很好地用于某些消费者死亡并且该消费者接管其现在落后的分区的情况。

> 注意
>
> 如果您更改正在获取的分区，则消耗可能会略有暂停。当消费者请求获取没有可用数据的分区时，就会发生这种情况。考虑设置一个相对较低的 `fetch_max_wait_ms` 来避免这种情况。
>
> `async for` 接口不能与显式分区过滤一起使用，只需使用 `consumer.getone()` 代替。

#### 读取事务信息

事务是在 Kafka 0.11.0 中引入的，其中应用程序可以原子地写入多个主题和分区。为了使其工作，从这些分区读取的消费者应配置为仅读取已提交的数据。这可以通过在消费者的配置中设置`isolation_level=read_committed `来实现：

```python
consumer = aiokafka.AIOKafkaConsumer(
    "my_topic",
    bootstrap_servers='localhost:9092',
    isolation_level="read_committed"
)
await consumer.start()
async for msg in consumer:  # Only read committed tranasctions
    pass
```

在 `read_committed` 模式下，消费者将只读取那些已成功提交的事务消息。它将像以前一样继续读取非事务性消息。在 `read_committed` 模式下没有客户端缓冲。相反，`read_committed `消费者的分区的结束偏移量将是属于打开事务的分区中第一条消息的偏移量。此偏移量称为最后稳定偏移量 (LSO)。

`read_committed` 消费者只会读取 LSO 并过滤掉任何已中止的事务消息。 LSO 还影响 `read_committed` 消费者的 `seek_to_end(*partitions)`和 `end_offsets(partitions)` 行为，详细信息在每个方法的文档中。最后，添加 `last_stable_offset()` API 类似于`highwater()`API 以查询当前分配事务的 lSO：

```python
async for msg in consumer:  # Only read committed tranasctions
    tp = TopicPartition(msg.topic, msg.partition)
    lso = consumer.last_stable_offset(tp)
    lag = lso - msg.offset
    print(f"Consumer is behind by {lag} messages")

    end_offsets = await consumer.end_offsets([tp])
    assert end_offsets[tp] == lso

await consumer.seek_to_end(tp)
position = await consumer.position(tp)
```

具有事务性消息的分区将包括指示事务结果的提交或中止标记。标记不会返回给应用程序，但在日志中有一个偏移量。因此，从带有事务性消息的主题中读取的应用程序将看到消耗的偏移量中的差距。这些丢失的消息将是事务标记，并且它们在两个隔离级别中都会被消费者过滤掉。此外，由于事务中止，使用 `read_committed` 消费者的应用程序也可能会看到间隙，因为这些消息不会被消费者返回，但会有有效的偏移量。

#### 检测消费者故障

使用过 kafka-python 或 Java Client 的人可能知道 `poll()` API 旨在确保消费者组的活跃度。换句话说，消费者只有在消费消息时才会被认为是活着的。 aiokafka 不一样，更多细节请阅读 aiokafka 和 kafka-python 的区别。

aiokafka 将在 `consumer.start()` 上加入组并在后台发送心跳，保持组活跃，与 Java Client 相同。但是在重新平衡的情况下，它也会在后台完成。

自动提交模式下的偏移提交是严格按时间在后台完成的（在 Java 客户端中，如果您不再次调用 `poll()`，则不会进行自动提交）。

## 示例

### 序列化与压缩

Kafka 支持多种压缩类型：“gzip”、“snappy”和“lz4”。您只需要在Kafka Producer中指定压缩方式，Consumer会自动解压。注意：消息是分批次压缩的，所以大批量的时候效率会更高。可以考虑设置 `linger_ms` 在发送前批量处理更多数据。默认情况下，返回的 msg 实例的 `msg.value` 和 `msg.key` 属性是字节。您可以使用自定义序列化器/反序列化器挂钩对对象而不是这些属性中的字节进行操作。

生产者

```python
import json
import asyncio
from aiokafka import AIOKafkaProducer

def serializer(value):
    return json.dumps(value).encode()

async def produce():
    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=serializer,
        compression_type="gzip")

    await producer.start()
    data = {"a": 123.4, "b": "some string"}
    await producer.send('foobar', data)
    data = [1,2,3,4]
    await producer.send('foobar', data)
    await producer.stop()

```

消费者

```python
import json
import asyncio
from kafka.common import KafkaError
from aiokafka import AIOKafkaConsumer

def deserializer(serialized):
    return json.loads(serialized)

async def consume():
    # consumer will decompress messages automatically
    # in accordance to compression type specified in producer
    consumer = AIOKafkaConsumer(
        'foobar',
        bootstrap_servers='localhost:9092',
        value_deserializer=deserializer,
        auto_offset_reset='earliest')
    await consumer.start()
    data = await consumer.getmany(timeout_ms=10000)
    for tp, messages in data.items():
        for message in messages:
            print(type(message.value), message.value)
    await consumer.stop()

asyncio.run(consume())
```

执行

```shell
>>>python3 producer.py
>>>python3 consumer.py
<class 'dict'> {'a': 123.4, 'b': 'some string'}
<class 'list'> [1,2,3,4]
```

### 手动提交

在处理更敏感的数据时，消费者的 `enable_auto_commit=False` 模式可能会导致在严重故障情况下数据丢失。为了避免这种情况，我们可以在处理后手动提交偏移量。请注意，这是从最多一次交付到至少一次交付的权衡，要实现恰好一次，您需要在目标数据库中保存偏移量并自行验证这些偏移量。

> 注意
>
> 在 Kafka Broker 0.11 版和 aiokafka==0.5.0 之后，可以使用Transactional Producer来实现恰好一次交付。请参阅事务生产部分。

消费者

```python
import json
import asyncio
from kafka.common import KafkaError
from aiokafka import AIOKafkaConsumer

async def consume():
    consumer = AIOKafkaConsumer(
        'foobar',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        group_id="some-consumer-group",
        enable_auto_commit=False)
    await consumer.start()
    # we want to consume 10 messages from "foobar" topic
    # and commit after that
    for i in range(10):
        msg = await (consumer.getone()
    await consumer.commit()

    await consumer.stop()

asyncio.run(consume())
```

### 组消费者

从 Kafka 9.0 开始，消费者可以同时消费同一主题。这是通过 Kafka 代理节点（协调器）之一协调消费者来实现的。该节点将执行分区分配的同步（您的分区将由 python 代码分配）并且消费者将始终为分配的分区返回消息。

> 注意
>
> 尽管消费者永远不会从未分配的分区返回消息，但如果您处于 `autocommit=Fals`e 模式，则应在处理 `getmany() `调用返回的下一条消息之前重新检查分配。

生产者

```python
import sys
import asyncio
from aiokafka import AIOKafkaProducer

async def produce(value, partition):
    producer = AIOKafkaProducer(bootstrap_servers='localhost:9092')

    await producer.start()
    await producer.send('some-topic', value, partition=partition)
    await producer.stop()

if len(sys.argv) != 3:
    print("usage: producer.py <partition> <message>")
    sys.exit(1)
value = sys.argv[2].encode()
partition = int(sys.argv[1])

asyncio.run(produce(value, partition))
```

消费者

```python
import sys
import asyncio
from aiokafka import AIOKafkaConsumer

async def consume():
    consumer = AIOKafkaConsumer(
        'some-topic',
        group_id=group_id,
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest')
    await consumer.start()
    for _ in range(msg_cnt):
        msg = await consumer.getone()
        print(f"Message from partition [{msg.partition}]: {msg.value}")
    await consumer.stop()

if len(sys.argv) < 3:
    print("usage: consumer.py <group_id> <wait messages count>")
    sys.exit(1)
group_id = sys.argv[1]
msg_cnt = int(sys.argv[2])

asyncio.run(consume(group_id, msg_cnt))
```

执行

```shell
# creating topic “some-topic” with 2 partitions using standard Kafka utility:

bin/kafka-topics.sh --create --zookeeper localhost:2181 --replication-factor 1 --partitions 2 --topic some-topic

# terminal#1:
python3 consumer.py TEST_GROUP 2

# terminal#2:
python3 consumer.py TEST_GROUP 2

# terminal#3:
python3 consumer.py OTHER_GROUP 4

# terminal#4:
python3 producer.py 0 'message #1'
python3 producer.py 0 'message #2'
python3 producer.py 1 'message #3'
python3 producer.py 1 'message #4'
```

### 自定义分区器

如果您考虑将分区用作逻辑实体，而不是纯粹用于负载平衡，您可能需要对将消息路由到分区有更多的控制。默认使用散列算法。

生产者

```python
import asyncio
import random
from aiokafka import AIOKafkaProducer

def my_partitioner(key, all_partitions, available_partitions):
   if key == b'first':
       return all_partitions[0]
   elif key == b'last':
       return all_partitions[-1]
   return random.choice(all_partitions)

async def produce_one(producer, key, value):
    future = await producer.send('foobar', value, key=key)
    resp = await future
    print("'%s' produced in partition: %i"%(value.decode(), resp.partition))

async def produce_task():
    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9092',
        partitioner=my_partitioner)

    await producer.start()
    await produce_one(producer, b'last', b'1')
    await produce_one(producer, b'some', b'2')
    await produce_one(producer, b'first', b'3')
    await producer.stop()

asyncio.run(produce_task())
```

使用

```python
>>>python3 producer.py
'1' produced in partition: 9
'2' produced in partition: 6
'3' produced in partition: 0
```

### SSL使用

使用 aiokafka 的 SSL 使用示例。请阅读摘要以获取更多信息。

```python
import asyncio
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.helpers import create_ssl_context
from kafka.common import TopicPartition

context = create_ssl_context(
    cafile="./ca-cert",  # CA used to sign certificate.
                         # `CARoot` of JKS store container
    certfile="./cert-signed",  # Signed certificate
    keyfile="./cert-key",  # Private Key file of `certfile` certificate
    password="123123"
)

async def produce_and_consume():
    # Produce
    producer = AIOKafkaProducer(
        bootstrap_servers='localhost:9093',
        security_protocol="SSL", ssl_context=context)

    await producer.start()
    try:
        msg = await producer.send_and_wait(
            'my_topic', b"Super Message", partition=0)
    finally:
        await producer.stop()

    consumer = AIOKafkaConsumer(
        "my_topic", bootstrap_servers='localhost:9093',
        security_protocol="SSL", ssl_context=context)
    await consumer.start()
    try:
        consumer.seek(TopicPartition('my_topic', 0), msg.offset)
        fetch_msg = await consumer.getone()
    finally:
        await consumer.stop()

    print("Success", msg, fetch_msg)

if __name__ == "__main__":
    asyncio.run(produce_and_consume())
```

使用

```shell
>>>python3 ssl_consume_produce.py
Success RecordMetadata(topic='my_topic', partition=0, topic_partition=TopicPartition(topic='my_topic', partition=0), offset=32) ConsumerRecord(topic='my_topic', partition=0, offset=32, timestamp=1479393347381, timestamp_type=0, key=None, value=b'Super Message', checksum=469650252, serialized_key_size=-1, serialized_value_size=13)
```

### 本地状态消费者

虽然 Kafka 应用程序的默认设置是在 Kafka 的内部存储中存储提交点，但您可以禁用它并使用 seek() 移动到存储点。如果您想将偏移量作为计算结果存储在同一系统中（下面示例中的文件系统），这是有道理的。但这就是说，您可能仍然希望使用协调消费者组功能。

此示例显示了广泛使用ConsumerRebalanceListener 来控制重新平衡之前和之后所做的事情。

本地状态消费者

```python
import asyncio
from aiokafka import AIOKafkaConsumer, ConsumerRebalanceListener
from aiokafka.errors import OffsetOutOfRangeError


import json
import pathlib
from collections import Counter

FILE_NAME_TMPL = "/tmp/my-partition-state-{tp.topic}-{tp.partition}.json"


class RebalanceListener(ConsumerRebalanceListener):

    def __init__(self, consumer, local_state):
        self.consumer = consumer
        self.local_state = local_state

    async def on_partitions_revoked(self, revoked):
        print("Revoked", revoked)
        self.local_state.dump_local_state()

    async def on_partitions_assigned(self, assigned):
        print("Assigned", assigned)
        self.local_state.load_local_state(assigned)
        for tp in assigned:
            last_offset = self.local_state.get_last_offset(tp)
            if last_offset < 0:
                await self.consumer.seek_to_beginning(tp)
            else:
                self.consumer.seek(tp, last_offset + 1)


class LocalState:

    def __init__(self):
        self._counts = {}
        self._offsets = {}

    def dump_local_state(self):
        for tp in self._counts:
            fpath = pathlib.Path(FILE_NAME_TMPL.format(tp=tp))
            with fpath.open("w+") as f:
                json.dump({
                    "last_offset": self._offsets[tp],
                    "counts": dict(self._counts[tp])
                }, f)

    def load_local_state(self, partitions):
        self._counts.clear()
        self._offsets.clear()
        for tp in partitions:
            fpath = pathlib.Path(FILE_NAME_TMPL.format(tp=tp))
            state = {
                "last_offset": -1,  # Non existing, will reset
                "counts": {}
            }
            if fpath.exists():
                with fpath.open("r+") as f:
                    try:
                        state = json.load(f)
                    except json.JSONDecodeError:
                        pass
            self._counts[tp] = Counter(state['counts'])
            self._offsets[tp] = state['last_offset']

    def add_counts(self, tp, counts, last_offset):
        self._counts[tp] += counts
        self._offsets[tp] = last_offset

    def get_last_offset(self, tp):
        return self._offsets[tp]

    def discard_state(self, tps):
        for tp in tps:
            self._offsets[tp] = -1
            self._counts[tp] = Counter()


async def save_state_every_second(local_state):
    while True:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            break
        local_state.dump_local_state()


async def consume():
    consumer = AIOKafkaConsumer(
        bootstrap_servers='localhost:9092',
        group_id="my_group",           # Consumer must be in a group to commit
        enable_auto_commit=False,      # Will disable autocommit
        auto_offset_reset="none",
        key_deserializer=lambda key: key.decode("utf-8") if key else "",
    )
    await consumer.start()

    local_state = LocalState()
    listener = RebalanceListener(consumer, local_state)
    consumer.subscribe(topics=["test"], listener=listener)

    save_task = asyncio.create_task(save_state_every_second(local_state))

    try:

        while True:
            try:
                msg_set = await consumer.getmany(timeout_ms=1000)
            except OffsetOutOfRangeError as err:
                # This means that saved file is outdated and should be
                # discarded
                tps = err.args[0].keys()
                local_state.discard_state(tps)
                await consumer.seek_to_beginning(*tps)
                continue

            for tp, msgs in msg_set.items():
                counts = Counter()
                for msg in msgs:
                    print("Process", tp, msg.key)
                    counts[msg.key] += 1
                local_state.add_counts(tp, counts, msg.offset)

    finally:
        await consumer.stop()
        save_task.cancel()
        await save_task


if __name__ == "__main__":
    asyncio.run(consume())
```

在这个例子中有几个有趣的点： 

- 我们实现了 `RebalanceListener` 以在重新平衡之前转储所有计数和偏移量。重新平衡后，我们从相同的文件加载它们。这是一种避免重新读取所有消息的缓存。
- 我们通过设置 `auto_offset_reset="none"` 手动控制偏移重置策略。我们需要它来捕获 OffsetOutOfRangeError，这样我们就可以在文件很旧并且 Kafka 中不再存在这样的偏移量时清除缓存。
- 由于我们在这里计算键，它们将始终分区到生产时的同一分区。我们不会在不同的文件中有重复的计数。

执行

```shell
# 第一个消费者输出
>>>python examples/local_state_consumer.py
Revoked set()
Assigned {TopicPartition(topic='test', partition=0), TopicPartition(topic='test', partition=1), TopicPartition(topic='test', partition=2)}
Heartbeat failed for group my_group because it is rebalancing
Revoked {TopicPartition(topic='test', partition=0), TopicPartition(topic='test', partition=1), TopicPartition(topic='test', partition=2)}
Assigned {TopicPartition(topic='test', partition=0), TopicPartition(topic='test', partition=2)}
Process TopicPartition(topic='test', partition=2) 123
Process TopicPartition(topic='test', partition=2) 9999
Process TopicPartition(topic='test', partition=2) 1111
Process TopicPartition(topic='test', partition=0) 4444
Process TopicPartition(topic='test', partition=0) 123123
Process TopicPartition(topic='test', partition=0) 5555
Process TopicPartition(topic='test', partition=2) 88891823
Process TopicPartition(topic='test', partition=2) 2

# 第二个消费者输出
>>>python examples/local_state_consumer.py
Revoked set()
Assigned {TopicPartition(topic='test', partition=1)}
Process TopicPartition(topic='test', partition=1) 321
Process TopicPartition(topic='test', partition=1) 777

# 结果创建了这样的文件
>>>cat /tmp/my-partition-state-test-0.json && echo
{"last_offset": 4, "counts": {"123123": 1, "4444": 1, "321": 2, "5555": 1}}
```

### 批量生产者

如果您的应用程序需要精确控制批量创建和提交，并且您愿意放弃自动序列化和分区选择的优点，则可以使用简单的 `create_batch()` 和 `send_batch()` 接口。

消费者

```python
import asyncio
import random
from aiokafka.producer import AIOKafkaProducer

async def send_many(num):
    topic  = "my_topic"
    producer = AIOKafkaProducer()
    await producer.start()

    batch = producer.create_batch()

    i = 0
    while i < num:
        msg = ("Test message %d" % i).encode("utf-8")
        metadata = batch.append(key=None, value=msg, timestamp=None)
        if metadata is None:
            partitions = await producer.partitions_for(topic)
            partition = random.choice(tuple(partitions))
            await producer.send_batch(batch, topic, partition=partition)
            print("%d messages sent to partition %d"
                  % (batch.record_count(), partition))
            batch = producer.create_batch()
            continue
        i += 1
    partitions = await producer.partitions_for(topic)
    partition = random.choice(tuple(partitions))
    await producer.send_batch(batch, topic, partition=partition)
    print("%d messages sent to partition %d"
          % (batch.record_count(), partition))
    await producer.stop()

asyncio.run(send_many(1000))
```

执行

```shell
>>>python3 batch_produce.py
329 messages sent to partition 2
327 messages sent to partition 0
327 messages sent to partition 0
17 messages sent to partition 1
```

### 事务消费处理生产

如果您有一种模式，您想从一个主题消费、处理数据并生产到另一个主题，那么您真的很想使用事务生产者来做到这一点。在下面的示例中，我们从 IN_TOPIC 读取、处理数据并将结果以事务方式生成到 OUT_TOPIC。

```python
import asyncio
from collections import defaultdict, Counter

from aiokafka import TopicPartition, AIOKafkaConsumer, AIOKafkaProducer


IN_TOPIC = "in_topic"
GROUP_ID = "processing-group"
OUT_TOPIC = "out_topic"
TRANSACTIONAL_ID = "my-txn-id"
BOOTSTRAP_SERVERS = "localhost:9092"

POLL_TIMEOUT = 60_000


def process_batch(msgs):
    # Group by key do simple count sampling by a minute window
    buckets_by_key = defaultdict(Counter)
    for msg in msgs:
        timestamp = (msg.timestamp // 60_000) * 60
        buckets_by_key[msg.key][timestamp] += 1

    res = []
    for key, counts in buckets_by_key.items():
        for timestamp, count in counts.items():
            value = str(count).encode()
            res.append((key, value, timestamp))

    return res


async def transactional_process():
    consumer = AIOKafkaConsumer(
        IN_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        enable_auto_commit=False,
        group_id=GROUP_ID,
        isolation_level="read_committed"  # <-- This will filter aborted txn's
    )
    await consumer.start()

    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        transactional_id=TRANSACTIONAL_ID
    )
    await producer.start()

    try:
        while True:
            msg_batch = await consumer.getmany(timeout_ms=POLL_TIMEOUT)

            async with producer.transaction():
                commit_offsets = {}
                in_msgs = []
                for tp, msgs in msg_batch.items():
                    in_msgs.extend(msgs)
                    commit_offsets[tp] = msgs[-1].offset + 1

                out_msgs = process_batch(in_msgs)
                for key, value, timestamp in out_msgs:
                    await producer.send(
                        OUT_TOPIC, value=value, key=key,
                        timestamp_ms=int(timestamp * 1000)
                    )
                # We commit through the producer because we want the commit
                # to only succeed if the whole transaction is done
                # successfully.
                await producer.send_offsets_to_transaction(
                    commit_offsets, GROUP_ID)
    finally:
        await consumer.stop()
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(transactional_process())
```



