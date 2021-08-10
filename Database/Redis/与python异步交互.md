# 与python异步交互

[官网](https://aioredis.readthedocs.io/en/latest/)

## 概述

安装

```shell
pip install aioredis
```

使用

```python
import aioredis
import asyncio

class Redis:
    _redis = None

    async def get_redis_pool(self, *args, **kwargs):
        if not self._redis:
            self._redis = await aioredis.create_redis_pool(*args, **kwargs)
        return self._redis

    async def close(self):
        if self._redis:
            self._redis.close()
            await self._redis.wait_closed()


async def get_value(key):
    redis = Redis()
    r = await redis.get_redis_pool(('127.0.0.1', 6379), db=7, minisize=8, maxsize=32, encoding='utf-8')
    value = await r.get(key)
    print(f'{key!r}: {value!r}')
    await redis.close()         

if __name__ == '__main__':
    asyncio.run(get_value('key'))
```

## 连接

```python
import asyncio

import aioredis


async def main():
    redis = aioredis.from_url("redis://localhost")
    await redis.set("my-key", "value")
    value = await redis.get("my-key")
    print(value)


asyncio.run(main())
```

特定数据库

```python
# 方法一
redis = await aioredis.from_url("redis://localhost",  db=1)
# 方法二
redis = await aioredis.from_url("redis://localhost/1")
```

需验证数据库

```python
# keywords
redis = await aioredis.from_url(
    "redis://localhost", username="user", password="sEcRet"
)

# Auth of the URI
redis = await aioredis.from_url("redis://user:sEcRet@localhost/")
```

## 操作

### String

- set

```python
set(key, value, *, expire=0, pexpire=0)
setex(key, seconds, value)
psetex(key, milliseconds, value)

setnx(key, value)
setbit(key, offset, value)
setrange(key, offset, valuee)

mset(*args)
msetnx(key, value, *pairs)

incr(key)
incrby(key increment)
incrbyfloat(key, increment)
desc(key)
descrby(key, decrement)
append(key, value)

getset(key, value)
```

- get

```python
get(key)
getbit(key, offset)
getrange(key, start, end)

mget(key, *keys)

bicount(key, start=None, end=None)
strlen(key)
```

### Hash

- set

```python
hset(key, field=None, value=None, mapping=None)
hsetnx(key, field, value)

hmset(key, field, value, *pairs)  # 不建议
hmset_dict(key, *args, **kwargs)  # 不建议

hincrby(key, field, increment=1)
hincrbyfloat(key, field, increment=1.0)

hdel(key, filed, *fields)
```

- get

```python
hget(key, field, *)
hgetall(key, *)
hkeys(key, *)
hlen(key)
hvals(key, *)
hscan(keey, cursor=0, match=None, count=None)
hexists(key, field)

hmget(key, field, *fields)
```

### List

- set

```python
linsert(key, pivot, value, before=False)
lpush(key, value, *values)
rpush(key, value, *values)
lpushx(key, value)
rpushx(key, value)
lset(key, index, value)

lpop(key, *)
rpop(key, *)
rpoplpush(sourcekey, destkey, *)

blpop(key, *keys, timeout=0)
brpop(key, *keys, timeout=0)
brpoplpush(sourcekey, destkey, timeout=0)

lrem(key, count, value)
ltrim(key, start, stop)
```

- get

```python
lindex(key, index, *)
llen(key)
lrange(key, start, stop, *)
```

### Set

- set

```python
sadd(key, member, *members)
sdiffstore(destkey, key, *keys)
sintersotre(destkey, key, *keys)
sunionstore(destkey, key, *keys)

smove(sourcekey, destkey, member)
spop(key, count=None, *)
sreem(key, member, *members)
```

- get

```python
scard(key)
smembers(key, *)
sismemebr(key, member)
srandmember(key, count=None, *)

sdiff(key, *keys)
sinter(key, *keys)

sunion(key, *keys)

sscan(key, cursor=0, math=None, count=None)
isscan(key, *, match=None, count=None)
```

### Zset

- set

```python
zadd(key, score, member, *pairs, exist=None)
zincrby(key, increment, member)

zinterstore(destkey, key, *keys, with_weights=False, aggregate=None)

zunionstore(destkey, keys, *keeys, with_weights=False, aggregate=None)

zrem(key, member, *members)
zremrangebylex(key, min=b"-", max=b"+")
zremrangebyrank(key, start, stop)
zremrangebyscore(key, min=float("-inf"), max=float("inf"), *, exclude=None)

zpopmin(key, count=None, *)
zpopmax(key, count=None, *)
bzpopmin(key, *keys, timeout=0)
bzpopmax(key, *keys, timeout=0)
```

- get

```python
zcard(key)
zcount(key, min=xx, max=xx, *, exclude=None)
zlexcount(key, min=b"-", max=b"+")

zscore(key, meember)

zrange(key, start=0, stop=-1, withscores=False)
zrangebylex(key,min=b"-",max=b"+",offset=None,count=None)
zrangebyscore(key,min=float("-inf"),max=float("inf"),withscores=False,offset=None,count=None,*,exclude=None)
zrevrange(key, start, stop, withscores=False)
zrevrangebyscore(key, max=float("inf"), min=float("-inf"), *, exclude=None, withscores=False,offset=None,count=None)
zrevrangebylex(key,min=b"-",max=b"+",offset=None,count=None)

zrank(key, member)
zreevrank(key, meember)

zscan(key, cursor=0, match=None, count=None)
izscan(key, *, match=None, count=None)
```

###  通用

- set

```python
delete(key, *keys)
dump(key)

expire(key, timeout)  # 秒
expireat(key, timestamp)  # 秒级别的时间戳
persist(key)
pexpire(key, timeout)	# 毫秒
pexpireat(key, timestamp)  # 毫秒级别的时间戳

rename(key, newkey)
renamenx(key, newkey)

reestore(key, ttl, value)
move(key, db)
```

- get

```python
exists(key, *keys)
randomkey()
keys(pattern)
scan(cursor=0, match=None, count=None, key_type=None)
iscan(*, match=None, count=None)
sort( key,*get_patterns,by=None,offset=None,count=None,asc=None,alpha=False,store=None)

ttl(key)
touch(key, *keys)
type(key)
```

## 返回值

默认情况下，aioredis 将为大多数返回字符串回复的 Redis 命令返回`bytes`。已知 Redis 错误回复是有效的 UTF-8 字符串，因此会自动解码错误消息。

如果您知道 Redis 中的数据是有效字符串，您可以通过在命令调用中传递 `decode_responses=True` 来告诉 aioredis 解码结果

```python
import asyncio
import aioredis


async def main():
    redis = aioredis.from_url("redis://localhost")
    await redis.set("key", "string-value")
    bin_value = await redis.get("key")
    assert bin_value == b"string-value"
    redis = aioredis.from_url("redis://localhost", decode_responses=True)
    str_value = await redis.get("key")
    assert str_value == "string-value"

    await redis.close()


asyncio.run(main())
```

默认情况，`aioredis`会自动解码`lists,hashes,sets`等

```python
import asyncio
import aioredis


async def main():
    redis = aioredis.from_url("redis://localhost")
    await redis.hmset_dict("hash", key1="value1", key2="value2", key3=123)

    result = await redis.hgetall("hash", encoding="utf-8")
    assert result == {
        "key1": "value1",
        "key2": "value2",
        "key3": "123",  # note that Redis returns int as string
    }

    await redis.close()


asyncio.run(main())
```

## 事务(Multi/Exec)

```python
import asyncio
import aioredis


async def main():
    redis = await aioredis.from_url("redis://localhost")
    async with redis.pipeline(transaction=True) as pipe:
        ok1, ok2 = await (pipe.set("key1", "value1").set("key2", "value2").execute())
    assert ok1
    assert ok2
    
    await redis.delete("foo", "bar")
    async with redis.pipeline(transaction=True) as pipe:
        res = await pipe.incr("foo").incr("bar").execute()
    print(res)


asyncio.run(main())
```

`aioredis.Redis.pipeline` 将返回一个 `aioredis.Pipeline` 对象，该对象将缓冲内存中的所有命令，并使用 `Redis Bulk String` 协议将它们编译成批处理。此外，每个命令都将返回 Pipeline 实例，允许您链接命令，如 `p.set('foo', 1).set('bar', 2).mget('foo', 'bar')`。

在调用 `execute()` 并等待之前，这些命令不会反映在 Redis 中。

通常，在执行批量操作时，需要利用“事务”（例如，Multi/Exec），因为它还会添加一个批量操作的原子性层。

## 发布订阅

aioredis 为 Redis 发布/订阅消息提供支持。

例一

```python
import asyncio

import async_timeout

import aioredis

STOPWORD = "STOP"


async def reader(channel: aioredis.client.PubSub):
    while True:
        try:
            async with async_timeout.timeout(1):
                message = await channel.get_message(ignore_subscribe_messages=True)
                if message is not None:
                    print(f"(Reader) Message Received: {message}")
                    if message["data"] == STOPWORD:
                        print("(Reader) STOP")
                        break
                await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            pass


async def main():
    redis = aioredis.from_url("redis://localhost")
    pubsub = redis.pubsub()
    
    # 订阅特定频道
    await pubsub.subscribe("channel:1", "channel:2")
	# 订阅与glob样式模式匹配的频道
    # await pubsub.psubscribe("channel:*")
    
    asyncio.create_task(reader(pubsub))

    await redis.publish("channel:1", "Hello")
    await redis.publish("channel:2", "World")
    await redis.publish("channel:1", STOPWORD)


asyncio.run(main())
```

例二

```python
import asyncio

import async_timeout

import aioredis

STOPWORD = "STOP"


async def pubsub():
    redis = aioredis.Redis.from_url(
        "redis://localhost", max_connections=10, decode_responses=True
    )
    psub = redis.pubsub()

    async def reader(channel: aioredis.client.PubSub):
        while True:
            try:
                async with async_timeout.timeout(1):
                    message = await channel.get_message(ignore_subscribe_messages=True)
                    if message is not None:
                        print(f"(Reader) Message Received: {message}")
                        if message["data"] == STOPWORD:
                            print("(Reader) STOP")
                            break
                    await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                pass

    async with psub as p:
        await p.subscribe("channel:1")
        await reader(p)  # wait for reader to complete
        await p.unsubscribe("channel:1")

    # closing all open connections
    await psub.close()


async def main():
    tsk = asyncio.create_task(pubsub())

    async def publish():
        pub = aioredis.Redis.from_url("redis://localhost", decode_responses=True)
        while not tsk.done():
            # wait for clients to subscribe
            while True:
                subs = dict(await pub.pubsub_numsub("channel:1"))
                if subs["channel:1"] == 1:
                    break
                await asyncio.sleep(0)
            # publish some messages
            for msg in ["one", "two", "three"]:
                print(f"(Publisher) Publishing Message: {msg}")
                await pub.publish("channel:1", msg)
            # send stop word
            await pub.publish("channel:1", STOPWORD)
        await pub.close()

    await publish()


if __name__ == "__main__":
    import os

    if "redis_version:2.6" not in os.environ.get("REDIS_VERSION", ""):
        asyncio.run(main())
```



## 哨兵客户端

```python
import asyncio

import aioredis.sentinel


async def main():
    
    # 哨兵客户端需要一个 Redis 哨兵地址列表来连接并开始发现服务。
    sentinel = aioredis.sentinel.Sentinel(
        ["redis://localhost:26379", "redis://sentinel2:26379"]
    )
    
    # aioredis.sentinel.Sentinel.master_for,aioredis.sentinel.Sentinel.slave_for将返回连接到哨兵监控的指定服务的 Redis 客户端
    # 哨兵客户端将检测故障转移并自动重新连接 Redis 客户端。
    redis = sentinel.master_for("mymaster")

    ok = await redis.set("key", "value")
    assert ok
    val = await redis.get("key", encoding="utf-8")
    assert val == "value"


asyncio.run(main())
```

 ## 连接池

```python
import asyncio

import aioredis


async def main():
    redis = aioredis.from_url("redis://localhost", max_connections=10)
    await redis.execute_command("set", "my-key", "value")
    val = await redis.execute_command("get", "my-key")
    print("raw value:", val)


async def main_pool():
    pool = aioredis.ConnectionPool.from_url("redis://localhost", max_connections=10)
    redis = aioredis.Redis(connection_pool=pool)
    await redis.execute_command("set", "my-key", "value")
    val = await redis.execute_command("get", "my-key")
    print("raw value:", val)


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main_pool())
```

## 扫描

```python
import asyncio

import aioredis


async def main():
    """Scan command example."""
    redis = aioredis.from_url("redis://localhost")

    await redis.mset({"key:1": "value1", "key:2": "value2"})
    async with redis.client() as conn:
        cur = b"0"  # set initial cursor to 0
        while cur:
            cur, keys = await conn.scan(cur, match="key:*")
            print("Iteration results:", keys)


if __name__ == "__main__":
    import os

    if "redis_version:2.6" not in os.environ.get("REDIS_VERSION", ""):
        asyncio.run(main())
```

## 阻塞命令

```python
import asyncio

import aioredis


async def blocking_commands():
    # Redis client bound to pool of connections (auto-reconnecting).
    redis = aioredis.Redis.from_url("redis://localhost")

    async def get_message():
        # Redis blocking commands block the connection they are on
        # until they complete. For this reason, the connection must
        # not be returned to the connection pool until we've
        # finished waiting on future created by brpop(). To achieve
        # this, 'await redis' acquires a dedicated connection from
        # the connection pool and creates a new Redis command object
        # using it. This object is a context manager and the
        # connection will be released back to the pool at the end of
        # the with block."
        with await redis as r:
            return await r.brpop("my-key")

    future = asyncio.create_task(get_message())
    await redis.lpush("my-key", "value")
    await future
    print(future.result())

    # gracefully closing underlying connection
    await redis.close()


if __name__ == "__main__":
    asyncio.run(blocking_commands())
```

## 高级客户端

```python
import asyncio

import aioredis


async def main():
    # Redis client bound to single connection (no auto reconnection).
    redis = aioredis.from_url(
        "redis://localhost", encoding="utf-8", decode_responses=True
    )
    async with redis.client() as conn:
        await conn.set("my-key", "value")
        val = await conn.get("my-key")
    print(val)


async def redis_pool():
    # Redis client bound to pool of connections (auto-reconnecting).
    redis = aioredis.from_url(
        "redis://localhost", encoding="utf-8", decode_responses=True
    )
    await redis.set("my-key", "value")
    val = await redis.get("my-key")
    print(val)


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(redis_pool())
```

## 低级连接

```python
import asyncio

import aioredis


async def main():
    # Create a redis client bound to a connection pool.
    redis = aioredis.from_url(
        "redis://localhost", encoding="utf-8", decode_responses=True
    )
    # get a redis client bound to a single connection.
    async with redis.client() as conn:
        ok = await conn.execute_command("set", "my-key", "some value")
        assert ok is True

        str_value = await conn.execute_command("get", "my-key")
        assert str_value == "some value"

        print("str value:", str_value)
    # The connection is automatically release to the pool


async def main_single():
    # Create a redis client with only a single connection.
    redis = aioredis.Redis(
        host="localhost",
        encoding="utf-8",
        decode_responses=True,
        single_connection_client=True,
    )
    ok = await redis.execute_command("set", "my-key", "some value")
    assert ok is True

    str_value = await redis.execute_command("get", "my-key")
    assert str_value == "some value"

    print("str value:", str_value)
    # the connection is automatically closed by GC.


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(main_single())
```



