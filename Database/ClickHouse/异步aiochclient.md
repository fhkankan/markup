# aiochclient

异步http(s) clickhouse client for python 3.6+具有双向转换类型、流式支持、对选定查询的延迟解码和全类型接口

## 安装

```shell
pip install aiochclient
```

安装时附加加速要求：

```
pip install aiochclient[speedups]
```

它将另外安装[cChardet](https://pypi.python.org/pypi/cchardet) 和[aiodns](https://pypi.python.org/pypi/aiodns)加速 和[ciso8601](https://github.com/closeio/ciso8601)用于超快 从ClickHouse解码数据时进行DateTime分析。

同时，在安装时，它将尝试建立cython扩展以提高速度（约30%）。

## 使用

### 连接

`aiochclient`连接需要`aiohttp.ClientSession`：

```python
from aiochclient import ChClient
from aiohttp import ClientSession

	async def main():
        async with ClientSession() as s:
            client = ChClient(s)
            assert await client.is_alive() # returns True if connection is Ok
```

### 查询

- 增加

```python
await client.execute("CREATE TABLE t (a UInt8, b Tuple(Date, Nullable(Float32))) ENGINE = Memory")
```

对于insert查询，可以将值作为`*args`传递。值应为iterables:

```python
await client.execute("INSERT INTO t VALUES",(1,(dt.date(2018,9,7),None)),(2,(dt.date(2018,9,8),3.14)),)
```

- 查询

要同时获取所有行，请使用`fetch`方法：

```python
all_rows = await client.fetch("SELECT * FROM t")
```

要从结果中获取第一行，请使用`fetchrow`方法：

```python
row = awaitclient.fetchrow("SELECT * FROM t WHERE a=1")
assert row[0] == 1
assert row["b"]==(dt.date(2018,9,7),None)
```

您还可以使用`fetchval`方法，它返回查询结果第一行的第一个值：

```python
val = await client.fetchval("SELECT b FROM t WHERE a=2")
assertval == (dt.date(2018,9,8),3.14)
```

通过查询结果的异步迭代，您可以获取 多行而不同时将它们全部加载到内存中：

```python
async for row in client.iterate("SELECT number, number*2 FROM system.numbers LIMIT 10000"):
    assert row[0]*2 == row[1]
```

使用`fetch`/`fetchrow`/`fetchval`/`iterate`进行选择查询 和`execute`或最后一个`for insert`和所有其他查询。

### 使用查询结果

所有fetch查询都将行作为轻量级内存返回有效对象（来自v`1.0.0`，在它之前-只有元组） 具有完整的映射接口，其中 您可以按名称或索引获取字段

```python
row = await client.fetchrow("SELECT a, b FROM t WHERE a=1")
assert row["a"] == 1
assert row[0] == 1
assert row[:] == (1,(dt.date(2018,9,8),3.14))
assert list(row.keys()) == ["a","b"]
assert list(row.values()) == [1,(dt.date(2018,9,8),3.14)]
```

## 连接池

如果要更改连接池大小，可以使用 [aiohttp.TCPConnector](https://docs.aiohttp.org/en/stable/client_advanced.html#limiting-connection-pool-size)。 请注意，默认池限制为100个连接。

## 速度

使用`uvloop`并与`aiochclient[speedups]`一起安装 为了速度，强烈推荐使用。

至于最后一个版本的`aiochclient`它的速度 使用一个任务（没有集合或并行 客户等）是关于180k-220k行/秒关于select和about50k-80k行/秒关于插入查询 取决于其环境和clickhouse设置。