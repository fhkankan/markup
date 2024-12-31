# Logstash

[文档](https://www.elastic.co/guide/en/logstash/current/introduction.html)

## 概述

`Logstash` 是一个具有实时管道功能的开源数据收集引擎。`Logstash`可以动态统一来自不同来源的数据，并将数据规范化到您选择的目标中。为了多样化的高级下游分析和可视化用例，清理和使所有数据平等化。

虽然` Logstash` 最初在日志收集方面推动了创新，但它的能力远远超出了该用例。任何类型的事件都可以通过广泛的输入、过滤和输出插件进行增强和转换，许多本地编解码器进一步简化了摄入过程。`Logstash` 通过利用更多的数据量和种类加速您的洞察力。

## 原理

Logstash 事件处理管道有三个阶段：输入 → 过滤器 → 输出。

inputs 模块负责收集数据，filters 模块可以对收集到的数据进行格式化、过滤、简单的数据处理，outputs 模块负责将数据同步到目的地，Logstash的处理流程，就像管道一样，数据从管道的一端，流向另外一端。

inputs 和 outputs 支持编解码器，使您能够在数据进入或离开管道时对数据进行编码或解码，而无需使用单独的过滤器。

## python

### 安装

```
# 同步处理
pip install python-stash

# 异步处理
pip install python-logstash-async
```

`logstash`的配置

```
input {
  udp {
    port => 5959
    codec => json
  }
}
output {
  stdout {
    codec => rubydebug
  }
}
```

### 使用

同步使用

```python
import logging
import logstash
import sys

host = 'localhost'

test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
test_logger.addHandler(logstash.LogstashHandler(host, 5959, version=1))  # 是UDPLogstashHandler
# test_logger.addHandler(logstash.UDPLogstashHandler(host, 5959, version=1))
# test_logger.addHandler(logstash.TCPLogstashHandler(host, 5959, version=1))
# test_logger.addHandler(logstash.AMQPLogstashHandler(host='localhost', version=1))  # 需要pip install pika

test_logger.error('python-logstash: test logstash error message.')
test_logger.info('python-logstash: test logstash info message.')
test_logger.warning('python-logstash: test logstash warning message.')

# add extra field to logstash message
extra = {
    'test_string': 'python version: ' + repr(sys.version_info),
    'test_boolean': True,
    'test_dict': {'a': 1, 'b': 'c'},
    'test_float': 1.23,
    'test_integer': 123,
    'test_list': [1, 2, '3'],
}
test_logger.info('python-logstash: test extra fields', extra=extra)  # extra中不要使用预留字段
```

异步使用

```python
import logging
import sys
from logstash_async.handler import AsynchronousLogstashHandler

host = 'localhost'
port = 5959

test_logger = logging.getLogger('python-logstash-logger')
test_logger.setLevel(logging.INFO)
test_logger.addHandler(AsynchronousLogstashHandler(host, port, database_path='logstash.db'))  # 默认TcpTransport
# from logstash_async.transport import HttpTransport
# transport = HttpTransport(*)
# test_logger.addHandler(AsynchronousLogstashHandler(host, port,None, transport=transport))
# from logstash_async.transport import UdpTransport
# transport = UdpTransport(*)
# test_logger.addHandler(AsynchronousLogstashHandler(host, port,None, transport=transport))

# If you don't want to write to a SQLite database, then you do
# not have to specify a database_path.
# NOTE: Without a database, messages are lost between process restarts.
# test_logger.addHandler(AsynchronousLogstashHandler(host, port))

test_logger.error('python-logstash-async: test logstash error message.')
test_logger.info('python-logstash-async: test logstash info message.')
test_logger.warning('python-logstash-async: test logstash warning message.')

# add extra field to logstash message
extra = {
    'test_string': 'python version: ' + repr(sys.version_info),
    'test_boolean': True,
    'test_dict': {'a': 1, 'b': 'c'},
    'test_float': 1.23,
    'test_integer': 123,
    'test_list': [1, 2, '3'],
}
test_logger.info('python-logstash: test extra fields', extra=extra)  # extra中不要使用预留字段
```

