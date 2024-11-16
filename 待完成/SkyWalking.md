# SkyWalking

[文档](https://skywalking.apache.org/zh/2020-04-19-skywalking-quick-start/)

## 概述

SkyWalking 是什么？

> FROM http://skywalking.apache.org/
>
> 分布式系统的应用程序性能监视工具，专为微服务、云原生架构和基于容器（Docker、K8s、Mesos）架构而设计。
>
> 提供分布式追踪、服务网格遥测分析、度量聚合和可视化一体化解决方案。

SkyWalking 有哪些功能？

> FROM http://skywalking.apache.org/
>
> - 多种监控手段。可以通过语言探针和 service mesh 获得监控是数据。
> - 多个语言自动探针。包括 Java，.NET Core 和 Node.JS。
> - 轻量高效。无需大数据平台，和大量的服务器资源。
> - 模块化。UI、存储、集群管理都有多种机制可选。
> - 支持告警。
> - 优秀的可视化解决方案。

整个架构，分成上、下、左、右四部分：

> 考虑到让描述更简单，我们舍弃掉 Metric 指标相关，而着重在 Tracing 链路相关功能。

- 上部分 **Agent** ：负责从应用中，收集链路信息，发送给 SkyWalking OAP 服务器。目前支持 SkyWalking、Zikpin、Jaeger 等提供的 Tracing 数据信息。而我们目前采用的是，SkyWalking Agent 收集 SkyWalking Tracing 数据，传递给服务器。
- 下部分 **SkyWalking OAP** ：负责接收 Agent 发送的 Tracing 数据信息，然后进行分析(Analysis Core) ，存储到外部存储器( Storage )，最终提供查询( Query )功能。
- 右部分 **Storage** ：Tracing 数据存储。目前支持 ES、MySQL、Sharding Sphere、TiDB、H2 多种存储器。而我们目前采用的是 ES ，主要考虑是 SkyWalking 开发团队自己的生产环境采用 ES 为主。
- 左部分 **SkyWalking UI** ：负责提供控台，查看链路等等。

## python

### 安装

```
# 同步
pip install "apache-skywalking"  # 默认gRPC
pip install "apache-skywalking[http]"
pip install "apache-skywalking[kafka]"
pip install "apache-skywalking[all]"  # gRPC, HTTP, Kafka

# 异步
pip install fastapi_skywalking_middleware  # Fastapi
pip install sanic-skywalking-middleware
```

### 使用

同步使用

```python
from skywalking import agent, config

config.init(agent_collector_backend_services='localhost:11800', agent_protocol='grpc',
            agent_name='great-app-provider-grpc',
            kafka_bootstrap_servers='localhost:9094',  # If you use kafka, set this
            agent_instance_name='instance-01',
            agent_experimental_fork_support=True,
            agent_logging_level='DEBUG',
            agent_log_reporter_active=True,
            agent_meter_reporter_active=True,
            agent_profile_active=True)


agent.start()
```

异步使用

```python
# fastapi
from fastapi_skywalking_middleware.middleware import FastAPISkywalkingMiddleware

app.add_middleware(FastAPISkywalkingMiddleware, collector="10.30.8.116:30799", service='your awesome service', instance=f'your instance name - pid: {os.getpid()}')

# sanic
from sanic_skywalking_middleware import SanicSkywalingMiddleware

SanicSkywalingMiddleware(app, service='Sanic Skywalking Demo Service', collector='127.0.0.1:11800')
```

