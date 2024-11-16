# Nacos

[文档](https://nacos.io/docs/latest/overview/)

## 概述

Nacos 致力于帮助您发现、配置和管理微服务。Nacos 提供了一组简单易用的特性集，帮助您快速实现动态服务发现、服务配置、服务元数据及流量管理。

Nacos 支持几乎所有主流类型的“服务”的发现、配置和管理：

```
Kubernetes Service
gRPC & Dubbo RPC Service
Spring Cloud RESTful Service
```

关键特性：服务发现和服务健康监测、动态配置服务、动态DNS服务、服务及其元数据管理

部署模式：单机模式、集群模式

## 安装部署

### 软件

下载软件包

解压

```
unzip nacos-server-$version.zip
# 或者 tar -xvf nacos-server-$version.tar.gz
cd nacos/bin
```

启动服务

```
# unix
sh startup.sh -m standalone
# windows
startup.cmd -m standalone
```

验证是否ok

```
进入${nacos.home}/logs/ 目录下， 使用tail -f start.out 查看日志，如果看到如下日志，说明服务启动成功。
Nacos started successfully in stand alone mode. use embedded storage
```

关闭服务

```
# unix
sh shutdown.sh
# windows
shutdown.cmd
```

web

```
http://127.0.0.1:8848/nacos
```

### docker

下载

```
git clone https://github.com/nacos-group/nacos-docker.git
cd nacos-docker
```

启动服务

```
docker-compose -f example/standalone-derby.yaml up
```

验证是否ok

```shell
docker logs -f $container_id # 查看日志，看是否有如下信息：Nacos started successfully in stand alone mode. use embedded storage
```

关闭服务

```
docker-compose down
```

### k8s

下载

```
git clone https://github.com/nacos-group/nacos-k8s.git
cd nacos-k8s
```

启动服务

```shell
# 未使用持久化卷，存在数据丢失风险
cd nacos-k8s
chmod +x quick-startup.sh
./quick-startup.sh
```

验证是否ok

```shell
kubectl logs -f $pod_name   # 查看日志，如果看到如下日志，说明服务启动成功。Nacos started successfully in stand alone mode. use embedded storage
```

## 使用

###  openapi

#### 配置管理

配置发布

```
curl -X POST "http://127.0.0.1:8848/nacos/v1/cs/configs?dataId=nacos.cfg.dataId&group=test&content=HelloWorld"

curl -d 'dataId=nacos.example' \
  -d 'group=DEFAULT_GROUP' \
  -d 'namespaceId=public' \
  -d 'content=contentTest' \
  -X POST 'http://127.0.0.1:8848/nacos/v2/cs/config'
```

配置删除

````
curl -X DELETE 'http://127.0.0.1:8848/nacos/v2/cs/config?dataId=nacos.example&group=DEFAULT_GROUP&namespaceId=public'
````

配置获取

```
curl -X GET "http://127.0.0.1:8848/nacos/v1/cs/configs?dataId=nacos.cfg.dataId&group=test"

curl -X GET 'http://127.0.0.1:8848/nacos/v2/cs/config?dataId=nacos.example&group=DEFAULT_GROUP&namespaceId=public'
```

配置查询列表

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/cs/history/list?dataId=nacos.example&group=com.alibaba.nacos&namespaceId='
```

配置查询具体版本历史列表

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/cs/history?dataId=nacos.example&group=com.alibaba.nacos&namespaceId=&nid=203'
```

配置查询上一版本

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/cs/history/previous?id=309135486247505920&dataId=nacos.example&group=com.alibaba.nacos&namespaceId='
```

#### 服务发现

- 实例

实例注册

```
curl -X POST 'http://127.0.0.1:8848/nacos/v1/ns/instance?serviceName=nacos.naming.serviceName&ip=20.18.7.10&port=8080'

curl -d 'serviceName=test_service' \
  -d 'ip=127.0.0.1' \
  -d 'port=8090' \
  -d 'weight=0.9' \
  -d 'ephemeral=true' \
  -X POST 'http://127.0.0.1:8848/nacos/v2/ns/instance'
```

实例注销

```
curl -d 'serviceName=test_service' \
  -d 'ip=127.0.0.1' \
  -d 'port=8090' \
  -d 'weight=0.9' \
  -d 'ephemeral=true' \
  -X DELETE 'http://127.0.0.1:8848/nacos/v2/ns/instance'
```

实例更新

```
curl -d 'serviceName=test_service' \
  -d 'ip=127.0.0.1' \
  -d 'port=8090' \
  -d 'weight=0.9' \
  -d 'ephemeral=true' \
  -X PUT 'http://127.0.0.1:8848/nacos/v2/ns/instance'
```

实例查询详情

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/instance?namespaceId=public&groupName=&serviceName=test_service&ip=127.0.0.1&port=8080'
```

指定服务的实例列表查询

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/instance/list?serviceName=test_service&ip=127.0.0.1'
```

批量更新实例元数据

```
curl -d 'serviceName=test_service' \
  -d 'consistencyType=ephemeral' \
  -d 'instances=[{"ip":"3.3.3.3","port": "8080","ephemeral":"true","clusterName":"xxxx-cluster"},{"ip":"2.2.2.2","port":"8080","ephemeral":"true","clusterName":"xxxx-cluster"}]' \
  -d 'metadata={"age":"20","name":"cocolan"}' \
  -X PUT 'http://127.0.0.1:8848/nacos/v2/ns/instance/metadata/batch'
```

批量删除实例元数据

```
curl -d 'serviceName=test_service' \
  -d 'consistencyType=ephemeral' \
  -d 'instances=[{"ip":"3.3.3.3","port": "8080","ephemeral":"true","clusterName":"xxxx-cluster"},{"ip":"2.2.2.2","port":"8080","ephemeral":"true","clusterName":"xxxx-cluster"}]' \
  -d 'metadata={"age":"20","name":"cocolan"}' \
  -X DELETE 'http://127.0.0.1:8848/nacos/v2/ns/instance/metadata/batch'
```

- 服务

服务创建

```
curl -d 'serviceName=nacos.test.1' \
  -d 'ephemeral=true' \
  -d 'metadata={"k1":"v1"}' \
  -X POST 'http://127.0.0.1:8848/nacos/v2/ns/service'
```

服务删除

```
curl -X DELETE 'http://127.0.0.1:8848/nacos/v2/ns/service?serviceName=nacos.test.1'
```

服务修改

```
curl -d 'serviceName=nacos.test.1' \
  -d 'metadata={"k1":"v2"}' \
  -X PUT 'http://127.0.0.1:8848/nacos/v2/ns/service'
```

服务详情查询

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/service?serviceName=nacos.test.1'
```

服务列表查询

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/service/list'
```

更新实例健康状态

```
curl -d 'serviceName=nacos.test.1' \
  -d 'ip=127.0.0.1' \
  -d 'port=8080' \
  -d 'healthy=false' \
  -X PUT 'http://127.0.0.1:8848/nacos/v2/ns/health/instance'
```

- 客户端

客户端列表查询

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/client/list'
```

客户端详情查询

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/client?clientId=1664527081276_127.0.0.1_4400'
```

客户端注册信息查询

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/client/publish/list?clientId=1664527081276_127.0.0.1_4400'
```

客户端订阅信息查询

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/client/subscribe/list?clientId=1664527081276_127.0.0.1_4400'
```

查询注册指定服务的客户端信息

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/client/service/publisher/list?serviceName=nacos.test.1&ip=&port='
```

查询订阅指定服务的客户端信息

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/ns/client/service/subscriber/list?serviceName=nacos.test.1&ip=&port='
```

#### 命名空间

列表查询

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/console/namespace/list'
```

详情查询

```
curl -X GET 'http://127.0.0.1:8848/nacos/v2/console/namespace?namespaceId=test_namespace'
```

创建

```
curl -d 'namespaceId=test_namespace' \
  -d 'namespaceName=test' \
  -X POST 'http://127.0.0.1:8848/nacos/v2/console/namespace'
```

编辑

```
curl -d 'namespaceId=test_namespace' \
  -d 'namespaceName=test.nacos' \
  -X PUT 'http://127.0.0.1:8848/nacos/v2/console/namespace'
```

删除

```
curl -d 'namespaceId=test_namespace' \
  -X DELETE 'http://127.0.0.1:8848/nacos/v2/console/namespace'
```

### SDK-python

#### 安装

```
pip install nacos-sdk-python
```

配置

```python
client = NacosClient(server_addresses, namespace=your_ns, ak=your_ak, sk=your_sk)
"""
客户端配置
参数：server_addresses/namespace/ak/sk/log_level/log_rotation_backup_count
"""

client.set_options({key}={value})
"""
额外配置
参数：default_timeout/pulling_timeout/pulling_config_size/callback_thread_num/failover_base/snapshot_base/no_snapshot/proxies
"""
```

使用

```python
import nacos

# Both HTTP/HTTPS protocols are supported, if not set protocol prefix default is HTTP, and HTTPS with no ssl check(verify=False)
# "192.168.3.4:8848" or "https://192.168.3.4:443" or "http://192.168.3.4:8848,192.168.3.5:8848" or "https://192.168.3.4:443,https://192.168.3.5:443"
SERVER_ADDRESSES = "server addresses split by comma"
NAMESPACE = "namespace id"

# no auth mode
client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE)
# auth mode
#client = nacos.NacosClient(SERVER_ADDRESSES, namespace=NAMESPACE, ak="{ak}", sk="{sk}")

# get config
data_id = "config.nacos"
group = "group"
print(client.get_config(data_id, group))
```

#### API

```python
# 配置获取
NacosClient.get_config(data_id, group, timeout, no_snapshot)
# 配置增加监听
NacosClient.add_config_watchers(data_id, group, cb_list)
# 配置移除监听
NacosClient.remove_config_watcher(data_id, group, cb, remove_all)
# 配置发布
NacosClient.publish_config(data_id, group, content, timeout)
# 配置删除
NacosClient.remove_config(data_id, group, timeout)


# 服务实例注册
NacosClient.add_naming_instance(service_name, ip, port, cluster_name, weight, metadata, enable, healthy,ephemeral,group_name,heartbeat_interval)
# 服务实例取消注册
NacosClient.remove_naming_instance(service_name, ip, port, cluster_name)
# 服务实例修改
NacosClient.modify_naming_instance(service_name, ip, port, cluster_name, weight, metadata, enable)
# 服务实例查询列表
NacosClient.list_naming_instance(service_name, clusters, namespace_id, group_name, healthy_only)
# 服务实例查询详情
NacosClient.get_naming_instance(service_name, ip, port, cluster_name)
# 服务实例监听
NacosClient.subscribe(listener_fn, listener_interval=7, *args, **kwargs)
# 服务实例取消监听
NacosClient.unsubscribe(service_name, listener_name)
# 所有服务停止监听
NacosClient.stop_subscribe()
# 主动发送心跳
NacosClient.send_heartbeat(service_name, ip, port, cluster_name, weight, metadata)
```

#### 使用

同步使用

```python
# 导入Flask库，用于构建web应用
from flask import Flask

# 导入NacosClient，用于与Nacos服务器交互，实现配置管理和服务发现功能
from nacos import NacosClient

# 初始化Flask应用实例
app = Flask(__name__)

# 设置Nacos服务器地址，请替换为实际的Nacos服务器地址
SERVER_ADDRESSES = "Your nacos server address"

# 设置Nacos命名空间ID，请替换为实际的命名空间ID
NAMESPACE = "Your nacos namespace"

# 设置Nacos用户名和密码，用于认证访问Nacos服务器
USERNAME = 'Your user name'
PASSWORD = 'Your password'

client = NacosClient(server_addresses=SERVER_ADDRESSES, namespace=NAMESPACE, username=USERNAME, password=PASSWORD,
                     log_level='INFO')


# 定义路由，访问根路径'/'时返回的消息，展示来自Nacos的配置信息
@app.route('/')
def hello_world():
    # 使用flask的config对象获取名为"config"的配置项，展示配置内容
    return f'Hello, World! Config from Nacos: {app.config.get("config")}'


def init():
    # 服务注册：让Flask应用在启动时自动注册到Nacos，实现服务发现的自动化。heartbeat_interval可以调整后台心跳间隔时间，默认为5秒。
    client.add_naming_instance(service_name='my-flask-service', ip='localhost', port=5000, heartbeat_interval=5)

    # 设置Nacos中配置数据的数据ID和分组，默认分组为'DEFAULT_GROUP'
    data_id = 'test'
    group = 'DEFAULT_GROUP'

    # 从Nacos获取配置，并更新到Flask应用的config对象中，以便在应用中使用这些配置
    app.config.update(config=client.get_config(data_id=data_id, group=group))

    # 添加配置监听器，当Nacos中的配置发生变化时，自动更新Flask应用的config对象
    client.add_config_watcher(data_id=data_id, group=group,
                              cb=lambda cfg: app.config.update(config=cfg['content']))


if __name__ == '__main__':
    init()
    app.run()
```

- 异步使用

fastapi

```python
# 导入FastAPI库，用于构建API服务
from fastapi import FastAPI
from nacos import NacosClient

app = FastAPI()

# 设置Nacos服务器地址，请替换为实际的Nacos服务器地址
SERVER_ADDRESSES = "Your nacos server address"

# 设置Nacos命名空间ID，请替换为实际的命名空间ID
NAMESPACE = "Your nacos namespace"

# 设置Nacos用户名和密码，用于认证访问Nacos服务器
USERNAME = 'Your user name'
PASSWORD = 'Your password'

client = NacosClient(server_addresses=SERVER_ADDRESSES, namespace=NAMESPACE, username=USERNAME, password=PASSWORD,
                     log_level='INFO')


# 定义一个异步函数，在FastAPI应用启动时执行
@app.on_event("startup")
async def startup_event():
    # 在启动时创建一个任务来初始化配置
    asyncio.create_task(init())


# 通过NacosClient获取配置，并存储在应用的状态(state)中，以便后续使用
async def load_config(data_id, group):
    app.state = {'config_data': client.get_config(data_id=data_id, group=group)}


# 异步函数，用于初始化应用所需的各种配置和监听器
async def init():
    data_id = 'test'
    group = 'DEFAULT_GROUP'
    await load_config(data_id, group)

    def on_config_change(cfg):
        # 当Nacos中的配置发生变化时，更新应用状态中的配置数据
        app.state = {'config_data': cfg['content']}

    client.add_config_watcher(data_id, group, cb=on_config_change)
    client.add_naming_instance(service_name='my-flask-service', ip='localhost', port=8000, heartbeat_interval=5)

# 定义一个GET请求的路由，返回简单的欢迎信息及当前从Nacos获取的配置数据
@app.get("/")
def hello_world():
    return f'Hello, World! Config from Nacos: {app.state["config_data"]}'


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
```

sanic

```
```

