# zeep

[官网](https://docs.python-zeep.org/en/master/index.html)

## 简介

一个快速而现代的Python SOAP客户端

- 亮点

与Python 3.7、3.8、3.9、3.10、3.11和PyPy兼容

基于lxml和请求构建

支持Soap 1.1、Soap 1.2和HTTP绑定

支持WS-Addressing标头

支持WSSE（UserNameToken/x.509签名）

通过httpx支持异步

对XOP消息的实验性支持

## 安装

安装

```
pip install zeep
```

简单示例

```python
from zeep import Client

wsdl = 'http://www.soapclient.com/xml/soapresponder.wsdl'
client = Client(wsdl=wsdl)
print(client.service.Method1('Zeep', 'is cool'))
```

要获得端点上可用服务的概述，可以在终端中运行以下命令

```shell
python -mzeep http://www.soapclient.com/xml/soapresponder.wsdl
```

## Client

`Client`是与SOAP服务器交互的主要接口。它提供了一个`service`属性，该属性引用客户端的默认绑定（通过ServiceProxy对象）。在启动客户端时，可以通过传递`service_name`和`port_name`来指定默认绑定。否则，第一个服务和该服务中的第一个端口将用作默认端口。

### 协议文件缓存

当客户端初始化时，它将自动检索作为参数传递的WSDL文件。此WSDL文件通常引用各种其他WSDL和XSD文件。默认情况下，Zeep不会缓存这些文件，但出于性能原因，建议启用此功能。使用方法见`Transports`

### 配置

Client类接受用于配置客户端的设置参数。您可以使用以下代码初始化对象

```python
from zeep import Client, Settings

settings = Settings(strict=False, xml_huge_tree=True)
client = Client('http://my-wsdl/wsdl', settings=settings)
```

### 异步

AsyncClient允许您以异步方式执行操作。然而，有一个很大的警告：wsdl文档仍然使用同步方法加载。原因是代码库最初并不是为异步使用和支持而编写的，这需要做大量的工作。

要使用异步操作，您需要使用`AsyncClient()`和相应的`AsyncTransport()`（这是AsyncClient的默认传输）

```python
client = zeep.AsyncClient("http://localhost:8000/?wsdl")

response = await client.service.myoperation()
```

### 严格模式

默认情况下，zeep将在“严格”模式下运行。如果您使用的SOAP服务器不符合严格设置的标准，则可以禁用此设置。请参见设置。

禁用严格模式将更改以下行为：1.在启用恢复模式的情况下解析XML；2.xsd:sequences中允许缺少非可选元素

请注意，禁用严格模式应该被视为最后的手段，因为它可能会导致XML和返回的响应之间的数据丢失。

### ServiceProxy

`ServiceProxy`对象是一个简单的对象，它将检查请求的属性或项是否存在操作。如果操作存在，则它将返回一个`OperationProxy`对象（可调用），该对象负责调用绑定上的操作。

```python
from zeep import Client
from zeep import xsd

client = Client('http://my-endpoint.com/production.svc?wsdl')

# service is a ServiceProxy object.  It will check if there
# is an operation with the name `X` defined in the binding
# and if that is the case it will return an OperationProxy
client.service.X()

# The operation can also be called via an __getitem__ call.
# This is useful if the operation name is not a valid
# python attribute name.
client.service['X-Y']()
```

### 无默认绑定

默认情况下，Zeep会选择WSDL中的第一个绑定作为默认绑定。此绑定可通过`client.service`获得。若要使用特定绑定，可以在客户端对象上使用`bind()`方法

```python
from zeep import Client
from zeep import xsd

client = Client('http://my-endpoint.com/production.svc?wsdl')

service2 = client.bind('SecondService', 'Port12')
service2.someOperation(myArg=1)
```

例如，wsdl若是包含如下定义

```xml
<wsdl:service name="ServiceName">
<wsdl:port name="PortName" binding="tns:BasicHttpsBinding_IServiziPartner">
<soap:address location="https://aaa.bbb.ccc/ddd/eee.svc"/>
</wsdl:port>
<wsdl:port name="PortNameAdmin" binding="tns:BasicHttpsBinding_IServiziPartnerAdmin">
<soap:address location="https://aaa.bbb.ccc/ddd/eee.svc/admin"/>
</wsdl:port>
</wsdl:service>
```

并且需要调用`https://aaa.bbb.ccc/ddd/eee.svc/admin`中定义的方法

```python
client = Client("https://www.my.wsdl") # this will use default binding
client_admin = client.bind('ServiceName', 'PortNameAdmin')
client_admin.method1() #this will call method1 defined in service name ServiceName and port PortNameAdmin
```

### 创建新ServiceProxy

在某些情况下，您要么需要更改WSDL中定义的SOAP地址，要么WSDL没有定义任何服务元素。这可以通过使用`Client.create_service()`方法创建一个新的`ServiceProxy`来完成。

```python
from zeep import Client
from zeep import xsd

client = Client('http://my-endpoint.com/production.svc?wsdl')
service = client.create_service(
    '{http://my-target-namespace-here}myBinding',
    'http://my-endpoint.com/acceptance/')

service.submit('something')
```

### 创建原始XML文档

当您希望zeep构建并返回XML而不是将其发送到服务器时，可以使用`Client.create_message()`调用。它要求`ServiceProxy`作为第一个参数，操作名称作为第二个参数。

```python
from zeep import Client

client = Client('http://my-endpoint.com/production.svc?wsdl')
node = client.create_message(client.service, 'myOperation', user='hi')
```

## 设置

您可以在客户端上将各种选项直接设置为属性，也可以通过上下文管理器进行设置。

例如，要让zeep直接返回原始响应而不是处理它，可以执行以下操作：

```python
from zeep import Client
from zeep import xsd

client = Client('http://my-endpoint.com/production.svc?wsdl')

with client.settings(raw_response=True):
    response = client.service.myoperation()
    # response is now a regular requests.Response object
    assert response.status_code == 200
    assert response.content
```

## Transport

如果需要更改缓存、超时或TLS（或SSL）验证等选项，则需要自己创建·Transport·类的实例。

### TLS

如果您需要验证TLS连接（以防您的主机有自签名证书），最好的方法是创建请求。会话实例，并将信息添加到该会话中，使其保持持久性：

```python
from requests import Session
from zeep import Client
from zeep.transports import Transport

session = Session()
session.verify = 'path/to/my/certificate.pem'
transport = Transport(session=session)
client = Client(
    'http://my.own.sslhost.local/service?WSDL',
    transport=transport)
```

禁用TLS

```python
session = Session()
session.verify = False
# 或者
client.transport.session.verify = False
```

### 会话超时

要设置用于加载wsdl sfn xsd文档的传输超时，请使用timeout选项。默认超时为300秒：

```python
from zeep import Client
from zeep.transports import Transport

transport = Transport(timeout=10)
client = Client(
    'http://www.webservicex.net/ConvertSpeed.asmx?WSDL',
    transport=transport)
```

要将超时传递给底层`POST/GET`请求，请使用`operation_timeout`。此选项默认为`None`。

### HTTP/SOCKS

默认情况下，zeep使用`requests`作为传输层，`requests`允许使用`requests.Session`的`proxies`属性定义代理。

```python
from zeep import Client

client = Client(
    'http://my.own.sslhost.local/service?WSDL')

client.transport.session.proxies = {
    # Utilize for all http/https connections
    'http': 'foo.bar:3128',
    'https': 'foo.bar:3128',
    # Utilize for certain URL
    'http://specific.host.example': 'foo.bar:8080',
    # Or use socks5 proxy (requires requests[socks])
    'https://socks5-required.example': 'socks5://foo.bar:8888',
}
```

### 缓存

默认情况下，zeep不使用缓存后端。为了提高性能，建议使用`SqliteCache`后端。默认情况下，它缓存WSDL和XSD文件1小时。要使用缓存后端初始化客户端，请执行以下操作：

```python
from zeep import Client
from zeep.cache import SqliteCache
from zeep.transports import Transport

transport = Transport(cache=SqliteCache())
client = Client(
    'http://www.webservicex.net/ConvertSpeed.asmx?WSDL',
    transport=transport)
```

可以通过以下方式更改SqliteCache设置：

```python
from zeep import Client
from zeep.cache import SqliteCache
from zeep.transports import Transport
cache = SqliteCache(path='/tmp/sqlite.db', timeout=60)
transport = Transport(cache=cache)
client = Client(
    'http://www.webservicex.net/ConvertSpeed.asmx?WSDL',
    transport=transport)
```

另一种选择是使用`InMemoryCach`e后端。它在内部使用全局dict来存储带有相应内容的url。

```python
from zeep import Client
from zeep.cache import InMemoryCache
from zeep.transports import Transport

cache = InMemoryCache(timeout=60)
with open(cache_file, 'rb') as f:
    cache.add('http://schemas.xmlsoap.org/soap/encoding/', f.read())
transport = Transport(cache=cache)
client = Client(
    'http://www.webservicex.net/ConvertSpeed.asmx?WSDL',
    transport=transport)

```

### HTTP认证

虽然一些提供程序在SOAP消息的标头中包含了安全功能，但其他提供程序则使用HTTP身份验证标头。在后一种情况下，您可以只创建一个请求。具有身份验证集的会话对象，并将其传递给传输类。

```python
from requests import Session
from requests.auth import HTTPBasicAuth  # or HTTPDigestAuth, or OAuth1, etc.
from zeep import Client
from zeep.transports import Transport

session = Session()
session.auth = HTTPBasicAuth(user, password)
client = Client('http://my-endpoint.com/production.svc?wsdl',
    transport=Transport(session=session))
```

### 异步HTTP认证

zeep的异步客户端使用不同的后端，因此在这种情况下设置不同。需要使用`httpx`来创建`httpx.AsyncClient`对象，并将其传递给`zeep.AsyncTransport`。

```python
import httpx
import zeep
from zeep.transports import AsyncTransport

USER = 'username'
PASSWORD = 'password'

httpx_client = httpx.AsyncClient(auth=(USER, PASSWORD))

aclient = zeep.AsyncClient(
    "http://my-endpoint.com/production.svc?wsdl",
    transport=AsyncTransport(client=httpx_client)
)
```

### 日志信息

- debug

要查看发送到远程服务器的SOAP XML消息和接收到的响应，可以将`zeep.transports`模块的Python记录器级别设置为DEBUG。

```python
import logging.config

logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'verbose': {
            'format': '%(name)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'zeep.transports': {
            'level': 'DEBUG',
            'propagate': True,
            'handlers': ['console'],
        },
    }
})
```

- 插件获取

从0.15版本开始可以通过`HistoryPlugin`实现。

`HistoryPlugin`插件保存发送和接收请求的列表。默认情况下，最多保留一个事务（发送/接收）。但是，当您通过传递`maxlen`参数来创建插件时，可以更改这一点。

```python
from zeep import Client
from zeep.plugins import HistoryPlugin
from lxml import etree

history = HistoryPlugin()
client = Client(
    'http://examples.python-zeep.org/basic.wsdl',
    plugins=[history])
client.service.DoSomething()

last_sent = history.last_sent
last_received = history.last_received
envelop = last_sent.get("envelop")
http_header = last_sent.get("http_header")
print(etree.tostring(envelop, pretty_print=True))
print(etree.tostring(envelop, pretty_print=True))
```

